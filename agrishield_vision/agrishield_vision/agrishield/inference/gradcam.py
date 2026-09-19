"""Grad-CAM++ for the mobile backbone.

Why Grad-CAM++ and not plain Grad-CAM
-------------------------------------
Plain Grad-CAM averages gradients over the spatial map, which biases it toward
one dominant blob. Leaf disease is usually *multi-instance*: a dozen small
Septoria spots scattered over a leaf. Grad-CAM++ weights each spatial position
by a second-order term, which recovers all the lesions instead of the biggest
one. For the Round-1 judge feedback — "show the spot, not just the label" —
that difference is the whole point.

Honest caveat to keep in the pitch
----------------------------------
A CAM is an *explanation of the model*, not a segmentation of the pathology. On
cross-domain images, CAMs have been shown to drift off the lesion and onto the
background — which is diagnostic of the domain-shift problem, not evidence the
model found the disease. So: use the CAM as a trust signal in the app, and if
you want a defensible lesion mask, train a real segmentation head on PlantSeg
masks (see ARCHITECTURE.md, server tier). Do not claim the CAM is a lesion map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CAMResult:
    heatmap: np.ndarray          # [H, W] in [0, 1], at input resolution
    class_index: int
    peak_boxes: List[Tuple[int, int, int, int]]  # (x1, y1, x2, y2) lesion candidates
    coverage: float              # fraction of frame above threshold


class GradCAMPlusPlus:
    """Hook-based Grad-CAM++ for any CNN backbone.

    Usage::

        cam = GradCAMPlusPlus(student, target_layer=student.backbone.conv_head)
        result = cam(image_tensor, class_index=None)   # None -> argmax
        cam.close()
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, _module, _inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()

    def __call__(
        self,
        image: torch.Tensor,
        class_index: Optional[int] = None,
        threshold: float = 0.45,
    ) -> CAMResult:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        output = self.model(image)
        logits = output.class_logits if hasattr(output, "class_logits") else output
        if class_index is None:
            class_index = int(logits.argmax(dim=-1).item())
        logits[0, class_index].backward(retain_graph=False)

        activations = self.activations  # [1, C, h, w]
        gradients = self.gradients      # [1, C, h, w]
        if activations is None or gradients is None:
            raise RuntimeError("no activations captured - is target_layer inside the model?")

        # Grad-CAM++ alpha coefficients.
        grad2 = gradients.pow(2)
        grad3 = grad2 * gradients
        sum_activations = activations.sum(dim=(2, 3), keepdim=True)
        denominator = 2.0 * grad2 + sum_activations * grad3
        denominator = torch.where(denominator != 0, denominator, torch.ones_like(denominator))
        alpha = grad2 / denominator

        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)

        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        heatmap = cam.cpu().numpy()

        boxes = _peak_boxes(heatmap, threshold)
        return CAMResult(
            heatmap=heatmap,
            class_index=class_index,
            peak_boxes=boxes,
            coverage=float((heatmap > threshold).mean()),
        )


def _peak_boxes(heatmap: np.ndarray, threshold: float, min_area: int = 64) -> List[Tuple[int, int, int, int]]:
    """Connected components above threshold -> lesion candidate boxes.

    These boxes are what the app actually draws. A soft heatmap looks
    impressive in a demo but a farmer needs discrete markers they can count
    ("we found 6 spots"), and a count is the quantitative severity signal the
    judges asked for.
    """
    mask = heatmap > threshold
    if not mask.any():
        return []
    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    boxes: List[Tuple[int, int, int, int]] = []

    for sy in range(height):
        for sx in range(width):
            if not mask[sy, sx] or visited[sy, sx]:
                continue
            stack = [(sy, sx)]
            visited[sy, sx] = True
            ys, xs = [], []
            while stack:
                y, x = stack.pop()
                ys.append(y)
                xs.append(x)
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if len(ys) >= min_area:
                boxes.append((min(xs), min(ys), max(xs), max(ys)))
    return sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))[:12]


def overlay_heatmap(
    image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45
) -> np.ndarray:
    """Blend a turbo-ish colormap over the RGB image without pulling in matplotlib."""
    h = np.clip(heatmap, 0, 1)
    red = np.clip(1.5 - np.abs(4 * h - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * h - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * h - 1), 0, 1)
    colour = np.stack([red, green, blue], axis=-1)
    base = image_rgb.astype(np.float32) / 255.0 if image_rgb.dtype == np.uint8 else image_rgb
    mixed = (1 - alpha) * base + alpha * colour
    return (np.clip(mixed, 0, 1) * 255).astype(np.uint8)


__all__ = ["GradCAMPlusPlus", "CAMResult", "overlay_heatmap"]
