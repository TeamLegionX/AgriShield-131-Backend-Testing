"""Loss functions.

Loss selection, with reasons rather than folklore:

* **Label smoothing (0.1) on the hierarchical NLL** is the default classification
  term. It is doing double duty here: it improves accuracy slightly, and it
  directly attacks over-confidence, which is the specific failure mode that
  makes a lab-trained crop model dangerous in a field. A documented cross-domain
  study found a model losing 67.7 accuracy points under controlled-to-field
  shift while its mean predicted confidence stayed near 80%. A farmer cannot
  tell those two situations apart; the calibration stack starts here.

* **Focal loss is optional, not default.** Focal helps when the imbalance is
  extreme *and* the hard examples are genuine. In merged plant-disease data a
  large share of hard examples are mislabelled scraped images, and focal loss
  will faithfully devote capacity to fitting that noise. Prefer class-balanced
  sampling first; reach for focal only if tail-class recall is still flat.

* **CutMix, not MixUp, and only within a crop.** Blending a mango leaf with a
  rice leaf creates an image whose label is a lie. CutMix pasting a lesion patch
  into another leaf of the same crop is at least physically plausible and is a
  good proxy for multi-infection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalNLL(nn.Module):
    """NLL over hierarchical log-probs + an auxiliary crop-level CE term."""

    def __init__(self, label_smoothing: float = 0.1, crop_weight: float = 0.3) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing
        self.crop_weight = crop_weight

    def forward(
        self,
        class_log_probs: torch.Tensor,   # [B, N]
        crop_logits: torch.Tensor,       # [B, G]
        class_target: torch.Tensor,      # [B]
        crop_target: torch.Tensor,       # [B]
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n_classes = class_log_probs.size(-1)
        eps = self.label_smoothing
        nll = -class_log_probs.gather(1, class_target.unsqueeze(1)).squeeze(1)
        if eps > 0:
            smooth = -class_log_probs.mean(dim=-1)
            nll = (1.0 - eps) * nll + eps * smooth
        if sample_weight is not None:
            nll = nll * sample_weight
        loss = nll.mean()

        if self.crop_weight > 0:
            loss = loss + self.crop_weight * F.cross_entropy(
                crop_logits, crop_target, label_smoothing=eps
            )
        return loss


class FocalLoss(nn.Module):
    """Focal loss on log-probabilities. Off by default; see module docstring."""

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else torch.tensor([]))

    def forward(self, log_probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        p = logp.exp()
        loss = -((1.0 - p) ** self.gamma) * logp
        if self.alpha.numel():
            loss = loss * self.alpha.to(loss.device)[target]
        return loss.mean()


class DistillationLoss(nn.Module):
    """KL(teacher || student) on logits + cosine feature matching.

    Two channels, because they transfer different things:

    * **Logit KD** (temperature T, gradient rescaled by T^2) transfers the
      teacher's *dark knowledge*: that early blight is 60% likely but late
      blight is a real 25% possibility. That ranking structure is exactly what a
      farmer-facing "also consider" list needs, and hard labels destroy it.

    * **Feature KD** (cosine between the projected student embedding and the
      teacher CLS vector) transfers the representation geometry, which is where
      the domain robustness actually lives. This term is what lets unlabeled
      field photos contribute.
    """

    def __init__(
        self,
        temperature: float = 3.0,
        logit_weight: float = 1.0,
        feature_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.logit_weight = logit_weight
        self.feature_weight = feature_weight

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_projection: Optional[torch.Tensor] = None,
        teacher_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t = self.temperature
        kd = F.kl_div(
            F.log_softmax(student_logits / t, dim=-1),
            F.log_softmax(teacher_logits / t, dim=-1),
            reduction="batchmean",
            log_target=True,
        ) * (t * t)
        loss = self.logit_weight * kd

        if self.feature_weight > 0 and student_projection is not None and teacher_features is not None:
            cosine = F.cosine_similarity(student_projection, teacher_features.detach(), dim=-1)
            loss = loss + self.feature_weight * (1.0 - cosine).mean()
        return loss


def cutmix_within_crop(
    images: torch.Tensor,
    class_target: torch.Tensor,
    crop_target: torch.Tensor,
    alpha: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """CutMix restricted to pairs from the same crop.

    Returns ``(mixed_images, target_a, target_b, lam)``. Samples with no
    same-crop partner in the batch are paired with themselves (a no-op mix).
    """
    batch = images.size(0)
    lam = float(torch.distributions.Beta(alpha, alpha).sample()) if alpha > 0 else 1.0

    permutation = torch.arange(batch, device=images.device)
    for crop in crop_target.unique():
        idx = (crop_target == crop).nonzero(as_tuple=True)[0]
        if idx.numel() > 1:
            permutation[idx] = idx[torch.randperm(idx.numel(), device=images.device)]

    height, width = images.shape[-2:]
    cut_ratio = (1.0 - lam) ** 0.5
    cut_h, cut_w = int(height * cut_ratio), int(width * cut_ratio)
    if cut_h == 0 or cut_w == 0:
        return images, class_target, class_target, torch.tensor(1.0, device=images.device)

    cy = int(torch.randint(height, (1,)).item())
    cx = int(torch.randint(width, (1,)).item())
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, height)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, width)

    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[permutation][:, :, y1:y2, x1:x2]
    true_lam = 1.0 - ((y2 - y1) * (x2 - x1) / (height * width))
    return mixed, class_target, class_target[permutation], torch.tensor(true_lam, device=images.device)


@dataclass
class LossWeights:
    """Per-phase weights. Phase 2 is where the domain-shift win is bought."""

    supervised: float = 1.0
    distillation: float = 1.0
    prototype_separation: float = 0.05


__all__ = [
    "HierarchicalNLL",
    "FocalLoss",
    "DistillationLoss",
    "cutmix_within_crop",
    "LossWeights",
]
