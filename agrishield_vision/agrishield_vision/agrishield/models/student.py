"""The edge-tier student model.

Backbone choice, stated plainly
-------------------------------
Default: ``mobilenetv4_conv_small`` (~3.8M params). Conservative fallback:
``efficientnet_lite0``. Both are pure-convolution graphs, which matters more
than their paper accuracy for this deployment:

* A large share of Indian smartphones in the target segment run TFLite on CPU
  via XNNPACK. GPU/NNAPI delegates are inconsistent across low-end SoCs and
  vendor drivers, so CPU is the number you must design against.
* Mobile ViT-family models (MobileViT, EfficientFormer) look competitive on
  paper but their attention blocks quantise worse to int8, export less cleanly,
  and are slower on CPU at equal accuracy. They are a reasonable choice when you
  control the hardware. You do not.

The ViT capability we actually want — robust features under domain shift — is
imported through *distillation* from a frozen DINOv2 teacher rather than by
putting a transformer on the phone. That is the central architectural bet of
this design.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import timm
import torch
import torch.nn as nn

from .heads import HeadOutput, HierarchicalHybridHead, ProjectionHead

SAFE_BACKBONES = {
    # name -> (timm id, approx params M, note)
    "mnv4s": ("mobilenetv4_conv_small.e2400_r224_in1k", 3.8, "default; best acc/latency on CPU"),
    "lite0": ("efficientnet_lite0.ra_in1k", 4.7, "most export-friendly; use if conversion fights you"),
    "mnv3": ("mobilenetv3_large_100.ra_in1k", 5.5, "widest device support, oldest"),
    "effv2b0": ("tf_efficientnetv2_b0.in1k", 7.1, "if you can afford ~7MB int8"),
}


@dataclass
class StudentConfig:
    backbone: str = "mnv4s"
    n_classes: int = 48
    n_crops: int = 11
    class_to_crop: Sequence[int] = ()
    n_prototypes: int = 4
    teacher_dim: int = 1024       # DINOv2 ViT-L/14 CLS dimension
    drop_path_rate: float = 0.1
    dropout: float = 0.1
    pretrained: bool = True


class AgriShieldStudent(nn.Module):
    """Mobile backbone -> hierarchical hybrid head (+ distillation projection)."""

    def __init__(self, config: StudentConfig) -> None:
        super().__init__()
        self.config = config
        timm_id = SAFE_BACKBONES.get(config.backbone, (config.backbone, 0, ""))[0]

        self.backbone = timm.create_model(
            timm_id,
            pretrained=config.pretrained,
            num_classes=0,              # return pooled features
            drop_path_rate=config.drop_path_rate,
        )
        self.embed_dim: int = self.backbone.num_features

        self.head = HierarchicalHybridHead(
            dim=self.embed_dim,
            n_classes=config.n_classes,
            n_crops=config.n_crops,
            class_to_crop=config.class_to_crop,
            n_prototypes=config.n_prototypes,
            dropout=config.dropout,
        )
        self.projection = ProjectionHead(self.embed_dim, config.teacher_dim)

    def forward(self, images: torch.Tensor) -> HeadOutput:
        embedding = self.backbone(images)
        return self.head(embedding)

    def project(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.projection(embedding)

    # ---------------------------------------------------------- utilities
    def param_groups(self, backbone_lr: float, head_lr: float, weight_decay: float = 0.05):
        """Discriminative learning rates: the trunk is pretrained, the head is not."""
        decay, no_decay = [], []
        for name, param in self.backbone.named_parameters():
            if not param.requires_grad:
                continue
            (no_decay if param.ndim <= 1 or name.endswith(".bias") else decay).append(param)
        head_params = list(self.head.parameters()) + list(self.projection.parameters())
        return [
            {"params": decay, "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": no_decay, "lr": backbone_lr, "weight_decay": 0.0},
            {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
        ]

    def freeze_backbone(self, freeze: bool = True) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = not freeze


class ExportWrapper(nn.Module):
    """Deployment graph: uint8-friendly input, three tensors out.

    Normalisation is folded *into* the graph so the Android side only has to
    hand over a resized RGB float tensor in [0, 1]. Every preprocessing step
    left outside the model is a step someone will implement differently in
    Kotlin and silently lose two points of accuracy on.

    Outputs are deliberately pre-softmax plus the raw open-set score, so the app
    can apply the calibrated temperature and the OOD thresholds itself (both are
    scalars shipped in a JSON sidecar and tunable without re-exporting).
    """

    def __init__(
        self,
        model: AgriShieldStudent,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        self.backbone = model.backbone
        self.head = model.head
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """images: [B, 3, H, W] float in [0, 1]."""
        x = (images - self.mean) / self.std
        out = self.head(self.backbone(x))
        return out.class_logits, out.class_log_probs, out.max_prototype_cosine


def build_student(config: StudentConfig) -> AgriShieldStudent:
    return AgriShieldStudent(config)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "AgriShieldStudent",
    "StudentConfig",
    "ExportWrapper",
    "build_student",
    "count_parameters",
    "SAFE_BACKBONES",
]
