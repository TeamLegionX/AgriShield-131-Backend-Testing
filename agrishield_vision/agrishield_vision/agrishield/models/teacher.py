"""The frozen foundation-model teacher (and the optional server tier).

Why a *frozen* teacher rather than a fine-tuned one
---------------------------------------------------
Fine-tuning a ViT-L on 30k mixed lab/field images with free-tier compute is both
slow and a reliable way to overfit to lab backgrounds. Frozen self-supervised
features plus a small head is the better trade on this problem: on the largest
in-the-wild benchmark (PlantWild, 18.5k images / 89 classes), a frozen DINOv2
ViT-L/14 with a light head reaches ~77-78% top-1, above the text-augmented
CLIP-prototype baseline at 76.2%, and concatenating DINOv2 + DINOv3 + CLIP
reaches ~80%. Crucially the whole sweep runs in about a day on one consumer GPU
because the features are extracted once and cached.

So the teacher is: extract features once (a few GPU-hours), train a ~1M-param
head in minutes, and you have both (a) your server tier and (b) the source of
soft targets for the mobile student.

Multi-backbone fusion is supported but off by default: it roughly doubles or
triples feature-extraction time and storage for about +2.5 points. Turn it on
only once the single-backbone pipeline is green end to end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .heads import HierarchicalHybridHead

TEACHER_BACKBONES: Dict[str, str] = {
    "dinov2_l": "vit_large_patch14_dinov2.lvd142m",
    "dinov2_b": "vit_base_patch14_dinov2.lvd142m",     # use on a free-tier T4
    "clip_l": "vit_large_patch14_clip_224.openai",
    "convnext_l": "convnext_large.fb_in22k_ft_in1k",
}


@dataclass
class TeacherConfig:
    backbones: Sequence[str] = field(default_factory=lambda: ["dinov2_l"])
    image_size: int = 224
    batch_size: int = 32
    amp_dtype: torch.dtype = torch.float16
    l2_normalise: bool = True


class FrozenEncoderBank(nn.Module):
    """One or more frozen backbones; returns L2-normalised, concatenated CLS features."""

    def __init__(self, config: TeacherConfig) -> None:
        super().__init__()
        self.config = config
        self.encoders = nn.ModuleList()
        self.dims: List[int] = []
        for key in config.backbones:
            timm_id = TEACHER_BACKBONES.get(key, key)
            model = timm.create_model(timm_id, pretrained=True, num_classes=0)
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            self.encoders.append(model)
            self.dims.append(model.num_features)

    @property
    def output_dim(self) -> int:
        return sum(self.dims)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = []
        for encoder in self.encoders:
            feat = encoder(images)
            if self.config.l2_normalise:
                feat = F.normalize(feat, dim=-1)
            features.append(feat)
        return torch.cat(features, dim=-1)


class TeacherModel(nn.Module):
    """Frozen encoder bank + trainable hierarchical hybrid head."""

    def __init__(
        self,
        encoder: FrozenEncoderBank,
        n_classes: int,
        n_crops: int,
        class_to_crop: Sequence[int],
        n_prototypes: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = HierarchicalHybridHead(
            dim=encoder.output_dim,
            n_classes=n_classes,
            n_crops=n_crops,
            class_to_crop=class_to_crop,
            n_prototypes=n_prototypes,
            dropout=0.2,
        )

    def forward(self, images: torch.Tensor):
        with torch.no_grad():
            features = self.encoder(images)
        return self.head(features)

    def forward_features(self, features: torch.Tensor):
        """Head-only forward, for training against a cached feature matrix."""
        return self.head(features)


# --------------------------------------------------------------------------
# Feature caching - the thing that makes this affordable on free-tier compute
# --------------------------------------------------------------------------
@torch.no_grad()
def cache_features(
    encoder: FrozenEncoderBank,
    loader: DataLoader,
    out_path: str | Path,
    device: str = "cuda",
    amp: bool = True,
) -> Path:
    """Run every image through the frozen bank once and memory-map the result.

    Cost model that actually matters on a free Colab T4: DINOv2 ViT-L/14 at
    224px runs at roughly 60-110 images/s in fp16. 60k images is therefore
    ~10-17 minutes of GPU time, once. After that every head experiment — head
    type, prototype count K, class weighting, calibration — is seconds, not
    hours. Do this before you touch any hyperparameter.
    """
    encoder = encoder.to(device).eval()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunks: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    crops: List[np.ndarray] = []

    autocast = torch.autocast(device_type=device.split(":")[0], dtype=torch.float16, enabled=amp)
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with autocast:
            feats = encoder(images)
        chunks.append(feats.float().cpu().numpy().astype(np.float16))
        if "class_index" in batch:
            labels.append(batch["class_index"].numpy())
            crops.append(batch["crop_index"].numpy())

    payload = {"features": np.concatenate(chunks, axis=0)}
    if labels:
        payload["class_index"] = np.concatenate(labels, axis=0)
        payload["crop_index"] = np.concatenate(crops, axis=0)
    np.savez(out_path, **payload)
    print(f"[teacher] cached {payload['features'].shape} -> {out_path}")
    return out_path


def load_cached_features(path: str | Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key] for key in data.files}


@torch.no_grad()
def teacher_soft_targets(
    teacher: TeacherModel,
    images: torch.Tensor,
    temperature: float = 3.0,
) -> torch.Tensor:
    """Soft class distribution used as the distillation target."""
    out = teacher(images)
    return F.softmax(out.class_logits / temperature, dim=-1)


__all__ = [
    "TeacherConfig",
    "FrozenEncoderBank",
    "TeacherModel",
    "cache_features",
    "load_cached_features",
    "teacher_soft_targets",
    "TEACHER_BACKBONES",
]
