"""Classifier heads.

Three ideas are combined here, each earning its place:

1. **Prototype head.** Instead of one weight vector per class, learn K
   prototypes per class and score by maximum cosine similarity. On frozen
   self-supervised features this consistently beats a plain linear probe for
   in-the-wild plant disease, because a single class ("rice blast") genuinely
   has multiple visual modes (early pinpoint lesions vs. coalesced spindle
   lesions). Published ablations put the sweet spot at K in {2, 4}; larger K
   over-parameterises and degrades.

2. **Hybrid linear + prototype.** A fixed 0.5/0.5 mixture. The linear path wins
   where a class has many training images; the prototype path carries the tail
   classes. Adaptive per-class routing has been shown to add ~nothing, so we do
   not pay its complexity or its export cost.

3. **Hierarchical crop -> disease factorisation.**
       log p(class c) = log p(crop g(c)) + log p(c | crop g(c))
   This buys three concrete things: crop identity is far easier than disease and
   regularises the shared trunk; an impossible answer ("rice blast" on a mango
   leaf) becomes structurally unlikely rather than merely improbable; and the
   app can degrade gracefully to "this is a rice leaf, disease unclear", which
   is a genuinely useful answer for a farmer.

Cosine prototypes carry a bonus: max cosine similarity to any prototype is a
free, well-behaved open-set score (see inference/ood.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HeadOutput:
    """Everything downstream (loss, OOD, calibration) needs from one forward."""

    class_logits: torch.Tensor        # [B, N]  hybrid, pre-softmax, used for energy
    crop_logits: torch.Tensor         # [B, G]
    class_log_probs: torch.Tensor     # [B, N]  hierarchical, normalised over all N
    max_prototype_cosine: torch.Tensor  # [B]   open-set score in [-1, 1]
    embedding: torch.Tensor           # [B, D]


class PrototypeHead(nn.Module):
    """Cosine-similarity classifier with K learnable prototypes per class."""

    def __init__(self, dim: int, n_classes: int, n_prototypes: int = 4, scale: float = 16.0) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.n_prototypes = n_prototypes
        self.prototypes = nn.Parameter(torch.randn(n_classes, n_prototypes, dim) * 0.02)
        # Learnable temperature, stored in log space so it stays positive.
        self.log_scale = nn.Parameter(torch.tensor(float(scale)).log())

    def forward(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits [B, N], max cosine over all classes/prototypes [B])."""
        e = F.normalize(embedding, dim=-1)                      # [B, D]
        p = F.normalize(self.prototypes, dim=-1)                # [N, K, D]
        cos = torch.einsum("bd,nkd->bnk", e, p)                 # [B, N, K]
        per_class = cos.amax(dim=-1)                            # [B, N]
        logits = self.log_scale.exp() * per_class
        return logits, per_class.amax(dim=-1)

    def separation_loss(self) -> torch.Tensor:
        """Push prototypes of *different* classes apart.

        Without this, prototypes of look-alike diseases collapse onto each other
        and the cosine OOD score loses its discriminative power.
        """
        p = F.normalize(self.prototypes, dim=-1).flatten(0, 1)  # [N*K, D]
        gram = p @ p.t()
        n = gram.size(0)
        same_class = torch.zeros(n, n, dtype=torch.bool, device=p.device)
        for c in range(self.n_classes):
            lo, hi = c * self.n_prototypes, (c + 1) * self.n_prototypes
            same_class[lo:hi, lo:hi] = True
        off = gram.masked_fill(same_class, -1.0)
        return off.clamp(min=0.0).pow(2).mean()


class HierarchicalHybridHead(nn.Module):
    """Hybrid (linear + prototype) class scores, factorised through crop."""

    def __init__(
        self,
        dim: int,
        n_classes: int,
        n_crops: int,
        class_to_crop: Sequence[int],
        n_prototypes: int = 4,
        linear_weight: float = 0.5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if len(class_to_crop) != n_classes:
            raise ValueError("class_to_crop must have one entry per class")

        self.n_classes = n_classes
        self.n_crops = n_crops
        self.linear_weight = linear_weight

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(dim, n_classes)
        self.prototype = PrototypeHead(dim, n_classes, n_prototypes)
        self.crop_linear = nn.Linear(dim, n_crops)

        # [G, N] boolean membership mask, registered so it moves with .to(device)
        # and is baked into the exported graph.
        mask = torch.zeros(n_crops, n_classes, dtype=torch.bool)
        for class_index, crop_index in enumerate(class_to_crop):
            mask[crop_index, class_index] = True
        self.register_buffer("group_mask", mask, persistent=False)
        self.register_buffer(
            "class_to_crop_index", torch.as_tensor(list(class_to_crop), dtype=torch.long),
            persistent=False,
        )

    def forward(self, embedding: torch.Tensor) -> HeadOutput:
        x = self.dropout(embedding)
        linear_logits = self.linear(x)
        proto_logits, max_cos = self.prototype(x)
        class_logits = self.linear_weight * linear_logits + (1.0 - self.linear_weight) * proto_logits
        crop_logits = self.crop_linear(x)

        class_log_probs = self._hierarchical_log_probs(class_logits, crop_logits)
        return HeadOutput(
            class_logits=class_logits,
            crop_logits=crop_logits,
            class_log_probs=class_log_probs,
            max_prototype_cosine=max_cos,
            embedding=embedding,
        )

    def _hierarchical_log_probs(
        self, class_logits: torch.Tensor, crop_logits: torch.Tensor
    ) -> torch.Tensor:
        """log p(c) = log p(crop) + log softmax over classes *within* that crop.

        Implemented with a dense [B, G, N] masked logsumexp rather than
        ``scatter_reduce``. G and N are small (tens), the tensor is tiny, and
        every op here (broadcast, masked_fill, logsumexp, gather) survives
        torch.export -> TFLite, which scatter-based versions do not reliably do.
        """
        batch = class_logits.size(0)
        crop_log_probs = F.log_softmax(crop_logits, dim=-1)                 # [B, G]

        expanded = class_logits.unsqueeze(1).expand(batch, self.n_crops, self.n_classes)
        masked = expanded.masked_fill(~self.group_mask.unsqueeze(0), -1e4)
        group_lse = torch.logsumexp(masked, dim=-1)                          # [B, G]

        owner = self.class_to_crop_index.unsqueeze(0).expand(batch, self.n_classes)
        within = class_logits - group_lse.gather(1, owner)                   # log p(c | crop)
        return crop_log_probs.gather(1, owner) + within


class ProjectionHead(nn.Module):
    """Projects the student embedding into the teacher's feature space.

    Used only by the feature-matching term of the distillation loss and dropped
    before export, so it costs the mobile model nothing.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 1024) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


__all__ = ["PrototypeHead", "HierarchicalHybridHead", "ProjectionHead", "HeadOutput"]
