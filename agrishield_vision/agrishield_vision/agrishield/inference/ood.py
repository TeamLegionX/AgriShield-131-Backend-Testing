"""Open-set rejection.

Four rejection cases, and they are genuinely different problems
---------------------------------------------------------------
  R1  Not a plant at all (a hand, a cow, a floor, a screenshot).
  R2  A plant we do not support (an unsupported crop).
  R3  A supported crop with a disease outside the label set.
  R4  A supported crop and disease, but the photo is unusable (blur, dark, tiny).

R1 and R4 are *far* easier than R2 and R3, and honest reporting says so. R1 is
handled mostly by the leaf detector and is near-solved (AUROC typically
0.90-0.97). R4 is handled by a deterministic quality gate with no model at all.
R3 is near-OOD and hard: an unknown lesion on a tomato leaf looks like a known
lesion on a tomato leaf, and published numbers for near-OOD in fine-grained
settings sit closer to 0.70-0.80 AUROC. Claim that range, not 0.95.

Scores used, and why
--------------------
* **Energy** ``E(x) = -T * logsumexp(logits / T)``. Strictly better than max
  softmax probability: softmax is shift-invariant and throws away the magnitude
  information that actually separates in- from out-of-distribution. The original
  result reduced FPR@95TPR by ~18 points versus MSP on a standard benchmark, at
  zero extra cost.
* **Max prototype cosine**, free from the prototype head. Feature-space,
  complementary to the logit-space energy score.
* **Mahalanobis** distance to the nearest class-conditional Gaussian with a
  shared covariance. Strongest of the three on near-OOD, but needs the training
  feature matrix, so it lives in the *server* tier only.

They are fused as a z-scored sum with weights fitted on the field validation
split. Thresholds are set at a target true-positive rate on in-distribution
data, i.e. "reject at most 5% of good photos", which is the constraint a farmer
actually cares about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------- raw scores
def energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Lower energy = more in-distribution. Returned negated so higher = more ID."""
    return temperature * torch.logsumexp(logits / temperature, dim=-1)


def max_softmax_probability(logits: torch.Tensor) -> torch.Tensor:
    """Baseline only. Kept so the ablation table has an honest comparison row."""
    return F.softmax(logits, dim=-1).amax(dim=-1)


class MahalanobisScorer:
    """Class-conditional Gaussian with tied covariance, fitted on train features."""

    def __init__(self, shrinkage: float = 0.1) -> None:
        self.shrinkage = shrinkage
        self.means: Optional[np.ndarray] = None
        self.precision: Optional[np.ndarray] = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "MahalanobisScorer":
        features = features.astype(np.float64)
        classes = np.unique(labels)
        dim = features.shape[1]

        means = np.zeros((len(classes), dim))
        centred = np.empty_like(features)
        for i, cls in enumerate(classes):
            mask = labels == cls
            means[i] = features[mask].mean(axis=0)
            centred[mask] = features[mask] - means[i]

        covariance = centred.T @ centred / max(1, len(features) - len(classes))
        # Ledoit-Wolf style shrinkage keeps the inverse stable when some classes
        # have only a few dozen field examples, which is the normal case here.
        covariance = (1 - self.shrinkage) * covariance + self.shrinkage * np.trace(covariance) / dim * np.eye(dim)

        self.means = means
        self.precision = np.linalg.pinv(covariance)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        """Negative minimum Mahalanobis distance (higher = more in-distribution)."""
        if self.means is None or self.precision is None:
            raise RuntimeError("fit() the scorer before scoring")
        features = features.astype(np.float64)
        best = None
        for mean in self.means:
            delta = features - mean
            distance = np.einsum("ij,jk,ik->i", delta, self.precision, delta)
            best = distance if best is None else np.minimum(best, distance)
        return -np.sqrt(np.maximum(best, 0.0))


# --------------------------------------------------------------- fusion
@dataclass
class OODCalibration:
    """Everything the app needs to reproduce the decision, as plain numbers."""

    energy_mean: float
    energy_std: float
    cosine_mean: float
    cosine_std: float
    weights: Dict[str, float] = field(default_factory=lambda: {"energy": 0.5, "cosine": 0.5})
    reject_threshold: float = 0.0
    target_tpr: float = 0.95
    measured_auroc: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "energy_mean": self.energy_mean,
            "energy_std": self.energy_std,
            "cosine_mean": self.cosine_mean,
            "cosine_std": self.cosine_std,
            "weights": self.weights,
            "reject_threshold": self.reject_threshold,
            "target_tpr": self.target_tpr,
            "measured_auroc": self.measured_auroc,
        }


class OODScorer:
    """Z-scored fusion of energy and prototype-cosine, thresholded at a target TPR."""

    def __init__(self, calibration: Optional[OODCalibration] = None) -> None:
        self.calibration = calibration

    def fit(
        self,
        id_energy: np.ndarray,
        id_cosine: np.ndarray,
        ood_sets: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
        target_tpr: float = 0.95,
    ) -> "OODScorer":
        """Fit normalisation and the reject threshold on *in-distribution* data.

        ``ood_sets`` maps a name ("non_leaf", "unsupported_crop", "unknown_disease")
        to (energy, cosine) arrays. They are used only to *report* AUROC, never
        to set the threshold — you will not have a representative sample of
        every weird thing a farmer photographs, so tuning on one is overfitting
        to your own imagination.
        """
        calibration = OODCalibration(
            energy_mean=float(id_energy.mean()),
            energy_std=float(id_energy.std() + 1e-6),
            cosine_mean=float(id_cosine.mean()),
            cosine_std=float(id_cosine.std() + 1e-6),
            target_tpr=target_tpr,
        )
        self.calibration = calibration

        id_scores = self.score(id_energy, id_cosine)
        calibration.reject_threshold = float(np.quantile(id_scores, 1.0 - target_tpr))

        if ood_sets:
            for name, (energy, cosine) in ood_sets.items():
                scores = self.score(energy, cosine)
                calibration.measured_auroc[name] = auroc(id_scores, scores)
                calibration.measured_auroc[f"{name}__fpr@95tpr"] = fpr_at_tpr(
                    id_scores, scores, tpr=target_tpr
                )
        return self

    def score(self, energy: np.ndarray, cosine: np.ndarray) -> np.ndarray:
        """Fused in-distribution score; higher = more likely in-distribution."""
        c = self.calibration
        if c is None:
            raise RuntimeError("fit() or load a calibration first")
        z_energy = (energy - c.energy_mean) / c.energy_std
        z_cosine = (cosine - c.cosine_mean) / c.cosine_std
        return c.weights["energy"] * z_energy + c.weights["cosine"] * z_cosine

    def is_out_of_distribution(self, energy: np.ndarray, cosine: np.ndarray) -> np.ndarray:
        return self.score(energy, cosine) < self.calibration.reject_threshold


# --------------------------------------------------------------- metrics
def auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Rank-based AUROC; ID is the positive class."""
    labels = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    scores = np.concatenate([id_scores, ood_scores])
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos, n_neg = labels.sum(), len(labels) - labels.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fpr_at_tpr(id_scores: np.ndarray, ood_scores: np.ndarray, tpr: float = 0.95) -> float:
    """Fraction of OOD accepted when accepting ``tpr`` of in-distribution."""
    threshold = np.quantile(id_scores, 1.0 - tpr)
    return float((ood_scores >= threshold).mean())


__all__ = [
    "energy_score",
    "max_softmax_probability",
    "MahalanobisScorer",
    "OODScorer",
    "OODCalibration",
    "auroc",
    "fpr_at_tpr",
]
