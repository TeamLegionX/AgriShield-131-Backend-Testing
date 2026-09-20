"""Confidence calibration.

The problem, concretely
----------------------
A published controlled-to-field benchmark measured a model lose 67.7 accuracy
points when moved from lab images to field images while its *mean predicted
confidence stayed near 80%*. A farmer reading "Late blight, 92% sure" has no way
to distinguish that from a real 92%. Spraying on a wrong diagnosis costs money,
costs a residue violation, and costs trust in the app. Raw softmax is therefore
not shippable.

What this module does, and what it cannot do
--------------------------------------------
Temperature scaling learns a single scalar T on a held-out *field* validation
split and divides the logits by it. It is the right default: one parameter,
cannot change the argmax (so accuracy is untouched), and it substantially
reduces Expected Calibration Error.

It does **not** fix the underlying problem. In the same benchmark, temperature
scaling reduced ECE but left selective risk at 80% coverage above 60%. So
calibration is necessary and insufficient: the honest product answer is
calibration *plus* abstention (see ood.py and pipeline.py), plus never claiming
more than the coverage-risk curve supports.

Fit T on the field validation split. Fitting it on test is self-deception.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    """Single-parameter logit scaling, fitted by LBFGS on NLL."""

    def __init__(self, initial: float = 1.0) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(float(initial)).log())

    @property
    def temperature(self) -> float:
        return float(self.log_temperature.exp())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.log_temperature.exp().clamp(min=1e-2)

    def fit(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        max_iter: int = 200,
        verbose: bool = True,
    ) -> "TemperatureScaler":
        logits = logits.detach().float()
        targets = targets.detach().long()
        optimiser = torch.optim.LBFGS([self.log_temperature], lr=0.05, max_iter=max_iter)

        def closure() -> torch.Tensor:
            optimiser.zero_grad()
            loss = F.cross_entropy(self.forward(logits), targets)
            loss.backward()
            return loss

        before = expected_calibration_error(F.softmax(logits, -1).numpy(), targets.numpy())
        optimiser.step(closure)
        after = expected_calibration_error(
            F.softmax(self.forward(logits), -1).detach().numpy(), targets.numpy()
        )
        if verbose:
            print(f"[calibration] T={self.temperature:.3f}  ECE {before:.4f} -> {after:.4f}")
        return self


def expected_calibration_error(
    probabilities: np.ndarray, targets: np.ndarray, n_bins: int = 15
) -> float:
    """Standard equal-width ECE on top-1 confidence."""
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = (prediction == targets).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        if not in_bin.any():
            continue
        weight = in_bin.mean()
        ece += weight * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def reliability_curve(
    probabilities: np.ndarray, targets: np.ndarray, n_bins: int = 15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (bin_centres, accuracy, mean_confidence) for the reliability diagram.

    Plot this. It is the single most persuasive slide you can show a judging
    panel that is used to seeing "99.2% accuracy" and nothing else.
    """
    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == targets).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centres, accuracy, mean_conf = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        centres.append((lo + hi) / 2)
        accuracy.append(correct[in_bin].mean() if in_bin.any() else np.nan)
        mean_conf.append(confidence[in_bin].mean() if in_bin.any() else np.nan)
    return np.array(centres), np.array(accuracy), np.array(mean_conf)


def coverage_risk_curve(
    probabilities: np.ndarray, targets: np.ndarray, n_points: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """Selective prediction: error rate among the most-confident X% of inputs.

    This is the curve the *product* is actually built on. It answers the
    question the pitch needs: "if we only answer when we are sure, how often are
    we wrong?" Read off the threshold at your target risk and put that number,
    not top-1 accuracy, in the deck.
    """
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets
    order = np.argsort(-confidence)
    correct_sorted = correct[order]

    coverages = np.linspace(1.0 / n_points, 1.0, n_points)
    risks = []
    for coverage in coverages:
        k = max(1, int(round(coverage * len(correct_sorted))))
        risks.append(1.0 - correct_sorted[:k].mean())
    return coverages, np.array(risks)


def threshold_for_target_risk(
    probabilities: np.ndarray, targets: np.ndarray, target_risk: float = 0.15
) -> Tuple[float, float]:
    """Smallest confidence threshold achieving ``target_risk``; returns (threshold, coverage).

    Use this to set the app's "confident answer" bar empirically instead of
    picking 0.8 because it feels round.
    """
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets
    order = np.argsort(-confidence)
    conf_sorted, correct_sorted = confidence[order], correct[order]

    running_errors = np.cumsum(~correct_sorted)
    counts = np.arange(1, len(correct_sorted) + 1)
    risk = running_errors / counts
    feasible = np.where(risk <= target_risk)[0]
    if len(feasible) == 0:
        return 1.0, 0.0
    k = feasible[-1]
    return float(conf_sorted[k]), float((k + 1) / len(correct_sorted))


@dataclass
class CalibrationArtifacts:
    """Serialised alongside the TFLite model; tunable without re-export."""

    temperature: float
    confident_threshold: float
    tentative_threshold: float
    ece_before: float
    ece_after: float
    coverage_at_confident: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


__all__ = [
    "TemperatureScaler",
    "expected_calibration_error",
    "reliability_curve",
    "coverage_risk_curve",
    "threshold_for_target_risk",
    "CalibrationArtifacts",
]
