"""Evaluation protocol.

The reporting contract for this project
---------------------------------------
Headline number = **macro-F1 on the field-only test split**. Not accuracy, and
not anything computed on PlantVillage.

Accuracy is the wrong headline because the merged dataset is heavily imbalanced
(the largest class can be 20x the smallest), so accuracy is dominated by a
handful of common classes. Macro-F1 weights every disease equally, which matches
what a farmer experiences: they care whether *their* disease is detected.

PlantVillage numbers are the wrong headline for a blunter reason. Models trained
on it reach ~99% on its own test split and then fall to roughly 31% when
evaluated under different capture conditions, and below 40% on in-the-wild
images. Any 99% figure in a crop-disease pitch is either a lab number being
passed off as a field number or a leaked split. Do not put one in your deck; do
be ready to explain why everyone else's is there.

Report all seven of these:
  1. field macro-F1 and per-class F1
  2. confusion matrix restricted to declared look-alike groups
  3. ECE + reliability diagram (before and after temperature scaling)
  4. coverage-risk curve, with the operating point marked
  5. OOD AUROC / FPR@95TPR, broken out per rejection case
  6. on-device p50/p95 latency and model size
  7. the with/without-context fusion ablation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ------------------------------------------------------------- classification
def confusion_matrix(targets: np.ndarray, predictions: np.ndarray, n_classes: int) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (targets.astype(int), predictions.astype(int)), 1)
    return matrix


def per_class_f1(matrix: np.ndarray) -> np.ndarray:
    true_positive = np.diag(matrix).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    actual = matrix.sum(axis=1).astype(np.float64)
    precision = np.divide(true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0)
    recall = np.divide(true_positive, actual, out=np.zeros_like(true_positive), where=actual > 0)
    denominator = precision + recall
    return np.divide(
        2 * precision * recall, denominator, out=np.zeros_like(precision), where=denominator > 0
    )


def macro_f1(targets: np.ndarray, predictions: np.ndarray, n_classes: int,
             present_only: bool = True) -> float:
    """Macro-F1. ``present_only`` excludes classes absent from the test split,
    which otherwise contribute a silent zero and deflate the score meaninglessly."""
    matrix = confusion_matrix(targets, predictions, n_classes)
    f1 = per_class_f1(matrix)
    if present_only:
        present = matrix.sum(axis=1) > 0
        return float(f1[present].mean()) if present.any() else 0.0
    return float(f1.mean())


@dataclass
class ClassificationReport:
    macro_f1: float
    accuracy: float
    balanced_accuracy: float
    per_class: Dict[str, Dict[str, float]]
    support: Dict[str, int]
    n_samples: int

    def worst_classes(self, n: int = 10) -> List[Tuple[str, float]]:
        """The tail. This is what you fix next, and what an honest slide shows."""
        scored = [(name, stats["f1"]) for name, stats in self.per_class.items()
                  if self.support.get(name, 0) > 0]
        return sorted(scored, key=lambda kv: kv[1])[:n]

    def summary(self) -> str:
        lines = [
            f"samples={self.n_samples}  macro-F1={self.macro_f1:.4f}  "
            f"acc={self.accuracy:.4f}  balanced-acc={self.balanced_accuracy:.4f}",
            "weakest classes:",
        ]
        for name, score in self.worst_classes():
            lines.append(f"  {name:<40} F1={score:.3f}  n={self.support.get(name, 0)}")
        return "\n".join(lines)


def classification_report(
    targets: np.ndarray,
    predictions: np.ndarray,
    class_ids: Sequence[str],
) -> ClassificationReport:
    n_classes = len(class_ids)
    matrix = confusion_matrix(targets, predictions, n_classes)
    f1 = per_class_f1(matrix)

    predicted = matrix.sum(axis=0).astype(np.float64)
    actual = matrix.sum(axis=1).astype(np.float64)
    true_positive = np.diag(matrix).astype(np.float64)
    precision = np.divide(true_positive, predicted, out=np.zeros_like(f1), where=predicted > 0)
    recall = np.divide(true_positive, actual, out=np.zeros_like(f1), where=actual > 0)

    present = actual > 0
    return ClassificationReport(
        macro_f1=float(f1[present].mean()) if present.any() else 0.0,
        accuracy=float(true_positive.sum() / max(1, matrix.sum())),
        balanced_accuracy=float(recall[present].mean()) if present.any() else 0.0,
        per_class={
            class_ids[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
            }
            for i in range(n_classes)
        },
        support={class_ids[i]: int(actual[i]) for i in range(n_classes)},
        n_samples=int(matrix.sum()),
    )


def confusable_confusion(
    targets: np.ndarray,
    predictions: np.ndarray,
    class_ids: Sequence[str],
    groups: Dict[str, List[str]],
) -> Dict[str, np.ndarray]:
    """Sub-confusion matrices for declared look-alike clusters.

    Early blight vs late blight vs Septoria on tomato is the case that decides
    whether the recommended chemical is right. A global confusion matrix at 48
    classes is unreadable on a slide; these 3x3 and 4x4 blocks are not, and they
    are where the real failure modes live.
    """
    index = {name: i for i, name in enumerate(class_ids)}
    blocks: Dict[str, np.ndarray] = {}
    for anchor, members in groups.items():
        members = [m for m in members if m in index]
        if len(members) < 2:
            continue
        ids = [index[m] for m in members]
        mask = np.isin(targets, ids)
        if not mask.any():
            continue
        remap = {old: new for new, old in enumerate(ids)}
        block = np.zeros((len(ids), len(ids)), dtype=np.int64)
        for t, p in zip(targets[mask], predictions[mask]):
            if p in remap:
                block[remap[int(t)], remap[int(p)]] += 1
        blocks[anchor] = block
    return blocks


# ------------------------------------------------------------------ bundling
@dataclass
class EvaluationBundle:
    """Everything that goes on the results slide, in one object."""

    field_report: ClassificationReport
    lab_report: Optional[ClassificationReport] = None
    ece_before: float = 0.0
    ece_after: float = 0.0
    temperature: float = 1.0
    coverage_risk: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ood_auroc: Dict[str, float] = field(default_factory=dict)
    fusion_ablation: Dict[float, float] = field(default_factory=dict)
    latency_ms: Dict[str, float] = field(default_factory=dict)
    model_size_mb: float = 0.0

    def headline(self) -> str:
        lines = [
            "=" * 66,
            "AgriShield vision - evaluation (field-only test split)",
            "=" * 66,
            self.field_report.summary(),
        ]
        if self.lab_report is not None:
            gap = self.lab_report.macro_f1 - self.field_report.macro_f1
            lines.append(
                f"\nlab-split macro-F1 = {self.lab_report.macro_f1:.4f} "
                f"(domain gap {gap:+.4f}) - reported for transparency, never as the headline"
            )
        lines.append(f"\ncalibration: T={self.temperature:.3f}  ECE {self.ece_before:.4f} -> {self.ece_after:.4f}")
        if self.ood_auroc:
            lines.append("open-set:")
            for name, value in sorted(self.ood_auroc.items()):
                lines.append(f"  {name:<34} {value:.4f}")
        if self.fusion_ablation:
            lines.append("context fusion ablation (lambda -> top-1):")
            for lam, acc in sorted(self.fusion_ablation.items()):
                lines.append(f"  lambda={lam:<5} {acc:.4f}")
        if self.latency_ms:
            lines.append(
                f"latency: p50={self.latency_ms.get('p50_ms', 0):.1f} ms  "
                f"p95={self.latency_ms.get('p95_ms', 0):.1f} ms  size={self.model_size_mb:.2f} MB"
            )
        lines.append("=" * 66)
        return "\n".join(lines)


__all__ = [
    "confusion_matrix",
    "per_class_f1",
    "macro_f1",
    "classification_report",
    "ClassificationReport",
    "confusable_confusion",
    "EvaluationBundle",
]
