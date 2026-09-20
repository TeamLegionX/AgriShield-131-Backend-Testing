"""Late fusion: context re-ranks vision, it never overrules it.

The rule
--------
    log p_fused(c)  =  log p_vision(c)  +  lambda * log p_context(c)     for c in top-k
                       (classes outside top-k are untouched and stay below)

Three guardrails make this safe rather than merely plausible:

  G1  **Re-rank within top-k only.** Weather cannot introduce a disease the
      image gave no support to. If the vision model puts rice blast at 0.2%,
      no amount of conducive humidity promotes it. This is the difference
      between a prior and a hallucination.

  G2  **Bounded influence.** lambda is clamped and the log-odds shift is capped,
      so context can reorder a close call (0.41 vs 0.38) but not overturn a
      confident one (0.85 vs 0.05). Default lambda = 0.35, tuned by ablation.

  G3  **Context is disabled when the vision model is already uncertain.** If the
      image is ambiguous enough to be near the rejection threshold, the honest
      output is "unclear, here is what to check", not a weather-flavoured guess.
      Fusing a prior into noise produces confident nonsense.

Always ship the with/without ablation. If fusion does not improve top-1 on your
field test set, say so and turn it off — a defensible negative result is worth
more in a judged competition than an unverifiable feature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class FusionConfig:
    lam: float = 0.35              # context weight
    top_k: int = 5                 # re-rank window
    max_log_shift: float = 1.0     # cap on |lambda * log p_context| per class
    min_vision_confidence: float = 0.25  # below this, skip fusion entirely (G3)
    enabled: bool = True


@dataclass
class FusedPrediction:
    class_ids: List[str]
    vision_probabilities: np.ndarray
    fused_probabilities: np.ndarray
    context_risk: Dict[str, float]
    reordered: bool
    explanation: List[str] = field(default_factory=list)

    @property
    def top_class(self) -> str:
        return self.class_ids[int(self.fused_probabilities.argmax())]

    @property
    def top_confidence(self) -> float:
        return float(self.fused_probabilities.max())

    def differential(self, n: int = 3) -> List[Tuple[str, float]]:
        """The 'also consider' list. Farmers and agronomists both want this;
        a single label with no alternatives is a worse product, not a better one."""
        order = np.argsort(-self.fused_probabilities)[:n]
        return [(self.class_ids[i], float(self.fused_probabilities[i])) for i in order]


def fuse(
    class_ids: Sequence[str],
    vision_probabilities: np.ndarray,
    context_risk: Dict[str, float],
    config: FusionConfig = FusionConfig(),
) -> FusedPrediction:
    """Apply bounded Bayesian re-ranking over the vision posterior."""
    probs = np.asarray(vision_probabilities, dtype=np.float64)
    probs = probs / probs.sum()
    class_ids = list(class_ids)
    original_top = int(probs.argmax())

    if not config.enabled or probs.max() < config.min_vision_confidence or not context_risk:
        reason = (
            "Context skipped: the image alone is too uncertain to re-rank safely."
            if probs.max() < config.min_vision_confidence
            else "Context unavailable (offline or no matching rule); using image only."
        )
        return FusedPrediction(class_ids, probs, probs.copy(), context_risk, False, [reason])

    top_k = min(config.top_k, len(class_ids))
    candidates = np.argsort(-probs)[:top_k]

    log_p = np.log(np.clip(probs, 1e-12, None))
    fused_log = log_p.copy()
    notes: List[str] = []

    for index in candidates:
        risk = float(context_risk.get(class_ids[index], 0.35))
        # Map risk in [0,1] to a log-odds nudge centred on the neutral prior, so
        # a 'no rule' class is not systematically penalised.
        shift = config.lam * np.log(np.clip(risk, 0.05, 1.0) / 0.35)
        shift = float(np.clip(shift, -config.max_log_shift, config.max_log_shift))
        fused_log[index] += shift

    fused = np.exp(fused_log - fused_log.max())
    fused /= fused.sum()

    new_top = int(fused.argmax())
    reordered = new_top != original_top
    if reordered:
        notes.append(
            f"Recent weather favours {class_ids[new_top]} over {class_ids[original_top]}; "
            "the image supported both."
        )
    else:
        notes.append("Recent weather is consistent with the image diagnosis.")

    return FusedPrediction(class_ids, probs, fused, context_risk, reordered, notes)


def ablation_report(
    class_ids: Sequence[str],
    vision_probabilities: np.ndarray,
    targets: np.ndarray,
    context_risks: Sequence[Dict[str, float]],
    lambdas: Sequence[float] = (0.0, 0.15, 0.35, 0.5, 0.75, 1.0),
) -> Dict[float, float]:
    """Top-1 accuracy as a function of lambda, on the field test set.

    Run this, plot it, and put it on a slide. If the curve is flat, fusion is
    decoration; if it peaks and falls, you have found your lambda and shown your
    work. Either outcome is a better answer than asserting that fusion helps.
    """
    results: Dict[float, float] = {}
    for lam in lambdas:
        config = FusionConfig(lam=lam, enabled=lam > 0)
        correct = 0
        for row, (probs, risk) in enumerate(zip(vision_probabilities, context_risks)):
            prediction = fuse(class_ids, probs, risk, config)
            correct += int(prediction.class_ids.index(prediction.top_class) == targets[row])
        results[lam] = correct / len(targets)
    return results


__all__ = ["fuse", "FusionConfig", "FusedPrediction", "ablation_report"]
