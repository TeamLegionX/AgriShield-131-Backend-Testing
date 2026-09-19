"""End-to-end diagnosis pipeline and the decision policy.

Flow
----
    photo
      -> G0  quality gate           (rule-based, ~5 ms, no model)
      -> G1  leaf detector          (YOLOv8n, optional; crops the subject)
      -> G2  classifier             (student, hierarchical + prototype)
      -> G3  temperature scaling    (calibrated probabilities)
      -> G4  open-set scoring       (energy + prototype cosine)
      -> G5  context re-ranking     (weather prior, bounded)
      -> G6  decision policy        -> CONFIDENT / TENTATIVE / CROP_ONLY / REJECT
      -> treatment suggestion -> spray log -> residue countdown

The decision policy is the product. A model that is right 70% of the time and
knows which 70% is far more useful to a farmer than one that is right 75% of the
time and never says "I'm not sure".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..fusion.late_fusion import FusionConfig, FusedPrediction, fuse
from ..fusion.risk_priors import DiseaseRiskModel
from ..fusion.weather import WeatherSeries
from ..taxonomy import Taxonomy
from .calibration import CalibrationArtifacts
from .ood import OODScorer, energy_score
from .quality import QualityReport, QualityThresholds, assess_quality


class Decision(str, Enum):
    CONFIDENT = "confident"          # show the diagnosis and the treatment
    TENTATIVE = "tentative"          # show a differential, recommend confirmation
    CROP_ONLY = "crop_only"          # we know the crop, not the disease
    REJECT_QUALITY = "reject_quality"
    REJECT_NOT_LEAF = "reject_not_leaf"
    REJECT_UNKNOWN = "reject_unknown"  # out of distribution: unsupported crop/disease


#: i18n keys. English strings live in the app bundle alongside Hindi, Kannada,
#: Marathi. Never build farmer-facing sentences by concatenating in Python.
MESSAGE_KEYS = {
    Decision.CONFIDENT: "diagnosis.confident",
    Decision.TENTATIVE: "diagnosis.tentative",
    Decision.CROP_ONLY: "diagnosis.crop_only",
    Decision.REJECT_QUALITY: "diagnosis.reject_quality",
    Decision.REJECT_NOT_LEAF: "diagnosis.reject_not_leaf",
    Decision.REJECT_UNKNOWN: "diagnosis.reject_unknown",
}


@dataclass
class DiagnosisResult:
    decision: Decision
    message_key: str
    class_id: Optional[str] = None
    crop_id: Optional[str] = None
    confidence: float = 0.0
    differential: List[Tuple[str, float]] = field(default_factory=list)
    quality: Optional[QualityReport] = None
    ood_score: float = 0.0
    lesion_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    lesion_count: int = 0
    context_notes: List[str] = field(default_factory=list)
    latency_ms: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "message_key": self.message_key,
            "class_id": self.class_id,
            "crop_id": self.crop_id,
            "confidence": round(self.confidence, 4),
            "differential": [{"class_id": c, "p": round(p, 4)} for c, p in self.differential],
            "lesion_count": self.lesion_count,
            "lesion_boxes": self.lesion_boxes,
            "ood_score": round(self.ood_score, 4),
            "quality": self.quality.to_dict() if self.quality else None,
            "context_notes": self.context_notes,
            "latency_ms": {k: round(v, 1) for k, v in self.latency_ms.items()},
        }


@dataclass
class PolicyConfig:
    confident_threshold: float = 0.70     # set empirically from the coverage-risk curve
    tentative_threshold: float = 0.40
    crop_confidence_threshold: float = 0.75
    suppress_lab_only_classes: bool = True
    min_detector_confidence: float = 0.35


class DiagnosisPipeline:
    """Orchestrates the edge tier. Every stage is optional except the classifier."""

    def __init__(
        self,
        model: torch.nn.Module,
        taxonomy: Taxonomy,
        calibration: CalibrationArtifacts,
        ood_scorer: Optional[OODScorer] = None,
        detector: Optional["LeafDetector"] = None,
        risk_model: Optional[DiseaseRiskModel] = None,
        policy: PolicyConfig = PolicyConfig(),
        fusion_config: FusionConfig = FusionConfig(),
        quality_thresholds: QualityThresholds = QualityThresholds(),
        transform=None,
        device: str = "cpu",
    ) -> None:
        self.model = model.eval().to(device)
        self.taxonomy = taxonomy
        self.calibration = calibration
        self.ood_scorer = ood_scorer
        self.detector = detector
        self.risk_model = risk_model or DiseaseRiskModel()
        self.policy = policy
        self.fusion_config = fusion_config
        self.quality_thresholds = quality_thresholds
        self.transform = transform
        self.device = device
        self._lab_only = set(taxonomy.lab_only_indices())

    @torch.no_grad()
    def diagnose(
        self,
        image: Image.Image,
        weather: Optional[WeatherSeries] = None,
        observed_at: Optional[datetime] = None,
        want_lesions: bool = True,
    ) -> DiagnosisResult:
        timings: Dict[str, float] = {}

        # ---- G0 quality ---------------------------------------------------
        t0 = _now_ms()
        quality = assess_quality(image, self.quality_thresholds)
        timings["quality"] = _now_ms() - t0
        if not quality.passed:
            return DiagnosisResult(
                decision=Decision.REJECT_QUALITY,
                message_key=MESSAGE_KEYS[Decision.REJECT_QUALITY],
                quality=quality,
                latency_ms=timings,
            )

        # ---- G1 detection -------------------------------------------------
        working = image
        if self.detector is not None:
            t0 = _now_ms()
            crop, det_confidence = self.detector.best_crop(image)
            timings["detect"] = _now_ms() - t0
            if det_confidence < self.policy.min_detector_confidence:
                return DiagnosisResult(
                    decision=Decision.REJECT_NOT_LEAF,
                    message_key=MESSAGE_KEYS[Decision.REJECT_NOT_LEAF],
                    quality=quality,
                    latency_ms=timings,
                )
            working = crop

        # ---- G2/G3 classify + calibrate ------------------------------------
        t0 = _now_ms()
        tensor = self.transform(working).unsqueeze(0).to(self.device)
        out = self.model(tensor)
        timings["classify"] = _now_ms() - t0

        logits = out.class_logits / max(self.calibration.temperature, 1e-2)
        probabilities = F.softmax(logits, dim=-1)[0].cpu().numpy()
        crop_probabilities = F.softmax(out.crop_logits, dim=-1)[0].cpu().numpy()

        if self.policy.suppress_lab_only_classes and self._lab_only:
            # Lab-only classes stay in the graph (so field photos are not pushed
            # into a wrong neighbour) but are renormalised out of the answer.
            mask = np.ones_like(probabilities)
            mask[list(self._lab_only)] = 0.05  # heavy discount, not a hard zero
            probabilities = probabilities * mask
            probabilities /= probabilities.sum()

        # ---- G4 open-set ---------------------------------------------------
        ood_value = 0.0
        if self.ood_scorer is not None:
            energy = float(energy_score(out.class_logits)[0].cpu())
            cosine = float(out.max_prototype_cosine[0].cpu())
            ood_value = float(self.ood_scorer.score(np.array([energy]), np.array([cosine]))[0])
            if ood_value < self.ood_scorer.calibration.reject_threshold:
                return DiagnosisResult(
                    decision=Decision.REJECT_UNKNOWN,
                    message_key=MESSAGE_KEYS[Decision.REJECT_UNKNOWN],
                    crop_id=self.taxonomy.crops[int(crop_probabilities.argmax())].id,
                    quality=quality,
                    ood_score=ood_value,
                    latency_ms=timings,
                )

        # ---- G5 context fusion ---------------------------------------------
        class_ids = self.taxonomy.index_to_class
        top_k_ids = [class_ids[i] for i in np.argsort(-probabilities)[: self.fusion_config.top_k]]
        risk = {
            k: v.risk
            for k, v in self.risk_model.assess(top_k_ids, weather, observed_at).items()
        }
        fused: FusedPrediction = fuse(class_ids, probabilities, risk, self.fusion_config)

        # ---- G6 decision ----------------------------------------------------
        confidence = fused.top_confidence
        crop_index = int(crop_probabilities.argmax())
        crop_id = self.taxonomy.crops[crop_index].id
        crop_confidence = float(crop_probabilities[crop_index])

        if confidence >= self.policy.confident_threshold:
            decision = Decision.CONFIDENT
        elif confidence >= self.policy.tentative_threshold:
            decision = Decision.TENTATIVE
        elif crop_confidence >= self.policy.crop_confidence_threshold:
            decision = Decision.CROP_ONLY
        else:
            decision = Decision.REJECT_UNKNOWN

        result = DiagnosisResult(
            decision=decision,
            message_key=MESSAGE_KEYS[decision],
            class_id=fused.top_class if decision in {Decision.CONFIDENT, Decision.TENTATIVE} else None,
            crop_id=crop_id,
            confidence=confidence,
            differential=fused.differential(3),
            quality=quality,
            ood_score=ood_value,
            context_notes=fused.explanation,
            latency_ms=timings,
        )

        # ---- lesion localisation (the "show the spot" requirement) ----------
        if want_lesions and decision in {Decision.CONFIDENT, Decision.TENTATIVE}:
            t0 = _now_ms()
            boxes = self._localise(tensor, int(np.argmax(fused.fused_probabilities)))
            timings["localise"] = _now_ms() - t0
            result.lesion_boxes = boxes
            result.lesion_count = len(boxes)

        return result

    def _localise(self, tensor: torch.Tensor, class_index: int) -> List[Tuple[int, int, int, int]]:
        """Grad-CAM++ boxes. Requires gradients, so it runs outside no_grad."""
        from .gradcam import GradCAMPlusPlus

        target_layer = _last_conv_layer(self.model)
        if target_layer is None:
            return []
        with torch.enable_grad():
            cam = GradCAMPlusPlus(self.model, target_layer)
            try:
                result = cam(tensor.clone().requires_grad_(True), class_index=class_index)
                return result.peak_boxes
            finally:
                cam.close()


class LeafDetector:
    """Thin wrapper over an Ultralytics YOLOv8n model.

    Train it as a *single-class* "leaf" detector, not a per-disease detector.
    Reasons:
      * PlantDoc's 8,851 boxes plus PlantSeg masks give plenty of leaf boxes,
        far more than any single disease has.
      * Localisation and classification are better decoupled: a published
        two-stage pipeline on PlantDoc reports ~92.9 mAP@0.5 for detection and
        ~78.5% classification accuracy downstream, and the decoupling is what
        makes the classifier robust to background.
      * One class means the detector transfers to crops the classifier has never
        seen, which is exactly what the "unsupported crop" rejection path needs.

    Train with::

        yolo detect train model=yolov8n.pt data=leaf.yaml imgsz=640 epochs=60
        yolo export model=best.pt format=tflite int8=True imgsz=320
    """

    def __init__(self, weights: str, confidence: float = 0.25, image_size: int = 320) -> None:
        from ultralytics import YOLO  # imported lazily: heavy, optional

        self.model = YOLO(weights)
        self.confidence = confidence
        self.image_size = image_size

    def best_crop(self, image: Image.Image, padding: float = 0.08) -> Tuple[Image.Image, float]:
        """Return (cropped leaf, detector confidence). Falls back to the full frame."""
        results = self.model.predict(
            image, imgsz=self.image_size, conf=self.confidence, verbose=False
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return image, 0.0

        boxes = results[0].boxes
        scores = boxes.conf.cpu().numpy()
        best = int(scores.argmax())
        x1, y1, x2, y2 = boxes.xyxy[best].cpu().numpy().tolist()

        pad_x = (x2 - x1) * padding
        pad_y = (y2 - y1) * padding
        crop = image.crop(
            (
                max(0, int(x1 - pad_x)),
                max(0, int(y1 - pad_y)),
                min(image.width, int(x2 + pad_x)),
                min(image.height, int(y2 + pad_y)),
            )
        )
        return crop, float(scores[best])


def _last_conv_layer(model: torch.nn.Module) -> Optional[torch.nn.Module]:
    last = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last = module
    return last


def _now_ms() -> float:
    import time

    return time.perf_counter() * 1000.0


__all__ = [
    "DiagnosisPipeline",
    "DiagnosisResult",
    "Decision",
    "PolicyConfig",
    "LeafDetector",
    "MESSAGE_KEYS",
]
