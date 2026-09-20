"""Pre-inference image quality gate.

Deliberately model-free. Roughly a third of rejected real-world submissions to
plant-diagnosis apps fail for reasons a five-line numeric check catches: the
photo is blurred, too dark, shot from three metres away, or the leaf occupies
5% of the frame. Spending a neural network on that is wasteful, and — more
importantly — a rule-based rejection can tell the farmer *what to fix*
("move closer", "the photo is blurry, hold still") in their own language, while
a model can only say "not confident".

Every threshold below is a starting point calibrated against a few hundred
photos. Re-measure them on your own field photo set before the demo; they are
device- and resolution-dependent.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


class QualityIssue(str, Enum):
    TOO_SMALL = "too_small"
    BLURRY = "blurry"
    TOO_DARK = "too_dark"
    TOO_BRIGHT = "too_bright"
    LOW_CONTRAST = "low_contrast"
    NO_VEGETATION = "no_vegetation"
    SUBJECT_TOO_FAR = "subject_too_far"


#: Farmer-facing guidance keyed by issue. The app resolves these keys against
#: its i18n bundle (Hindi / Kannada / Marathi); English is the fallback.
GUIDANCE_KEYS = {
    QualityIssue.TOO_SMALL: "quality.too_small",
    QualityIssue.BLURRY: "quality.blurry",
    QualityIssue.TOO_DARK: "quality.too_dark",
    QualityIssue.TOO_BRIGHT: "quality.too_bright",
    QualityIssue.LOW_CONTRAST: "quality.low_contrast",
    QualityIssue.NO_VEGETATION: "quality.no_vegetation",
    QualityIssue.SUBJECT_TOO_FAR: "quality.subject_too_far",
}

GUIDANCE_EN = {
    QualityIssue.TOO_SMALL: "Photo resolution is too low. Use the camera, not a screenshot.",
    QualityIssue.BLURRY: "The photo is blurry. Hold the phone steady and tap to focus on the leaf.",
    QualityIssue.TOO_DARK: "Too dark. Move into daylight or use the flash.",
    QualityIssue.TOO_BRIGHT: "Too bright or washed out. Shade the leaf with your hand.",
    QualityIssue.LOW_CONTRAST: "The leaf is hard to see. Try a plain background behind the leaf.",
    QualityIssue.NO_VEGETATION: "No plant detected. Point the camera at a single leaf.",
    QualityIssue.SUBJECT_TOO_FAR: "Move closer. One leaf should fill most of the frame.",
}


@dataclass
class QualityReport:
    passed: bool
    issues: List[QualityIssue]
    blur_score: float
    brightness: float
    contrast: float
    vegetation_fraction: float
    width: int
    height: int

    def primary_guidance(self) -> Optional[str]:
        return GUIDANCE_EN[self.issues[0]] if self.issues else None

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "issues": [i.value for i in self.issues],
            "guidance_key": GUIDANCE_KEYS[self.issues[0]] if self.issues else None,
            "blur_score": round(self.blur_score, 2),
            "brightness": round(self.brightness, 3),
            "contrast": round(self.contrast, 3),
            "vegetation_fraction": round(self.vegetation_fraction, 3),
        }


@dataclass
class QualityThresholds:
    min_side: int = 224
    min_blur_variance: float = 60.0      # variance of Laplacian on 0-255 grey
    min_brightness: float = 0.12
    max_brightness: float = 0.92
    min_contrast: float = 0.045          # std of luminance
    min_vegetation: float = 0.04         # ExG-positive pixel fraction
    min_subject_fraction: float = 0.10   # largest vegetation blob / frame


def _laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the Laplacian - the standard cheap sharpness proxy."""
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    padded = np.pad(gray, 1, mode="edge")
    response = (
        padded[:-2, 1:-1] * kernel[0, 1]
        + padded[1:-1, :-2] * kernel[1, 0]
        + padded[1:-1, 1:-1] * kernel[1, 1]
        + padded[1:-1, 2:] * kernel[1, 2]
        + padded[2:, 1:-1] * kernel[2, 1]
    )
    return float(response.var())


def _vegetation_mask(rgb: np.ndarray) -> np.ndarray:
    """ExG > 0 is a serviceable 'is there green stuff here' test.

    Note the honest limitation: severely chlorotic or necrotic leaves are brown,
    not green, so this is a *permissive* gate. It is tuned to catch "photo of a
    wall", not to segment disease.
    """
    exg = 2.0 * rgb[..., 1] - rgb[..., 0] - rgb[..., 2]
    brown = (rgb[..., 0] > rgb[..., 2] + 0.06) & (rgb[..., 0] > 0.2) & (rgb[..., 1] > 0.12)
    return (exg > 0.02) | brown


def _largest_blob_fraction(mask: np.ndarray, downsample: int = 8) -> float:
    """Approximate largest connected component as a fraction of the frame.

    Uses a coarse grid flood fill: exact connected components are overkill for a
    gate that only needs to answer "is the subject big enough".
    """
    small = mask[::downsample, ::downsample]
    if small.sum() == 0:
        return 0.0
    visited = np.zeros_like(small, dtype=bool)
    best = 0
    height, width = small.shape
    for start_y in range(height):
        for start_x in range(width):
            if not small[start_y, start_x] or visited[start_y, start_x]:
                continue
            stack = [(start_y, start_x)]
            visited[start_y, start_x] = True
            size = 0
            while stack:
                y, x = stack.pop()
                size += 1
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and small[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            best = max(best, size)
    return best / small.size


def assess_quality(
    image: Image.Image, thresholds: QualityThresholds = QualityThresholds()
) -> QualityReport:
    """Run all gates. Cheap enough to run on every camera frame if you want to."""
    width, height = image.size
    work = image.convert("RGB")
    if max(work.size) > 512:  # analysis resolution; keeps this ~5ms
        scale = 512 / max(work.size)
        work = work.resize((int(work.width * scale), int(work.height * scale)), Image.BILINEAR)

    rgb = np.asarray(work, dtype=np.float32) / 255.0
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)

    blur = _laplacian_variance(gray * 255.0)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    veg = _vegetation_mask(rgb)
    veg_fraction = float(veg.mean())
    subject_fraction = _largest_blob_fraction(veg)

    issues: List[QualityIssue] = []
    if min(width, height) < thresholds.min_side:
        issues.append(QualityIssue.TOO_SMALL)
    if blur < thresholds.min_blur_variance:
        issues.append(QualityIssue.BLURRY)
    if brightness < thresholds.min_brightness:
        issues.append(QualityIssue.TOO_DARK)
    elif brightness > thresholds.max_brightness:
        issues.append(QualityIssue.TOO_BRIGHT)
    if contrast < thresholds.min_contrast:
        issues.append(QualityIssue.LOW_CONTRAST)
    if veg_fraction < thresholds.min_vegetation:
        issues.append(QualityIssue.NO_VEGETATION)
    elif subject_fraction < thresholds.min_subject_fraction:
        issues.append(QualityIssue.SUBJECT_TOO_FAR)

    return QualityReport(
        passed=not issues,
        issues=issues,
        blur_score=blur,
        brightness=brightness,
        contrast=contrast,
        vegetation_fraction=veg_fraction,
        width=width,
        height=height,
    )


__all__ = ["assess_quality", "QualityReport", "QualityIssue", "QualityThresholds", "GUIDANCE_EN"]
