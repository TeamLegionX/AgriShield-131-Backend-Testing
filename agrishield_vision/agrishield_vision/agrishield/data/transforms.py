"""Field-style augmentation.

The single highest-leverage augmentation for this problem is **background
replacement**, not RandAugment. PlantVillage leaves sit on a uniform grey sheet;
a model trained on them learns "uniform background + leaf shape" and, when shown
a real field photo, attends to soil and sky instead of lesions. Pasting the
segmented leaf onto real field backgrounds removes that shortcut directly.

Two mask sources:
  * PlantSeg-style ground-truth masks where available (best).
  * ExG (excess-green) + Otsu on uniform-background lab images (works very well
    on PlantVillage precisely *because* its background is uniform; it is not
    reliable on field photos, and we do not use it there).

Everything else here is standard phone-camera realism: motion blur, JPEG
recompression, shadows, exposure error, occlusion.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------- masking
def estimate_leaf_mask_exg(image: Image.Image, min_area: float = 0.02) -> Optional[Image.Image]:
    """Foreground mask via excess-green index + Otsu threshold.

    ExG = 2G - R - B. On a grey/white studio background the leaf separates
    cleanly. Returns ``None`` when the result looks implausible (too small or
    nearly the whole frame), which is the common failure mode on field photos.
    """
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    exg = 2.0 * arr[..., 1] - arr[..., 0] - arr[..., 2]
    exg = (exg - exg.min()) / (exg.ptp() + 1e-6)

    thresh = _otsu(exg)
    mask = (exg > thresh).astype(np.uint8) * 255

    frac = float(mask.mean()) / 255.0
    if frac < min_area or frac > 0.95:
        return None

    pil = Image.fromarray(mask, mode="L")
    # Close pinholes and soften the cut line so the paste does not leave a halo
    # the network can trivially key on.
    pil = pil.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))
    pil = pil.filter(ImageFilter.GaussianBlur(1.2))
    return pil


def _otsu(gray: np.ndarray, bins: int = 256) -> float:
    hist, edges = np.histogram(gray, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.5
    centres = (edges[:-1] + edges[1:]) / 2.0
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    valid = (weight_bg > 0) & (weight_fg > 0)
    mean_bg = np.cumsum(hist * centres) / np.maximum(weight_bg, 1e-9)
    total_mean = (hist * centres).sum() / total
    mean_fg = (total_mean * total - np.cumsum(hist * centres)) / np.maximum(weight_fg, 1e-9)
    variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    variance[~valid] = -1.0
    return float(centres[int(np.argmax(variance))])


class BackgroundReplace:
    """Composite the masked leaf onto a random real-world background."""

    def __init__(self, probability: float = 0.5, allow_exg_fallback: bool = True) -> None:
        self.probability = probability
        self.allow_exg_fallback = allow_exg_fallback

    def __call__(
        self,
        image: Image.Image,
        mask: Optional[Image.Image],
        backgrounds: Sequence[Path],
    ) -> Image.Image:
        if not backgrounds or random.random() > self.probability:
            return image
        if mask is None and self.allow_exg_fallback:
            mask = estimate_leaf_mask_exg(image)
        if mask is None:
            return image

        bg_path = random.choice(list(backgrounds))
        try:
            background = Image.open(bg_path).convert("RGB")
        except Exception:
            return image

        background = _random_crop_to(background, image.size)
        # Match global colour temperature so the composite is not a giveaway.
        background = ImageEnhance.Color(background).enhance(random.uniform(0.7, 1.3))
        mask = mask.resize(image.size, Image.BILINEAR)
        return Image.composite(image, background, mask)


def _random_crop_to(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    target_w, target_h = size
    scale = max(target_w / image.width, target_h / image.height) * random.uniform(1.0, 1.4)
    image = image.resize((max(target_w, int(image.width * scale)),
                          max(target_h, int(image.height * scale))), Image.BILINEAR)
    left = random.randint(0, image.width - target_w)
    top = random.randint(0, image.height - target_h)
    return image.crop((left, top, left + target_w, top + target_h))


# ----------------------------------------------------------- phone realism
class PhoneCameraNoise:
    """Motion blur, exposure error, shadow, and JPEG recompression."""

    def __init__(self, probability: float = 0.7) -> None:
        self.probability = probability

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.probability:
            return image
        if random.random() < 0.4:
            image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.4, 2.0)))
        if random.random() < 0.5:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.55, 1.5))
        if random.random() < 0.3:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.6, 1.4))
        if random.random() < 0.3:
            image = _paste_shadow(image)
        if random.random() < 0.5:
            image = _jpeg_roundtrip(image, quality=random.randint(30, 85))
        return image


def _paste_shadow(image: Image.Image) -> Image.Image:
    """Hard-edged directional shadow, as cast by a hand or a neighbouring plant."""
    w, h = image.size
    overlay = Image.new("L", (w, h), 0)
    pts = [(random.randint(-w // 2, w), random.randint(-h // 2, h)) for _ in range(3)]
    pts.append((pts[0][0] + random.randint(w // 3, w), pts[0][1] + random.randint(h // 3, h)))
    from PIL import ImageDraw

    ImageDraw.Draw(overlay).polygon(pts, fill=random.randint(60, 140))
    overlay = overlay.filter(ImageFilter.GaussianBlur(random.uniform(2, 12)))
    dark = ImageEnhance.Brightness(image).enhance(0.55)
    return Image.composite(dark, image, overlay)


def _jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    import io

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ------------------------------------------------------------- pipelines
class TrainTransform:
    """Mask-aware training pipeline. Call signature matches UnifiedPlantDataset."""

    def __init__(
        self,
        image_size: int = 224,
        bg_probability: float = 0.5,
        randaugment_ops: int = 2,
        randaugment_magnitude: int = 9,
        erasing_probability: float = 0.25,
    ) -> None:
        self.bg = BackgroundReplace(probability=bg_probability)
        self.phone = PhoneCameraNoise()
        self.geometric = v2.Compose(
            [
                # Aggressive lower bound: field photos are often a close crop of
                # one lesion, or a wide shot with the leaf occupying ~20% of frame.
                v2.RandomResizedCrop(image_size, scale=(0.25, 1.0), ratio=(0.75, 1.33)),
                v2.RandomHorizontalFlip(0.5),
                v2.RandomVerticalFlip(0.2),
                v2.RandomRotation(25, expand=False),
            ]
        )
        self.photometric = v2.Compose(
            [
                v2.RandAugment(num_ops=randaugment_ops, magnitude=randaugment_magnitude),
                v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.35, hue=0.05),
            ]
        )
        self.finalise = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                v2.RandomErasing(p=erasing_probability, scale=(0.02, 0.15), value="random"),
            ]
        )

    def __call__(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        backgrounds: Sequence[Path] = (),
    ) -> torch.Tensor:
        image = self.bg(image, mask, backgrounds)
        image = self.geometric(image)
        image = self.photometric(image)
        image = self.phone(image)
        return self.finalise(image)


class EvalTransform:
    """Deterministic resize + centre crop. Also used as the TTA base view."""

    def __init__(self, image_size: int = 224, crop_pct: float = 0.9) -> None:
        resize = int(round(image_size / crop_pct))
        self.pipeline = v2.Compose(
            [
                v2.Resize(resize, antialias=True),
                v2.CenterCrop(image_size),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __call__(self, image: Image.Image, mask=None, backgrounds=()) -> torch.Tensor:
        return self.pipeline(image)


def collect_backgrounds(roots: Sequence[Path]) -> List[Path]:
    """Gather background plates (soil, canopy, sky, hands, mulch, tarpaulin).

    Cheapest viable source: the *field* datasets themselves. Crop random patches
    from PlantDoc/PlantWild images that contain no annotated leaf, or simply
    shoot 200 backgrounds on a phone in an afternoon. 200 is enough; diversity of
    texture matters far more than count.
    """
    out: List[Path] = []
    for root in roots:
        out.extend(
            p for p in Path(root).rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
    return out


__all__ = [
    "TrainTransform",
    "EvalTransform",
    "BackgroundReplace",
    "PhoneCameraNoise",
    "estimate_leaf_mask_exg",
    "collect_backgrounds",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]
