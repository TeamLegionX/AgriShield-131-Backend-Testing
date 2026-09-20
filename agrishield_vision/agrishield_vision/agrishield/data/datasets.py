"""Manifest construction and the unified multi-source dataset.

Everything upstream of training collapses into a single tidy manifest CSV with
these columns:

    path, dataset, raw_label, class_id, class_index, crop_index,
    domain (lab|field), group_key, mask_path (optional)

Keeping the manifest as an explicit artifact matters for three reasons:
  1. Splits become reproducible and auditable (you can diff two manifests).
  2. The lab/field domain flag is what lets us report *field-only* metrics,
     which is the only number honest enough to put in a pitch deck.
  3. `group_key` is what stops near-duplicate leakage across the split.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..taxonomy import Taxonomy

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Which sources are lab-condition. PlantVillage is the only large one, and it is
# also the one that produces the famous 99% -> ~31% cliff when you move to field
# photos, so it is quarantined from every reported metric.
LAB_SOURCES = {"plantvillage"}


@dataclass(frozen=True)
class SourceSpec:
    """Declares one dataset root laid out as ``root/<raw_label>/<image>``."""

    name: str
    root: Path
    domain: str = "field"  # "lab" | "field"
    mask_root: Optional[Path] = None  # PlantSeg-style segmentation masks
    label_from_path: Optional[Callable[[Path], str]] = None

    def raw_label(self, image_path: Path) -> str:
        if self.label_from_path is not None:
            return self.label_from_path(image_path)
        return image_path.parent.name


def build_manifest(
    sources: Sequence[SourceSpec],
    taxonomy: Taxonomy,
    strict: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Walk every source root and produce the unified manifest.

    ``strict=False`` keeps ingestion moving during a hackathon but *reports*
    every dropped label, so unmapped classes are a visible decision rather than
    a silent loss.
    """
    rows: List[Dict[str, object]] = []
    dropped: Dict[Tuple[str, str], int] = {}

    for spec in sources:
        if not spec.root.exists():
            raise FileNotFoundError(f"source {spec.name!r} root not found: {spec.root}")
        for path in sorted(spec.root.rglob("*")):
            if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
                continue
            raw = spec.raw_label(path)
            class_id = taxonomy.map_raw(spec.name, raw, strict=strict)
            if class_id is None:
                dropped[(spec.name, raw)] = dropped.get((spec.name, raw), 0) + 1
                continue
            idx = taxonomy.class_to_index[class_id]
            mask_path = _find_mask(path, spec)
            rows.append(
                {
                    "path": str(path),
                    "dataset": spec.name,
                    "raw_label": raw,
                    "class_id": class_id,
                    "class_index": idx,
                    "crop_index": taxonomy.class_to_crop[idx],
                    "domain": spec.domain,
                    "mask_path": str(mask_path) if mask_path else "",
                }
            )

    if verbose and dropped:
        print("[manifest] unmapped labels (add to taxonomy.yaml or ignore deliberately):")
        for (ds, raw), n in sorted(dropped.items(), key=lambda kv: -kv[1]):
            print(f"  {ds:<22} {raw:<45} {n:>6} images")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("manifest is empty - check source roots and taxonomy aliases")
    if verbose:
        print(f"[manifest] {len(df)} images | {df.class_id.nunique()} classes")
        print(df.groupby("domain").size().to_string())
    return df


def _find_mask(image_path: Path, spec: SourceSpec) -> Optional[Path]:
    if spec.mask_root is None:
        return None
    for suffix in (".png", ".jpg"):
        candidate = spec.mask_root / image_path.parent.name / (image_path.stem + suffix)
        if candidate.exists():
            return candidate
    return None


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class UnifiedPlantDataset(Dataset):
    """Serves (image, labels, metadata) from a manifest slice.

    The transform contract is deliberately mask-aware: field-style augmentation
    gets much stronger when a foreground mask is available (see
    ``transforms.BackgroundReplace``), and PlantSeg supplies real ones.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        transform: Optional[Callable] = None,
        background_pool: Optional[Sequence[Path]] = None,
        return_path: bool = False,
    ) -> None:
        self.df = manifest.reset_index(drop=True)
        self.transform = transform
        self.background_pool = list(background_pool) if background_pool else []
        self.return_path = return_path

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        row = self.df.iloc[index]
        image = Image.open(row["path"]).convert("RGB")

        mask: Optional[Image.Image] = None
        mask_path = str(row.get("mask_path", "") or "")
        if mask_path:
            mask = Image.open(mask_path).convert("L")

        if self.transform is not None:
            image = self.transform(image, mask=mask, backgrounds=self.background_pool)

        sample: Dict[str, torch.Tensor | str] = {
            "image": image,
            "class_index": torch.tensor(int(row["class_index"]), dtype=torch.long),
            "crop_index": torch.tensor(int(row["crop_index"]), dtype=torch.long),
            "is_field": torch.tensor(float(row["domain"] == "field")),
        }
        if self.return_path:
            sample["path"] = str(row["path"])
        return sample

    # --------------------------------------------------------------- utils
    def class_counts(self, n_classes: int) -> np.ndarray:
        counts = np.zeros(n_classes, dtype=np.int64)
        vals, freq = np.unique(self.df["class_index"].to_numpy(), return_counts=True)
        counts[vals] = freq
        return counts

    def sampling_weights(self, n_classes: int, power: float = 0.5) -> np.ndarray:
        """Per-sample weights for a ``WeightedRandomSampler``.

        ``power=0.5`` (square-root inverse frequency) is the pragmatic middle
        ground: full inverse frequency over-samples 30-image classes so hard
        that the model memorises them and macro-F1 gets worse, not better.
        """
        counts = self.class_counts(n_classes).astype(np.float64)
        counts[counts == 0] = 1.0
        per_class = (1.0 / counts) ** power
        per_class /= per_class.sum()
        return per_class[self.df["class_index"].to_numpy()]


class UnlabeledImageDataset(Dataset):
    """Field photos with no labels - used for teacher-supervised distillation.

    This is where most of the domain-shift win comes from: you can collect a few
    thousand unlabeled phone photos in a week, and the teacher's soft targets
    turn them into training signal without anyone hand-labelling a thing.
    """

    def __init__(self, roots: Sequence[Path], transform: Optional[Callable] = None) -> None:
        self.paths: List[Path] = []
        for root in roots:
            self.paths.extend(
                p for p in sorted(Path(root).rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES
            )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image = Image.open(self.paths[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image, mask=None, backgrounds=[])
        return {"image": image}


def file_sha1(path: str | Path, chunk: int = 1 << 16) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


__all__ = [
    "SourceSpec",
    "build_manifest",
    "UnifiedPlantDataset",
    "UnlabeledImageDataset",
    "file_sha1",
    "LAB_SOURCES",
]
