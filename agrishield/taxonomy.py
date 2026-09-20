"""Unified label taxonomy across heterogeneous plant-disease datasets.

Design note
-----------
Every public plant-disease dataset invents its own folder names for the same
biological entity ("Tomato___Early_blight" vs "Tomato Early blight leaf" vs
"tomato_early_blight"). Merging them by string similarity is how leakage and
silent label corruption get introduced. We therefore require an *explicit*
alias table (configs/taxonomy.yaml): an unmapped raw label raises, it does not
get quietly bucketed.

The taxonomy also carries `field_support`, which is the honesty valve for the
whole product: a class trained only on PlantVillage lab photos is kept in the
output space (so field photos are not force-fitted into a neighbour) but is
never surfaced to a farmer as a confident diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import yaml

FieldSupport = str  # Literal["strong", "weak", "lab"]

_SUPPORT_ORDER: Dict[str, int] = {"lab": 0, "weak": 1, "strong": 2}


@dataclass(frozen=True)
class CropInfo:
    """A crop species plus its farmer-facing names."""

    id: str
    name_en: str
    name_hi: str = ""
    name_kn: str = ""


@dataclass(frozen=True)
class ClassInfo:
    """One canonical (crop, condition) class."""

    id: str
    crop: str
    pathogen: Optional[str] = None
    is_healthy: bool = False
    is_pest: bool = False
    field_support: FieldSupport = "lab"
    confusable_with: Sequence[str] = field(default_factory=tuple)
    aliases: Dict[str, Sequence[str]] = field(default_factory=dict)

    @property
    def condition(self) -> str:
        return self.id.split("::", 1)[1]


class Taxonomy:
    """Loads taxonomy.yaml and provides raw-label -> canonical-index mapping.

    Index conventions used everywhere downstream:
      * ``class_index``: 0..n_classes-1, sorted by canonical id (stable across runs)
      * ``crop_index``:  0..n_crops-1, sorted by crop id
      * ``class_to_crop[i]``: crop index of class i (drives the hierarchical head)
    """

    def __init__(self, crops: Sequence[CropInfo], classes: Sequence[ClassInfo]) -> None:
        self.crops: List[CropInfo] = sorted(crops, key=lambda c: c.id)
        self.classes: List[ClassInfo] = sorted(classes, key=lambda c: c.id)

        self.crop_to_index: Dict[str, int] = {c.id: i for i, c in enumerate(self.crops)}
        self.class_to_index: Dict[str, int] = {c.id: i for i, c in enumerate(self.classes)}
        self.index_to_class: List[str] = [c.id for c in self.classes]

        unknown_crops = {c.crop for c in self.classes} - set(self.crop_to_index)
        if unknown_crops:
            raise ValueError(f"classes reference undeclared crops: {sorted(unknown_crops)}")

        self.class_to_crop: List[int] = [self.crop_to_index[c.crop] for c in self.classes]

        # (dataset, lowercased raw label) -> canonical id
        self._alias_index: Dict[tuple, str] = {}
        for info in self.classes:
            for dataset, raws in (info.aliases or {}).items():
                for raw in raws:
                    key = (dataset.lower(), _normalise(raw))
                    if key in self._alias_index and self._alias_index[key] != info.id:
                        raise ValueError(
                            f"alias collision: {dataset}/{raw!r} maps to both "
                            f"{self._alias_index[key]} and {info.id}"
                        )
                    self._alias_index[key] = info.id

    # ------------------------------------------------------------------ io
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Taxonomy":
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        crops = [CropInfo(**c) for c in raw["crops"]]
        classes = [
            ClassInfo(
                id=c["id"],
                crop=c["crop"],
                pathogen=c.get("pathogen"),
                is_healthy=bool(c.get("is_healthy", False)),
                is_pest=bool(c.get("is_pest", False)),
                field_support=c.get("field_support", "lab"),
                confusable_with=tuple(c.get("confusable_with", ())),
                aliases={k: tuple(v) for k, v in (c.get("aliases") or {}).items()},
            )
            for c in raw["classes"]
        ]
        return cls(crops, classes)

    # ------------------------------------------------------------- mapping
    def map_raw(self, dataset: str, raw_label: str, strict: bool = True) -> Optional[str]:
        """Map a source dataset's raw label to a canonical class id.

        Returns ``None`` (or raises, if ``strict``) when the label is unknown.
        An unknown label is a data-prep bug, not a runtime condition.
        """
        canonical = self._alias_index.get((dataset.lower(), _normalise(raw_label)))
        if canonical is None and strict:
            raise KeyError(
                f"no taxonomy entry for dataset={dataset!r} label={raw_label!r}. "
                "Add it to configs/taxonomy.yaml or exclude it explicitly."
            )
        return canonical

    # ------------------------------------------------------------- queries
    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def n_crops(self) -> int:
        return len(self.crops)

    def get(self, class_id: str) -> ClassInfo:
        return self.classes[self.class_to_index[class_id]]

    def support_at_least(self, level: FieldSupport) -> List[str]:
        """Class ids whose field support is at least ``level``."""
        floor = _SUPPORT_ORDER[level]
        return [c.id for c in self.classes if _SUPPORT_ORDER[c.field_support] >= floor]

    def lab_only_indices(self) -> List[int]:
        """Indices the decision policy must never report as a confident answer."""
        return [i for i, c in enumerate(self.classes) if c.field_support == "lab"]

    def confusion_groups(self) -> Dict[str, List[str]]:
        """Look-alike clusters, for the targeted confusion matrix in eval."""
        groups: Dict[str, List[str]] = {}
        for c in self.classes:
            if c.confusable_with:
                groups[c.id] = [c.id, *c.confusable_with]
        return groups

    def crop_group_mask(self) -> List[List[bool]]:
        """``mask[g][i] == True`` iff class i belongs to crop g.

        Used by the hierarchical head; returned as plain lists so the caller
        decides the tensor dtype/device.
        """
        return [
            [self.class_to_crop[i] == g for i in range(self.n_classes)]
            for g in range(self.n_crops)
        ]

    def describe(self) -> str:
        by_support: Dict[str, int] = {}
        for c in self.classes:
            by_support[c.field_support] = by_support.get(c.field_support, 0) + 1
        parts = ", ".join(f"{k}={v}" for k, v in sorted(by_support.items()))
        return f"Taxonomy(v2): {self.n_classes} classes / {self.n_crops} crops ({parts})"


def _normalise(label: str) -> str:
    """Whitespace/underscore/case-insensitive comparison key."""
    return " ".join(label.replace("_", " ").split()).lower()


def load_taxonomy(path: str | Path = "configs/taxonomy.yaml") -> Taxonomy:
    return Taxonomy.from_yaml(path)


__all__ = ["Taxonomy", "ClassInfo", "CropInfo", "load_taxonomy"]
