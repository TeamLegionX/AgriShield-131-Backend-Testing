"""Leakage-aware splitting.

Why not ``train_test_split(shuffle=True)``
------------------------------------------
PlantVillage contains many near-identical shots of the *same physical leaf*
(same specimen, small rotations, colour-cast variants). A random split puts
sibling frames on both sides of the fence, and the reported accuracy measures
memorisation of specimens, not recognition of disease. The same applies to
scraped datasets, where one blog photo is often duplicated across classes.

Two defences, both cheap:
  1. ``group_key`` = perceptual-hash bucket. Visually near-duplicate images get
     the same key and therefore land in the same split.
  2. Group-aware splitting on that key, stratified by class where possible.

And one policy: the *test* split is field-only. Lab images may train, may
validate for sanity, but never score the headline number.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image


# ---------------------------------------------------------------- hashing
def dhash(image: Image.Image, hash_size: int = 8) -> int:
    """64-bit difference hash. Robust to resize/JPEG, sensitive to content."""
    small = image.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = np.asarray(small, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    bits = diff.flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def assign_group_keys(
    manifest: pd.DataFrame,
    hash_size: int = 8,
    max_distance: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """Add a ``group_key`` column clustering near-duplicate images.

    Candidate generation uses **LSH banding**, not an exact hash-prefix match.
    The 64-bit dhash is cut into ``n_bands`` bands; two hashes are compared only
    if they agree *exactly* on at least one band. By pigeonhole, any pair within
    Hamming distance ``d`` must share a band whenever ``d < n_bands``, so with 8
    bands every pair at distance <= 5 is guaranteed to become a candidate.

    (The obvious-looking alternative — bucketing on the high 32 bits — silently
    misses most true duplicates, because a single flipped bit in the top half
    sends the pair to different buckets and it is never compared. That failure
    is invisible: you get a clean-looking split and inflated test metrics.)
    """
    df = manifest.copy()
    hashes: List[int] = []
    for path in df["path"]:
        try:
            with Image.open(path) as im:
                hashes.append(dhash(im, hash_size))
        except Exception:  # unreadable file -> unique group, never merged
            hashes.append(-abs(hash(path)))
    df["_dhash"] = hashes

    parent: Dict[int, int] = {i: i for i in range(len(df))}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    values = [int(v) for v in df["_dhash"].tolist()]
    total_bits = hash_size * hash_size
    n_bands = max(max_distance + 1, 8)
    band_bits = max(1, total_bits // n_bands)

    buckets: Dict[Tuple[int, int], List[int]] = {}
    for i, h in enumerate(values):
        if h < 0:  # unreadable file sentinel - never merged with anything
            continue
        for band in range(n_bands):
            key = (band, (h >> (band * band_bits)) & ((1 << band_bits) - 1))
            buckets.setdefault(key, []).append(i)

    for members in buckets.values():
        if len(members) < 2:
            continue
        if len(members) > 400:
            # Degenerate bucket (e.g. thousands of near-black thumbnails).
            # Comparing them all is quadratic and buys nothing.
            continue
        for a_pos in range(len(members)):
            for b_pos in range(a_pos + 1, len(members)):
                i, j = members[a_pos], members[b_pos]
                if find(i) != find(j) and hamming(values[i], values[j]) <= max_distance:
                    union(i, j)

    df["group_key"] = [f"g{find(i)}" for i in range(len(df))]
    df = df.drop(columns=["_dhash"])

    if verbose:
        n_groups = df["group_key"].nunique()
        dup = len(df) - n_groups
        print(f"[splits] {len(df)} images -> {n_groups} groups ({dup} near-duplicates absorbed)")
    return df


# ---------------------------------------------------------------- splitting
def group_aware_split(
    manifest: pd.DataFrame,
    val_fraction: float = 0.12,
    test_fraction: float = 0.15,
    field_only_test: bool = True,
    seed: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Add a ``split`` column in {train, val, test}.

    Greedy class-balanced group assignment: groups are walked rarest-class-first
    and pushed into whichever split is furthest below its quota for that class.
    Simple, deterministic, and it keeps tail classes present in val/test — which
    plain ``GroupShuffleSplit`` does not guarantee and which silently destroys
    macro-F1 reporting.
    """
    if "group_key" not in manifest.columns:
        raise ValueError("call assign_group_keys() before group_aware_split()")

    df = manifest.copy()
    rng = np.random.default_rng(seed)

    eligible_test = df["domain"] == "field" if field_only_test else pd.Series(True, index=df.index)

    group_info = (
        df.assign(_test_ok=eligible_test)
        .groupby("group_key")
        .agg(
            size=("path", "size"),
            class_index=("class_index", lambda s: s.mode().iat[0]),
            test_ok=("_test_ok", "all"),
        )
        .reset_index()
    )

    class_freq = df["class_index"].value_counts().to_dict()
    group_info["_rarity"] = group_info["class_index"].map(lambda c: class_freq.get(c, 0))
    group_info = group_info.sort_values(
        ["_rarity", "size"], ascending=[True, False], kind="mergesort"
    )

    quotas = {"train": 1.0 - val_fraction - test_fraction, "val": val_fraction, "test": test_fraction}
    filled: Dict[Tuple[int, str], float] = {}
    totals: Dict[int, float] = {}
    assignment: Dict[str, str] = {}

    for row in group_info.itertuples(index=False):
        cls = int(row.class_index)
        totals[cls] = totals.get(cls, 0.0) + row.size
        candidates = ["train", "val", "test"] if row.test_ok else ["train", "val"]
        deficits = []
        for split in candidates:
            have = filled.get((cls, split), 0.0)
            want = quotas[split] * totals[cls]
            deficits.append((want - have + rng.normal(0, 1e-6), split))
        _, chosen = max(deficits)
        assignment[row.group_key] = chosen
        filled[(cls, chosen)] = filled.get((cls, chosen), 0.0) + row.size

    df["split"] = df["group_key"].map(assignment)

    if field_only_test:
        # Belt and braces: no lab image ever reaches the test set.
        leaked = (df["split"] == "test") & (df["domain"] != "field")
        df.loc[leaked, "split"] = "train"

    if verbose:
        print("[splits] images per split / domain:")
        print(df.groupby(["split", "domain"]).size().unstack(fill_value=0).to_string())
        missing = _classes_missing_from(df, "test")
        if missing:
            print(f"[splits] WARNING: {len(missing)} classes absent from the field test set "
                  f"(they cannot be reported on): {missing[:10]}{'...' if len(missing) > 10 else ''}")
    return df


def _classes_missing_from(df: pd.DataFrame, split: str) -> List[str]:
    present = set(df.loc[df["split"] == split, "class_id"])
    return sorted(set(df["class_id"]) - present)


def verify_no_leakage(df: pd.DataFrame) -> None:
    """Hard assertion: no group_key straddles two splits."""
    straddling = df.groupby("group_key")["split"].nunique()
    bad = straddling[straddling > 1]
    if len(bad):
        raise AssertionError(f"{len(bad)} groups appear in more than one split, e.g. {bad.index[:5].tolist()}")


__all__ = ["dhash", "hamming", "assign_group_keys", "group_aware_split", "verify_no_leakage"]
