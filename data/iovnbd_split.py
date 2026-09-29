"""
data/iovnbd_split.py — deterministic, sequence-level train/val/test split
for the real IO-VNBD dataset.

Splits at the RUN level (never rows within a run), so no temporal leakage
is possible between train/val/test: a run's rows never appear in more
than one split. Stratified by driver_code where a category has enough
runs to split across all three partitions; small categories are kept
whole in a single split rather than being fragmented.

Usage:
    python -m data.iovnbd_split --zip "raw/Synchronised V abd S datasets.zip" \
        --out_json results/iovnbd/split_manifest.json
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from data.iovnbd import load_iovnbd_dataset, RunMeta

logger = logging.getLogger(__name__)

SEED = 42
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def make_split(metas: List[RunMeta], seed: int = SEED,
                val_frac: float = VAL_FRAC, test_frac: float = TEST_FRAC
                ) -> Dict[str, List[str]]:
    """Deterministic sequence-level split, stratified by driver_code.

    Only runs with n_rows_used > 0 (i.e. successfully parsed) are eligible.
    A driver category with fewer than 3 usable runs is placed entirely in
    train (too few sequences to safely hold out a representative val/test
    slice from it alone) — this is disclosed in the manifest, not hidden.
    """
    rng = random.Random(seed)
    usable = [m for m in metas if m.n_rows_used > 0]

    by_cat: Dict[str, List[RunMeta]] = defaultdict(list)
    for m in usable:
        by_cat[m.driver_code].append(m)

    train, val, test = [], [], []
    small_categories = []

    for cat, runs in by_cat.items():
        runs = sorted(runs, key=lambda m: m.name)  # deterministic order before shuffle
        rng.shuffle(runs)
        n = len(runs)
        if n < 3:
            train.extend(m.name for m in runs)
            small_categories.append(cat)
            continue
        n_test = max(1, round(n * test_frac))
        n_val = max(1, round(n * val_frac))
        n_test = min(n_test, n - 2)   # always leave >=2 for train
        n_val = min(n_val, n - n_test - 1)
        test.extend(m.name for m in runs[:n_test])
        val.extend(m.name for m in runs[n_test:n_test + n_val])
        train.extend(m.name for m in runs[n_test + n_val:])

    manifest = {
        "seed": seed,
        "val_frac_target": val_frac,
        "test_frac_target": test_frac,
        "split_level": "sequence (run)",
        "n_total_usable_runs": len(usable),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "small_categories_kept_in_train": small_categories,
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
    }
    return manifest


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--zip", required=True)
    p.add_argument("--out_json", default="results/iovnbd/split_manifest.json")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    _, metas = load_iovnbd_dataset(args.zip)
    manifest = make_split(metas, seed=args.seed)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"train={manifest['n_train']}  val={manifest['n_val']}  test={manifest['n_test']}")
    print(f"small categories kept whole in train: {manifest['small_categories_kept_in_train']}")
    print(f"Written to {args.out_json}")
