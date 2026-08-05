#!/usr/bin/env python
"""02_build_prototypes.py — Phase 3: prototype construction from cached features.

Builds text prototypes (M=20 prompt templates) and visual prototypes (k in
{1,3,5} support boxes, pre-registered outlier rejection) from the feature
cache produced by 01_extract_and_cache.py.

Usage:
    python scripts/02_build_prototypes.py \
        --cache-dir cached_features \
        --dataset-config configs/datasets/dfire.yaml \
        --shots 5 \
        --seed 0 \
        --out cached_features/prototypes_k5_seed0.json

Outputs a JSON file consumed by 03_run_fusion.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Bootstrap so ``uadapt`` is importable when this script runs from any CWD
# (Colab, shell, notebook subprocess) without PYTHONPATH=src.
_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("02_build_prototypes")


def load_yaml(path: Path) -> dict:
    import yaml

    with open(path) as fh:
        return yaml.safe_load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build prototypes from cached features.")
    parser.add_argument("--cache-dir", default="cached_features")
    parser.add_argument("--dataset-config", required=True, type=Path)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--no-outlier-rejection", action="store_true")
    args = parser.parse_args()

    from uadapt.features.cache_engine import load_cache
    from uadapt.prototypes.prototype_builder import build_visual_prototypes

    dataset_cfg = load_yaml(args.dataset_config)
    # Configs may declare classes as a plain list (``classes: [fire, smoke]``)
    # or as a dict of {class: {retained: bool}} for partial-class runs. Both
    # forms are accepted here; retained=False entries are excluded.
    raw_classes = dataset_cfg["classes"]
    if isinstance(raw_classes, dict):
        classes = [c for c, info in raw_classes.items() if info.get("retained", True)]
    else:
        classes = list(raw_classes)

    records = load_cache(args.cache_dir, split="train")
    rng = __import__("numpy").random.default_rng(args.seed)
    prototypes = build_visual_prototypes(
        records,
        classes,
        shots=args.shots,
        rng=rng,
        reject_outliers_flag=not args.no_outlier_rejection,
    )

    payload = {
        "shots": args.shots,
        "seed": args.seed,
        "dataset": dataset_cfg["name"],
        "classes": classes,
        "prototypes": {
            cls: {
                "centroid": p.centroid.tolist(),
                "sigma_visual": p.sigma_visual,
                "n_kept": p.n_kept,
                "support_ids": p.support_ids,
            }
            for cls, p in prototypes.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("wrote prototypes for %d classes -> %s", len(prototypes), args.out)


if __name__ == "__main__":
    main()
