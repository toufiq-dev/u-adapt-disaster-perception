#!/usr/bin/env python
"""01_extract_and_cache.py — Phase 1-2: candidate generation + feature caching.

Runs the FROZEN backbone once per image, limits proposals to top-k (k=100
primary; k=300 ablation only), extracts box features, and caches everything to
disk OUTSIDE the repository.

Usage:
    python scripts/01_extract_and_cache.py \
        --model-config configs/models/grounding_dino_swinT.yaml \
        --dataset-config configs/datasets/dfire.yaml \
        --split train \
        --cache-dir cached_features \
        --limit 200

Cached features are then consumed by 02_build_prototypes.py (no backbone
re-runs). See docs/pre_registration.md §Feature Caching.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Bootstrap so ``uadapt`` is importable when this script runs from any CWD
# (Colab, shell, notebook subprocess) without PYTHONPATH=src.
_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("01_extract_and_cache")


def load_yaml(path: Path) -> dict:
    import yaml

    with open(path) as fh:
        return yaml.safe_load(fh)


def iter_image_pairs(split_dir: Path, limit: int | None):
    """Lazily yield ``(image_rgb, image_id)`` for readable images in split_dir.

    RAM-safe streaming (2026-08-05): reads/decodes ONE image at a time and
    drops it after yielding, so peak image RAM is ~1 frame instead of the
    whole split. The old ``load_images`` decoded the entire split into a list
    first — a full LADD train split (~1,200 aerial images) would need tens of
    GB of decoded RAM and OOM on a laptop. Image ids are filename stems (the
    cache/GT key used by demo_mode_a_end_to_end.py).
    """
    import cv2

    paths = sorted(p for p in split_dir.iterdir() if p.suffix in {".jpg", ".jpeg", ".png"})
    if limit:
        paths = paths[:limit]
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            logger.warning("could not read %s", p)
            continue
        yield cv2.cvtColor(img, cv2.COLOR_BGR2RGB), p.stem


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract + cache features (frozen backbone).")
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--dataset-config", required=True, type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--cache-dir", default="cached_features")
    parser.add_argument("--top-k", type=int, default=None, help="default from config; 300 only as ablation")
    parser.add_argument("--limit", type=int, default=None, help="max images (pilot)")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    model_cfg = load_yaml(args.model_config)
    dataset_cfg = load_yaml(args.dataset_config)
    classes = list(dataset_cfg["classes"])

    from uadapt.features.cache_engine import DEFAULT_TOP_K, ABLATION_TOP_K, FeatureCacheEngine
    from uadapt.models.backbone_loader import load_backbone

    top_k = args.top_k or model_cfg["inference"].get("top_k", DEFAULT_TOP_K)
    if top_k not in (DEFAULT_TOP_K, ABLATION_TOP_K):
        logger.warning(
            "top_k=%d is not a pre-registered value (primary=%d, ablation=%d)",
            top_k, DEFAULT_TOP_K, ABLATION_TOP_K,
        )

    split_dir = Path(dataset_cfg["splits"][args.split])
    # Stream images one at a time (RAM-safe; see iter_image_pairs).
    pairs = iter_image_pairs(split_dir, args.limit)
    logger.info("loading backbone %s (device from config)", model_cfg["name"])
    backbone = load_backbone(model_cfg, device=model_cfg["inference"].get("device", "cuda"))

    engine = FeatureCacheEngine(
        backbone=backbone,
        cache_dir=args.cache_dir,
        top_k=top_k,
    )
    out = engine.extract_and_cache(
        pairs, classes, split=args.split, image_ids=None, resume=not args.no_resume
    )
    logger.info("done. cache at %s", out)


if __name__ == "__main__":
    main()
