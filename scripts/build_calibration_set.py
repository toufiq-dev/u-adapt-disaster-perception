#!/usr/bin/env python
"""build_calibration_set.py — Mode B calibration-set sampling (pre-registered).

Samples the 20-box-per-class calibration set for the Mode B gate from the
cached TRAIN split, strictly disjoint from the test split and from the
seed's k-shot support examples:

    * boxes are sampled per class (stratified, up to ``--boxes-per-class``),
      seeded and deterministic;
    * only same-class GT-matched boxes (IoU >= 0.5) are eligible ("labeled
      boxes", proposal §5.4.2);
    * the support image ids recorded in the ``--prototypes`` payload
      (02_build_prototypes.py output) are excluded.

Usage:
    python scripts/build_calibration_set.py \
        --cache-dir cached_features/ladd \
        --ground-truth data/annotations/ladd_train.json \
        --prototypes cached_features/ladd/prototypes_k5_seed0.json \
        --boxes-per-class 20 \
        --seed 0 \
        --out cached_features/ladd/calibration_k5_seed0.json

Output JSON schema: see src/uadapt/fusion/calibration.py module docstring
(--calibration file), extended with a ``sampling`` audit block.

NOTE (n=100 pilot scale): with the tiny pilot train caches (LADD 10 images,
D-Fire 9 images) some classes may yield fewer than 20 eligible boxes; the
sampler keeps ALL eligible boxes and records the true counts under
``sampling.per_class_eligible``.
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
logger = logging.getLogger("build_calibration_set")


def load_json(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def coco_to_gt(coco: dict) -> list[dict]:
    """COCO annotations -> shared gt schema {image_id, class, bbox}.

    Mirrors ``scripts/04_evaluate.py::_coco_to_gt`` (including the
    sequential-COCO-int -> filename-stem remap for D-Fire) so the sampler and
    the evaluator agree on image ids.
    """
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    id_to_stem = {
        str(img.get("id")): Path(img["file_name"]).stem
        for img in coco.get("images", [])
        if img.get("file_name")
    }
    gts: list[dict] = []
    for ann in coco["annotations"]:
        b = ann["bbox"]
        gts.append(
            {
                "image_id": id_to_stem.get(str(ann["image_id"]), str(ann["image_id"])),
                "class": cat_name.get(ann["category_id"], "?"),
                "bbox": [b[0], b[1], b[0] + b[2], b[1] + b[3]],
            }
        )
    return gts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Mode B 20-box/class calibration set from the "
                    "cached train split (pre-registration §5)."
    )
    parser.add_argument("--cache-dir", required=True, type=Path,
                        help="dataset feature cache (contains train/ split)")
    parser.add_argument("--ground-truth", required=True, type=Path,
                        help="COCO train-split annotation JSON")
    parser.add_argument("--prototypes", required=True, type=Path,
                        help="JSON from 02_build_prototypes.py (support ids + "
                             "prototype stats)")
    parser.add_argument("--boxes-per-class", type=int, default=20,
                        help="pre-registered calibration size (default: 20)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for deterministic sampling")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    from uadapt.features.cache_engine import load_cache
    from uadapt.fusion.calibration_set import build_calibration_set

    records = load_cache(args.cache_dir, split="train")
    gt = coco_to_gt(load_json(args.ground_truth))
    protos = load_json(args.prototypes)
    payload = build_calibration_set(
        records,
        gt,
        protos,
        boxes_per_class=args.boxes_per_class,
        seed=args.seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)

    per_class = payload["sampling"]["per_class_sampled"]
    logger.info(
        "wrote calibration set (%d samples, per class %s) -> %s",
        len(payload["samples"]), per_class, args.out,
    )


if __name__ == "__main__":
    main()
