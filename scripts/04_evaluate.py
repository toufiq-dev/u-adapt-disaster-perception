#!/usr/bin/env python
"""04_evaluate.py — Phase 5 evaluation + baselines + diagnostics.

Evaluates, on the test split:
  * zero-shot Grounding DINO (raw detector scores)
  * text-only baseline (S_text)
  * visual-only baseline (S_visual)
  * naive averaging baseline (T-Rex2-style w = 0.5)
  * U-ADAPT Mode A (and Mode B / C when fused scores are provided)

Metrics reported: mAP50, Gap Recovery, ECE (15 bins), Brier score,
uncertainty AUROC, proposal recall (ceiling), and diagnostics D1-D5.

Usage:
    python scripts/04_evaluate.py \
        --predictions outputs/scores_modeA.json \
        --ground-truth data/annotations/dfire_test.json \
        --zero-shot-map50 27.5 --oracle-map50 65.6 \
        --out outputs/results_modeA.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("04_evaluate")


def load_json(path: Path):
    with open(path) as fh:
        return json.load(fh)


def _coco_to_gt(coco: Dict) -> List[Dict]:
    """Convert COCO-style annotations to the shared gt schema."""
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    gts: List[Dict] = []
    for ann in coco["annotations"]:
        b = ann["bbox"]
        gts.append(
            {
                "image_id": str(ann["image_id"]),
                "class": cat_name.get(ann["category_id"], "?"),
                "bbox": [b[0], b[1], b[0] + b[2], b[1] + b[3]],
            }
        )
    return gts


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate U-ADAPT vs baselines.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--zero-shot-map50", type=float, help="pre-registered zero-shot floor")
    parser.add_argument("--oracle-map50", type=float, help="pre-registered transfer upper bound")
    parser.add_argument("--out", default="outputs/results.json", type=Path)
    args = parser.parse_args()

    from uadapt.metrics.calibration_metrics import brier_score, ece, uncertainty_auroc
    from uadapt.metrics.detection_metrics import compute_map50, gap_recovery, proposal_recall

    preds = load_json(args.predictions)
    gts = _coco_to_gt(load_json(args.ground_truth))

    # Rename fused-score key into the metric schema ('score').
    for p in preds:
        if "score" not in p and "fused_score" in p:
            p["score"] = p["fused_score"]

    map50 = compute_map50(preds, gts)
    recall = proposal_recall(preds, gts)
    confs = np.asarray([p["score"] for p in preds], dtype=float)
    # Correctness needs GT matching; simplified here as proposal-level IoU.
    correct = _proposal_correct(preds, gts)

    results = {
        "mAP50": map50,
        "proposal_recall_ceiling": recall,
        "ECE_15bin": ece(confs, correct),
        "brier": brier_score(confs, correct),
        "uncertainty_auroc": uncertainty_auroc(1.0 - confs, correct),
    }
    if args.zero_shot_map50 is not None and args.oracle_map50 is not None:
        results["gap_recovery"] = gap_recovery(map50, args.zero_shot_map50, args.oracle_map50)
        results["zero_shot_floor"] = args.zero_shot_map50
        results["oracle_ceiling"] = args.oracle_map50

    # Diagnostics D1-D5 (pre-registered; computed AFTER main results).
    results["diagnostics"] = _run_diagnostics(preds, correct)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps(results, indent=2))


def _proposal_correct(preds: List[Dict], gts: List[Dict]) -> np.ndarray:
    """Proposal-level correctness: IoU >= 0.5 with a GT box of the SAME class
    (per pre-registered D1/D2 criterion)."""
    from uadapt.metrics.detection_metrics import _iou

    gt_by = {}
    for g in gts:
        gt_by.setdefault(g["image_id"], []).append(g)
    correct = []
    for p in preds:
        hit = False
        for g in gt_by.get(p["image_id"], []):
            if g["class"] == p["class"] and _iou(np.asarray(p["bbox"]), np.asarray(g["bbox"])) >= 0.5:
                hit = True
                break
        correct.append(hit)
    return np.asarray(correct, dtype=bool)


def _run_diagnostics(preds: List[Dict], correct: np.ndarray) -> Dict:
    from uadapt.metrics.diagnostics import (
        d1_text_uncertainty_accuracy,
        d2_visual_uncertainty_accuracy,
        d3_gate_favorability,
        d4_affinity_diagnostic,
        d5_variance_distribution,
    )

    # Placeholders: real normalized variances flow from Mode A wiring (Milestone 5).
    n = len(preds)
    norm_text_var = np.full(n, 0.5)
    norm_visual_var = np.full(n, 0.5)
    affinities = np.asarray([p.get("affinity", 0.5) for p in preds], dtype=float)
    w = np.asarray([p.get("gate_weight", 0.5) for p in preds], dtype=float)

    out = {}
    for diag in (
        d1_text_uncertainty_accuracy(norm_text_var, correct),
        d2_visual_uncertainty_accuracy(norm_visual_var, correct),
        d3_gate_favorability(w[w < 0.5], w[w > 0.5]),
        d4_affinity_diagnostic(w, np.full_like(w, 0.5), affinities),
        d5_variance_distribution(norm_text_var, norm_visual_var),
    ):
        out[diag.name] = {"summary": diag.summary, "flag": diag.flag}
    return out


if __name__ == "__main__":
    main()
