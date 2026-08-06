#!/usr/bin/env python
"""04_evaluate.py — Phase 5 evaluation + baselines + diagnostics.

Evaluates, on the test split:
  * zero-shot Grounding DINO (raw detector scores)
  * text-only baseline (S_text)
  * visual-only baseline (S_visual)
  * naive averaging baseline (T-Rex2-style w = 0.5)
  * U-ADAPT Mode A (and Mode B / C when fused scores are provided)

Metrics reported: mAP50, mAP50:95, per-class AP, Gap Recovery, ECE (15
bins), Brier score, uncertainty AUROC, proposal recall (ceiling), and
diagnostics D1-D5.

Usage:
    python scripts/04_evaluate.py \
        --predictions outputs/scores_modeA.json \
        --ground-truth data/annotations/dfire_test.json \
        --zero-shot-map50 27.5 --oracle-map50 65.6 \
        --out outputs/results_modeA.json

    # Pooled D1/D2/D3 (pre-registration deviation 2026-08-03, §10): D-Fire
    # alone has only 2 classes -> 2 distinct variance values, so D1/D2/D3 are
    # structurally underpowered. Evaluate on D-Fire while pooling diagnostics
    # across LADD (3 distinct classes: pedestrian, fire, smoke):
    python scripts/04_evaluate.py \
        --predictions outputs/scores_dfire.json \
        --ground-truth data/annotations/dfire_test.json \
        --pool-predictions outputs/scores_ladd.json \
        --pool-ground-truth data/annotations/ladd_test.json \
        --out outputs/results_dfire_pooled.json
    # Result JSON then contains diagnostics_ladd (per-dataset),
    # diagnostics_dfire (per-dataset) and diagnostics_pooled (PRIMARY claim).
    # Note: in a pooled run the full metric set (mAP50, ECE, ...) is computed
    # for the PRIMARY dataset only; the second dataset contributes its
    # diagnostic arrays. Run the script twice (once with each dataset as
    # --predictions) to get full per-dataset metrics for both.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Bootstrap so ``uadapt`` is importable when this script runs from any CWD
# (Colab, shell, notebook subprocess) without PYTHONPATH=src.
_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("04_evaluate")


def load_json(path: Path):
    with open(path) as fh:
        return json.load(fh)


def _coco_to_gt(coco: Dict) -> List[Dict]:
    """Convert COCO-style annotations to the shared gt schema.

    Cache image ids are filename stems (01_extract_and_cache.py) while GT
    image ids may be sequential COCO ints (mask->box conversion; D-Fire) or
    already stems (LADD). The images-table remap below mirrors
    ``demo_mode_a_end_to_end.py::_load_real_data`` (change_log 2026-08-07):
    without it, D-Fire proposals (stem ids) never matched GT (int ids) and
    every scripted-path mAP50 was 0.0. LADD-style stem ids are not keys of
    the remap and pass through unchanged.
    """
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    id_to_stem = {
        str(img.get("id")): Path(img["file_name"]).stem
        for img in coco.get("images", [])
        if img.get("file_name")
    }
    gts: List[Dict] = []
    for ann in coco["annotations"]:
        b = ann["bbox"]
        gts.append(
            {
                "image_id": id_to_stem.get(
                    str(ann["image_id"]), str(ann["image_id"])
                ),
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
    # Pooled D1/D2/D3 (pre-registration deviation 2026-08-03, §10). On D-Fire
    # alone (2 classes -> 2 distinct variance values) the diagnostics are
    # structurally underpowered; the pre-registered protocol evaluates them
    # pooled across LADD + D-Fire and reports the pooled values as the
    # primary claim (per-dataset values are still reported).
    parser.add_argument(
        "--pool-predictions",
        type=Path,
        help="predictions of the second dataset (e.g. LADD) used to compute "
        "pooled D1/D2/D3 diagnostics",
    )
    parser.add_argument(
        "--pool-ground-truth",
        type=Path,
        help="ground truth of the second dataset (must be given together with "
        "--pool-predictions)",
    )
    parser.add_argument(
        "--pool-primary-name",
        default="ladd",
        help="suffix of the primary dataset's per-dataset diagnostics key "
        "(default: ladd) -> diagnostics_ladd",
    )
    parser.add_argument(
        "--pool-secondary-name",
        default="dfire",
        help="suffix of the second dataset's per-dataset diagnostics key "
        "(default: dfire) -> diagnostics_dfire",
    )
    args = parser.parse_args()

    if (args.pool_predictions is None) != (args.pool_ground_truth is None):
        parser.error(
            "--pool-predictions and --pool-ground-truth must be provided together"
        )

    from uadapt.metrics.calibration_metrics import brier_score, ece, uncertainty_auroc
    from uadapt.metrics.detection_metrics import (
        compute_map50,
        compute_map50_95,
        compute_per_class_ap,
        gap_recovery,
        proposal_recall,
    )

    preds = _load_preds(args.predictions)
    gts = _coco_to_gt(load_json(args.ground_truth))

    pool_preds: Optional[List[Dict]] = None
    pool_gts: Optional[List[Dict]] = None
    if args.pool_predictions is not None:
        pool_preds = _load_preds(args.pool_predictions)
        pool_gts = _coco_to_gt(load_json(args.pool_ground_truth))

    map50 = compute_map50(preds, gts)
    map50_95 = compute_map50_95(preds, gts)
    per_class_ap = compute_per_class_ap(preds, gts)
    recall = proposal_recall(preds, gts)
    confs = np.asarray([p["score"] for p in preds], dtype=float)
    # Correctness needs GT matching; simplified here as proposal-level IoU.
    correct = _proposal_correct(preds, gts)

    results = {
        "mAP50": map50,
        "mAP50_95": map50_95,
        "per_class_AP": per_class_ap,
        "proposal_recall_ceiling": recall,
        "ECE_15bin": ece(confs, correct),
        "brier": brier_score(confs, correct),
        "uncertainty_auroc": uncertainty_auroc(1.0 - confs, correct),
    }
    if args.zero_shot_map50 is not None and args.oracle_map50 is not None:
        results["gap_recovery"] = gap_recovery(map50, args.zero_shot_map50, args.oracle_map50)
        results["zero_shot_floor"] = args.zero_shot_map50
        results["oracle_ceiling"] = args.oracle_map50

    # Diagnostics D1-D5 (pre-registered; computed AFTER main results). When a
    # second dataset is supplied, D1/D2/D3 are ALSO computed pooled across
    # both datasets (PRIMARY claim, deviation 2026-08-03) and reported
    # per-dataset under diagnostics_{ladd,dfire} plus pooled under
    # diagnostics_pooled. Without pooling args the output is unchanged.
    if pool_preds is not None:
        pooled = _run_diagnostics(
            preds, correct, pool_with=(pool_preds, _proposal_correct(pool_preds, pool_gts))
        )
        results[f"diagnostics_{args.pool_primary_name}"] = pooled["primary"]
        results[f"diagnostics_{args.pool_secondary_name}"] = pooled["secondary"]
        results["diagnostics_pooled"] = pooled["pooled"]
    else:
        results["diagnostics"] = _run_diagnostics(preds, correct)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps(results, indent=2))


def _load_preds(path: Path) -> List[Dict]:
    """Load predictions and rename the fused-score key into 'score'."""
    preds = load_json(path)
    for p in preds:
        if "score" not in p and "fused_score" in p:
            p["score"] = p["fused_score"]
    return preds


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


def _diag_arrays(preds: List[Dict], correct: np.ndarray) -> Dict[str, np.ndarray]:
    """Extract the variance / gate arrays a diagnostic run needs.

    03_run_fusion.py emits real per-proposal ``norm_text_var`` (normalized
    class-similarity entropy) and ``norm_visual_var`` (support dispersion)
    since change_log 2026-08-05; when a predictions file predates that
    (or was produced by a path without the terms), the old neutral 0.5 is
    kept so the machinery stays runnable. The D4 counterfactual
    ``w_gamma_0`` (gate weight with the affinity term zeroed, gamma = 0)
    is emitted since change_log 2026-08-07; stale files without it fall
    back to 0.5 (the pre-2026-08-07 constant used by D4).

    D3's disagreeing subsets follow the pipeline /
    compute_pooled_diagnostics.py convention (change_log 2026-08-05):
    ``text_ok`` = the text modality's top-1 class matched a GT box
    (== ``correct`` on the real path) and ``visual_ok`` = affinity
    >= ``VISUAL_CORRECT_AFFINITY`` (0.65, the same pre-registered threshold
    as ``src/uadapt/demo/pipeline.py``). Explicit per-proposal
    ``text_correct`` / ``visual_correct`` fields win when a predictions file
    carries them; otherwise they are derived here so the disagreement
    subsets stay defined for any input. Note: a predictions file WITHOUT
    ``affinity`` degrades ``visual_ok`` to all-False (one-sided text-only
    subset) — well-defined, but unlike the old w-split which was trivially
    favorable by construction.
    """
    # Pre-registered D3 convention (demos/pipeline.py defines the canonical
    # VISUAL_CORRECT_AFFINITY; kept local here so this standalone eval script
    # does not import the demo module).
    VISUAL_CORRECT_AFFINITY = 0.65

    return {
        "norm_text_var": np.asarray(
            [p.get("norm_text_var", 0.5) for p in preds], dtype=float
        ),
        "norm_visual_var": np.asarray(
            [p.get("norm_visual_var", 0.5) for p in preds], dtype=float
        ),
        # Single fallback for a missing affinity (0.0 = "not visual"), used by
        # both D4's binning and the D3 visual_ok derivation below.
        "affinities": np.asarray([p.get("affinity", 0.0) for p in preds], dtype=float),
        "w": np.asarray([p.get("gate_weight", 0.5) for p in preds], dtype=float),
        # D4 counterfactual weight w_{gamma=0} (affinity term zeroed); stale
        # files fall back to the old constant 0.5.
        "w_gamma_0": np.asarray(
            [p.get("w_gamma_0", 0.5) for p in preds], dtype=float
        ),
        "correct": np.asarray(correct, dtype=bool),
        # D3 disagreeing subsets (pipeline / compute_pooled convention).
        "text_ok": np.asarray(
            [p.get("text_correct", c) for p, c in zip(preds, correct)], dtype=bool
        ),
        "visual_ok": np.asarray(
            [
                p.get(
                    "visual_correct",
                    p.get("affinity", 0.0) >= VISUAL_CORRECT_AFFINITY,
                )
                for p in preds
            ],
            dtype=bool,
        ),
    }


def _diag_dict(r) -> Dict:
    """Serialize one DiagnosticResult into the JSON schema (summary + flag + raw)."""
    out = {"summary": r.summary, "flag": r.flag}
    if r.raw is not None:
        out["raw"] = r.raw
    return out


def _per_dataset_dict(
    preds: List[Dict], correct: np.ndarray, arr: Optional[Dict[str, np.ndarray]] = None
) -> Dict:
    """Run D1-D5 on ONE dataset (no pooling).

    Args:
        preds: proposals of the dataset.
        correct: per-proposal correctness.
        arr: optional precomputed :func:`_diag_arrays` output, to avoid
            re-deriving the same arrays when the caller already has them.
    """
    from uadapt.metrics.diagnostics import (
        d1_text_uncertainty_accuracy,
        d2_visual_uncertainty_accuracy,
        d3_gate_favorability,
        d4_affinity_diagnostic,
        d5_variance_distribution,
    )

    a = arr if arr is not None else _diag_arrays(preds, correct)
    d1 = d1_text_uncertainty_accuracy(a["norm_text_var"], a["correct"])
    d2 = d2_visual_uncertainty_accuracy(a["norm_visual_var"], a["correct"])
    # D3: disagreeing-proposal weight subsets, aligned with the pipeline /
    # compute_pooled convention (change_log 2026-08-05). The previous
    # w < 0.5 / w > 0.5 split counted every proposal as favorable by
    # construction (any weight is on one side of 0.5), so D3 could never
    # drop below 100% — it measured nothing.
    text_better = a["text_ok"] & ~a["visual_ok"]
    visual_better = a["visual_ok"] & ~a["text_ok"]
    d3 = d3_gate_favorability(a["w"][text_better], a["w"][visual_better])
    # D4: the TRUE pre-registered counterfactual — the per-proposal gate
    # weight with the affinity term zeroed (w_gamma_0, emitted by
    # 03_run_fusion.py). Previously a constant 0.5 array was used, which
    # measured w - 0.5 (a monotone transform of w) instead of the
    # affinity-induced shift Delta w = w - w_{gamma=0} (2026-08-07).
    d4 = d4_affinity_diagnostic(a["w"], a["w_gamma_0"], a["affinities"])
    # NOTE (2026-08-05): this D5 runs on the normalized arrays carried in the
    # predictions JSON, while compute_pooled_diagnostics.py runs D5 on the raw
    # absolute-scale per-proposal values (text_entropy, distance/2.0). The two
    # scripts intentionally use different scales for the same diagnostic name;
    # the pooled-diagnostics path is the PRIMARY real-data one.
    d5 = d5_variance_distribution(a["norm_text_var"], a["norm_visual_var"])
    return {
        d1.name: _diag_dict(d1),
        d2.name: _diag_dict(d2),
        d3.name: _diag_dict(d3),
        d4.name: _diag_dict(d4),
        d5.name: _diag_dict(d5),
    }


def _run_diagnostics(
    preds: List[Dict],
    correct: np.ndarray,
    pool_with: Optional[Tuple[List[Dict], np.ndarray]] = None,
) -> Dict:
    """Compute diagnostics D1-D5 for one dataset, optionally pooled across two.

    Args:
        preds: proposals of the PRIMARY dataset.
        correct: per-proposal correctness of the PRIMARY dataset.
        pool_with: optional ``(pool_preds, pool_correct)`` of a SECOND dataset
            (e.g. LADD when the primary is D-Fire). When provided, returns
            ``{"primary": {...}, "secondary": {...}, "pooled": {...}}`` where
            ``pooled`` holds D1/D2/D3 computed on the CONCATENATED proposals
            (the PRIMARY diagnostic claim, deviation 2026-08-03) and
            primary/secondary hold per-dataset D1-D5 for reporting.

    Returns:
        Per-dataset D1-D5 dict when ``pool_with`` is None (backward
        compatible), else the structured primary/secondary/pooled dict.
    """
    if pool_with is None:
        return _per_dataset_dict(preds, correct)

    from uadapt.metrics.diagnostics import (
        d1_text_uncertainty_accuracy,
        d2_visual_uncertainty_accuracy,
        d3_gate_favorability,
    )

    pool_preds, pool_correct = pool_with
    pa = _diag_arrays(preds, correct)
    sa = _diag_arrays(pool_preds, pool_correct)
    # The D-functions themselves validate and concatenate the arrays; the
    # pooled result is the pre-registered PRIMARY claim.
    d1 = d1_text_uncertainty_accuracy(
        pa["norm_text_var"], pa["correct"],
        pool_with=(sa["norm_text_var"], sa["correct"]),
    )
    d2 = d2_visual_uncertainty_accuracy(
        pa["norm_visual_var"], pa["correct"],
        pool_with=(sa["norm_visual_var"], sa["correct"]),
    )
    # D3 pooled on the disagreeing-proposal weight subsets of BOTH datasets
    # (same convention as the per-dataset path, change_log 2026-08-05).
    p_text_better = pa["text_ok"] & ~pa["visual_ok"]
    p_visual_better = pa["visual_ok"] & ~pa["text_ok"]
    s_text_better = sa["text_ok"] & ~sa["visual_ok"]
    s_visual_better = sa["visual_ok"] & ~sa["text_ok"]
    d3 = d3_gate_favorability(
        pa["w"][p_text_better], pa["w"][p_visual_better],
        pool_with=(sa["w"][s_text_better], sa["w"][s_visual_better]),
    )
    return {
        "primary": _per_dataset_dict(preds, correct, arr=pa),
        "secondary": _per_dataset_dict(pool_preds, pool_correct, arr=sa),
        "pooled": {
            "D1_text_uncertainty_accuracy": _diag_dict(d1["pooled"]),
            "D2_visual_uncertainty_accuracy": _diag_dict(d2["pooled"]),
            "D3_gate_favorability": _diag_dict(d3["pooled"]),
        },
    }


if __name__ == "__main__":
    main()
