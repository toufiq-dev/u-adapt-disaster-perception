#!/usr/bin/env python
"""03_run_fusion.py — Phase 4: uncertainty-gated fusion over cached features.

Applies Mode A (analytic, training-free), Mode B (logreg / MLP gates trained
on the calibration set), or Mode C (source-learned coefficients, exploratory)
to produce fused detection scores for every cached proposal.

Usage:
    # Mode A (primary, strict few-shot, T=1)
    python scripts/03_run_fusion.py --mode A \
        --cache-dir cached_features \
        --prototypes cached_features/prototypes_k5_seed0.json \
        --mode-config configs/modes/mode_A_analytic.yaml \
        --out outputs/scores_modeA.json

    # Mode B (lightweight calibration, reported separately)
    python scripts/03_run_fusion.py --mode B \
        --mode-config configs/modes/mode_B_logreg.yaml \
        --calibration cached_features/calibration_set.json --out outputs/scores_modeB.json

Outputs per-proposal fused scores + gate weights as JSON for 04_evaluate.py.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("03_run_fusion")


def load_yaml(path: Path) -> dict:
    import yaml

    with open(path) as fh:
        return yaml.safe_load(fh)


def load_json(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run uncertainty-gated fusion on cached features.")
    parser.add_argument("--mode", choices=["A", "B", "C"], required=True)
    parser.add_argument("--cache-dir", default="cached_features")
    parser.add_argument("--prototypes", type=Path, help="JSON from 02_build_prototypes.py")
    parser.add_argument("--mode-config", required=True, type=Path)
    parser.add_argument("--calibration", type=Path, help="Mode B: 20-box/class calibration set JSON")
    parser.add_argument("--source-artifacts", type=Path, help="Mode C: source-learned coefficients/MLP")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    from uadapt.features.cache_engine import load_cache
    from uadapt.fusion.mode_a_analytic import ModeAGate

    cfg = load_yaml(args.mode_config)
    records = load_cache(args.cache_dir, split=args.split)

    if args.mode == "A":
        if not args.prototypes:
            raise SystemExit("Mode A requires --prototypes")
        payload = load_json(args.prototypes)
        gate = ModeAGate(
            alpha=cfg["coefficients"]["alpha"],
            beta=cfg["coefficients"]["beta"],
            gamma=cfg["coefficients"]["gamma"],
            temperature=cfg.get("temperature", 1.0),
        )
        results = _run_mode_a(records, payload, gate)
    elif args.mode == "B":
        raise SystemExit(
            "Mode B wiring (logreg/MLP gate + calibration set) lands in Milestone 6; "
            "see tests/test_mode_a_gate.py for the gate unit tests."
        )
    else:  # Mode C
        raise SystemExit("Mode C (source-domain transfer) lands in Milestone 8 (exploratory).")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("wrote %d fused scores -> %s", len(results), args.out)


def _run_mode_a(records, prototype_payload, gate: ModeAGate) -> list[dict]:
    """Analytic gate over cached records using the text/visual prototypes.

    NOTE: full wiring (text-prototype similarities from CLIP embeddings,
    support-set normalization statistics) is completed in Milestone 5; the
    structure below is the intended data flow.
    """
    import numpy as np

    from uadapt.uncertainty.variance_estimators import visual_affinity

    results = []
    for rec in records:
        proto = prototype_payload["prototypes"].get(rec.class_name)
        if proto is None:
            continue
        affinity = visual_affinity(rec.visual_feature, np.asarray(proto["centroid"]))
        # Placeholders until Milestone 5: normalized text/visual variances from
        # cached text_similarities + prototype sigma values.
        norm_text_var = 0.5
        norm_visual_var = min(1.0, float(proto["sigma_visual"]))
        w = gate.weight(norm_text_var, norm_visual_var, affinity)
        results.append(
            {
                "image_id": rec.image_id,
                "class": rec.class_name,
                "score": float(rec.score),
                "bbox": rec.bbox.tolist(),
                "gate_weight": float(w),
                "affinity": float(affinity),
            }
        )
    return results


if __name__ == "__main__":
    main()
