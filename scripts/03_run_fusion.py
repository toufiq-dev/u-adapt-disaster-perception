#!/usr/bin/env python
"""03_run_fusion.py — Phase 4: uncertainty-gated fusion over cached features.

Applies Mode A (analytic, training-free) or Mode B (logreg / MLP gates trained
on the calibration set, optionally initialized from COCO/LVIS-pretrained
weights — the former Mode C, now a Mode B initialization ablation per proposal
§5.4.3) to produce fused detection scores for every cached proposal.

Usage:
    # Mode A (primary, strict few-shot, T=1)
    python scripts/03_run_fusion.py --mode A \
        --cache-dir cached_features \
        --prototypes cached_features/prototypes_k5_seed0.json \
        --mode-config configs/modes/mode_A_analytic.yaml \
        --out outputs/scores_modeA.json

    # Mode B (lightweight calibration, reported separately)
    python scripts/03_run_fusion.py --mode B \
        --cache-dir cached_features \
        --prototypes cached_features/prototypes_k5_seed0.json \
        --mode-config configs/modes/mode_B_logreg.yaml \
        --calibration cached_features/calibration_set.json --out outputs/scores_modeB.json

    # Mode B with COCO/LVIS-pretrained gate init (ablation; the former Mode C)
    python scripts/03_run_fusion.py --mode B \
        --cache-dir cached_features \
        --prototypes cached_features/prototypes_k5_seed0.json \
        --mode-config configs/modes/mode_B_coco_lvis_init.yaml \
        --calibration cached_features/calibration_set.json \
        --gate-init cached_features/gate_coco_lvis_init.json --out outputs/scores_modeB_init.json

Mode B calibration JSON schema (``--calibration``): see the module docstring of
src/uadapt/fusion/calibration.py — 20 labeled boxes per class with normalized
5-D gate inputs plus text/visual correctness flags.

Note on score scale: Mode A now emits the FUSED score
``S_final = (1-w)*S_text + w*S_visual`` where ``S_text`` is the cached
per-class text similarity of the predicted class and ``S_visual`` is the
visual affinity (the documented proxy — see ``_run_mode_a``). Mode B emits
its min-max-normalized fused score. Both are in [0, 1] but on different
scales, so the modes are reported separately per pre-registration and the two
fields are not directly comparable numerically.

Outputs per-proposal fused scores + gate weights as JSON for 04_evaluate.py.
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
    parser.add_argument("--mode", choices=["A", "B"], required=True)
    parser.add_argument("--cache-dir", default="cached_features")
    parser.add_argument("--prototypes", type=Path, help="JSON from 02_build_prototypes.py")
    parser.add_argument("--mode-config", required=True, type=Path)
    parser.add_argument("--calibration", type=Path, help="Mode B: 20-box/class calibration set JSON")
    parser.add_argument(
        "--gate-init", type=Path, help="Mode B: COCO/LVIS-pretrained gate initialization (ablation; former Mode C)"
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--gate-type",
        choices=["analytic", "beta_fallback"],
        default="analytic",
        help="Mode A gate: analytic (default, pre-registered) or beta_fallback "
        "(pre-registered D5 Beta-regression variant for boundary-clustered "
        "variances)",
    )
    args = parser.parse_args()

    from uadapt.features.cache_engine import load_cache
    from uadapt.fusion.calibration import run_mode_b
    from uadapt.fusion.mode_a_analytic import BetaGate, ModeAGate

    cfg = load_yaml(args.mode_config)
    records = load_cache(args.cache_dir, split=args.split)

    if args.mode == "A":
        if not args.prototypes:
            raise SystemExit("Mode A requires --prototypes")
        payload = load_json(args.prototypes)
        if args.gate_type == "beta_fallback":
            # Pre-registered D5 fallback (docs/pre_registration.md §10):
            # Beta-linked gate for the boundary-clustered-variance regime.
            gate = BetaGate(
                alpha=cfg["coefficients"]["alpha"],
                beta=cfg["coefficients"]["beta"],
                gamma=cfg["coefficients"]["gamma"],
            )
            logger.info("Mode A using Beta-regression fallback gate")
        else:
            gate = ModeAGate(
                alpha=cfg["coefficients"]["alpha"],
                beta=cfg["coefficients"]["beta"],
                gamma=cfg["coefficients"]["gamma"],
                temperature=cfg.get("temperature", 1.0),
            )
        results = _run_mode_a(records, payload, gate)
    else:  # Mode B
        if not args.prototypes:
            raise SystemExit("Mode B requires --prototypes (from 02_build_prototypes.py)")
        if not args.calibration:
            raise SystemExit("Mode B requires --calibration (20-box/class calibration set JSON)")
        calibration = load_json(args.calibration)
        prototype_payload = load_json(args.prototypes)
        gate_init = load_json(args.gate_init) if args.gate_init is not None else None
        outcome = run_mode_b(
            records,
            calibration,
            prototype_payload,
            cfg,
            gate_init_payload=gate_init,
        )
        results = outcome.scores
        cv = outcome.cv_mean_std()
        gate_init_label = "coco_lvis" if gate_init is not None else "random"
        if cv is not None:
            logger.info(
                "Mode B (%s, gate-init=%s): T=%.3f, 5-fold CV MSE %.4f±%.4f",
                cfg.get("gate", "logistic_regression"),
                gate_init_label,
                outcome.temperature,
                cv[0],
                cv[1],
            )
        else:
            logger.info(
                "Mode B (%s, gate-init=%s): T=%.3f",
                cfg.get("gate", "logistic_regression"),
                gate_init_label,
                outcome.temperature,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("wrote %d fused scores -> %s", len(results), args.out)


def _run_mode_a(records, prototype_payload, gate) -> list[dict]:
    """Apply the analytic (or Beta-fallback) gate over cached records.

    Emits the FUSED score ``S_final = (1 - w) * S_text + w * S_visual``:

      * S_text   — cached per-class text similarity of the proposal's
                   predicted class (raw similarity in [0, 1]; same semantics
                   as ``uadapt.demo.pipeline._class_text_score``).
      * S_visual — visual affinity ``a = (1 + cos(f_box, p_visual)) / 2``.
                   There is no dedicated visual-only score in the cache, so
                   the affinity serves as the visual-score proxy — the
                   documented choice shared by ``uadapt.demo.pipeline`` and
                   Mode B (``uadapt.fusion.calibration``).

    Per-proposal variance terms (``norm_text_var`` / ``norm_visual_var``)
    are still emitted for ``scripts/04_evaluate.py``'s D1/D2 arrays
    (backward compatible: absent keys fall back to the neutral 0.5).
    """
    import numpy as np

    from uadapt.features.cache_engine import class_text_score
    from uadapt.fusion.mode_a_analytic import fuse_scores
    from uadapt.uncertainty.variance_estimators import (
        proposal_text_variance,
        visual_affinity,
    )

    results = []
    for rec in records:
        proto = prototype_payload["prototypes"].get(rec.class_name)
        if proto is None:
            continue
        affinity = visual_affinity(rec.visual_feature, np.asarray(proto["centroid"]))
        # REAL per-proposal text variance from the cached class-similarity
        # vector (normalized entropy, [0, 1]) — replaces the 0.5 placeholder
        # that zeroed D1 (change_log.md 2026-08-05). Visual variance remains
        # the class-level support dispersion (sigma_visual); a per-proposal
        # box-to-support term needs the support features persisted in the
        # prototype payload (02_build_prototypes.py), which is a follow-up.
        norm_text_var = proposal_text_variance(rec.text_similarities)
        norm_visual_var = min(1.0, float(proto["sigma_visual"]))
        w = gate.weight(norm_text_var, norm_visual_var, affinity)
        s_text = class_text_score(rec)
        s_visual = affinity
        fused = fuse_scores(s_text, s_visual, w)
        results.append(
            {
                "image_id": rec.image_id,
                "class": rec.class_name,
                "score": float(fused),
                "s_text": float(s_text),
                "s_visual": float(s_visual),
                "bbox": rec.bbox.tolist(),
                "gate_weight": float(w),
                "affinity": float(affinity),
                # Per-proposal variance terms consumed by 04_evaluate.py's
                # D1/D2 arrays (backward compatible: absent keys fall back to
                # the old neutral 0.5).
                "norm_text_var": float(norm_text_var),
                "norm_visual_var": float(norm_visual_var),
            }
        )
    return results


if __name__ == "__main__":
    main()
