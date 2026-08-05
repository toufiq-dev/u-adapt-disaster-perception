#!/usr/bin/env python
"""compute_pooled_diagnostics.py — pooled D1/D2/D3 across LADD + D-Fire.

Pre-registration deviation 2026-08-03 (§10, docs/change_log.md): D-Fire's 2
classes (fire, smoke) yield only 2 distinct normalized variance values, so
D1/D2/D3 computed on D-Fire alone are STRUCTURALLY UNDERPOWERED — no
meaningful Spearman correlation / gate-favorability trend can be computed
from 2 data points. The pre-registered protocol therefore computes D1/D2/D3
POOLED across LADD + D-Fire (3 distinct classes: pedestrian, fire, smoke);
the pooled values are the PRIMARY diagnostic claim, per-dataset values are
still reported.

This script consumes the outputs of ``demo_mode_a_end_to_end.py``:

    * ``results.json``          — summary metrics (map50, gap_recovery, meta)
    * ``proposal_level.json``   — per-proposal rows (normalized variances,
                                  text/visual correctness flags, gate weights)

and computes the pooled diagnostics with the ``pool_with`` functionality in
``src/uadapt/metrics/diagnostics.py``. Correctness follows the
PRE-REGISTRATION (and ``scripts/04_evaluate.py``): D1/D2 correlate each
variance term with the per-proposal correctness ``gt_correct``
(IoU >= 0.5 with same-class GT). On the real-cache path the per-modality
flags are degenerate (text_ok == gt_correct by construction; the affinity
threshold saturates on RoI features) — see pipeline.py and change_log.md
2026-08-05. D3 uses the disagreeing-proposal weight subsets
(text_correct vs visual_correct flags).

Usage:
    python scripts/compute_pooled_diagnostics.py \\
        --ladd-results  outputs/real_data/ladd/results.json \\
        --dfire-results outputs/real_data/dfire/results.json \\
        --out outputs/real_data/pooled_diagnostics.json

Outputs a structured JSON with per-dataset D1-D3 (``per_dataset``), pooled
D1-D3 (``pooled`` — PRIMARY claim), a per-dataset D5 variance-distribution
sentinel, and a flat ``summary`` block for the report generator.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# Bootstrap so ``uadapt`` is importable without PYTHONPATH=src.
_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("compute_pooled_diagnostics")

# Diagnostic dict keys (match src/uadapt/demo/pipeline.py).
D1 = "D1_text_uncertainty_accuracy"
D2 = "D2_visual_uncertainty_accuracy"
D3 = "D3_gate_favorability"
D5 = "D5_variance_distribution"

DIAG_KEYS = (D1, D2, D3)


def load_json(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _resolve_proposals(results_path: Path, proposals_path: Optional[Path]) -> Path:
    """Default proposal_level.json to the sibling of the results file."""
    if proposals_path is not None:
        return proposals_path
    sibling = results_path.parent / "proposal_level.json"
    if not sibling.exists():
        raise FileNotFoundError(
            f"no proposal-level file found next to {results_path} ({sibling}); "
            "pass --<dataset>-proposals explicitly (demo_mode_a_end_to_end.py "
            "writes proposal_level.json by default)"
        )
    return sibling


def _load_dataset(name: str, results_path: Path, proposals_path: Optional[Path]) -> Dict:
    """Load + validate one dataset's demo outputs; returns meta + arrays."""
    if not results_path.exists():
        raise FileNotFoundError(
            f"--{name}-results points to a missing file: {results_path} "
            "(run scripts/demo_mode_a_end_to_end.py on real cached features first)"
        )
    results = load_json(results_path)
    if "map50" not in results or "meta" not in results:
        raise ValueError(
            f"{results_path} is not a demo results.json (missing map50/meta); "
            "expected the output of scripts/demo_mode_a_end_to_end.py"
        )
    proposals_path = _resolve_proposals(results_path, proposals_path)
    payload = load_json(proposals_path)
    proposals = payload.get("proposals")
    if not proposals:
        raise ValueError(
            f"{proposals_path} contains no proposals; cannot compute diagnostics"
        )
    required = {"norm_text_var", "norm_visual_var", "w",
                "text_correct", "visual_correct", "gt_correct",
                # raw per-proposal values for the honest D5 sentinel
                "text_entropy", "visual_distance_raw"}
    missing = required - set(proposals[0].keys())
    if missing:
        raise ValueError(
            f"{proposals_path} rows are missing required fields {sorted(missing)}; "
            "expected the proposal_level.json written by demo_mode_a_end_to_end.py "
            "(re-run it — the raw per-proposal fields were added 2026-08-05)"
        )
    return {
        "name": name,
        "results": results,
        "meta": results.get("meta", {}),
        "proposals": proposals,
    }


def _extract_arrays(ds: Dict) -> Dict[str, np.ndarray]:
    """Per-proposal arrays from the demo proposal rows (pipeline.py schema)."""
    rows = ds["proposals"]
    return {
        "norm_text_var": np.asarray([r["norm_text_var"] for r in rows], dtype=float),
        "norm_visual_var": np.asarray([r["norm_visual_var"] for r in rows], dtype=float),
        "text_ok": np.asarray([r["text_correct"] for r in rows], dtype=bool),
        "visual_ok": np.asarray([r["visual_correct"] for r in rows], dtype=bool),
        # gt_correct (IoU-based proposal correctness) is the PRE-REGISTERED
        # D1/D2 correctness label (matches 04_evaluate.py). The per-modality
        # flags drive D3's disagreeing subsets only.
        "gt_correct": np.asarray([r["gt_correct"] for r in rows], dtype=bool),
        "w": np.asarray([r["w"] for r in rows], dtype=float),
        # RAW per-proposal values for the D5 sentinel: D5 must run on the
        # absolute scale (text entropy in [0, 1]; visual distance / 2.0) —
        # min-max normalized arrays are spread across [0, 1] BY CONSTRUCTION
        # and would defeat the Taylor-validity clustering flag (2026-08-05).
        "text_entropy": np.asarray([r["text_entropy"] for r in rows], dtype=float),
        "visual_distance_raw": np.asarray(
            [r["visual_distance_raw"] for r in rows], dtype=float
        ),
    }


def _diag_dict(r) -> Dict:
    """Serialize a DiagnosticResult into the JSON schema (summary + flag + raw)."""
    out = {"summary": r.summary, "flag": r.flag}
    if r.raw is not None:
        out["raw"] = r.raw
    return out


def _per_dataset_diagnostics(a: Dict[str, np.ndarray]) -> Dict:
    """D1/D2/D3 (per dataset, no pooling) + the D5 variance sentinel."""
    from uadapt.metrics.diagnostics import (
        d1_text_uncertainty_accuracy,
        d2_visual_uncertainty_accuracy,
        d3_gate_favorability,
        d5_variance_distribution,
    )

    text_better = a["text_ok"] & ~a["visual_ok"]
    visual_better = a["visual_ok"] & ~a["text_ok"]
    # D1/D2 use the pre-registered proposal correctness (gt_correct) — the
    # per-modality flags are degenerate on the real cache (pipeline.py,
    # change_log.md 2026-08-05); D3 keeps the disagreeing subsets.
    d1 = d1_text_uncertainty_accuracy(a["norm_text_var"], a["gt_correct"])
    d2 = d2_visual_uncertainty_accuracy(a["norm_visual_var"], a["gt_correct"])
    d3 = d3_gate_favorability(a["w"][text_better], a["w"][visual_better])
    # D5 on the ABSOLUTE scale (raw values): text entropy is already in
    # [0, 1]; the raw box-to-support distance (range [0, 2]) is divided by
    # its max. Min-max normalized arrays would defeat the sentinel.
    d5 = d5_variance_distribution(
        a["text_entropy"], a["visual_distance_raw"] / 2.0
    )
    return {
        D1: _diag_dict(d1),
        D2: _diag_dict(d2),
        D3: _diag_dict(d3),
        D5: _diag_dict(d5),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute pooled D1/D2/D3 diagnostics across LADD + D-Fire "
                    "(pre-registration deviation 2026-08-03, §10)."
    )
    parser.add_argument("--ladd-results", required=True, type=Path,
                        help="LADD results.json from demo_mode_a_end_to_end.py")
    parser.add_argument("--dfire-results", required=True, type=Path,
                        help="D-Fire results.json from demo_mode_a_end_to_end.py")
    parser.add_argument("--ladd-proposals", type=Path, default=None,
                        help="LADD proposal_level.json (default: sibling of --ladd-results)")
    parser.add_argument("--dfire-proposals", type=Path, default=None,
                        help="D-Fire proposal_level.json (default: sibling of --dfire-results)")
    parser.add_argument("--out", default="outputs/real_data/pooled_diagnostics.json", type=Path)
    args = parser.parse_args()

    from uadapt.metrics.diagnostics import (
        d1_text_uncertainty_accuracy,
        d2_visual_uncertainty_accuracy,
        d3_gate_favorability,
    )

    ladd = _load_dataset("ladd", args.ladd_results, args.ladd_proposals)
    dfire = _load_dataset("dfire", args.dfire_results, args.dfire_proposals)
    la, da = _extract_arrays(ladd), _extract_arrays(dfire)

    # Pooled D1/D2/D3 — PRIMARY claim (deviation 2026-08-03, §10). LADD is the
    # primary dataset, D-Fire the secondary (matches 04_evaluate defaults).
    # Correctness = pre-registered proposal correctness (gt_correct), per the
    # pipeline/04_evaluate convention (change_log.md 2026-08-05).
    d1 = d1_text_uncertainty_accuracy(
        la["norm_text_var"], la["gt_correct"],
        pool_with=(da["norm_text_var"], da["gt_correct"]),
    )
    d2 = d2_visual_uncertainty_accuracy(
        la["norm_visual_var"], la["gt_correct"],
        pool_with=(da["norm_visual_var"], da["gt_correct"]),
    )
    l_text_better = la["text_ok"] & ~la["visual_ok"]
    l_visual_better = la["visual_ok"] & ~la["text_ok"]
    d_text_better = da["text_ok"] & ~da["visual_ok"]
    d_visual_better = da["visual_ok"] & ~da["text_ok"]
    d3 = d3_gate_favorability(
        la["w"][l_text_better], la["w"][l_visual_better],
        pool_with=(da["w"][d_text_better], da["w"][d_visual_better]),
    )

    # Sanity: pooled run should use the intended norm strategies (absolute is
    # required for 2-class D-Fire, deviation 2026-08-03 §2).
    for ds in (ladd, dfire):
        ns = ds["meta"].get("norm_strategy")
        if ds["name"] == "dfire" and ns != "absolute":
            logger.warning(
                "D-Fire run used norm_strategy=%r — absolute (x/2.0) is required "
                "for the 2-class variance fix (§2 deviation); min-max collapses "
                "the variance terms to {0, 1}.",
                ns,
            )
        if ds["name"] == "ladd" and ns not in ("min-max", "absolute"):
            logger.warning("LADD run used unknown norm_strategy=%r", ns)

    # Consistency check: per-dataset diagnostics recomputed here should match
    # the ones the demo already stored in results.json.
    for ds, arrays in ((ladd, la), (dfire, da)):
        stored = ds["results"].get("diagnostics", {})
        if stored:
            recomputed = _per_dataset_diagnostics(arrays)
            for key in DIAG_KEYS:
                s1 = stored.get(key, {}).get("summary", {})
                s2 = recomputed.get(key, {}).get("summary", {})
                if s1 and s2 and any(
                    abs(float(s1.get(k, 0.0)) - float(s2.get(k, 0.0))) > 1e-9
                    for k in s1 if k in s2
                ):
                    logger.warning(
                        "%s stored diagnostics (%s) disagree with the recomputed "
                        "values — the results.json may come from an older run.",
                        ds["name"], key,
                    )

    pooled = {
        D1: _diag_dict(d1["pooled"]),
        D2: _diag_dict(d2["pooled"]),
        D3: _diag_dict(d3["pooled"]),
    }
    out = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": "pooled D1/D2/D3 across LADD + D-Fire "
                        "(pre-registration deviation 2026-08-03, §10)",
            "primary_claim": "pooled",
            "ladd": {
                "data_source": ladd["meta"].get("data_source"),
                "norm_strategy": ladd["meta"].get("norm_strategy"),
                "classes": ladd["meta"].get("classes"),
                "n_proposals": len(ladd["proposals"]),
            },
            "dfire": {
                "data_source": dfire["meta"].get("data_source"),
                "norm_strategy": dfire["meta"].get("norm_strategy"),
                "classes": dfire["meta"].get("classes"),
                "n_proposals": len(dfire["proposals"]),
            },
        },
        "per_dataset": {
            "ladd": _per_dataset_diagnostics(la),
            "dfire": _per_dataset_diagnostics(da),
        },
        "pooled": pooled,
        "summary": {
            "D1_spearman_rho": pooled[D1]["summary"]["spearman_rho"],
            "D1_n": pooled[D1]["summary"]["n"],
            "D2_spearman_rho": pooled[D2]["summary"]["spearman_rho"],
            "D2_n": pooled[D2]["summary"]["n"],
            "D3_favorability_fraction": pooled[D3]["summary"]["favorability_fraction"],
            "D3_binomial_pvalue": pooled[D3]["summary"]["binomial_pvalue"],
            "D3_n": pooled[D3]["summary"]["n"],
            "n_proposals_ladd": len(ladd["proposals"]),
            "n_proposals_dfire": len(dfire["proposals"]),
            "n_pooled": int(pooled[D1]["summary"]["n"]),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    logger.info("wrote pooled diagnostics -> %s", args.out)
    s = out["summary"]
    print("Pooled D1/D2/D3 (PRIMARY claim, deviation 2026-08-03):")
    print(f"  D1 spearman_rho = {s['D1_spearman_rho']:+.3f}  (n={s['D1_n']:.0f})")
    print(f"  D2 spearman_rho = {s['D2_spearman_rho']:+.3f}  (n={s['D2_n']:.0f})")
    print(f"  D3 favorability = {s['D3_favorability_fraction']:.1%}  "
          f"(n={s['D3_n']:.0f}, p={s['D3_binomial_pvalue']:.3g})")


if __name__ == "__main__":
    main()
