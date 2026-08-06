#!/usr/bin/env python
"""run_10_seed_protocol.py — pre-registered 10-seed paired statistical protocol.

Pre-registration §9 (docs/pre_registration.md) mandates, for the PRIMARY
comparison **U-ADAPT Mode A vs naive averaging (w = 0.5)**, per dataset and
per shot k: **10 random seeds** with a **paired two-sided t-test AND a
Wilcoxon signed-rank test** across seeds, **Cohen's d** effect sizes, and
**Benjamini-Hochberg FDR control (q = 0.05)** across the full comparison
family (2 datasets x 3 shots x 2 tests).

This driver orchestrates the existing SCRIPTED pipeline per seed — the same
path the review's fused-score fix exercised (not the demo path):

    per (dataset, seed, shots k):
      1. scripts/02_build_prototypes.py --seed <s>  -- k-shot support
         sampling (seed-dependent; same function the demo path uses)
      2. scripts/03_run_fusion.py --gate-type <gate> --prototypes <1>
         -- Mode A fused scores (S_final = (1-w)*S_text + w*S_visual)
      3. scripts/03_run_fusion.py --gate-type naive   --prototypes <1>
         -- the w = 0.5 baseline (NaiveGate) through the SAME scoring path
      4. scripts/04_evaluate.py on both score files -> mAP50 per method

and then computes the paired statistics across seeds for every
(dataset, shots) cell.

NOTE on subsets: the scripted path (02 -> 03 -> 04) evaluates the FULL
cached test split — it does not apply the demo path's ``--n-test-images``
image subset. The n=100 pilot caches ARE the subset extraction; for the
full-scale run the complete test split is evaluated, per the
pre-registration.

Usage:
    # Full protocol (hours on the full data — do NOT run ad hoc)
    python scripts/run_10_seed_protocol.py \
        --datasets ladd dfire --shots 1 3 5 --max-seeds 10 \
        --gate-type analytic \
        --work-dir outputs/real_data/ten_seed_protocol \
        --out outputs/real_data/ten_seed_protocol/stats.json

    # Smoke test (2 seeds, one dataset, k=5)
    python scripts/run_10_seed_protocol.py \
        --datasets ladd --shots 5 --max-seeds 2 \
        --work-dir /tmp/ten_seed_smoke --out /tmp/ten_seed_smoke_stats.json
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("run_10_seed_protocol")


# ---------------------------------------------------------------------------
# Orchestration helpers
# ---------------------------------------------------------------------------
def _run(cmd: List[str]) -> str:
    """Run a subprocess, raising with a traceback tail on failure."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
        raise RuntimeError(
            f"command failed (rc={proc.returncode}): {' '.join(cmd)}\n"
            + "\n".join(tail)
        )
    return proc.stdout


def _map50_of(eval_path: Path) -> float:
    with open(eval_path) as fh:
        return float(json.load(fh)["mAP50"])


def _run_cell(
    py: str,
    ds: str,
    seed: int,
    shots: int,
    gate_type: str,
    cache_root: Path,
    annotations_dir: Path,
    mode_config: Path,
    work: Path,
) -> None:
    """One (dataset, seed, shots) cell: prototypes -> Mode A + naive scores
    -> evaluation of both. Writes intermediates under ``work``."""
    proto = work / "prototypes" / f"{ds}_k{shots}_seed{seed}.json"
    scores_a = work / "scores" / f"{ds}_k{shots}_seed{seed}.json"
    scores_naive = work / "scores_naive" / f"{ds}_k{shots}_seed{seed}.json"
    eval_a = work / "eval" / f"{ds}_k{shots}_seed{seed}.json"
    eval_naive = work / "eval_naive" / f"{ds}_k{shots}_seed{seed}.json"
    for d in (proto.parent, scores_a.parent, scores_naive.parent,
              eval_a.parent, eval_naive.parent):
        d.mkdir(parents=True, exist_ok=True)

    dataset_config = _ROOT / "configs" / "datasets" / f"{ds}.yaml"
    gt = annotations_dir / f"{ds}_test.json"

    _run([
        py, str(_ROOT / "scripts" / "02_build_prototypes.py"),
        "--cache-dir", str(cache_root / ds),
        "--dataset-config", str(dataset_config),
        "--shots", str(shots),
        "--seed", str(seed),
        "--out", str(proto),
    ])
    for gate, out in (("A", scores_a), ("naive", scores_naive)):
        gtype = gate_type if gate == "A" else "naive"
        _run([
            py, str(_ROOT / "scripts" / "03_run_fusion.py"),
            "--mode", "A",
            "--cache-dir", str(cache_root / ds),
            "--prototypes", str(proto),
            "--mode-config", str(mode_config),
            "--gate-type", gtype,
            "--out", str(out),
        ])
    for preds, out in ((scores_a, eval_a), (scores_naive, eval_naive)):
        _run([
            py, str(_ROOT / "scripts" / "04_evaluate.py"),
            "--predictions", str(preds),
            "--ground-truth", str(gt),
            "--out", str(out),
        ])


# ---------------------------------------------------------------------------
# Statistics (pre-registration §9)
# ---------------------------------------------------------------------------
def _paired_stats(mode_a: List[float], naive: List[float]) -> Dict:
    """Paired t-test + Wilcoxon + Cohen's d (paired, d_z) over seeds."""
    from scipy import stats as st

    a = np.asarray(mode_a, dtype=float)
    b = np.asarray(naive, dtype=float)
    out: Dict = {
        "n_seeds": int(len(a)),
        "mode_a_map50_mean": float(a.mean()) if len(a) else float("nan"),
        "naive_map50_mean": float(b.mean()) if len(b) else float("nan"),
        "mean_diff": float((a - b).mean()) if len(a) else float("nan"),
    }
    if len(a) < 2:
        out["note"] = "need >= 2 seeds for paired statistics"
        return out
    t, p_t = st.ttest_rel(a, b)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            w_stat, p_w = st.wilcoxon(a, b, alternative="two-sided")
        except ValueError as exc:  # e.g. all differences are zero
            w_stat, p_w = float("nan"), float("nan")
            out["wilcoxon_note"] = str(exc)
    diff = a - b
    sd = float(diff.std(ddof=1))
    if sd > 0.0:
        d = float(diff.mean() / sd)
        d_note = None
    else:
        d = float("inf") if diff.mean() != 0.0 else float("nan")
        d_note = "undefined (sd = 0 across seeds)"
    out.update(
        {
            "paired_ttest": {"t": float(t), "p": float(p_t)},
            "wilcoxon": {"statistic": float(w_stat), "p": float(p_w)},
            "cohens_d": d,
            "cohens_d_note": d_note,
        }
    )
    return out


def _bh_fdr(pvals: List[float]) -> List[float]:
    """Benjamini-Hochberg adjusted q-values (scipy >= 1.11)."""
    from scipy import stats as st

    finite = [p for p in pvals if p is not None and p == p]
    if not finite:
        return [float("nan")] * len(pvals)
    adj = st.false_discovery_control(finite, method="bh")
    it = iter(adj)
    return [float("nan") if (p is None or p != p) else next(it) for p in pvals]


def _cell_key(ds: str, shots: int) -> str:
    return f"{ds}_k{shots}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-registered 10-seed paired statistical protocol "
                    "(Mode A vs naive averaging, §9)."
    )
    parser.add_argument("--datasets", nargs="+", default=["ladd", "dfire"])
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--max-seeds", type=int, default=10)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--gate-type", choices=["analytic", "beta_fallback"],
                        default="analytic",
                        help="Mode A gate variant to run against the naive "
                             "w = 0.5 baseline (pre-registered primary: analytic)")
    parser.add_argument("--cache-root", default="cached_features", type=Path)
    parser.add_argument("--annotations-dir", default="data/annotations", type=Path)
    parser.add_argument("--mode-config",
                        default="configs/modes/mode_A_analytic.yaml", type=Path)
    parser.add_argument("--work-dir",
                        default="outputs/real_data/ten_seed_protocol", type=Path)
    parser.add_argument("--out", default="outputs/real_data/ten_seed_protocol/stats.json",
                        type=Path)
    parser.add_argument("--py", default=sys.executable,
                        help="interpreter for the pipeline subprocesses "
                             "(default: the interpreter running this script)")
    args = parser.parse_args()

    if args.max_seeds < 1:
        parser.error("--max-seeds must be >= 1")

    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed0, args.seed0 + args.max_seeds))
    logger.info(
        "10-seed protocol scaffold: %d seed(s) x %d dataset(s) x %d shot(s) "
        "= %d cells (gate=%s)",
        len(seeds), len(args.datasets), len(args.shots),
        len(seeds) * len(args.datasets) * len(args.shots), args.gate_type,
    )

    cells: Dict[str, Dict] = {}
    n_cells = 0
    for ds in args.datasets:
        for shots in args.shots:
            cells[_cell_key(ds, shots)] = {"mode_a": [], "naive": []}
    for ds in args.datasets:
        for seed in seeds:
            for shots in args.shots:
                n_cells += 1
                logger.info("[%d] %s seed=%d shots=%d ...", n_cells, ds, seed, shots)
                _run_cell(args.py, ds, seed, shots, args.gate_type,
                          args.cache_root, args.annotations_dir,
                          args.mode_config, work)
                cells[_cell_key(ds, shots)]["mode_a"].append(
                    _map50_of(work / "eval" / f"{ds}_k{shots}_seed{seed}.json")
                )
                cells[_cell_key(ds, shots)]["naive"].append(
                    _map50_of(work / "eval_naive" / f"{ds}_k{shots}_seed{seed}.json")
                )

    # Per-cell paired statistics.
    p_pool: List[float] = []
    order: List[str] = []
    for ds in args.datasets:
        for shots in args.shots:
            key = _cell_key(ds, shots)
            c = cells[key]
            stats = _paired_stats(c["mode_a"], c["naive"])
            cells[key]["stats"] = stats
            order.append(key)
            for fam in ("paired_ttest", "wilcoxon"):
                p_pool.append(stats.get(fam, {}).get("p", float("nan")))

    q_pool = _bh_fdr(p_pool)
    q_iter = iter(q_pool)
    for key in order:
        for fam in ("paired_ttest", "wilcoxon"):
            cells[key]["stats"][fam]["q_bh"] = next(q_iter)

    # Summary table.
    print("\n" + "=" * 100)
    print(f"10-seed protocol (gate={args.gate_type}) — Mode A vs naive averaging (w=0.5)")
    print("=" * 100)
    print(f"{'cell':<12}{'n':>3}{'ModeA mean':>11}{'Naive mean':>11}"
          f" {'d':>7} {'t':>8} {'p(t)':>9} {'q(t)':>8} {'W':>7} {'p(W)':>9} {'q(W)':>8}")
    significant: List[str] = []
    for key in order:
        s = cells[key]["stats"]
        if "note" in s:
            print(f"{key:<12}{s['n_seeds']:>3}  {s.get('note', '')}")
            continue
        t = s["paired_ttest"]
        w = s["wilcoxon"]
        d = s["cohens_d"]
        dcell = f"{d:+.2f}" if d == d and abs(d) != float("inf") else "inf" if d > 0 else "n/a"
        print(f"{key:<12}{s['n_seeds']:>3}{s['mode_a_map50_mean']:>11.4f}"
              f"{s['naive_map50_mean']:>11.4f} {dcell:>7} {t['t']:>8.3f}"
              f" {t['p']:>9.3g} {t['q_bh']:>8.3g} {w['statistic']:>7.1f}"
              f" {w['p']:>9.3g} {w['q_bh']:>8.3g}")
        if t["q_bh"] == t["q_bh"] and t["q_bh"] < 0.05 or \
           w["q_bh"] == w["q_bh"] and w["q_bh"] < 0.05:
            significant.append(key)
    print("=" * 100)
    print("Benjamini-Hochberg FDR (q = 0.05) over",
          len(p_pool), "comparisons (2 tests x cells).")
    print("Cells with any q < 0.05:", significant or "none")
    print("NOTE: smoke runs (< 10 seeds) are for pipeline verification only — "
          "the pre-registered protocol needs all 10 seeds.")

    out = {
        "meta": {
            "protocol": "pre-registration §9 — 10-seed paired test, "
                        "Mode A vs naive averaging (w = 0.5)",
            "gate_type": args.gate_type,
            "datasets": args.datasets,
            "shots": args.shots,
            "seed0": args.seed0,
            "n_seeds": len(seeds),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "fdr": "Benjamini-Hochberg, q = 0.05",
            "pipeline": ["02_build_prototypes.py", "03_run_fusion.py",
                         "04_evaluate.py"],
            "work_dir": str(work),
        },
        "cells": cells,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    logger.info("wrote protocol stats -> %s", args.out)


if __name__ == "__main__":
    main()
