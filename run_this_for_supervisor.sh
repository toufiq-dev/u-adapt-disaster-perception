#!/usr/bin/env bash
# run_this_for_supervisor.sh — execute the full U-ADAPT supervisor demo.
#
#   1. scripts/demo_mode_a_end_to_end.py   -> results.json + proposal_level.json
#      (deterministic seed=0; synthetic world unless a real cache exists)
#   2. notebooks/supervisor_demo_visualizations.ipynb (executed via jupyter)
#      -> outputs/supervisor_demo/figures/*.png
#   3. prints a summary pointing to docs/supervisor_demo_report.md
#
# Usage:
#   bash run_this_for_supervisor.sh            # full run (figures + report)
#   bash run_this_for_supervisor.sh --no-figs  # results + console table only
#
# Requirements: python3 with numpy/matplotlib/scipy/pyyaml (+ jupyter for
# figure rendering). On Colab everything except jupyter is preinstalled.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
OUT_DIR="outputs/supervisor_demo"
NO_FIGS=0
[[ "${1:-}" == "--no-figs" ]] && NO_FIGS=1

mkdir -p "$OUT_DIR"

echo "==> [1/3] Running U-ADAPT Mode A demo (seed=0, synthetic/real auto) ..."
"$PY" scripts/demo_mode_a_end_to_end.py \
    --out "$OUT_DIR/results.json" \
    --proposal-out "$OUT_DIR/proposal_level.json"

if [[ "$NO_FIGS" -eq 1 ]]; then
    echo "==> [2/3] Skipped figures (--no-figs)."
else
    echo "==> [2/3] Rendering Figures 1-6 ..."
    if "$PY" -c "import jupyter, nbformat, nbclient" 2>/dev/null; then
        "$PY" -m jupyter nbconvert \
            --to notebook \
            --execute \
            --inplace \
            notebooks/supervisor_demo_visualizations.ipynb \
            --ExecutePreprocessor.timeout=1200
    else
        echo "    jupyter not installed — rendering figures directly from the script..."
        "$PY" scripts/demo_mode_a_end_to_end.py \
            --out "$OUT_DIR/results.json" \
            --proposal-out "$OUT_DIR/proposal_level.json" \
            --figures-dir "$OUT_DIR/figures"
    fi
fi

echo "==> [3/3] Done."
echo ""
echo "  Results JSON : $OUT_DIR/results.json"
echo "  Proposal data: $OUT_DIR/proposal_level.json"
if [[ "$NO_FIGS" -eq 0 ]]; then
    echo "  Figures      : $OUT_DIR/figures/ (figure1..figure6)"
fi
echo "  Report       : docs/supervisor_demo_report.md"
echo ""
echo "Open docs/supervisor_demo_report.md (or render it to PDF) for the 2-page supervisor summary."
