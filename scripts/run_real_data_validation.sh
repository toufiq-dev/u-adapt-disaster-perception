#!/usr/bin/env bash
# run_real_data_validation.sh — real-data U-ADAPT validation pipeline (critical path).
#
#   [0] Prerequisites     raw LADD/D-Fire images + annotations present,
#                         docs/licenses.md updated (Milestone 1 gates)
#   [1] Feature extraction scripts/01_extract_and_cache.py (Grounding DINO
#                         Swin-T, top-k=100) for LADD and D-Fire, train+test,
#                         skipped when the cache already exists
#   [2] Prototypes        scripts/02_build_prototypes.py, shots k=1/3/5
#   [3] Mode A evaluation scripts/demo_mode_a_end_to_end.py on REAL cached
#                         features — LADD with min-max (1 class), D-Fire with
#                         absolute (required for the 2-class variance fix,
#                         pre-registration deviation 2026-08-03 §2)
#   [4] Pooled D1/D2/D3   scripts/compute_pooled_diagnostics.py — pooled across
#                         LADD + D-Fire (PRIMARY claim, deviation §10)
#   [5] Report            scripts/generate_real_data_report.py ->
#                         docs/real_data_results.md
#
# Usage:
#   bash scripts/run_real_data_validation.sh
#
# Env overrides (all optional):
#   PYTHON            python interpreter          (default: python3)
#   CACHE_ROOT        feature-cache root          (default: cached_features)
#   OUT_ROOT          evaluation output root      (default: outputs/real_data)
#   MODEL_CONFIG      backbone config             (default: configs/models/grounding_dino_swinT.yaml)
#   TOP_K             proposal limit              (default: 100; 300 only as ablation)
#   SHOTS             support examples k          (default: 5)
#   SEED              RNG seed                    (default: 0)
#   N_TEST_IMAGES     test subset size            (default: 100000 = all images)
#   LADD_GT / DFIRE_GT  COCO annotation JSONs     (default: data/annotations/{ladd,dfire}_test.json)
#   LADD_NORM / DFIRE_NORM  normalization strategy (default: min-max / absolute)
#   SKIP_PREREQS      set to 1 to skip step [0] gates (dry-run / CI only)
#
# Every step echoes progress; failures abort immediately (set -e).
# Runtime gate: torch + transformers are checked before step [1]; the pipeline
# must not run without them.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
CACHE_ROOT="${CACHE_ROOT:-cached_features}"
OUT_ROOT="${OUT_ROOT:-outputs/real_data}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/models/grounding_dino_swinT.yaml}"
TOP_K="${TOP_K:-100}"
SHOTS="${SHOTS:-5}"
SEED="${SEED:-0}"
N_TEST_IMAGES="${N_TEST_IMAGES:-100000}"
LADD_GT="${LADD_GT:-data/annotations/ladd_test.json}"
DFIRE_GT="${DFIRE_GT:-data/annotations/dfire_test.json}"
LADD_NORM="${LADD_NORM:-min-max}"
DFIRE_NORM="${DFIRE_NORM:-absolute}"
SKIP_PREREQS="${SKIP_PREREQS:-0}"

step() { echo; echo "=== [$1/6] $2 ==="; }
die() { echo "ERROR: $*" >&2; exit 1; }

count_images() {
    # Number of jpg/jpeg/png files under a directory (0 when absent).
    local dir="$1"
    if [[ ! -d "$dir" ]]; then echo 0; return; fi
    find "$dir" -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' \) | wc -l | tr -d ' '
}

# ---------------------------------------------------------------------------
# [0/6] Prerequisites
# ---------------------------------------------------------------------------
step 0 "Prerequisites (raw data + licenses + annotations)"
if [[ "$SKIP_PREREQS" == "1" ]]; then
    echo "  SKIP_PREREQS=1 — skipping prerequisite gates."
else
    for ds in ladd dfire; do
        n=$(count_images "data/raw/$ds")
        if [[ "$n" -eq 0 ]]; then
            die "data/raw/$ds is missing or contains no images (found 0). Download the \
dataset and verify it before running the real-data pipeline (docs/datasets.md)."
        fi
        echo "  OK  data/raw/$ds ($n images)"
    done

    # docs/licenses.md must have Milestone-1 status for the primary datasets.
    # NOTE: this matches the markdown table format ("| LADD | <license> | ...");
    # if the table is reformatted, re-check the pattern here.
    if grep -E '^\| (LADD|D-Fire) \|' docs/licenses.md | grep -qE 'TBD|To verify'; then
        die "docs/licenses.md is not updated: LADD/D-Fire license status is still \
TBD / 'To verify (issue #1)'. Fill in the verified license + date in \
docs/licenses.md (Milestone 1) before running the real-data pipeline."
    fi
    echo "  OK  docs/licenses.md updated (LADD/D-Fire license status filled in)"

    for gt in "$LADD_GT" "$DFIRE_GT"; do
        if [[ ! -f "$gt" ]]; then
            die "ground-truth annotations missing: $gt (see docs/datasets.md \
'Layout'; mask-to-box conversions live in data/annotations/)."
        fi
        echo "  OK  $gt"
    done
fi

mkdir -p "$CACHE_ROOT" "$OUT_ROOT"

# Runtime dependency gate: steps [1]-[3] run torch/transformers inference, so
# fail early with a helpful message instead of a confusing import traceback.
if ! "$PY" -c "import torch" >/dev/null 2>&1; then
    die "PyTorch not found. Please run 'pip install torch torchvision torchaudio' \
before running the full pipeline. (transformers and the Grounding DINO weights \
are also required for step [1]; see docs/datasets.md.)"
fi
if ! "$PY" -c "import transformers" >/dev/null 2>&1; then
    die "transformers not found. Please run 'pip install transformers' before \
running the full pipeline (required for the Grounding DINO backbone)."
fi
echo "  OK  PyTorch + transformers available ($("$PY" -c "import torch; print('torch ' + torch.__version__)" 2>/dev/null))"

# ---------------------------------------------------------------------------
# [1/6] Feature extraction (skipped when the split cache already exists)
# ---------------------------------------------------------------------------
step 1 "Feature extraction (Grounding DINO Swin-T, top-k=$TOP_K)"
for ds in ladd dfire; do
    for split in train test; do
        manifest="$CACHE_ROOT/$ds/$split/manifest.json"
        if [[ -f "$manifest" ]]; then
            echo "  cache exists ($manifest) — skipping extraction"
            continue
        fi
        echo "  extracting $ds/$split -> $CACHE_ROOT/$ds ..."
        "$PY" scripts/01_extract_and_cache.py \
            --model-config "$MODEL_CONFIG" \
            --dataset-config "configs/datasets/$ds.yaml" \
            --split "$split" \
            --cache-dir "$CACHE_ROOT/$ds" \
            --top-k "$TOP_K"
    done
done

# ---------------------------------------------------------------------------
# [2/6] Prototype construction (k = 1, 3, 5)
# ---------------------------------------------------------------------------
step 2 "Prototype construction (shots k=1/3/5, seed=$SEED)"
for ds in ladd dfire; do
    for k in 1 3 5; do
        out="$CACHE_ROOT/$ds/prototypes_k${k}_seed${SEED}.json"
        if [[ -f "$out" ]]; then
            echo "  prototypes exist ($out) — skipping"
            continue
        fi
        echo "  building $ds prototypes k=$k ..."
        "$PY" scripts/02_build_prototypes.py \
            --cache-dir "$CACHE_ROOT/$ds" \
            --dataset-config "configs/datasets/$ds.yaml" \
            --shots "$k" \
            --seed "$SEED" \
            --out "$out"
    done
done

# ---------------------------------------------------------------------------
# [3/6] Mode A evaluation on real cached features
# ---------------------------------------------------------------------------
step 3 "Mode A evaluation (real cache; LADD=$LADD_NORM, D-Fire=$DFIRE_NORM)"
run_eval() {
    local ds="$1" norm="$2" gt="$3"
    echo "  evaluating $ds (norm-strategy=$norm, shots=$SHOTS, n-test-images=$N_TEST_IMAGES) ..."
    "$PY" scripts/demo_mode_a_end_to_end.py \
        --cache-dir "$CACHE_ROOT/$ds" \
        --dataset-config "configs/datasets/$ds.yaml" \
        --ground-truth "$gt" \
        --norm-strategy "$norm" \
        --shots "$SHOTS" \
        --seed "$SEED" \
        --n-test-images "$N_TEST_IMAGES" \
        --out "$OUT_ROOT/$ds/results.json" \
        --proposal-out "$OUT_ROOT/$ds/proposal_level.json" \
        --no-figures
}
run_eval ladd  "$LADD_NORM" "$LADD_GT"
run_eval dfire "$DFIRE_NORM" "$DFIRE_GT"

# ---------------------------------------------------------------------------
# [4/6] Pooled D1/D2/D3 diagnostics (PRIMARY claim, deviation §10)
# ---------------------------------------------------------------------------
step 4 "Pooled D1/D2/D3 diagnostics (LADD + D-Fire)"
"$PY" scripts/compute_pooled_diagnostics.py \
    --ladd-results "$OUT_ROOT/ladd/results.json" \
    --dfire-results "$OUT_ROOT/dfire/results.json" \
    --out "$OUT_ROOT/pooled_diagnostics.json"

# ---------------------------------------------------------------------------
# [5/6] Markdown report
# ---------------------------------------------------------------------------
step 5 "Generate report (docs/real_data_results.md)"
"$PY" scripts/generate_real_data_report.py \
    --ladd-results "$OUT_ROOT/ladd/results.json" \
    --dfire-results "$OUT_ROOT/dfire/results.json" \
    --pooled-diagnostics "$OUT_ROOT/pooled_diagnostics.json" \
    --out docs/real_data_results.md

# ---------------------------------------------------------------------------
# [6/6] Summary
# ---------------------------------------------------------------------------
step 6 "Done"
echo "  Results JSON   : $OUT_ROOT/{ladd,dfire}/results.json"
echo "  Proposal data  : $OUT_ROOT/{ladd,dfire}/proposal_level.json"
echo "  Pooled diag    : $OUT_ROOT/pooled_diagnostics.json"
echo "  Report         : docs/real_data_results.md"
echo ""
echo "Next: 10 seeds + paired tests (pre-registration §9), mAP50:95/ECE/Brier via"
echo "scripts/04_evaluate.py, and cross-backbone repeats (RQ5)."
