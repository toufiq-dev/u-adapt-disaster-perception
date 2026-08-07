# Colab execution guide — Mode B 10-seed protocol (pre-registered contingency, Risk R3)

This runbook executes the **Mode B 10-seed paired statistical protocol** (the
past prompt: logistic-regression gate trained on 20 boxes/class, evaluated
against Mode A and the naive w = 0.5 baseline) entirely on Google Colab.

**TL;DR** — the protocol is 100% cache-based: no backbone, no GPU, no raw
images. You upload a small bundle (repo + ~5 MB of feature caches + 4 GT JSONs),
run two commands, download the outputs. A free **CPU** runtime is sufficient.

---

## 1. What you need to understand first

- `scripts/run_10_seed_protocol.py --mode B` now runs the full per-cell loop
  (02 prototypes → **05 calibration sampling** → 03 Mode B fusion → 04 eval),
  plus a naive w = 0.5 baseline and a per-dataset zero-shot mAP50.
- `scripts/05_build_calibration.py` samples the 20-box/class calibration set
  per seed from the **train split**, stratified by class, strictly disjoint
  from that seed's k-shot support and from the test split (pre-registration §5).
- `scripts/generate_real_data_report.py --mode-b-stats` writes the comparative
  report `docs/real_data_results_modeB.md`.
- **Nothing here runs a deep model.** The caches already contain Grounding DINO
  features; the Mode B gate is a 6-parameter numpy logistic regression. That is
  why Colab CPU is more than enough and why the upload is tiny.

> ⚠️ **Pilot-scale reality check (read before interpreting results):** the
> n=100 pilot train caches are tiny — LADD 10 images, D-Fire 9 images. The
> sampler therefore produces far fewer than 20 eligible boxes per class
> (LADD ≈ 5–6 `person` boxes; D-Fire ≈ 1). The report audits the true counts
> per cell, and every statistical conclusion must be read with this caveat. A
> full-scale run (full train split) is required before Mode B conclusions carry
> research weight.

---

## 2. Prepare the upload bundle (on your Mac)

From the repo parent directory, create a tarball of the repo (working tree
**including** the new Mode B tooling and the soft-target fix), excluding the
9.6 GB of raw images and local cruft:

```bash
cd /Users/toufiq/Developer
tar -czf uadapt_modeB_colab.tar.gz \
  --exclude='u-adapt-disaster-perception/.git' \
  --exclude='u-adapt-disaster-perception/.venv' \
  --exclude='u-adapt-disaster-perception/.pytest_cache' \
  --exclude='u-adapt-disaster-perception/data/raw' \
  --exclude='u-adapt-disaster-perception/data/.staging' \
  --exclude='*/__pycache__*' \
  u-adapt-disaster-perception
ls -lh uadapt_modeB_colab.tar.gz   # expect a few MB to ~40 MB (outputs/ figures)
```

This bundle already contains the previous 10-seed runs under
`outputs/real_data/ten_seed_protocol/` (analytic) and
`outputs/real_data/ten_seed_protocol_beta/` (beta), so the report's Mode A
column works out of the box.

Upload the tarball to your Google Drive (or keep it local and use the
`files.upload()` option in cell 0).

---

## 3. Colab notebook setup

1. **New notebook** → `Runtime ▸ Change runtime type` → **CPU** (a T4 GPU is
   fine but unused; CPU avoids quota).
2. Run these cells in order.

**Cell 0 — mount Drive + unpack (or use the upload button):**

```python
#@title Unpack the bundle into /content
from google.colab import drive
drive.mount('/content/drive')
!tar -xzf '/content/drive/MyDrive/uadapt_modeB_colab.tar.gz' -C /content
%cd /content/u-adapt-disaster-perception
!pwd && ls cached_features/ladd scripts/05_build_calibration.py
```

*Prefer browser upload (no Drive)?*
```python
#@title Unpack from browser upload
from google.colab import files
up = files.upload()                    # pick uadapt_modeB_colab.tar.gz
!tar -xzf {list(up)[0]} -C /content
%cd /content/u-adapt-disaster-perception
!pwd
```

**Cell 1 — install dependencies (torch is preinstalled and NOT needed here):**

```python
#@title Install deps
!pip install -q -e .                     # numpy scipy pyyaml tqdm (all the protocol needs)
# Optional (only if you ever re-run backbone extraction here):
# !pip install -q "transformers>=4.44" "opencv-python-headless>=4.9"
```

**Cell 2 — sanity checks:**

```python
#@title Verify imports, caches, and the Mode B unit tests
import numpy, scipy, yaml, uadapt
print("scipy", scipy.__version__, "(needs >= 1.11 for false_discovery_control)")
!python scripts/05_build_calibration.py --help > /dev/null && echo "05 script OK"
!ls cached_features/ladd/test/manifest.json cached_features/dfire/test/manifest.json
!ls data/annotations/ladd_train.json data/annotations/dfire_train.json
!python -m pytest tests/test_calibration_set.py tests/test_mode_b_calibration.py -q
```

---

## 4. Smoke test first (≈ 1 minute)

Verifies the whole per-cell chain (02 → 05 → 03 → 04) including calibration
sampling and zero-shot, on 2 seeds × one dataset × k=5:

```python
#@title 2-seed smoke run (ladd, k=5)
!python scripts/run_10_seed_protocol.py \
    --datasets ladd --shots 5 --max-seeds 2 --mode B \
    --work-dir /tmp/ten_seed_modeB_smoke \
    --out /tmp/ten_seed_modeB_smoke/stats.json
```

Expected log highlights: `ladd zero-shot mAP50 (raw, test cache) ≈ 0.81`, a
`Mode B ... vs naive averaging` summary table, and a note that < 10 seeds is a
smoke run. **If this fails, fix before the full run (see §7).**

---

## 5. Full Mode B 10-seed protocol (the actual run)

```python
#@title Run the full 60-cell Mode B protocol
!mkdir -p outputs/real_data/ten_seed_protocol_modeB
!python scripts/run_10_seed_protocol.py \
    --datasets ladd dfire --shots 1 3 5 --max-seeds 10 --mode B \
    --work-dir outputs/real_data/ten_seed_protocol_modeB \
    --out outputs/real_data/ten_seed_protocol_modeB/stats.json \
    2>&1 | tee outputs/real_data/ten_seed_protocol_modeB/run.log
```

- **Expected runtime:** the local 2-cell smoke takes ~4 s; the full run is 60
  cells over 2 × 100-image test caches → **roughly 5–30 minutes** on a free
  Colab CPU (well under the runtime limit).
- **Determinism:** every stage is seeded, so a Colab run on the same caches
  reproduces the same `stats.json` as a local run bit-for-bit.
- Outputs land in `outputs/real_data/ten_seed_protocol_modeB/` (per-cell
  `prototypes/`, `calibration/`, `scores/`, `scores_naive/`, `eval/`,
  `eval_naive/`, zero-shot evals, and `stats.json`).

---

## 6. Generate the comparative report

```python
#@title Mode B vs naive + four-way comparison report
!python scripts/generate_real_data_report.py \
    --mode-b-stats outputs/real_data/ten_seed_protocol_modeB/stats.json \
    --analytic-stats outputs/real_data/ten_seed_protocol/stats.json \
    --out docs/real_data_results_modeB.md
```

The report contains:

1. **Mode B vs naive averaging** — per-cell (ladd/dfire × k=1/3/5) paired
   t-test, Wilcoxon signed-rank, Cohen's d, and BH-FDR q-values over the full
   12-test family (2 tests × 6 cells).
2. **Calibration-set audit** — the true per-seed sampled box counts per cell
   (read this before trusting "20 boxes/class").
3. **Four-way comparison** — per-cell mAP50 means for zero-shot / naive /
   Mode A / Mode B, plus Δ(B − N) in percentage points.
4. **Verdict** — data-driven: whether Mode B beat naive (and significantly so),
   with the pilot-scale caveats and the pre-registered fallback-narrative
   guidance (Risk R3).

Optional: omit `--analytic-stats` for a Mode-B-only report, or add
`--report-date YYYY-MM-DD`.

---

## 7. Download the results back

```python
#@title Download results + report
!tar -czf /content/modeB_results.tar.gz \
    outputs/real_data/ten_seed_protocol_modeB \
    docs/real_data_results_modeB.md
from google.colab import files
files.download('/content/modeB_results.tar.gz')
```

On your Mac:

```bash
cd /Users/toufiq/Developer/u-adapt-disaster-perception
tar -xzf ~/Downloads/modeB_results.tar.gz
```

---

## 8. How to read the results (honest interpretation checklist)

Answer the past prompt's question 4 with the report's tables:

- **Did Mode B beat naive averaging?** Look at the paired table's
  `Mode B mAP50` vs `Naive mAP50` means and the verdict section. If d < 0 and
  q < 0.05 in every cell, Mode B is *significantly worse* — the pre-registered
  fallback (plain-confidence margin / acknowledged limitation) should be
  elevated in the thesis narrative, as the report states.
- **Calibration size caveat:** check the audit table. At n=100 pilot scale the
  gate trains on ~5 (LADD) to ~1 (D-Fire) boxes — it has almost no learning
  signal, so "Mode B loses" at this scale is **not evidence against the method**.
  A full-scale run (full train split, 20 boxes/class genuinely available) is
  the deciding experiment.
- **Soft-target fix:** this branch fixed the Mode B soft-target mapping to the
  pre-registered formula (proposal §5.4.2) on 2026-08-07. Do **not** compare
  these numbers against any pre-fix Mode B run.
- **Affinity saturation:** `visual_correct` ≈ True for every sampled box
  (affinity ≥ 0.65), so the soft targets collapse toward σ(S_visual − S_text)
  — a degenerate learning signal, also noted in the report.
- **Zero-shot column** is the pilot-scale raw-detector mAP50 on the full cached
  test split (LADD ≈ 81.3%, D-Fire ≈ 73.4% for the n=100 caches) — not the
  literature floors (61.0% / 27.5%, full test set).

---

## 9. Troubleshooting / FAQ

| Symptom | Fix |
|---|---|
| `FalseDiscoveryControl`/scipy error | `!pip install -q "scipy>=1.11"` |
| `no cache found at cached_features/<ds>/test` | Bundle is stale — re-run §2 and re-upload (caches are only ~5 MB) |
| `class 'X' has N cached proposals < shots=5` | The train cache for that dataset is too small for k=5 support — expected on the pilot caches for some seeds; use the same caches as the local 10-seed runs |
| Protocol "succeeds" but all mAP50 = 0.0 on D-Fire | Outdated `04_evaluate.py` (needs the 2026-08-07 image-id remap); re-bundle from the current tree |
| Runtime disconnects mid-run | Not expected (< 30 min). If it happens, just re-run the full `--max-seeds 10` invocation — the protocol is per-cell idempotent and overwrites `stats.json`. Do NOT split seeds to "merge": a `--seed0 0 --max-seeds 5` run writes a 5-seed `stats.json` that the second half would simply overwrite (cell arrays hold only 5 entries each) |
| Disk space | The bundle is tiny; no Colab quota concern |
| `05_build_calibration.py` produces 0 eligible boxes | That dataset's train cache has no GT-matched proposals (possible on tiny pilot caches); the protocol then errors clearly — full-scale caches do not have this issue |

---

## 10. After the run (back on your Mac)

1. Review `docs/real_data_results_modeB.md` and the stats summary from `run.log`.
2. Log the outcome + verdict in `docs/change_log.md` (match the existing
   2026-08-07 entries).
3. If the verdict is "Mode B did not beat naive," update the thesis narrative
   per the pre-registered Risk R3 fallback (plain-confidence margin /
   acknowledged limitation) — the report's Verdict section gives the exact
   wording to fold in.
