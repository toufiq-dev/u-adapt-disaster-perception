# Colab execution guide — Mode B 10-seed protocol (free T4)

Executes the **Mode B 10-seed paired statistical protocol** (pre-registered
contingency, Risk R3): the 6-parameter logistic-regression gate trained on
20 boxes/class, compared against Mode A and the naive w = 0.5 baseline.

**Key facts:** the protocol is 100% cache-based — no backbone, no GPU, no raw
images. A free **T4** (or CPU) runtime is enough; total upload is ~5 MB;
runtime is minutes. Every stage is seeded, so Colab reproduces the local
`stats.json` bit-for-bit.

---

## 0. One-time local prep (on your Mac)

**a) Push the committed code** (the repo — including the Mode B tooling and the
soft-target fix — is committed locally; Colab clones it):

```bash
cd /Users/toufiq/Developer/u-adapt-disaster-perception
git push origin main
```

**b) Zip the gitignored data assets** (the clone does NOT contain these —
`cached_features/`, `data/annotations/`, and the previous 10-seed `stats.json`
files are all gitignored by design):

```bash
cd /Users/toufiq/Developer/u-adapt-disaster-perception
zip -r uadapt_assets.zip \
  cached_features \
  data/annotations \
  outputs/real_data/ten_seed_protocol/stats.json \
  outputs/real_data/ten_seed_protocol_beta/stats.json
ls -lh uadapt_assets.zip      # ~2 MB
```

Upload `uadapt_assets.zip` anywhere convenient (Google Drive recommended).

---

## 1. Colab notebook setup

**Runtime:** `Runtime ▸ Change runtime type ▸ T4 GPU` (free; the GPU is unused
but satisfies your environment preference — CPU also works).

**Cell 0 — clone + mount Drive + unpack assets:**

```python
#@title Clone repo, mount Drive, unpack assets
from google.colab import drive
drive.mount('/content/drive')

# Clone the repo (adjust the URL to your fork if needed)
%cd /content
!git clone https://github.com/toufiq-dev/u-adapt-disaster-perception.git
%cd /content/u-adapt-disaster-perception

# Unpack the gitignored assets (caches + annotations + previous stats)
!unzip -o '/content/drive/MyDrive/uadapt_assets.zip' -d /content/u-adapt-disaster-perception
!ls cached_features/ladd/test/manifest.json data/annotations/ladd_train.json
```

*No Drive? Use browser upload instead of cells 0:*

```python
#@title Clone repo + browser-upload assets (alternative)
%cd /content
!git clone https://github.com/toufiq-dev/u-adapt-disaster-perception.git
%cd /content/u-adapt-disaster-perception
from google.colab import files
up = files.upload()                      # pick uadapt_assets.zip
!unzip -o {list(up)[0]} -d /content/u-adapt-disaster-perception
```

**Cell 1 — install dependencies (torch/torchvision already preinstalled and
NOT required by this protocol):**

```python
#@title Install deps
# pyproject.toml core deps = numpy, scipy, pyyaml, tqdm — no torch.
!pip install -q -e .
!python -c "import scipy; assert tuple(map(int, scipy.__version__.split('.')[:2])) >= (1, 11), 'need scipy>=1.11 for BH-FDR'; print('scipy OK', scipy.__version__)"
```

**Cell 2 — sanity checks:**

```python
#@title Verify imports, assets, and the Mode B unit tests
import uadapt, numpy, yaml
!python -m pytest tests/test_calibration_set.py tests/test_mode_b_calibration.py -q
```

---

## 2. Calibration-set sampler (standalone demo + sanity)

The protocol builds a fresh calibration set per seed automatically. This cell
runs the sampler once so you can inspect the JSON the gate is trained on
(20 boxes/class requested; at n=100 pilot scale LADD yields ~6, D-Fire ~1 —
recorded honestly in the `sampling` block):

```python
#@title Build + inspect one calibration set (ladd, k=5, seed 0)
!python scripts/build_calibration_set.py \
    --cache-dir cached_features/ladd \
    --ground-truth data/annotations/ladd_train.json \
    --prototypes cached_features/ladd/prototypes_k5_seed0.json \
    --boxes-per-class 20 \
    --seed 0 \
    --out /tmp/cal_ladd_k5_s0.json
!python -c "import json; d=json.load(open('/tmp/cal_ladd_k5_s0.json')); print('n samples:', len(d['samples'])); print('sampling:', json.dumps(d['sampling'], indent=1))"
```

---

## 3. Mode B 10-seed protocol

**Step 1 — smoke test first (≈ 1 min; 2 seeds × ladd × k=5):**

```python
#@title 2-seed smoke run (pipeline verification)
!python scripts/run_10_seed_protocol.py \
    --datasets ladd --shots 5 --max-seeds 2 --mode B \
    --work-dir /tmp/ten_seed_modeB_smoke \
    --out /tmp/ten_seed_modeB_smoke/stats.json
```

Expected: a `Mode B ... vs naive averaging` summary table and the note that
< 10 seeds is a smoke run. **Do not read statistics from this.**

**Step 2 — full 10-seed run (the actual protocol; ≈ 5–30 min on T4/CPU):**

```python
#@title Full 60-cell Mode B protocol (ladd + dfire, k=1/3/5, 10 seeds)
!mkdir -p outputs/real_data/ten_seed_protocol_modeB
!python scripts/run_10_seed_protocol.py \
    --datasets ladd dfire --shots 1 3 5 --max-seeds 10 --mode B \
    --work-dir outputs/real_data/ten_seed_protocol_modeB \
    --out outputs/real_data/ten_seed_protocol_modeB/stats.json \
    2>&1 | tee outputs/real_data/ten_seed_protocol_modeB/run.log
```

This writes `stats.json` with per-cell paired t-test, Wilcoxon, Cohen's d, and
BH-FDR q-values (the naive baseline and zero-shot mAP50 are included), plus
per-cell calibration audits.

---

## 4. Comparative report + download

```python
#@title Generate the Mode B report (vs naive + Mode A + zero-shot)
!python scripts/generate_real_data_report.py \
    --mode-b-stats outputs/real_data/ten_seed_protocol_modeB/stats.json \
    --analytic-stats outputs/real_data/ten_seed_protocol/stats.json \
    --out docs/real_data_results_modeB.md

#@title Download stats.json + report
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

## 5. Reading the results (honest checklist)

1. **Did Mode B beat naive?** — paired table (`Mode B mAP50` vs `Naive
   mAP50`) + the Verdict section. d < 0 and q < 0.05 in every cell ⇒
   significantly worse ⇒ elevate the pre-registered fallback narrative
   (plain-confidence margin / acknowledged limitation) in the thesis.
2. **Calibration size** — the audit table shows the TRUE per-seed sampled
   counts. At n=100 pilot scale the gate trains on ~5 (LADD) / ~1 (D-Fire)
   boxes, so a pilot "loss" is **not evidence against the method**; the
   full-scale run (real 20 boxes/class) is the deciding experiment.
3. **Soft-target fix** (2026-08-07) — do not compare with any pre-fix Mode B
   run; the mapping now matches proposal §5.4.2 exactly.
4. **Zero-shot column** is pilot-scale raw-detector mAP50 on the full cached
   test split (LADD ≈ 81.3%, D-Fire ≈ 73.4%) — not the literature floors
   (61.0% / 27.5%).

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `no cache found at cached_features/<ds>/test` | Re-upload `uadapt_assets.zip` (it is gitignored, never cloned) |
| scipy `false_discovery_control` missing | `!pip install -q "scipy>=1.11"` |
| `class 'X' has N cached proposals < shots=5` | Pilot train cache too small for that seed; expected on the pilot caches |
| All D-Fire mAP50 = 0.0 | Stale clone (missing the 2026-08-07 image-id remap) — `git pull` |
| Runtime disconnects mid-run | Not expected (< 30 min). Re-run the full `--max-seeds 10` command (idempotent per cell); do NOT split seeds — a 5-seed run writes a separate 5-seed `stats.json`, not a merged 10-seed one |
| `build_calibration_set.py` → 0 eligible boxes | That dataset's tiny pilot train cache has no GT-matched proposals; full-scale caches are unaffected |
