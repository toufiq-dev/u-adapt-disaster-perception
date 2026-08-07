# Full-scale Colab guide — extraction (LADD 1,365 / D-Fire 21,527) + Mode B 10-seed protocol

Runs the **full-scale** U-ADAPT pipeline on Google Colab (free T4): download the
complete raw datasets, extract features for every image with the frozen
Grounding DINO Swin-T backbone, then run the pre-registered **Mode B 10-seed
paired protocol** on the full caches — the deciding experiment for the
learned logistic-regression gate (20 boxes/class calibration), which the
n=100 pilot could not properly test.

The work is split into **two clearly separated phases**:

| Phase | What | Where | Runtime | GPU needed |
|---|---|---|---|---|
| **A — Extraction** | download full raw data + extract + cache features | Colab session #1 | **many hours** (see §5) | yes (T4) |
| **B — Mode B protocol** | 10-seed paired stats on the full caches | Colab session #2 (fresh) | ~1–3 h | no (CPU fine) |

The two phases are decoupled by the `cached_features/` zip: Phase A produces
it, Phase B consumes it. **You do not need the raw images in Phase B.**

---

## 0. One-time local prep (on your Mac)

Push the current code first — Colab clones the repo, so the full-scale tooling
(the resume-safe extractor, the n=100-free report labels, the Mode B protocol)
must be on GitHub:

```bash
cd /Users/toufiq/Developer/u-adapt-disaster-perception
git push origin main
```

> ℹ️ **Prefer upload over download?** If you already have the complete LADD
> raw data locally (1,365 images ≈ 9.3 GB, as on the development Mac), skip
> §A3.2 entirely and follow **`docs/colab_upload_raw_guide.md`** — zip LADD
> on the Mac, upload to Drive, pull into Colab, and run the D-Fire mirror
> download concurrently. That guide replaces §A3.2 (manual URL + train GT)
> with a zero-download, zero-risk path.
>
> Otherwise: before you run Phase A, make sure you can obtain a **verified
> LADD download URL**. The official LADD repo (huyhieupham/LADD) is offline;
> the script refuses to guess a URL (Milestone-1 policy). The working source
> is the Kaggle **"Lacmus Drone Dataset (LaDD)"** page (or the author page /
> Zenodo). You also need a **LADD train COCO GT** — see §A3.2. D-Fire needs
> nothing special (anonymous HF mirror).

---

# Phase A — Extraction (Colab session #1)

## A1. Runtime

`Runtime ▸ Change runtime type ▸ T4 GPU` (free). Verify GPU + disk:

```python
#@title Runtime sanity
import torch, shutil
print("GPU:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
total, used, free = shutil.disk_usage("/content")
print(f"disk free: {free / 1e9:.0f} GB")
```

D-Fire raw images + caches need roughly 20–40 GB; the free runtime provides
~78 GB. If free space looks tight, `!rm -rf /content/u-adapt-disaster-perception` before re-extracting.

## A2. Clone + install dependencies

```python
#@title Clone repo
%cd /content
!git clone https://github.com/toufiq-dev/u-adapt-disaster-perception.git
%cd /content/u-adapt-disaster-perception
!git log -1 --oneline
```

```python
#@title Install deps (torch/torchvision are PREINSTALLED on Colab — do not reinstall)
# Grounding DINO + extraction need transformers + opencv. The repo's pip
# deps skip torch/torchvision on Colab by design (requirements.txt header).
!pip install -q "transformers>=4.44" "opencv-python-headless>=4.9" "pyyaml>=6.0" "numpy>=1.26" "scipy>=1.11" "tqdm>=4.66"
# Install the repo package (no torch — core deps are numpy/scipy/pyyaml/tqdm):
!pip install -q -e . --no-deps
!python -c "import transformers, torch; print('transformers', transformers.__version__, '| torch', torch.__version__)"
```

## A3. Download the FULL raw datasets

### A3.1 D-Fire (automated, resumable)

```python
#@title Download full D-Fire from the anonymous HF mirror (17,221 train / 4,306 test)
# Resumable (2026-08-07): already-downloaded images are skipped on re-run, so
# a Colab disconnect does not force re-downloading all 21,527 images.
!python data/download_scripts/download_datasets.py --dataset dfire --dfire-mirror
```

This writes `data/raw/dfire/{train,test}/` images and COCO-style
`data/annotations/dfire_{train,test}.json` (YOLO → COCO conversion, class order
0=smoke, 1=fire — verified 2026-08-05). If the session dies mid-download,
**re-run this same cell** — it skips what is already on disk.

### A3.2 LADD (manual URL + train GT)

The official LADD repo is offline, so this needs a verified URL you obtained
(Kaggle LaDD page etc.):

```python
#@title LADD download (replace <URL> with your verified link)
!python data/download_scripts/download_datasets.py --dataset ladd --ladd-url "<URL>"
```

**LADD train GT is required** by the Mode B calibration sampler
(`data/annotations/ladd_train.json`). The script now auto-extracts BOTH
`ladd_train.json` and `ladd_test.json` from an archive that ships
`annotations/{train,test}.json` (Kaggle LaDD layout), remapping
`Pedestrian → person`:

```python
#@title If your LADD archive ships annotations/{train,test}.json (Kaggle LaDD)
!python data/download_scripts/download_datasets.py --dataset ladd \
    --ladd-archive /path/to/ladd.zip \
    --ladd-gt-remap "Pedestrian=person"
```

If your GT JSONs are separate, place them directly:

```python
#@title Manual GT placement (alternative)
!mkdir -p data/annotations
# upload ladd_train.json / ladd_test.json here, e.g. from Drive:
# !cp /content/drive/MyDrive/ladd_train.json data/annotations/
```

### A3.3 Verify all four GT files + image counts

```python
#@title Verify data
!ls -la data/raw/ladd data/raw/dfire | head -30
!ls -la data/annotations/
!python -c "
import json
for ds in ('ladd', 'dfire'):
    for split in ('train', 'test'):
        p = f'data/annotations/{ds}_{split}.json'
        try:
            d = json.load(open(p))
            print(f'{ds}/{split}: {len(d[\"images\"])} images, {len(d[\"annotations\"])} boxes, cats={[c[\"name\"] for c in d[\"categories\"]]}')
        except FileNotFoundError:
            print(f'{ds}/{split}: MISSING {p}')
"
```

Expected: D-Fire ~17,221 train / ~4,306 test images; LADD 1,365 total across
train/test.

## A4. Feature extraction — the long pole

This is the many-hours step. **It is RAM-safe** (the repo streams **one image
at a time** — peak image RAM ≈ 1 frame, see `01_extract_and_cache.py`), and it
is **resume-safe** (2026-08-07): a disconnected run leaves `records.json`
behind, and a re-run continues from there instead of restarting at image 0.

> ⚠️ **About `batch_size=8`:** the repo has **no `--batch-size` flag** — the
> RAM-safe streaming decoder processes images **singly** (even more RAM-safe
> than a batch of 8). Batching was deliberately not added: Grounding DINO pads
> each image individually, and a batch would silently change the cached
> features vs. the pilot — breaking the n=100 → full-scale comparability. Use
> the single-image streaming path below.

Run each of the four splits (`ladd train/test`, `dfire train/test`) with the
**resume loop** below. The loop re-invokes the extractor until the split's
`manifest.json` exists, so any disconnect just means you re-run the cell and
it picks up where it left off:

```python
#@title Resume-safe extraction for ONE split — edit the ds/split variables
ds    = "dfire"     # "ladd" | "dfire"
split = "test"      # "train" | "test"

import os, time
manifest = f"cached_features/{ds}/{split}/manifest.json"
os.makedirs(os.path.dirname(manifest), exist_ok=True)
attempt = 0
while not os.path.exists(manifest):
    attempt += 1
    print(f"[{time.strftime('%H:%M:%S')}] attempt {attempt}: extracting {ds}/{split} ...")
    rc = os.system(
        f"python scripts/01_extract_and_cache.py "
        f"--model-config configs/models/grounding_dino_swinT.yaml "
        f"--dataset-config configs/datasets/{ds}.yaml "
        f"--split {split} --cache-dir cached_features/{ds} --top-k 100"
    )
    if rc != 0:
        print(f"[{time.strftime('%H:%M:%S')}] extraction exited rc={rc}; retrying in 20s")
        time.sleep(20)
        continue
    # manifest.json is only written on completion; if missing, the run was
    # interrupted (resume-safe), so loop again.
    if not os.path.exists(manifest):
        print(f"[{time.strftime('%H:%M:%S')}] interrupted before manifest — resuming")
        time.sleep(10)
print(f"[{time.strftime('%H:%M:%S')}] DONE {ds}/{split}")
```

**Disconnect survival guide** (Colab free-T4 reality):

- **Max session ~12 h** and an **idle timeout ~90 min**. D-Fire is 21,527
  images at roughly 0.5–2 s/image on a T4 ⇒ **~3–12 h per D-Fire split**. It
  may not finish in one session.
- The resume loop above is your safety net: if the session dies, open a **new
  session**, re-run cells A1–A2 (clone + install), re-run this cell with the
  same `ds`/`split` — extraction continues from `records.json`. Do **not**
  delete `cached_features/` between sessions.
- Keep the tab open; optionally add an auto-clicker/`Ctrl+Shift+Enter` or just
  accept re-running after a disconnect. A paid Colab runtime (24 h limit)
  removes most of this pain.
- Run **LADD first** (~1,365 images, ~15–60 min) so you have a complete,
  usable cache early, then do D-Fire train, then D-Fire test.

While waiting, you can watch progress (the extractor logs every 50 images):

```python
#@title Progress monitor (run in a new cell while extraction runs)
import json, time
def cached_images(ds, split):
    try:
        d = json.load(open(f"cached_features/{ds}/{split}/records.json"))
        return len(d["records"])
    except Exception:
        return 0
while True:
    print(time.strftime("%H:%M:%S"), {f"{d}/{s}": cached_images(d, s)
          for d in ("ladd", "dfire") for s in ("train", "test")})
    time.sleep(60)
```

## A5. Verify the four full caches

```python
#@title Verify caches are complete (n_images must equal the split sizes)
!cat cached_features/ladd/train/manifest.json cached_features/ladd/test/manifest.json
!cat cached_features/dfire/train/manifest.json cached_features/dfire/test/manifest.json
```

Expected: LADD train+test ≈ 1,365 total; D-Fire train ≈ 17,221, test ≈ 4,306.

## A6. Zip + get the caches to your Mac

```python
#@title Zip caches + annotations (raw images are NOT needed for Phase B)
!zip -r /content/full_caches.zip \
    cached_features \
    data/annotations
!ls -lh /content/full_caches.zip
```

Expect ~0.5–1.5 GB (D-Fire dominates). Download to your Mac — Drive is the
most reliable for large files:

```python
#@title Download to Mac via Drive (recommended for large zips)
from google.colab import drive
drive.mount("/content/drive")
!cp /content/full_caches.zip "/content/drive/MyDrive/full_caches.zip"
print("uploaded to Drive — download from drive.google.com on your Mac")
```

```python
#@title ...or direct browser download (smaller zips only)
from google.colab import files
files.download("/content/full_caches.zip")
```

On your Mac, keep the zip somewhere convenient (e.g. `~/Downloads/`). **Phase A
is complete.**

---

# Phase B — Mode B 10-seed protocol (Colab session #2)

A **fresh** session. CPU-only runtime is fine (the protocol is 100%
cache-based — no backbone, no GPU, no raw images). This phase evaluates the
**FULL cached test split** (the scripted path 02 → 05 → 03 → 04 ignores
`--n-test-images`; only the caches you extracted decide the size).

## B1. Setup

```python
#@title Runtime
print("CPU is fine here — the protocol is cache-only.")
```

```python
#@title Clone repo + install
%cd /content
!git clone https://github.com/toufiq-dev/u-adapt-disaster-perception.git
%cd /content/u-adapt-disaster-perception
!pip install -q "numpy>=1.26" "scipy>=1.11" "pyyaml>=6.0" "tqdm>=4.66"
!pip install -q -e . --no-deps
!python -c "import scipy; assert tuple(map(int, scipy.__version__.split('.')[:2])) >= (1, 11); print('scipy OK', scipy.__version__)"
```

```python
#@title Get the full caches (Drive or browser upload)
from google.colab import drive
drive.mount("/content/drive")
!unzip -o "/content/drive/MyDrive/full_caches.zip" -d /content/u-adapt-disaster-perception
!cat cached_features/dfire/test/manifest.json
```

> No Drive? Use `from google.colab import files; up = files.upload()` and
> `!unzip -o {list(up)[0]}`. For a >1 GB zip, Drive is strongly recommended.

```python
#@title Sanity: caches + annotations + tests
!ls cached_features/ladd/test/manifest.json data/annotations/ladd_train.json
!python -m pytest tests/test_calibration_set.py tests/test_mode_b_calibration.py tests/test_cache_engine.py -q
```

## B2. Smoke test first (2 seeds × ladd × k=5)

```python
#@title 2-seed smoke run (≈ 1–5 min)
!python scripts/run_10_seed_protocol.py \
    --datasets ladd --shots 5 --max-seeds 2 --mode B \
    --work-dir /tmp/ten_seed_modeB_smoke \
    --out /tmp/ten_seed_modeB_smoke/stats.json
```

The calibration audit in the run log should now show the **pre-registered
20 boxes/class** (or close to it) — unlike the pilot, where LADD yielded ~6
and D-Fire ~1. **Do not read statistics from a 2-seed run.**

## B3. Full 10-seed Mode B protocol (the deciding experiment)

```python
#@title Full 60-cell Mode B protocol (ladd + dfire, k=1/3/5, 10 seeds)
!mkdir -p outputs/real_data/ten_seed_protocol_modeB
!python scripts/run_10_seed_protocol.py \
    --datasets ladd dfire --shots 1 3 5 --max-seeds 10 --mode B \
    --work-dir outputs/real_data/ten_seed_protocol_modeB \
    --out outputs/real_data/ten_seed_protocol_modeB/stats.json \
    2>&1 | tee outputs/real_data/ten_seed_protocol_modeB/run.log
```

At full scale each cell processes the whole test split (D-Fire ~4,306 images,
~430k proposals), so budget **~1–3 h** for the 60 cells (10 seeds × 6 cells).
The stats.json records per-cell paired t-test, Wilcoxon, Cohen's d, BH-FDR
q-values, zero-shot mAP50, and the per-seed calibration audit.

## B4. Report + download

```python
#@title Generate the Mode B report (vs naive + zero-shot; add Mode A if you ran it)
!python scripts/generate_real_data_report.py \
    --mode-b-stats outputs/real_data/ten_seed_protocol_modeB/stats.json \
    --out docs/real_data_results_modeB.md
```

The report auto-detects full scale from `meta.n_test_images` — it will say
"full cached test split" / "full scale", not the pilot's "n=100 subset".
Download:

```python
#@title Download stats.json + report
!tar -czf /content/modeB_fullscale_results.tar.gz \
    outputs/real_data/ten_seed_protocol_modeB \
    docs/real_data_results_modeB.md
from google.colab import files
files.download("/content/modeB_fullscale_results.tar.gz")
```

On your Mac:

```bash
cd /Users/toufiq/Developer/u-adapt-disaster-perception
tar -xzf ~/Downloads/modeB_fullscale_results.tar.gz
```

---

## 5. Expected runtimes (honest estimates, free T4)

| Step | Volume | Estimate |
|---|---|---|
| D-Fire download (HF mirror) | 21,527 images | 1–4 h (resumable) |
| LADD download | 1,365 images | minutes after you have the URL |
| LADD extraction | 1,365 images | ~15–60 min |
| D-Fire train extraction | ~17,221 images | ~2–10 h |
| D-Fire test extraction | ~4,306 images | ~1–3 h |
| Zip + Drive transfer | 0.5–1.5 GB | ~10–30 min upload |
| Mode B protocol (60 cells, full caches) | CPU | ~1–3 h |
| Report | — | < 1 min |

## 6. Reading the results (honest checklist)

1. **Calibration audit is now real.** Full-scale train caches give the
   pre-registered 20 boxes/class; the audit table in `stats.json` (and the
   report) shows the true per-seed counts. A Mode B verdict here is the
   deciding evidence for Risk R3 — unlike the pilot.
2. **Did Mode B beat naive?** Paired table + Verdict section. d < 0 and
   q < 0.05 in every cell ⇒ significantly worse ⇒ the learned gate does not
   recover value at full scale ⇒ elevate the plain-confidence-margin /
   acknowledged-limitation fallback in the thesis narrative.
3. **Zero-shot column** is now the raw-detector mAP50 over the full test split
   — directly comparable to the Michailidou et al. floors (LADD 61.0%,
   D-Fire 27.5%) in a way the pilot could not be.
4. **Soft-target fix** (2026-08-07) — do not compare with any pre-fix Mode B
   run.
5. **Comparability with the pilot:** extraction is deterministic and
   per-image, so full-scale and pilot caches use the same feature convention
   (no batching, same top-k=100, same backbone).

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `no cache found at cached_features/<ds>/test` | Phase B zip was not unpacked (`unzip -o` into the repo root) — re-run B1 |
| `manifest.json` missing after extraction | Extraction was interrupted; re-run the §A4 cell (resume-safe) |
| D-Fire mirror download restarting | Re-run the §A3.1 cell — it now skips existing images |
| `ladd_train.json` missing | LADD GT must be provided (manual URL / archive annotations / manual copy) — §A3.2 |
| Extraction OOM / runtime disconnect | Re-run §A4 with the same `ds`/`split` — resumes from `records.json`; do not delete `cached_features/` |
| Session hit 12 h cap mid-D-Fire | Expected; new session + re-run §A4 loop for the unfinished split only |
| `class 'X' has N cached proposals < shots=5` | Only possible if the train cache is smaller than the shots demand; at full scale this should not happen |
| All D-Fire mAP50 = 0.0 | Stale clone — `git pull` (needs the 2026-08-07 image-id remap) |
| Disk full during download | Free runtime ~78 GB is enough for D-Fire; if a previous attempt left junk, `!rm -rf /content/u-adapt-disaster-perception data/raw/.staging` |
