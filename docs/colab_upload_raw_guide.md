# Colab guide — upload LADD from your Mac instead of downloading it (faster path)

**2026-08-07.** The in-Colab LADD download requires a manually sourced URL
(the official LADD repo is offline) and a train GT — the single most fragile
step of the full-scale run. This guide removes it entirely: **you already have
the complete LADD dataset locally**, so upload it once to Google Drive and pull
it into Colab at Google-internal speed.

The recommended fastest path is:

1. **Mac (background):** zip LADD (~8.1 GB, val excluded) + the two LADD GT
   JSONs, upload both to Drive (≈ 12–40 min depending on your upload speed).
2. **Colab session #1:** copy the zip from Drive, unzip, verify, **start LADD
   extraction immediately** (~15–60 min on T4) and, in parallel, start the
   **D-Fire mirror download in the background** (~1–4 h, resumable).
3. Then extract D-Fire train/test with the resume loop (the long pole,
   ~3–13 h, possibly across 2–3 sessions).
4. Zip the caches → Drive → Mac → Phase B (Mode B 10-seed protocol, CPU-only,
   per `docs/colab_full_scale_guide.md`).

> **D-Fire is NOT uploadable** — your local `data/raw/dfire` is only the n=100
> pilot subset (3,739 of 21,527 train images). The full D-Fire must be
> downloaded in Colab (`--dfire-mirror`, resumable since 2026-08-07: existing
> files are skipped and writes are atomic `.part` renames).

---

## 1. Is it worth it? (honest math)

| Path | Cost |
|---|---|
| In-Colab LADD download | Time for the download **plus** sourcing a verified URL **plus** ensuring `ladd_train.json` (risk of getting stuck) |
| Upload from Mac → Drive → Colab | `8.1 GB ÷ upload_speed` one-time, **zero risk** (data already verified locally) |

Upload time at typical home upload speeds (8.1 GB = 65 Gbit):

| Upload speed | Time |
|---|---|
| 10 Mbps | ~1.8 h |
| 30 Mbps | ~40 min |
| 50 Mbps | ~22 min |
| 100 Mbps | ~11 min |

**Decision rule:** upload wins unless your upload is slower than ~15 Mbps —
and even then it is usually worth it because it eliminates the manual-URL/GT
dependency entirely.

> ⚠️ Upload only shortens the *download* phase. Extraction is still the long
> pole (LADD ~15–60 min; D-Fire ~3–13 h) — see §5. Uploading lets you start
> LADD extraction **now** instead of waiting for any download.

---

## 2. Mac side — package LADD for upload

### 2.1 Verify what you have (30 s)

```bash
cd /Users/toufiq/Developer/u-adapt-disaster-perception
for d in train test; do echo "ladd/$d: $(ls data/raw/ladd/$d/*.jpg | wc -l) images"; done
python3 -c "
import json
for k in ('ladd_train', 'ladd_test'):
    d = json.load(open(f'data/annotations/{k}.json'))
    print(f'{k}.json: {len(d[\"images\"])} images, {len(d[\"annotations\"])} boxes')"
```

Expected: `ladd/train: 1220 images`, `ladd/test: 202 images`,
`ladd_train.json: 1220 images`, `ladd_test.json: 202 images`.

### 2.2 Zip the LADD images — STORE mode, val excluded

```bash
mkdir -p ~/uadapt_raw
cd data/raw
# -0 = store (no compression): jpgs don't compress, and store mode avoids
# ~30+ min of wasted CPU for ~0% size gain. -x drops ladd/val, which is a
# byte-identical duplicate of test (202 files, 1.2 GB) and never used by the
# pipeline (configs only reference train/test).
zip -0 -r ~/uadapt_raw/ladd_raw.zip ladd -x 'ladd/val/*'
ls -lh ~/uadapt_raw/ladd_raw.zip        # expect ~8.1 GB
```

### 2.3 Zip the LADD GT JSONs (tiny, but required by Mode B calibration)

```bash
cd /Users/toufiq/Developer/u-adapt-disaster-perception
zip ~/uadapt_raw/ladd_annotations.zip data/annotations/ladd_train.json data/annotations/ladd_test.json
ls -lh ~/uadapt_raw/ladd_annotations.zip   # ~1.2 MB
```

### 2.4 Upload to Google Drive

1. Check Drive quota first — free tier is **15 GB shared with Gmail/Photos**:
   `https://drive.google.com/drive/my-drive` ▸ Storage. 8.1 GB fits, but
   confirm you aren't near the cap.
2. Create a folder **`My Drive/uadapt_raw/`**.
3. Drag `ladd_raw.zip` + `ladd_annotations.zip` into it.
   - **Recommended:** the **Drive for Desktop** app (drag into the
     `My Drive/uadapt_raw` folder). It **resumes interrupted uploads**, which
     matters for an 8 GB file.
   - Browser drag-drop on drive.google.com also works; if it stalls, don't
     delete the partial — re-drop the same file (the desktop app resumes it).
   - ⚠️ Do **not** use Colab's `files.upload()` for these — browser-mediated
     uploads are unreliable above ~2 GB.

---

## 3. Colab session #1 — pull LADD from Drive, extract, download D-Fire

Runtime: `Runtime ▸ Change runtime type ▸ T4 GPU` (free). Verify disk:
`!python -c "import shutil; print(shutil.disk_usage('/content'))"` — free
runtime has ~78 GB; LADD zip (8.1) + extracted (8.1) + D-Fire (~2) + caches
fits comfortably.

### 3.1 Clone + install (same as the full-scale guide)

```python
#@title Clone repo
%cd /content
!git clone https://github.com/toufiq-dev/u-adapt-disaster-perception.git
%cd /content/u-adapt-disaster-perception
!git log -1 --oneline
```

```python
#@title Install deps (torch/torchvision are PREINSTALLED on Colab)
!pip install -q "transformers>=4.44" "opencv-python-headless>=4.9" "pyyaml>=6.0" "numpy>=1.26" "scipy>=1.11" "tqdm>=4.66"
!pip install -q -e . --no-deps
!python -c "import transformers, torch; print('transformers', transformers.__version__, '| torch', torch.__version__)"
```

### 3.2 Pull the LADD zip from Drive → local VM disk

```python
#@title Mount Drive and copy LADD zips to /content (NOT extracted on Drive)
from google.colab import drive
drive.mount("/content/drive")
!cp "/content/drive/MyDrive/uadapt_raw/ladd_raw.zip" /content/
!cp "/content/drive/MyDrive/uadapt_raw/ladd_annotations.zip" /content/
!ls -lh /content/*.zip
```

> Always copy the zip to the VM disk first — never unzip directly on the Drive
> mount (FUSE is slow for many files). Copying one 8 GB file from Drive takes
> ~5–20 min; optionally use a **gdown shareable link** instead (faster, no
> mount): `!pip install -q gdown && !gdown --fuzzy "https://drive.google.com/file/d/<FILE_ID>/view" -O /content/ladd_raw.zip`.

### 3.3 Unzip into the expected layout + verify

```python
#@title Unzip (images to data/raw, GT to data/annotations) + verify
!mkdir -p data/raw data/annotations
!unzip -o /content/ladd_raw.zip -d data/raw
!unzip -o /content/ladd_annotations.zip -d .
!echo "--- raw layout ---"
!ls data/raw/ladd
!echo "--- counts ---"
!for d in train test; do echo "ladd/$d: $(ls data/raw/ladd/$d/*.jpg | wc -l) images"; done
!echo "--- GT ---"
!python -c "
import json
for split in ('train', 'test'):
    d = json.load(open(f'data/annotations/ladd_{split}.json'))
    print(f'ladd_{split}: {len(d[\"images\"])} images, cats={[c[\"name\"] for c in d[\"categories\"]]}')"
# Clean up the 8 GB zip now that it's unpacked:
!rm -f /content/ladd_raw.zip /content/ladd_annotations.zip
```

Expected: `ladd/train` 1220, `ladd/test` 202, `ladd_train: 1220`,
`ladd_test: 202`. **No LADD download was needed — this is the whole point.**

### 3.4 Start the D-Fire download in the BACKGROUND (while LADD extracts)

```python
#@title Kick off the full D-Fire mirror download in the background
# Resumable: re-run restarts nothing already on disk; atomic .part writes
# mean a truncated image can never be mistaken for complete.
!nohup python data/download_scripts/download_datasets.py --dataset dfire --dfire-mirror \
    > /content/dfire_download.log 2>&1 &
print("D-Fire download running in the background — it does not block extraction.")
```

```python
#@title Poll the download (run this cell whenever you want a status)
!tail -3 /content/dfire_download.log
# Done when the log's last line is the dfire_test.json write summary, e.g.
# "wrote .../data/annotations/dfire_test.json (4306 images, ... boxes)"
```

> D-Fire download is network-I/O bound; LADD extraction is GPU/CPU bound —
> running both concurrently is safe and saves hours. If the session dies
> mid-download, just re-run §3.4 in a fresh session (skips existing files).

### 3.5 Extract LADD now — the resume-safe loop (ds="ladd", both splits)

```python
#@title Resume-safe extraction for ONE split — edit ds/split
ds    = "ladd"      # "ladd" | "dfire"
split = "train"     # "train" | "test"

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
    if not os.path.exists(manifest):
        print(f"[{time.strftime('%H:%M:%S')}] interrupted before manifest — resuming")
        time.sleep(10)
print(f"[{time.strftime('%H:%M:%S')}] DONE {ds}/{split}")
```

Run it once with `split = "train"`, once with `split = "test"`. LADD finishes
in ~15–60 min total, giving you a **complete, usable LADD cache** while
D-Fire is still downloading/extracting. If the session dies, open a fresh one,
re-run §3.1–§3.5 — extraction continues from `records.json` (resume-safe since
2026-08-07). **Do not delete `cached_features/` between sessions.**

### 3.6 Then D-Fire extraction (the long pole)

When the §3.4 log shows the D-Fire download completed, extract D-Fire with the
**same §3.5 loop** (`ds="dfire"`, train then test) across as many sessions as
needed (~3–13 h total on a free T4). Everything else — cache verification,
zipping caches to your Mac, and Phase B (Mode B 10-seed protocol) — is
unchanged and documented in `docs/colab_full_scale_guide.md` (§A5–§B4).

---

## 4. What NOT to upload

| Don't upload | Why |
|---|---|
| `data/raw/dfire` | Only the n=100 pilot subset locally (3,739/21,527) — would silently run the protocol on pilot data |
| `data/raw/ladd/val` | Byte-identical duplicate of `test` (already excluded by the §2.2 zip) |
| `cached_features/` | Pilot caches are n=100 — the whole point of full-scale is re-extraction |
| `outputs/`, `.venv/`, `.staging/` | Gitignored scratch/junk |
| Your LADD archive (.zip of the dataset) | Already unpacked locally; only the layout in §2.2 is needed |

---

## 5. Expected runtimes (free T4, honest estimates)

| Step | Volume | Estimate |
|---|---|---|
| Zip LADD (store mode) | 8.1 GB | ~2–5 min |
| Upload LADD to Drive | 8.1 GB | ~12–40 min (upload speed) |
| Drive → Colab copy | 8.1 GB | ~5–20 min |
| Unzip + verify | 8.1 GB | ~2–5 min |
| **LADD extraction** | 1,422 images | **~15–60 min** |
| D-Fire download (background, resumable) | 21,527 images | ~1–4 h |
| D-Fire train extraction | ~17,221 images | ~2–10 h |
| D-Fire test extraction | ~4,306 images | ~1–3 h |
| Zip caches → Drive → Mac | 0.5–1.5 GB | ~10–30 min |
| Phase B: Mode B 10-seed protocol (CPU) | full caches | ~1–3 h |

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| Drive upload stalled at 8 GB | Use Drive for Desktop — it resumes; do not delete the partial file |
| "Drive storage full" | Free tier is 15 GB shared with Gmail/Photos — free space or use a second account |
| `cp` from Drive mount slow | Use the gdown shareable link (§3.2) instead of FUSE |
| `ladd/train` count ≠ 1220 after unzip | Re-unzip; verify §2.1 first — the zip is exactly what you uploaded |
| D-Fire download restarting after disconnect | Re-run §3.4 — it skips already-downloaded images |
| Session died mid-extraction | Fresh session, re-run §3.1 + §3.5 with the same `ds`/`split` — resumes from `records.json` |
| All D-Fire mAP50 = 0.0 later | Stale clone — `git pull` (needs the 2026-08-07 image-id remap) |
| Want results sooner | Run the LADD-only 10-seed protocol first (`--datasets ladd`) — the protocol and report support a single dataset |
