# Download Scripts

Milestone-1 (2026-08-04) download and organization tooling for the four
datasets. Raw data is **never** uploaded to GitHub — only scripts, checksums
(e.g., SHA-256), and documentation live in this repository.

## Policy (frozen)

1. **Never upload raw data to GitHub.** Only scripts, checksums (e.g.,
   SHA-256), and documentation live in this repository. `data/raw/`,
   `data/annotations/` and `*.json` are gitignored.
2. Each dataset directory below will contain:
   - `download.py` (or a section of `download_datasets.py`) — fetches the dataset
   - `sha256sums.txt` — integrity checksums (written after each download)
   - `README.md` — dataset-specific organization notes
3. Download and license status must be confirmed **before** the pilot
   experiment. If any dataset license restricts academic use, the dataset is
   replaced or dropped and logged in [`docs/change_log.md`](../../docs/change_log.md).
   All four licenses were verified on 2026-08-04 — see
   [`docs/licenses.md`](../../docs/licenses.md).

## Layout

```
download_scripts/
├── README.md             (this file)
├── download_datasets.py  (single entry point: LADD + D-Fire)
├── ladd/
│   └── sha256sums.txt     (written after a LADD download)
└── dfire/
    └── sha256sums.txt     (written after a D-Fire download)
```

## Usage — LADD and D-Fire (primary benchmarks)

```bash
# 1) Verify all configured URLs are reachable (no download)
python download_datasets.py --check-only

# 2) D-Fire pilot — 10 images per split (train/test) + YOLO->COCO conversion
python download_datasets.py --dataset dfire --subset 10

# 3) Full D-Fire (all images, all splits)
python download_datasets.py --dataset dfire

# 4) LADD — requires a verified download URL (official repo is offline,
#    see below). Placeholder must be replaced first:
python download_datasets.py --dataset ladd --ladd-url <VERIFIED_URL> --subset 10
```

Outputs (all gitignored):

| Output | Path |
|--------|------|
| LADD images | `data/raw/ladd/{train,val,test}/` |
| D-Fire images | `data/raw/dfire/{train,val,test}/` |
| D-Fire COCO GT | `data/annotations/dfire_{train,val,test}.json` |
| LADD COCO GT | `data/annotations/ladd_test.json` (user-provided) |
| Checksums | `download_scripts/{ladd,dfire}/sha256sums.txt` |

> **`--subset N` semantics:** the full official archive is still downloaded
> (OneDrive shares cannot be ranged/partially downloaded), so `--subset` does
> **not** save download bandwidth — it saves extraction/copy time and the disk
> used by `data/raw`/`data/annotations`. Only the first N sorted images per
> split are copied into `data/raw` and the COCO annotations are filtered to
> those images. This keeps the n=10 pilot fast and correct.

### D-Fire — sources (verified 2026-08-04)

Official repo: <https://github.com/gaia-solutions-on-demand/DFireDataset>
(authors' original `gaiasd/DFire` is archived/redirected). Download links from
the official README:

- Images + labels (OneDrive): <https://1drv.ms/u/c/c0bd25b6b048b01d/EbLgD7bES4FDvUN37Grxn8QBF5gIBBc7YV2qklF08GCiBw>
- Pre-split train/val/test (OneDrive folder): <https://1drv.ms/f/c/c0bd25b6b048b01d/Ema8FFze8mFIlM1Hn81BUUgBE3vnnmK4SQxybS-nHRt2pA?e=6rk0aN>
- Kaggle mirror: <https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo>

**Annotations are YOLO format** (normalized `class xc yc w h`; classes
`0=fire`, `1=smoke`) — `download_datasets.py` converts them to COCO-style
JSON for the evaluation pipeline. If the OneDrive download fails from a
script, click the link in a browser, save the zip, then re-run with
`--dfire-archive /path/to/DFire_images_labels.zip`.

### LADD — manual download required

The official repository (`huyhieupham/LADD`) returned **404 on 2026-08-04**
and no verified live URL could be found. Per the Milestone-1 no-guessing
policy, the script does **not** invent a download URL — it prints step-by-step
instructions and uses the placeholder `LADD_URL_PLACEHOLDER`. To proceed:

1. Obtain a verified LADD download link (author's academic site
   <https://huyhieupham.github.io/>, paper supplementary, or a Zenodo record).
2. Replace the placeholder (in the script or via `--ladd-url <URL>`).
3. Run `python download_datasets.py --dataset ladd --ladd-url <URL> --subset 10`.
4. Provide the COCO-style GT JSON (`--ladd-gt path.json` → copied to
   `data/annotations/ladd_test.json`) so the pipeline gate passes.

License: research use only (presumed — confirm exact terms at download);
tracked in [`docs/licenses.md`](../../docs/licenses.md).

## RescueNet and FloodNet+ (auxiliary segmentation datasets)

These two are downloaded manually (BinaLab Dropbox links; a click-through
may be required): <https://github.com/BinaLab/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation>
and <https://github.com/BinaLab/FloodNet-Supervised_v1.0>.

Once the masks and images are on disk, generate the frozen COCO-style
annotations with the pre-registered mask-to-box converter
(`data/mask_to_box/filter.py`, frozen rules — see
[`docs/datasets.md`](../../docs/datasets.md)):

```bash
# RescueNet: masks in data/raw/rescuenet/masks, images in data/raw/rescuenet/images
python data/mask_to_box/filter.py \
    --mask-root data/raw/rescuenet/masks \
    --image-root data/raw/rescuenet/images \
    --class-config configs/datasets/rescuenet.yaml \
    --out data/annotations/rescuenet_test.json

# FloodNet+: masks in data/raw/floodnet/masks, images in data/raw/floodnet/images
python data/mask_to_box/filter.py \
    --mask-root data/raw/floodnet/masks \
    --image-root data/raw/floodnet/images \
    --class-config configs/datasets/floodnet.yaml \
    --out data/annotations/floodnet_test.json
```

> Both commands are ready to run; they are blocked only on the raw masks +
> images being on disk (RescueNet/FloodNet are not part of the n=10 pilot —
> the pilot covers LADD + D-Fire only).

## Pilot runbook (n=10, GPU machine or Colab)

The full pipeline is orchestrated by
[`scripts/run_real_data_validation.sh`](../../scripts/run_real_data_validation.sh).
On a machine with torch installed (e.g., Colab T4 or a CUDA box — see
`requirements.txt`):

```bash
# 1) Data (D-Fire automatable; LADD manual per above)
python data/download_scripts/download_datasets.py --dataset dfire --subset 10
python data/download_scripts/download_datasets.py --dataset ladd --ladd-url <URL> --subset 10

# 2) Real-data pilot: 10 test images per dataset, gates enabled
N_TEST_IMAGES=10 SKIP_PREREQS=0 bash scripts/run_real_data_validation.sh

# 3) Generate the pilot report (the generator auto-labels it
#    "PILOT RESULTS (n=10 images)" whenever N_TEST_IMAGES < 100, so it can
#    never be confused with the final thesis report):
python scripts/generate_real_data_report.py \
    --ladd-results  outputs/real_data/ladd/results.json \
    --dfire-results outputs/real_data/dfire/results.json \
    --pooled-diagnostics outputs/real_data/pooled_diagnostics.json \
    --out docs/real_data_results_pilot.md
```

The pilot report title reads **"PILOT RESULTS (n=10 images)"** with a warning
banner, so it cannot be confused with the final thesis results (the report
generator adds this automatically when `N_TEST_IMAGES < 100`).
