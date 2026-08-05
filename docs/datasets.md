# Datasets

Download, organization, and pre-registered conversion notes for the four
datasets. **No raw data lives in the repository** — see
`data/download_scripts/` for scripts and checksums (Milestone 1).

## Primary detection benchmarks (few-shot)

### LADD — Large-scale Aerial Disaster Damage
| Field | Value |
|-------|-------|
| Images | 1,365 |
| Classes | 1 (pedestrian / search-and-rescue target) |
| Annotation | Boxes (COCO-style) |
| Zero-shot → transfer (Grounding DINO) | 61.0% → 92.2% mAP50 (gap 31.2 pp) |
| License | Research use (presumed; confirm at manual download) — Confirmed 2026-08-04 ([`docs/licenses.md`](licenses.md)) |
| Role | Primary few-shot detection benchmark |
| Provenance | Archive user-supplied (2026-08-05, `~/Downloads/archive.zip`); extracted via `download_datasets.py --dataset ladd --ladd-archive ... --ladd-gt-remap "Pedestrian=person"`; n=10 pilot subset in `data/raw/ladd`. See [`docs/change_log.md`](change_log.md) 2026-08-05 |

### D-Fire
| Field | Value |
|-------|-------|
| Images | 21,527 |
| Classes | 2 (fire, smoke) |
| Annotation | Boxes |
| Zero-shot → transfer (Grounding DINO) | 27.5% → 65.6% mAP50 (gap 38.1 pp) |
| License | Free for research use — Confirmed 2026-08-04 ([`docs/licenses.md`](licenses.md)) |
| Role | Primary few-shot detection benchmark |
| Annotation | Official labels are **YOLO format** (normalized `class xc yc w h`; `0=fire`, `1=smoke`); `download_datasets.py` converts to COCO-style JSON in `data/annotations/` |

## Auxiliary segmentation datasets (novel-category validation)

### RescueNet
| Field | Value |
|-------|-------|
| Images | 4,494 |
| Classes | 10 (semantic segmentation) |
| License | CC BY-NC-ND 4.0 — Confirmed 2026-08-04 ([`docs/licenses.md`](licenses.md)) |
| Conversion | `data/mask_to_box/filter.py` |
| Retained classes (frozen) | building, pool, vehicle, debris (region-level), roof |
| Excluded (stuff) | road, tree, grass, sand, water |

### FloodNet+
| Field | Value |
|-------|-------|
| Images | 2,289 |
| Classes | 9 (semantic segmentation) |
| License | CDLA-Permissive-1.0 — Confirmed 2026-08-04 ([`docs/licenses.md`](licenses.md)) |
| Conversion | `data/mask_to_box/filter.py` |
| Retained classes (frozen) | building-flooded, building-non-flooded (region-level damage), road-flooded, road-non-flooded (region-level), vehicle, pool |
| Excluded (stuff) | water, tree, grass |

> Damage-level classes are region-level targets: results on these are reported
> **separately** from the LADD/D-Fire benchmarks.

## Pre-registered mask-to-box rules (frozen)

Implemented in `data/mask_to_box/filter.py`; constants are module-level.

| Rule | Value |
|------|-------|
| Minimum box area | ≥ 32 px² |
| Maximum box area | < 50% of image area |
| Aspect ratio | 1:10 – 10:1 |
| Pure stuff classes | excluded |
| Minimum valid boxes per class | ≥ 10 across the whole dataset, else excluded and reported |
| Output | COCO-style JSON (`data/annotations/`, gitignored) |

## Layout (all outside git)

```
data/raw/{ladd,dfire,rescuenet,floodnet}/{train,val,test}/
data/processed/...            # converted/filtered artifacts
data/annotations/*.json       # COCO-style annotations
cached_features/{train,val,test}/  # extracted features
```

If any dataset license restricts academic use, the dataset is replaced or
dropped and logged in `docs/change_log.md` (none did — all four licenses
confirmed 2026-08-04).

## Milestone-1 download + pilot runbook (2026-08-04)

Downloading and organizing LADD/D-Fire (with `--subset N` pilot mode),
YOLO→COCO conversion, the verified source links, the LADD manual-download
steps, and the n=10 pilot runbook live in
[`data/download_scripts/README.md`](../data/download_scripts/README.md). The
orchestrated real-data pipeline is `scripts/run_real_data_validation.sh`
(`N_TEST_IMAGES=10` for the pilot); its report labels itself
**"PILOT RESULTS (n=10 images)"** so pilot numbers are never mistaken for
final thesis results.
