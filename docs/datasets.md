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
| License | **TBD — verify (issue #1)** |
| Role | Primary few-shot detection benchmark |

### D-Fire
| Field | Value |
|-------|-------|
| Images | 21,527 |
| Classes | 2 (fire, smoke) |
| Annotation | Boxes |
| Zero-shot → transfer (Grounding DINO) | 27.5% → 65.6% mAP50 (gap 38.1 pp) |
| License | **TBD — verify (issue #1)** |
| Role | Primary few-shot detection benchmark |

## Auxiliary segmentation datasets (novel-category validation)

### RescueNet
| Field | Value |
|-------|-------|
| Images | 4,494 |
| Classes | 10 (semantic segmentation) |
| License | **TBD — verify (issue #1)** |
| Conversion | `data/mask_to_box/filter.py` |
| Retained classes (frozen) | building, pool, vehicle, debris (region-level), roof |
| Excluded (stuff) | road, tree, grass, sand, water |

### FloodNet+
| Field | Value |
|-------|-------|
| Images | 2,289 |
| Classes | 9 (semantic segmentation) |
| License | **TBD — verify (issue #1)** |
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
dropped and logged in `docs/change_log.md`.
