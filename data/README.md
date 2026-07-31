# Data

**No raw data is stored in this repository.** This directory contains only
scripts, checksums, and documentation for acquiring and preparing datasets.

| Path | Purpose |
|------|---------|
| `download_scripts/` | Downloaders + checksums for LADD, D-Fire, RescueNet, FloodNet+ |
| `mask_to_box/` | Pre-registered segmentation-mask → bounding-box conversion with frozen filtering rules |

Refer to [`docs/datasets.md`](../docs/datasets.md) for dataset details and to
[`docs/licenses.md`](../docs/licenses.md) for license verification status.

## Expected on-disk layout (outside the repository)

```
data/raw/            # downloaded datasets (gitignored)
data/processed/      # mask->box JSON, filtered annotations (gitignored)
data/annotations/    # COCO-style JSON produced by mask_to_box/filter.py (gitignored)
cached_features/     # extracted features (gitignored)
```

Paths in `configs/datasets/*.yaml` are relative to the repository root and can
be overridden via environment variables or CLI flags (`--data-root`,
`--cache-dir`).
