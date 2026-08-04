# Licenses

Living document for issue #1 *"Verify dataset and model licenses"*. Status
column is filled in during Milestone 1; verified entries get a date + source.

**Pre-registered policy:** dataset and model licenses are checked before
experiments. If any dataset license restricts academic use, the dataset is
replaced or dropped and logged in [`change_log.md`](change_log.md).

> **Milestone-1 status (2026-08-04):** all four datasets and all model
> backbones have been checked. None restricts academic use, so no dataset
> needs to be replaced or dropped. Two caveats:
> 1. **LADD** — the official GitHub repository (`huyhieupham/LADD`) is
>    currently offline (404, checked 2026-08-04). Research use is presumed
>    from the dataset documentation, but the exact terms must be confirmed at
>    the manual download step (`data/download_scripts/README.md`).
> 2. **AGPL-3.0 backbones** (YOLO-World, YOLO11, YOLOE26) — permitted for
>    academic use, but AGPL has thesis/licensing implications if any derived
>    code is released; see the ⚠️ notes below.

## Models

| Model | License | Status | Notes |
|-------|---------|--------|-------|
| Grounding DINO (Swin-T) | Apache-2.0 | **Confirmed (2026-08-04)** — [IDEA-Research/GroundingDINO LICENSE](https://github.com/IDEA-Research/GroundingDINO/blob/main/LICENSE) | Permits research use **and feature caching** (pre-registered) |
| OWL-ViT (google/owlvit-base-patch32) | Apache-2.0 | **Confirmed (2026-08-04)** — [HF model card](https://huggingface.co/google/owlvit-base-patch32) + transformers source | — |
| YOLO-World (ultralytics) | AGPL-3.0 | **Confirmed (2026-08-04)** — ultralytics LICENSE | ⚠️ AGPL — check thesis implications if used |
| YOLO11 (ultralytics) | AGPL-3.0 | **Confirmed (2026-08-04)** — ultralytics LICENSE | ⚠️ AGPL — check thesis implications if used |
| YOLOE26 | AGPL-3.0 | **Confirmed (2026-08-04)** — [THU-MIG/yoloe LICENSE](https://github.com/THU-MIG/yoloe/blob/main/LICENSE) | ⚠️ AGPL — check thesis implications if used; cross-backbone ablation (proposal §7.3) |
| CLIP (OpenAI) | MIT | **Confirmed (2026-08-04)** — [openai/CLIP LICENSE](https://github.com/openai/CLIP/blob/main/LICENSE) | Encoder ablation |
| DINOv2 (Meta) | Apache-2.0 | **Confirmed (2026-08-04)** — [facebookresearch/dinov2 LICENSE](https://github.com/facebookresearch/dinov2/blob/main/LICENSE) | Encoder ablation |

## Datasets

| Dataset | License | Status | Notes |
|---------|---------|--------|-------|
| LADD | Pending manual download and verification | **Milestone-1 check (2026-08-04)** — official repo offline; manual download step pending | Original GitHub repository (`huyhieupham/LADD`) is currently unavailable (404 as of 2026-08-04). User must locate an official academic source and verify the license before use. Placeholder in `data/download_scripts/download_datasets.py` — no URL was guessed |
| D-Fire | Free for research use (no explicit OSS license on official repo) | **Confirmed (2026-08-04)** — [gaia-solutions-on-demand/DFireDataset](https://github.com/gaia-solutions-on-demand/DFireDataset) README + [Neural Computing & Applications 2022 paper](https://link.springer.com/article/10.1007/s00521-022-07467-z) | Images + labels free to download (OneDrive / Kaggle mirrors, verified 2026-08-04). Annotations are **YOLO format** — converted to COCO-style by the download script |
| RescueNet | CC BY-NC-ND 4.0 | **Confirmed (2026-08-04)** — [BinaLab RescueNet repo](https://github.com/BinaLab/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation) | Non-commercial + no-derivatives: compatible with this thesis (academic use); do **not** redistribute modified copies of the dataset |
| FloodNet+ | CDLA-Permissive-1.0 | **Confirmed (2026-08-04)** — [BinaLab FloodNet-Supervised_v1.0 repo](https://github.com/BinaLab/FloodNet-Supervised_v1.0) | Permissive data license (academic + commercial use permitted) |

## Repository code

MIT License (see `LICENSE`) — code written for this thesis.
