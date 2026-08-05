# Licenses

Living document for issue #1 *"Verify dataset and model licenses"*. Status
column is filled in during Milestone 1; verified entries get a date + source.

**Pre-registered policy:** dataset and model licenses are checked before
experiments. If any dataset license restricts academic use, the dataset is
replaced or dropped and logged in [`change_log.md`](change_log.md).

> **Milestone-1 status (2026-08-04, updated 2026-08-06):** all four datasets and
> all model backbones have been checked. None restricts academic use, so no
> dataset needs to be replaced or dropped. Two caveats:
> 1. **LADD — license verified 2026-08-06: GPL-3.0.** The official repository
>    (`lacmus-foundation/ladd-utils`) states: *"LADD is licensed under GNU
>    General Public License v3.0. ... This license applies not only to the
>    dataset, but also to ALL SOFTWARE products that use it to one degree or
>    another."* This is a **copyleft** license (not CC0/CC-BY as initially
>    assumed): the dataset itself may be used for academic research and
>    evaluation, but any derived *code* distributed to third parties carries
>    GPL-3.0 obligations. For this thesis (evaluation-only use, code released
>    under MIT separately from the dataset), use is compatible; do **not**
>    redistribute the dataset itself. The Kaggle mirror
>    ([`maxinstellar/lost-people-detection`](https://www.kaggle.com/datasets/maxinstellar/lost-people-detection),
>    9.98 GB, publisher Maksim Tingaev) labels the license "Other" with an
>    empty description, so the official repo is the authoritative source.
>    Original GitHub (`huyhieupham/LADD`) is offline (404, checked 2026-08-04).
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
| LADD | **GPL-3.0** (copyleft) | **Confirmed (2026-08-06)** — [lacmus-foundation/ladd-utils README](https://github.com/lacmus-foundation/ladd-utils): *"LADD is licensed under GNU General Public License v3.0 ... This license applies not only to the dataset, but also to ALL SOFTWARE products that use it to one degree or another."* | Evaluation-only academic use compatible with this thesis; **do not redistribute the dataset**. Kaggle mirror ([maxinstellar/lost-people-detection](https://www.kaggle.com/datasets/maxinstellar/lost-people-detection), 9.98 GB) lists "Other" — official repo is authoritative. `huyhieupham/LADD` is offline (404, 2026-08-04); user-supplied archive (`~/Downloads/archive.zip`) extracted 2026-08-05 (provenance: `docs/change_log.md`) |
| D-Fire | Free for research use (no explicit OSS license on official repo) | **Confirmed (2026-08-04)** — [gaia-solutions-on-demand/DFireDataset](https://github.com/gaia-solutions-on-demand/DFireDataset) README + [Neural Computing & Applications 2022 paper](https://link.springer.com/article/10.1007/s00521-022-07467-z) | Images + labels free to download (OneDrive / Kaggle mirrors, verified 2026-08-04). Annotations are **YOLO format** — converted to COCO-style by the download script |
| RescueNet | CC BY-NC-ND 4.0 | **Confirmed (2026-08-04)** — [BinaLab RescueNet repo](https://github.com/BinaLab/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation) | Non-commercial + no-derivatives: compatible with this thesis (academic use); do **not** redistribute modified copies of the dataset |
| FloodNet+ | CDLA-Permissive-1.0 | **Confirmed (2026-08-04)** — [BinaLab FloodNet-Supervised_v1.0 repo](https://github.com/BinaLab/FloodNet-Supervised_v1.0) | Permissive data license (academic + commercial use permitted) |

## Repository code

MIT License (see `LICENSE`) — code written for this thesis.
