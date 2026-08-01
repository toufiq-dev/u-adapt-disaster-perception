# Licenses

Living document for issue #1 *"Verify dataset and model licenses"*. Status
column is filled in during Milestone 1; verified entries get a date + source.

**Pre-registered policy:** dataset and model licenses are checked before
experiments. If any dataset license restricts academic use, the dataset is
replaced or dropped and logged in [`change_log.md`](change_log.md).

## Models

| Model | License | Status | Notes |
|-------|---------|--------|-------|
| Grounding DINO (Swin-T) | Apache-2.0 | **Confirmed (proposal: "Dataset and model licenses" section)** | Permits research use **and feature caching** (pre-registered) |
| OWL-ViT (google/owlvit-base-patch32) | Apache-2.0 | To verify (issue #1) | — |
| YOLO-World (ultralytics) | AGPL-3.0 | To verify (issue #1) | ⚠️ AGPL — check thesis implications if used |
| YOLO11 (ultralytics) | AGPL-3.0 | To verify (issue #1) | ⚠️ AGPL — check thesis implications if used |
| YOLOE26 (ultralytics) | TBD | To verify (issue #1) | Cross-backbone ablation (proposal §7.3); open-weight research license expected |
| CLIP (OpenAI) | MIT | To verify (issue #1) | Encoder ablation |
| DINOv2 (Meta) | Apache-2.0 | To verify (issue #1) | Encoder ablation |

## Datasets

| Dataset | License | Status | Notes |
|---------|---------|--------|-------|
| LADD | TBD | To verify (issue #1) | Academic-use check required |
| D-Fire | TBD | To verify (issue #1) | Academic-use check required |
| RescueNet | TBD | To verify (issue #1) | Academic-use check required |
| FloodNet+ | TBD | To verify (issue #1) | Academic-use check required |

## Repository code

MIT License (see `LICENSE`) — code written for this thesis.
