# Mask-to-Box Conversion (Pre-Registered)

Converts semantic-segmentation masks (RescueNet, FloodNet+) into COCO-style
bounding-box annotations for detection evaluation, using **frozen filtering
rules** that are pre-registered before any experimental results are collected.

## Frozen Filtering Rules (see `docs/pre_registration.md`)

| Rule | Value |
|------|-------|
| Minimum box area | ≥ 32 px² |
| Maximum box area | < 50% of image area |
| Aspect ratio | between 1:10 and 10:1 |
| Pure stuff classes | excluded (grass, tree, road, water, sand) |
| Damage-level classes | treated as region-level targets, reported separately |
| Output format | COCO-style JSON (`images`, `annotations`, `categories`) |

## Usage

```bash
# Convert segmentation masks to filtered COCO-style boxes
python data/mask_to_box/filter.py \
    --mask-root data/raw/rescuenet/masks/train \
    --image-root data/raw/rescuenet/images/train \
    --class-config configs/datasets/rescuenet.yaml \
    --out data/annotations/rescuenet_train.json
```

Unit tests: `tests/test_mask_to_box.py`.
