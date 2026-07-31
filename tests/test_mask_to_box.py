"""Unit tests for the pre-registered mask-to-box filtering rules.

Covers the frozen rules (docs/pre_registration.md §6):
  * minimum box area >= 32 px^2
  * maximum box area < 50% of image area
  * aspect ratio within 1:10 .. 10:1
  * pure stuff classes excluded
  * COCO-style JSON output

Masks are written as .npy so the tests run without cv2/scipy image I/O.
"""

import numpy as np
import pytest

from data.mask_to_box.filter import (
    MAX_ASPECT,
    MAX_BOX_AREA_FRACTION,
    MIN_ASPECT,
    MIN_BOX_AREA_PX2,
    build_class_lookup,
    connected_components_to_boxes,
    filter_boxes,
    masks_to_coco,
)

IMG_H, IMG_W = 512, 512


def _write_mask(tmp_path, mask, name="m.npy"):
    path = tmp_path / name
    np.save(path, mask)
    return path


def test_minimum_area_rule():
    boxes = np.array([[0, 0, 4, 4], [0, 0, 8, 8], [0, 0, 5.6, 5.8]])  # 16, 64, 32.48 px^2
    kept, mask = filter_boxes(boxes, IMG_H, IMG_W)
    # 16 px^2 excluded; 64 px^2 kept; 32.48 px^2 (>= 32) kept
    assert (kept[:, 2] - kept[:, 0]).tolist() == [8.0, 5.6]
    assert mask.tolist() == [False, True, True]


def test_maximum_area_fraction_rule():
    # Box covering 60% of image area -> excluded; 40% -> kept
    w_big = np.sqrt(0.6 * IMG_H * IMG_W)
    w_ok = np.sqrt(0.4 * IMG_H * IMG_W)
    boxes = np.array([[0, 0, w_big, w_big], [0, 0, w_ok, w_ok]])
    kept, mask = filter_boxes(boxes, IMG_H, IMG_W)
    assert mask.tolist() == [False, True]
    assert kept.shape[0] == 1


def test_aspect_ratio_rule():
    # 100:1 -> excluded; 5:1 -> kept; 8:1 -> kept; 20:1 -> excluded
    boxes = np.array(
        [
            [0, 0, 1, 100],
            [0, 0, 20, 100],
            [0, 0, 160, 20],
            [0, 0, 1, 20],
        ]
    )
    kept, mask = filter_boxes(boxes, IMG_H, IMG_W)
    assert mask.tolist() == [False, True, True, False]
    assert kept.shape[0] == 2


def test_constants_frozen():
    assert MIN_BOX_AREA_PX2 == 32.0
    assert MAX_BOX_AREA_FRACTION == 0.5
    assert MIN_ASPECT == pytest.approx(1 / 10)
    assert MAX_ASPECT == 10.0


def test_connected_components_and_stuff_class_exclusion(tmp_path):
    # Binary mask: one 40x40 component labeled class 1 (building, retained),
    # one component labeled class 2 (road, stuff -> excluded).
    mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    mask[100:140, 100:140] = 1
    mask[300:400, 300:350] = 2  # road region, 50x100 px

    cfg = {
        "classes": {
            "building": {"id": 1, "retained": True, "type": "object_like"},
            "road": {"id": 2, "retained": False, "type": "stuff"},
        }
    }
    lookup = build_class_lookup(cfg)
    mask_path = _write_mask(tmp_path, mask)

    coco = masks_to_coco([mask_path], [mask_path], [(IMG_H, IMG_W)], lookup)

    names = {c["name"] for c in coco["categories"]}
    assert names == {"building"}  # road (stuff) excluded
    assert len(coco["annotations"]) == 1
    ann = coco["annotations"][0]
    x1, y1, w, h = ann["bbox"]
    assert (w, h) == (40.0, 40.0)


def test_masks_to_coco_json_structure(tmp_path):
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[10:50, 10:50] = 7  # debris (region_level, retained)
    cfg = {
        "classes": {
            "debris": {"id": 7, "retained": True, "type": "region_level"},
            "water": {"id": 10, "retained": False, "type": "stuff"},
        }
    }
    lookup = build_class_lookup(cfg)
    mask_path = _write_mask(tmp_path, mask)

    coco = masks_to_coco([mask_path], [mask_path], [(200, 200)], lookup)
    assert set(coco.keys()) == {"info", "images", "annotations", "categories"}
    assert coco["images"][0]["height"] == 200
    assert coco["categories"][0]["region_level"] is True  # flagged region-level
    assert coco["info"]["rules"]["min_area_px2"] == 32.0


def test_shape_mismatch_raises(tmp_path):
    mask_path = _write_mask(tmp_path, np.zeros((100, 100), dtype=np.uint8))
    cfg = {"classes": {"a": {"id": 1, "retained": True, "type": "object_like"}}}
    lookup = build_class_lookup(cfg)
    with pytest.raises(ValueError):
        masks_to_coco([mask_path], [mask_path], [(200, 200)], lookup)


def test_connected_components_extraction():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[0:10, 0:10] = 1
    mask[30:40, 30:40] = 1  # disconnected component, same label
    boxes = connected_components_to_boxes(mask == 1, min_area=10.0)
    assert boxes.shape == (2, 4)
    # tiny noise component (3x3 < min_area) dropped
    mask[50:53, 50:53] = 1
    boxes = connected_components_to_boxes(mask == 1, min_area=10.0)
    assert boxes.shape == (2, 4)
