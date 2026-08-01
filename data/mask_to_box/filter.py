"""Pre-registered mask-to-box conversion with frozen filtering rules.

Converts semantic-segmentation masks (RescueNet, FloodNet+) into COCO-style
bounding-box annotations for detection evaluation.

Frozen rules (see docs/pre_registration.md, issue #3):
  * minimum box area  >= MIN_BOX_AREA_PX2 (32 px^2)
  * maximum box area  <  MAX_BOX_AREA_FRACTION * image area (50%)
  * aspect ratio      in [MIN_ASPECT, MAX_ASPECT] (1:10 .. 10:1)
  * pure stuff classes excluded (configured via retained class ids)
  * damage-level classes retained but flagged ``region_level``
  * classes with < MIN_VALID_BOXES_PER_CLASS (10) valid boxes across the
    whole dataset are excluded from evaluation and reported as such

The rules are module-level constants and MUST NOT change after the pre-
registration freeze; any change requires a docs/change_log.md entry.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy import ndimage

# --- Frozen pre-registration constants (do not edit post-freeze) ------------
MIN_BOX_AREA_PX2: float = 32.0
MAX_BOX_AREA_FRACTION: float = 0.5
MIN_ASPECT: float = 1.0 / 10.0
MAX_ASPECT: float = 10.0
MIN_VALID_BOXES_PER_CLASS: int = 10
EPS: float = 1e-9

# Class "types" used by the class tables in configs/datasets/*.yaml.
STUFF_TYPES = {"stuff"}
REGION_LEVEL_TYPES = {"region_level", "region_level_damage"}


def filter_boxes(
    boxes: np.ndarray,
    image_h: int,
    image_w: int,
    min_area: float = MIN_BOX_AREA_PX2,
    max_area_fraction: float = MAX_BOX_AREA_FRACTION,
    min_aspect: float = MIN_ASPECT,
    max_aspect: float = MAX_ASPECT,
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter boxes (N, 4) in [x1, y1, x2, y2] format.

    Returns (kept_boxes, mask) where ``mask`` is a boolean array aligned with
    the input rows. A box is kept iff ALL rules pass:
      * area >= min_area
      * area < max_area_fraction * (image_h * image_w)
      * min_aspect <= width / height <= max_aspect  (uses the wider dimension
        ratio so the rule is orientation-agnostic for swapped boxes)

    Note: the aspect-ratio rule is evaluated as
        min_aspect <= max(w,h)/min(w,h) <= max_aspect
    which is equivalent to the pre-registered 1:10..10:1 statement.
    """
    if boxes.size == 0:
        return boxes, np.zeros(0, dtype=bool)

    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    areas = widths * heights
    image_area = float(image_h * image_w)

    keep_area_min = areas >= min_area
    keep_area_max = areas < max_area_fraction * image_area

    with np.errstate(divide="ignore", invalid="ignore"):
        aspect = np.maximum(widths, heights) / np.maximum(np.minimum(widths, heights), EPS)
    keep_aspect = (aspect >= min_aspect) & (aspect <= max_aspect)

    mask = keep_area_min & keep_area_max & keep_aspect
    return boxes[mask], mask


def connected_components_to_boxes(
    mask: np.ndarray, min_area: float = MIN_BOX_AREA_PX2
) -> np.ndarray:
    """Extract bounding boxes (x1, y1, x2, y2) of connected components.

    ``mask`` is a binary HxW array. Components smaller than ``min_area`` are
    discarded here (the full area filter is re-applied downstream so the two
    are consistent).
    """
    labels, n = ndimage.label(mask)
    if n == 0:
        return np.zeros((0, 4), dtype=float)

    slices = ndimage.find_objects(labels)
    boxes: List[Tuple[float, float, float, float]] = []
    for sl in slices:
        if sl is None:
            continue
        # Exclusive coordinates (x2, y2 = stop) so that box area
        # (x2-x1)*(y2-y1) is consistent with filter_boxes() area.
        y1, x1 = sl[0].start, sl[1].start
        y2, x2 = sl[0].stop, sl[1].stop
        if (x2 - x1) * (y2 - y1) < min_area:
            continue
        boxes.append((float(x1), float(y1), float(x2), float(y2)))
    if not boxes:
        return np.zeros((0, 4), dtype=float)
    return np.asarray(boxes, dtype=float)


def build_class_lookup(class_config: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """Build {pixel_value: class_info} from a dataset config's ``classes`` map.

    Pixel values come from each class's ``id`` field. Returned entries contain
    ``name``, ``retained`` and ``type``.
    """
    lookup: Dict[int, Dict[str, Any]] = {}
    for name, info in class_config["classes"].items():
        pixel_value = int(info["id"])
        lookup[pixel_value] = {
            "name": name,
            "retained": bool(info.get("retained", False)),
            "type": info.get("type", "object_like"),
        }
    return lookup


def masks_to_coco(
    mask_paths: Sequence[Path],
    image_paths: Sequence[Path],
    image_sizes: Sequence[Tuple[int, int]],
    class_lookup: Dict[int, Dict[str, Any]],
    min_valid_boxes: int = MIN_VALID_BOXES_PER_CLASS,
) -> Dict[str, Any]:
    """Convert segmentation masks to a filtered COCO-style annotation dict.

    Args:
        mask_paths: paths to PNG/npz semantic masks (pixel value = class id).
        image_paths: parallel list of image paths (only for metadata).
        image_sizes: parallel list of (height, width).
        class_lookup: pixel value -> {name, retained, type} (from config).
        min_valid_boxes: classes with fewer than this many valid boxes across
            the whole dataset are excluded and listed in ``info``
            (pre-registered rule: 10).

    Returns a COCO-style dict with keys ``images``, ``annotations``,
    ``categories`` and ``info``. Only ``retained`` classes are kept; region-
    level classes are kept but tagged via the ``region_level`` flag on the
    category entry. Pure stuff classes are dropped. Classes that do not reach
    ``min_valid_boxes`` boxes are excluded and recorded in ``info``.
    """
    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    categories: List[Dict[str, Any]] = []
    cat_id_map: Dict[int, int] = {}
    next_cat_id = 1
    next_ann_id = 1

    for img_id, (mask_path, image_path, (h, w)) in enumerate(
        zip(mask_paths, image_paths, image_sizes)
    ):
        images.append(
            {
                "id": img_id,
                "file_name": str(image_path.name),
                "height": h,
                "width": w,
            }
        )
        mask = _load_mask(mask_path)
        if mask.shape != (h, w):
            raise ValueError(
                f"mask {mask_path} shape {mask.shape} != image size {(h, w)}"
            )

        for pixel_value, info in sorted(class_lookup.items()):
            if not info["retained"]:
                continue
            if pixel_value not in cat_id_map:
                cat_id_map[pixel_value] = next_cat_id
                categories.append(
                    {
                        "id": next_cat_id,
                        "name": info["name"],
                        "region_level": info["type"] in REGION_LEVEL_TYPES,
                    }
                )
                next_cat_id += 1

            binary = mask == pixel_value
            if not binary.any():
                continue
            boxes = connected_components_to_boxes(binary)
            if boxes.size == 0:
                continue
            boxes, _ = filter_boxes(boxes, image_h=h, image_w=w)

            for box in boxes:
                x1, y1, x2, y2 = box
                annotations.append(
                    {
                        "id": next_ann_id,
                        "image_id": img_id,
                        "category_id": cat_id_map[pixel_value],
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "area": float((x2 - x1) * (y2 - y1)),
                    }
                )
                next_ann_id += 1

    # Pre-registered min-valid-boxes rule: classes with fewer than
    # ``min_valid_boxes`` boxes across the dataset are excluded from evaluation
    # and reported (proposal §6 filtering rules, item 4). Every RETAINED class
    # is checked -- including classes that never appeared (0 boxes).
    counts: Dict[int, int] = {}
    for ann in annotations:
        counts[ann["category_id"]] = counts.get(ann["category_id"], 0) + 1

    excluded_low_count = []
    for pixel_value, info in class_lookup.items():
        if not info["retained"]:
            continue
        cat_id = cat_id_map.get(int(pixel_value))
        count = counts.get(cat_id, 0) if cat_id is not None else 0
        if count < min_valid_boxes:
            excluded_low_count.append(info["name"])
    excluded_low_count = sorted(excluded_low_count)

    keep_ids = {c["id"] for c in categories if counts.get(c["id"], 0) >= min_valid_boxes}
    if keep_ids != {c["id"] for c in categories}:
        categories = [c for c in categories if c["id"] in keep_ids]
        annotations = [a for a in annotations if a["category_id"] in keep_ids]

    return {
        "info": {
            "description": "U-ADAPT pre-registered mask-to-box conversion",
            "rules": {
                "min_area_px2": MIN_BOX_AREA_PX2,
                "max_area_fraction": MAX_BOX_AREA_FRACTION,
                "aspect_ratio": [MIN_ASPECT, MAX_ASPECT],
                "min_valid_boxes_per_class": min_valid_boxes,
            },
            "excluded_classes_below_min_valid": sorted(excluded_low_count),
        },
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def _load_mask(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(str(path))
    if path.suffix == ".npz":
        arr = np.load(str(path))
        return arr[arr.files[0]]
    # PNG/TIFF via cv2 (lazy import; not needed for unit tests)
    import cv2

    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pre-registered mask-to-box conversion (frozen rules)."
    )
    parser.add_argument("--mask-root", required=True, type=Path, help="dir of masks")
    parser.add_argument("--image-root", required=True, type=Path, help="dir of images")
    parser.add_argument(
        "--class-config", required=True, type=Path, help="dataset yaml with classes map"
    )
    parser.add_argument("--out", required=True, type=Path, help="output COCO JSON")
    parser.add_argument("--suffix", default=".png", help="mask file suffix")
    parser.add_argument(
        "--min-valid-boxes",
        type=int,
        default=None,
        help="classes below this box count are excluded (default: config or 10)",
    )
    args = parser.parse_args(argv)

    cfg = _load_yaml(args.class_config)
    min_valid = (
        args.min_valid_boxes
        if args.min_valid_boxes is not None
        else cfg.get("min_valid_boxes", MIN_VALID_BOXES_PER_CLASS)
    )
    class_lookup = build_class_lookup(cfg)
    mask_paths = sorted(args.mask_root.glob(f"*{args.suffix}"))
    if not mask_paths:
        print(f"ERROR: no masks found in {args.mask_root}", file=sys.stderr)
        return 1

    image_sizes: List[Tuple[int, int]] = []
    for mask_path in mask_paths:
        mask = _load_mask(mask_path)
        image_sizes.append((int(mask.shape[0]), int(mask.shape[1])))

    image_paths = [
        args.image_root / mask_path.name
        for mask_path in mask_paths
        if (args.image_root / mask_path.name).exists()
    ]
    coco = masks_to_coco(
        mask_paths, image_paths, image_sizes, class_lookup, min_valid_boxes=min_valid
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(coco, fh, indent=2)
    print(f"wrote {args.out} ({len(coco['annotations'])} annotations)")
    if coco["info"].get("excluded_classes_below_min_valid"):
        print(
            "excluded (below min_valid_boxes): "
            + ", ".join(coco["info"]["excluded_classes_below_min_valid"]),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
