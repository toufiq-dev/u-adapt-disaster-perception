"""Detection metrics: mAP50, mAP50:95, per-class AP, Gap Recovery, recall.

Prediction/ground-truth schemas (shared by scripts/04_evaluate.py and tests):

    preds: list of dicts per image::
        {"image_id": str, "class": str, "score": float, "bbox": [x1, y1, x2, y2]}
    gts: list of dicts per image::
        {"image_id": str, "class": str, "bbox": [x1, y1, x2, y2]}

Implemented per proposal §7.4: mAP50 (primary), mAP50:95 (where feasible),
and per-class AP. All use COCO-style 101-point recall interpolation with
per-image greedy assignment (each GT box matched to at most one prediction).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

RECALL_POINTS = 101  # COCO-style 101-point interpolation


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_ap(sorted_scores: np.ndarray, tps: np.ndarray, fps: np.ndarray) -> float:
    """Average precision via 101-point recall interpolation.

    Args:
        sorted_scores: detection scores (already sorted descending).
        tps: boolean TP flags aligned with sorted_scores.
        fps: boolean FP flags aligned with sorted_scores.
    """
    tp_cum = np.cumsum(tps.astype(float))
    fp_cum = np.cumsum(fps.astype(float))
    n_pos = float(tp_cum[-1]) if len(tp_cum) else 0.0
    if n_pos == 0:
        return 0.0
    recalls = tp_cum / n_pos
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1.0)
    # 101-point interpolation
    ap = 0.0
    for r in np.linspace(0.0, 1.0, RECALL_POINTS):
        idx = recalls >= r
        ap += (np.max(precisions[idx]) if idx.any() else 0.0) / RECALL_POINTS
    return float(ap)


def _per_class_ap(
    preds: Sequence[Dict],
    gts: Sequence[Dict],
    iou_threshold: float,
) -> Dict[str, float]:
    """Per-class COCO-style AP at a fixed IoU threshold.

    Greedy per-image assignment: each ground-truth box is matched to at most
    one prediction (sorted by score) with IoU >= threshold. The returned map
    covers ALL classes (union of prediction and ground-truth classes); a class
    present in GT with zero detections contributes AP = 0.0.
    """
    # Group ground truths by (image_id, class)
    gt_by_img: Dict[str, Dict[str, List[np.ndarray]]] = {}
    for g in gts:
        gt_by_img.setdefault(g["image_id"], {}).setdefault(g["class"], []).append(
            np.asarray(g["bbox"], dtype=float)
        )

    # Group predictions by class
    pred_by_class: Dict[str, list] = {}
    for p in preds:
        pred_by_class.setdefault(p["class"], []).append(p)

    # COCO semantics: mAP averages over ALL classes (union of prediction and
    # ground-truth classes). A class present in GT but with zero detections
    # contributes AP = 0 and must not be silently dropped.
    gt_classes = {g["class"] for g in gts}
    all_classes = sorted(set(pred_by_class) | gt_classes)

    aps: Dict[str, float] = {}
    for cls in all_classes:
        ps = pred_by_class.get(cls, [])
        if not ps:
            aps[cls] = 0.0
            continue
        ps = sorted(ps, key=lambda p: p["score"], reverse=True)
        tps: List[bool] = []
        fps: List[bool] = []
        for p in ps:
            img_key = p["image_id"]
            gt_list = gt_by_img.get(img_key, {}).get(cls, [])
            matched = False
            for i, gt in enumerate(gt_list):
                if gt is not None and _iou(np.asarray(p["bbox"]), gt) >= iou_threshold:
                    gt_list[i] = None  # mark used
                    matched = True
                    break
            tps.append(matched)
            fps.append(not matched)
        scores = np.asarray([p["score"] for p in ps], dtype=float)
        aps[cls] = compute_ap(
            scores, np.asarray(tps, dtype=bool), np.asarray(fps, dtype=bool)
        )
    return aps


def compute_map50(preds: Sequence[Dict], gts: Sequence[Dict], iou_threshold: float = 0.5) -> float:
    """COCO-style mAP@IoU=0.5, per-class AP averaged over classes."""
    aps = _per_class_ap(preds, gts, iou_threshold)
    return float(np.mean(list(aps.values()))) if aps else 0.0


def compute_map50_95(preds: Sequence[Dict], gts: Sequence[Dict]) -> float:
    """COCO-style mAP averaged over IoU thresholds 0.5:0.05:0.95 (10 values).

    Proposal §7.4: mAP50:95 "where feasible". Per-class AP is averaged over
    the 10 IoU thresholds, then over classes (equivalent to the grand mean
    over (class, threshold) pairs since every threshold uses the same class
    set).
    """
    iou_thresholds = np.linspace(0.5, 0.95, 10)  # 0.5, 0.55, ..., 0.95
    maps = [
        compute_map50(preds, gts, iou_threshold=float(t)) for t in iou_thresholds
    ]
    return float(np.mean(maps)) if maps else 0.0


def compute_per_class_ap(
    preds: Sequence[Dict],
    gts: Sequence[Dict],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Per-class COCO-style AP at a fixed IoU threshold (proposal §7.4).

    Returns {class: AP} over the union of prediction and ground-truth classes;
    classes in GT with no detections are kept with AP = 0.0.
    """
    return _per_class_ap(preds, gts, iou_threshold)


def gap_recovery(
    map_adapted: float,
    map_zero_shot: float,
    map_oracle: float,
) -> float:
    """Fraction of the zero-shot-to-transfer gap recovered.

    gap_recovery = (map_adapted - map_zero_shot) / (map_oracle - map_zero_shot).

    Negative values are allowed and reported as-is (pre-registered: a negative
    gap recovery means the adapter underperforms zero-shot and is a finding,
    not an error). Guard: if oracle == zero-shot, returns 0.0.
    """
    denom = map_oracle - map_zero_shot
    if abs(denom) < 1e-12:
        return 0.0
    return (map_adapted - map_zero_shot) / denom


def proposal_recall(
    preds: Sequence[Dict],
    gts: Sequence[Dict],
    iou_threshold: float = 0.5,
) -> float:
    """Raw proposal recall (ceiling analysis): fraction of ground-truth boxes
    covered by at least one predicted box at IoU >= threshold, regardless of
    score threshold. Establishes the maximum recoverable performance for any
    downstream re-scoring method."""
    gt_by_img: Dict[str, Dict[str, List[np.ndarray]]] = {}
    for g in gts:
        gt_by_img.setdefault(g["image_id"], {}).setdefault(g["class"], []).append(
            np.asarray(g["bbox"], dtype=float)
        )
    pred_by_img: Dict[str, List[np.ndarray]] = {}
    for p in preds:
        pred_by_img.setdefault(p["image_id"], []).append(np.asarray(p["bbox"], dtype=float))

    covered = 0
    total = 0
    for img_key, classes in gt_by_img.items():
        pred_boxes = pred_by_img.get(img_key, [])
        for cls, boxes in classes.items():
            for gt in boxes:
                total += 1
                if any(_iou(gt, pb) >= iou_threshold for pb in pred_boxes):
                    covered += 1
    return float(covered / total) if total else 0.0
