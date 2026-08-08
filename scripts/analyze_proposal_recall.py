#!/usr/bin/env python
"""analyze_proposal_recall.py — proposal-recall ceiling diagnostic.

For every ground-truth box in the evaluated split, checks whether ANY cached
proposal (top-k, score-filtered at extraction) overlaps it at IoU >= 0.5,
**regardless of score and predicted class** (geometry-only matching). This
answers the ceiling question: was the box EVER proposed? No re-scoring of the
cached proposal set (Mode A/B, naive averaging, or any re-ranker) can recover
a box that is not in the set.

Reports, per dataset:
  * overall + per-image recall (geometry-only, IoU >= 0.5),
  * recall by GT box size (small/medium/large) and per class,
  * recall vs proposal budget (top-1/3/5/10/20/50/100, score-ranked),
  * max covering-proposal score for covered GT boxes,
  * a COCO-correct AP50 cross-check (denominator = ALL GT boxes). The repo
    ``compute_ap`` normalizes recall by *matched* GT only (n_pos = tp_cum[-1]),
    so the reported mAP50 can exceed the true recall ceiling; this script
    reports both so the divergence is explicit.

Usage:
    python scripts/analyze_proposal_recall.py --cache cached_features/ladd \
        --gt data/annotations/ladd_test.json --name "LADD (person)"
    python scripts/analyze_proposal_recall.py --cache cached_features/dfire \
        --gt data/annotations/dfire_test.json --name "D-Fire (smoke/fire)"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from uadapt.metrics.detection_metrics import _iou  # noqa: E402


def load_gt(path: Path) -> Dict[str, List[Dict]]:
    """COCO GT -> {stem_id: [gt dicts]}, using the 04_evaluate id->stem remap.

    Cache image ids are filename stems (01_extract_and_cache.py) while GT
    image ids may be sequential COCO ints (D-Fire) or already stems (LADD);
    the remap below mirrors scripts/04_evaluate.py::_coco_to_gt.
    """
    coco = json.load(open(path))
    id_to_stem = {
        str(img.get("id")): Path(img["file_name"]).stem
        for img in coco.get("images", [])
        if img.get("file_name")
    }
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    w = {str(img["id"]): img.get("width", 1.0) for img in coco.get("images", [])}
    h = {str(img["id"]): img.get("height", 1.0) for img in coco.get("images", [])}
    gt_by_img: Dict[str, List[Dict]] = {}
    for ann in coco["annotations"]:
        b = ann["bbox"]
        iid = id_to_stem.get(str(ann["image_id"]), str(ann["image_id"]))
        gt_by_img.setdefault(iid, []).append(
            {
                "box": np.asarray([b[0], b[1], b[0] + b[2], b[1] + b[3]], dtype=float),
                "cls": cat_name.get(ann["category_id"], "?"),
                "area_frac": float(b[2] * b[3]) / float(w.get(str(ann["image_id"]), 1.0))
                / float(h.get(str(ann["image_id"]), 1.0)),
            }
        )
    return gt_by_img


def analyze(cache: Path, gt_path: Path, name: str, iou_thr: float) -> None:
    rec = json.load(open(cache / "test" / "records.json"))
    gt_by_img = load_gt(gt_path)
    prop_by_img = {
        iid: [np.asarray(r["bbox"], dtype=float) for r in recs]
        for iid, recs in rec["records"].items()
    }
    score_by_img = {
        iid: np.asarray([float(r["score"]) for r in recs], dtype=float)
        for iid, recs in rec["records"].items()
    }

    evaled = [i for i in gt_by_img if i in prop_by_img]
    rows: List[Tuple[str, str, float, bool]] = []  # (img, cls, area_frac, covered)
    for iid in evaled:
        props = prop_by_img[iid]
        for g in gt_by_img[iid]:
            ious = [_iou(g["box"], p) for p in props]
            rows.append((iid, g["cls"], g["area_frac"], bool(ious) and max(ious) >= iou_thr))

    total = len(rows)
    covered = sum(r[3] for r in rows)
    print(f"\n{'=' * 72}\n{name}  (top-k proposals, geometry-only recall, IoU >= {iou_thr})")
    print(f"cached images with GT: {len(evaled)}/{len(prop_by_img)}  "
          f"|  GT boxes evaluated: {total}")
    print(f"OVERALL RECALL: {covered / total * 100:.2f}%  ({covered}/{total})")

    img_rec: Dict[str, List[int]] = {}
    for iid, cls, af, cov in rows:
        img_rec.setdefault(iid, [0, 0])
        img_rec[iid][0] += int(cov)
        img_rec[iid][1] += 1
    per_img = [a / b for a, b in img_rec.values()]
    print(f"per-image recall: mean {np.mean(per_img) * 100:.1f}%, "
          f"median {np.median(per_img) * 100:.1f}%, "
          f"100% images: {sum(1 for v in per_img if v == 1.0)}/{len(per_img)}")

    for lo, hi, lab in [(0.0, 0.01, "small  (<1% img)"),
                        (0.01, 0.09, "medium (1-9%)"),
                        (0.09, 9.0, "large  (>=9%)")]:
        sub = [r for r in rows if lo <= r[2] < hi]
        if sub:
            c = sum(r[3] for r in sub)
            print(f"  {lab}: {c}/{len(sub)} = {c / len(sub) * 100:5.1f}% recall")

    classes = sorted({r[1] for r in rows})
    for cls in classes:
        sub = [r for r in rows if r[1] == cls]
        c = sum(r[3] for r in sub)
        print(f"  class {cls!r}: {c}/{len(sub)} = {c / len(sub) * 100:5.1f}% recall")

    # Recall vs proposal budget (score-ranked within each image)
    for budget in (1, 3, 5, 10, 20, 50, 100):
        n_gt = n_cov = 0
        for iid, gts in gt_by_img.items():
            if iid not in prop_by_img:
                continue
            props = prop_by_img[iid]
            order = np.argsort(-score_by_img[iid])[:budget]
            kept = [props[i] for i in order]
            for g in gts:
                n_gt += 1
                if any(_iou(g["box"], p) >= iou_thr for p in kept):
                    n_cov += 1
        mark = "  <-- cached top-100" if budget == 100 else ""
        print(f"  top-{budget:>3} recall: {n_cov / n_gt * 100:5.1f}%{mark}")

    # Max covering-proposal score for covered GT boxes (scoring-side signal)
    cover = []
    for iid, gts in gt_by_img.items():
        if iid not in prop_by_img:
            continue
        sc = score_by_img[iid]
        props = prop_by_img[iid]
        for g in gts:
            hits = np.nonzero([_iou(g["box"], p) >= iou_thr for p in props])[0]
            if hits.size:
                cover.append(sc[hits].max())
    if cover:
        cover = np.asarray(cover)
        print(f"covered GT boxes: {len(cover)}; max covering-proposal score "
              f">= 0.5: {(cover >= 0.5).mean() * 100:.1f}%, >= 0.7: "
              f"{(cover >= 0.7).mean() * 100:.1f}%")

    # COCO-correct AP50 cross-check (denominator = ALL GT; the repo metric
    # uses only matched GT, so its mAP50 can exceed this ceiling).
    gts_flat = [g for iid in evaled for g in gt_by_img[iid]]
    preds = [
        (iid, r["score"], np.asarray(r["bbox"], dtype=float))
        for iid, recs in rec["records"].items()
        for r in recs
    ]
    n_gt = len(gts_flat)
    gt_by_img2 = {iid: [g["box"] for g in gs] for iid, gs in gt_by_img.items()}
    preds_sorted = sorted(preds, key=lambda p: -p[1])
    tp_cum = 0
    matched = 0
    rc_points = []
    prec_vals = []
    fp_cum = 0
    for iid, sc, box in preds_sorted:
        gl = gt_by_img2.get(iid, [])
        hit = False
        for i, gt in enumerate(gl):
            if gt is not None and _iou(box, gt) >= iou_thr:
                gl[i] = None
                hit = True
                break
        tp_cum += int(hit)
        fp_cum += int(not hit)
        matched += int(hit)
        rc_points.append(tp_cum / n_gt)
        prec_vals.append(tp_cum / max(tp_cum + fp_cum, 1))
    ap = 0.0
    rc_points = np.asarray(rc_points)
    prec_vals = np.asarray(prec_vals)
    for r in np.linspace(0.0, 1.0, 101):
        idx = rc_points >= r
        ap += (np.max(prec_vals[idx]) if idx.any() else 0.0) / 101
    print(f"COCO-correct AP50 (denom = ALL GT): {ap * 100:.2f}%  "
          f"[max achievable = recall {matched / n_gt * 100:.2f}%]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Proposal-recall ceiling diagnostic.")
    parser.add_argument("--cache", required=True, type=Path,
                        help="cache dir containing a test/ split, e.g. cached_features/ladd")
    parser.add_argument("--gt", required=True, type=Path,
                        help="COCO-style GT JSON, e.g. data/annotations/ladd_test.json")
    parser.add_argument("--name", default="dataset",
                        help="display label for the report block")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="IoU threshold for 'covered' (default: 0.5)")
    args = parser.parse_args()
    analyze(args.cache, args.gt, args.name, args.iou)


if __name__ == "__main__":
    main()
