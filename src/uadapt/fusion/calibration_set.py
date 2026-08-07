"""Mode B calibration-set builder (proposal §5.4.2, pre-registration §5).

Builds the 20-box-per-class calibration set consumed by
``scripts/03_run_fusion.py --mode B --calibration`` from the cached TRAIN
split (the target-domain training split):

  * boxes are sampled per class (stratified), up to ``boxes_per_class``;
  * only boxes that match a same-class ground-truth box (IoU >= 0.5) are
    eligible — these are the pre-registered "labeled boxes";
  * the seed's k-shot support examples (image ids recorded in the prototype
    payload by ``02_build_prototypes.py``) are excluded, and the TEST split
    is never touched — the set is strictly disjoint from both;
  * sampling is seeded (``np.random.default_rng(seed)``) and deterministic,
    so the 10-seed protocol resamples a fresh calibration set per seed.

Per-box gate inputs mirror ``uadapt.fusion.calibration.record_gate_input``
so the gate is trained on the SAME feature distribution it sees at test time:

    x = [s_text, s_visual, sigma2_text, sigma2_visual, a_visual]

with the documented proxies: ``s_visual == a_visual == affinity`` (the
cached record has no dedicated visual-only score) and ``sigma2_text``
falling back to the neutral 0.5 (02_build_prototypes.py does not serialize
the text prototype). Correctness flags follow the real-cache convention of
``uadapt.demo.pipeline``: ``text_correct`` = same-class IoU >= 0.5 (True for
every sampled box, since only GT-matched boxes are eligible) and
``visual_correct`` = affinity >= 0.65 (``VISUAL_CORRECT_AFFINITY``).

Output schema matches the ``--calibration`` file documented in
``uadapt.fusion.calibration`` (``boxes_per_class``, ``classes``,
``samples``), extended with a ``sampling`` audit block recording per-class
eligibility counts, the seed, and the disjointness guarantees.

NOTE (n=100 pilot scale): the pilot train caches are small (LADD 10 images,
D-Fire 9 images). When fewer than ``boxes_per_class`` eligible boxes exist
for a class, the sampler keeps ALL of them and records the true count in
``sampling.per_class_eligible`` — reports must surface this before
asserting "20 samples per class".
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence

import numpy as np

from uadapt.features.cache_engine import FeatureRecord, class_text_score
from uadapt.metrics.detection_metrics import _iou
from uadapt.uncertainty.variance_estimators import visual_affinity

logger = logging.getLogger(__name__)

# Same pre-registered threshold as uadapt.demo.pipeline.VISUAL_CORRECT_AFFINITY.
VISUAL_CORRECT_AFFINITY = 0.65


def build_calibration_set(
    records: Sequence[FeatureRecord],
    ground_truth: Sequence[Dict],
    prototype_payload: Dict,
    boxes_per_class: int = 20,
    seed: int = 0,
    affinity_threshold: float = VISUAL_CORRECT_AFFINITY,
) -> Dict:
    """Build the Mode B calibration-set JSON from cached train features.

    Args:
        records: cached TRAIN-split feature records (``load_cache(split="train")``).
        ground_truth: COCO-style GT boxes: {image_id, class, bbox} (bbox as
            [x1, y1, x2, y2] — see ``scripts/04_evaluate.py::_coco_to_gt``).
        prototype_payload: JSON from ``02_build_prototypes.py`` — supplies the
            per-class support ids to exclude and the prototype centroid /
            ``sigma_visual`` used to build the gate inputs.
        boxes_per_class: pre-registered calibration size (20).
        seed: RNG seed for deterministic per-seed sampling.
        affinity_threshold: affinity at/above which ``visual_correct`` is True.

    Returns:
        The calibration JSON per the schema in ``uadapt.fusion.calibration``.

    Raises:
        ValueError: if no GT-matched boxes are available for any class.
    """
    protos = prototype_payload.get("prototypes", {})
    if not protos:
        raise ValueError("prototype_payload contains no prototypes")
    if boxes_per_class < 1:
        raise ValueError(f"boxes_per_class must be >= 1 (got {boxes_per_class})")

    gt_by = _index_gt(ground_truth)
    classes = list(protos)
    rng = np.random.default_rng(seed)
    samples: List[Dict] = []
    per_class_eligible: Dict[str, int] = {}

    for cls in classes:
        proto = protos[cls]
        support = set(proto.get("support_ids", []) or [])
        candidates = [r for r in records if r.class_name == cls]
        eligible = [
            r
            for r in candidates
            if r.image_id not in support and _gt_match(r, gt_by)
        ]
        per_class_eligible[cls] = len(eligible)
        rng.shuffle(eligible)
        chosen = eligible[:boxes_per_class]
        for r in chosen:
            feat = _gate_inputs(r, proto)
            samples.append(
                {
                    "class": cls,
                    # Audit fields (ignored by the gate machinery).
                    "image_id": r.image_id,
                    "bbox": r.bbox.astype(float).tolist(),
                    **feat,
                    "text_correct": True,  # sampled boxes are GT-matched ("labeled")
                    "visual_correct": bool(feat["a_visual"] >= affinity_threshold),
                }
            )

    if not samples:
        raise ValueError(
            "no eligible GT-matched boxes in the train cache for any class "
            "(check --cache-dir train split and --ground-truth train JSON)"
        )

    payload = {
        "boxes_per_class": boxes_per_class,
        "classes": classes,
        "samples": samples,
        "sampling": {
            "seed": seed,
            "source": "train split cache",
            "disjoint_from_support": True,
            "disjoint_from_test": True,
            "affinity_threshold": affinity_threshold,
            "per_class_eligible": per_class_eligible,
            "per_class_sampled": {
                cls: sum(1 for s in samples if s["class"] == cls) for cls in classes
            },
            "note": (
                "classes with fewer than boxes_per_class eligible boxes kept "
                "ALL of them (recorded in per_class_eligible)"
            ),
        },
    }
    logger.info(
        "calibration set: %d samples across %d classes (requested %d/class)",
        len(samples), len(classes), boxes_per_class,
    )
    return payload


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _gate_inputs(rec: FeatureRecord, proto: Dict) -> Dict[str, float]:
    """Normalized-ish 5-D gate inputs for one sampled box.

    Mirrors ``uadapt.fusion.calibration.record_gate_input`` so calibration
    and test rows share the same feature semantics. (Min-max normalization
    with calibration-set statistics happens downstream in
    ``build_calibration_matrices``/``min_max_stats``.)
    """
    centroid = np.asarray(proto["centroid"], dtype=np.float64)
    aff = float(visual_affinity(rec.visual_feature, centroid))
    s_text = float(class_text_score(rec))
    return {
        "s_text": s_text,
        "s_visual": aff,                    # visual-only score proxy (documented)
        "sigma2_text": float(proto.get("sigma_text", 0.5)),  # not serialized by 02
        "sigma2_visual": float(proto.get("sigma_visual", 0.0)),
        "a_visual": aff,
        "affinity": aff,                    # audit convenience
    }


def _index_gt(ground_truth: Sequence[Dict]) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    for g in ground_truth:
        out.setdefault(g["image_id"], []).append(g)
    return out


def _gt_match(rec: FeatureRecord, gt_by: Dict[str, List[Dict]]) -> bool:
    for g in gt_by.get(rec.image_id, []):
        if g["class"] == rec.class_name and _iou(rec.bbox, np.asarray(g["bbox"])) >= 0.5:
            return True
    return False
