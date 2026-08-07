"""Unit tests for the Mode B calibration-set sampler (proposal §5.4.2).

Covers:
  * schema + per-class stratified counts (up to boxes_per_class)
  * strict disjointness from the seed's support examples
  * determinism (same seed -> identical set; seeded sampling)
  * shortfall handling (fewer eligible boxes than requested -> keep all +
    record the true count)
  * correctness flags (text_correct True for GT-matched boxes;
    visual_correct = affinity >= threshold)
  * no eligible boxes -> ValueError
"""

import numpy as np
import pytest

from uadapt.features.cache_engine import FeatureRecord
from uadapt.fusion.calibration_set import (
    VISUAL_CORRECT_AFFINITY,
    build_calibration_set,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _make_record(image_id, class_name, feature, sims=0.9):
    return FeatureRecord(
        image_id=image_id,
        class_name=class_name,
        score=float(np.random.default_rng(0).uniform(0.2, 0.8)),
        bbox=np.asarray([0, 0, 10, 10], dtype=np.float32),
        visual_feature=np.asarray(feature, dtype=np.float32),
        text_similarities=np.asarray([sims, sims * 0.5], dtype=np.float32),
        classes=["fire", "smoke"],
    )


def _make_gt(image_ids):
    """GT boxes that perfectly overlap the record boxes above."""
    return [
        {"image_id": iid, "class": "fire", "bbox": [0, 0, 10, 10]}
        for iid in image_ids
    ]


def _unit(seed=None):
    """A unit-norm feature on the +x axis (affinity 1.0 to the centroid)."""
    rng = np.random.default_rng(seed)
    f = np.zeros(8)
    f[0] = 1.0
    return f + rng.normal(0, 0.01, size=8)


def _prototype_payload(centroid, support_ids=("img_support",)):
    return {
        "shots": 5,
        "prototypes": {
            "fire": {
                "centroid": centroid.tolist(),
                "sigma_visual": 0.3,
                "n_kept": 5,
                "support_ids": list(support_ids),
            }
        },
    }


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_schema_and_counts():
    records = [_make_record(f"img{i}", "fire", _unit(i)) for i in range(25)]
    gt = _make_gt([f"img{i}" for i in range(25)])
    payload = build_calibration_set(
        records, gt, _prototype_payload(_unit()), boxes_per_class=20, seed=0
    )
    assert payload["boxes_per_class"] == 20
    assert payload["classes"] == ["fire"]
    assert len(payload["samples"]) == 20  # capped at boxes_per_class
    assert payload["sampling"]["per_class_sampled"]["fire"] == 20
    assert payload["sampling"]["per_class_eligible"]["fire"] == 25
    for s in payload["samples"]:
        assert s["class"] == "fire"
        for k in ("s_text", "s_visual", "sigma2_text", "sigma2_visual", "a_visual"):
            assert k in s
            assert 0.0 <= s[k] <= 1.0
        assert set(("text_correct", "visual_correct")) <= set(s)


def test_disjoint_from_support():
    records = [_make_record(f"img{i}", "fire", _unit(i)) for i in range(20)]
    gt = _make_gt([f"img{i}" for i in range(20)])
    support = {f"img{i}" for i in range(5)}  # e.g. the k=5 support of seed 0
    payload = build_calibration_set(
        records, gt, _prototype_payload(_unit(), support_ids=sorted(support)),
        boxes_per_class=20, seed=0,
    )
    sampled = {s["image_id"] for s in payload["samples"]}
    assert not (sampled & support)
    # 20 eligible minus 5 support = 15 available -> all kept.
    assert len(sampled) == 15


def test_determinism_same_seed():
    records = [_make_record(f"img{i}", "fire", _unit(i)) for i in range(30)]
    gt = _make_gt([f"img{i}" for i in range(30)])
    a = build_calibration_set(records, gt, _prototype_payload(_unit()), seed=7)
    b = build_calibration_set(records, gt, _prototype_payload(_unit()), seed=7)
    assert [s["image_id"] for s in a["samples"]] == \
           [s["image_id"] for s in b["samples"]]


def test_different_seed_differs():
    records = [_make_record(f"img{i}", "fire", _unit(i)) for i in range(30)]
    gt = _make_gt([f"img{i}" for i in range(30)])
    a = build_calibration_set(records, gt, _prototype_payload(_unit()), seed=1)
    b = build_calibration_set(records, gt, _prototype_payload(_unit()), seed=2)
    assert {s["image_id"] for s in a["samples"]} != \
           {s["image_id"] for s in b["samples"]}


def test_stratified_across_classes():
    """Two classes each get up to boxes_per_class (independent per class)."""
    records = (
        [_make_record(f"a{i}", "fire", _unit(i)) for i in range(10)]
        + [_make_record(f"b{i}", "smoke", _unit(i)) for i in range(10)]
    )
    gt = (
        [{"image_id": f"a{i}", "class": "fire", "bbox": [0, 0, 10, 10]}
         for i in range(10)]
        + [{"image_id": f"b{i}", "class": "smoke", "bbox": [0, 0, 10, 10]}
           for i in range(10)]
    )
    protos = {
        "fire": {"centroid": _unit(1).tolist(), "sigma_visual": 0.3,
                 "support_ids": ["a0"], "n_kept": 1},
        "smoke": {"centroid": _unit(2).tolist(), "sigma_visual": 0.3,
                  "support_ids": ["b0"], "n_kept": 1},
    }
    payload = build_calibration_set(records, gt, {"prototypes": protos},
                                    boxes_per_class=4, seed=0)
    per = payload["sampling"]["per_class_sampled"]
    assert per["fire"] == 4
    assert per["smoke"] == 4
    assert {s["class"] for s in payload["samples"]} == {"fire", "smoke"}


def test_shortfall_keeps_all_and_records_count():
    records = [_make_record(f"img{i}", "fire", _unit(i)) for i in range(3)]
    gt = _make_gt([f"img{i}" for i in range(3)])
    payload = build_calibration_set(
        records, gt, _prototype_payload(_unit()), boxes_per_class=20, seed=0
    )
    assert len(payload["samples"]) == 3  # only 3 eligible
    assert payload["sampling"]["per_class_eligible"]["fire"] == 3
    assert payload["sampling"]["per_class_sampled"]["fire"] == 3


def test_correctness_flags():
    # Near-identical feature to the centroid -> affinity ~ 1.0 >= 0.65.
    records = [_make_record("img0", "fire", _unit(0))]
    gt = _make_gt(["img0"])
    payload = build_calibration_set(
        records, gt, _prototype_payload(_unit(0)), boxes_per_class=1, seed=0
    )
    s = payload["samples"][0]
    assert s["text_correct"] is True          # sampled boxes are GT-matched
    assert s["visual_correct"] is True        # affinity ~ 1.0
    assert s["a_visual"] >= VISUAL_CORRECT_AFFINITY
    # s_visual is the affinity proxy, mirroring record_gate_input.
    assert s["s_visual"] == pytest.approx(s["a_visual"])


def test_no_eligible_raises():
    # Records exist but none matches GT (empty GT list).
    records = [_make_record("img0", "fire", _unit(0))]
    payload_proto = _prototype_payload(_unit(0))
    with pytest.raises(ValueError, match="no eligible"):
        build_calibration_set(records, [], payload_proto, boxes_per_class=1)


def test_ignores_non_matching_class_records():
    """Records whose class has no GT match (or no prototype) are skipped."""
    records = [_make_record("img0", "fire", _unit(0))]
    protos = {"fire": {"centroid": _unit(0).tolist(), "sigma_visual": 0.3,
                       "support_ids": [], "n_kept": 1},
              "smoke": {"centroid": _unit(1).tolist(), "sigma_visual": 0.3,
                        "support_ids": [], "n_kept": 1}}
    payload = build_calibration_set(
        records, _make_gt(["img0"]), {"prototypes": protos}, boxes_per_class=1
    )
    assert {s["class"] for s in payload["samples"]} == {"fire"}
