"""Alignment tests for scripts/04_evaluate.py's D3 disagreeing-subset convention.

The pre-registered D3 definition (docs/pre_registration.md §7.6) is the
fraction of *disagreeing cases* — where the text and visual modalities
disagree on correctness — in which the gate assigns higher weight to the more
accurate modality. ``scripts/04_evaluate.py`` previously split gate weights by
``w < 0.5`` / ``w > 0.5``, which counts every proposal as favorable by
construction (any weight is on one side of 0.5) and therefore measures
nothing. These tests pin the aligned behavior: D3 subsets come from the
per-proposal ``text_ok`` / ``visual_ok`` flags (pipeline /
compute_pooled_diagnostics.py convention, change_log 2026-08-05).
"""

import importlib.util
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "04_evaluate.py"


def _load_04():
    """Import scripts/04_evaluate.py as a module (no CLI side effects)."""
    spec = importlib.util.spec_from_file_location("evaluate_04_script", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _make_world():
    """5 proposals where the disagreeing subsets differ from w<0.5/w>0.5.

    Affinity threshold 0.65 (VISUAL_CORRECT_AFFINITY) -> visual_ok. GT covers
    images a/b/c (correct); images d/e have no GT (incorrect).

      A: correct, visual_ok    (agree)      w=0.7  -> excluded by D3
      B: correct, visual_ok    (agree)      w=0.2  -> excluded by D3
      C: correct, text_better  (disagree)   w=0.7  -> gate points WRONG
      D: wrong,   visual_better (disagree)  w=0.2  -> gate points WRONG
      E: wrong,   not visual   (agree)      w=0.9  -> excluded by D3

    Disagreeing n=2, favorability 0.0. The old w<0.5/w>0.5 split would have
    reported n=5 with favorability 1.0.
    """
    gt = [
        {"image_id": "a", "class": "x", "bbox": [0, 0, 10, 10]},
        {"image_id": "b", "class": "x", "bbox": [0, 0, 10, 10]},
        {"image_id": "c", "class": "x", "bbox": [0, 0, 10, 10]},
    ]
    preds = [
        {"image_id": "a", "class": "x", "bbox": [0, 0, 9, 9], "score": 0.9,
         "affinity": 0.9, "gate_weight": 0.7},
        {"image_id": "b", "class": "x", "bbox": [0, 0, 9, 9], "score": 0.8,
         "affinity": 0.9, "gate_weight": 0.2},
        {"image_id": "c", "class": "x", "bbox": [0, 0, 9, 9], "score": 0.7,
         "affinity": 0.2, "gate_weight": 0.7},
        {"image_id": "d", "class": "x", "bbox": [0, 0, 9, 9], "score": 0.6,
         "affinity": 0.9, "gate_weight": 0.2},
        {"image_id": "e", "class": "x", "bbox": [0, 0, 9, 9], "score": 0.5,
         "affinity": 0.2, "gate_weight": 0.9},
    ]
    return preds, gt


def test_04_d3_uses_disagreeing_subsets():
    mod = _load_04()
    preds, gt = _make_world()
    correct = mod._proposal_correct(preds, gt)
    assert correct.tolist() == [True, True, True, False, False]

    out = mod._per_dataset_dict(preds, correct)
    d3 = out["D3_gate_favorability"]["summary"]
    # Only C (text_better) and D (visual_better) are disagreeing and both are
    # gated the WRONG way -> favorability 0.0 on n=2.
    assert d3["n"] == pytest.approx(2.0)
    assert d3["favorability_fraction"] == pytest.approx(0.0)


def test_04_d3_explicit_flags_win():
    """Explicit per-proposal flags are honored over derived ones."""
    mod = _load_04()
    preds, gt = _make_world()
    # Explicit flags intentionally DISAGREE with the derived labels
    # (correct/affinity) to prove precedence: C is text_better, D is
    # visual_better; A/B agree (both ok), E agrees (both wrong).
    for p in preds:
        p["text_correct"] = p["image_id"] in ("a", "b", "c")
        p["visual_correct"] = p["image_id"] in ("a", "b", "d")
    correct = mod._proposal_correct(preds, gt)
    out = mod._per_dataset_dict(preds, correct)
    d3 = out["D3_gate_favorability"]["summary"]
    # Disagreeing with the explicit flags: C (text_better, w=0.7) and D
    # (visual_better, w=0.2) -> n=2, both gated the WRONG way -> 0.0.
    assert d3["n"] == pytest.approx(2.0)
    assert d3["favorability_fraction"] == pytest.approx(0.0)


def test_04_d3_pooled_uses_disagreeing_subsets():
    mod = _load_04()
    preds, gt = _make_world()
    correct = mod._proposal_correct(preds, gt)

    # Second dataset mirrors the first but gates the two disagreeing
    # proposals the RIGHT way (C w=0.2, D w=0.9) -> favorable on both.
    preds2, gt2 = _make_world()
    preds2[2]["gate_weight"] = 0.2
    preds2[3]["gate_weight"] = 0.9
    correct2 = mod._proposal_correct(preds2, gt2)

    out = mod._run_diagnostics(preds, correct, pool_with=(preds2, correct2))
    primary = out["primary"]["D3_gate_favorability"]["summary"]
    secondary = out["secondary"]["D3_gate_favorability"]["summary"]
    pooled = out["pooled"]["D3_gate_favorability"]["summary"]
    assert primary["n"] == pytest.approx(2.0)
    assert primary["favorability_fraction"] == pytest.approx(0.0)
    assert secondary["n"] == pytest.approx(2.0)
    assert secondary["favorability_fraction"] == pytest.approx(1.0)
    # Pooled: 2 favorable of 4 disagreeing cases -> 0.5.
    assert pooled["n"] == pytest.approx(4.0)
    assert pooled["favorability_fraction"] == pytest.approx(0.5)


def test_04_d4_uses_true_gamma0_counterfactual():
    """D4 must use the per-proposal w_gamma_0 counterfactual (gamma=0).

    The pre-fix behavior passed a constant 0.5 array, so D4 measured
    w - 0.5 — a monotone transform of w that can only rank-correlate with
    affinity as strongly as w itself does. The true counterfactual
    Delta w = w - w_gamma_0 is NOT monotone in w, so its correlation with
    affinity differs (here: 0.5 vs the old 1.0).

      A: affinity 0.9, w 0.9, w_gamma_0 0.7 -> delta +0.2
      B: affinity 0.5, w 0.5, w_gamma_0 0.2 -> delta +0.3
      C: affinity 0.1, w 0.3, w_gamma_0 0.6 -> delta -0.3
    """
    import numpy as np

    from uadapt.metrics.diagnostics import _spearman_rho

    mod = _load_04()
    preds = [
        {"image_id": "a", "class": "x", "bbox": [0, 0, 10, 10], "score": 0.9,
         "affinity": 0.9, "gate_weight": 0.9, "w_gamma_0": 0.7},
        {"image_id": "b", "class": "x", "bbox": [0, 0, 10, 10], "score": 0.8,
         "affinity": 0.5, "gate_weight": 0.5, "w_gamma_0": 0.2},
        {"image_id": "c", "class": "x", "bbox": [0, 0, 10, 10], "score": 0.7,
         "affinity": 0.1, "gate_weight": 0.3, "w_gamma_0": 0.6},
    ]
    gt = [
        {"image_id": "a", "class": "x", "bbox": [0, 0, 10, 10]},
        {"image_id": "b", "class": "x", "bbox": [0, 0, 10, 10]},
        {"image_id": "c", "class": "x", "bbox": [0, 0, 10, 10]},
    ]
    correct = mod._proposal_correct(preds, gt)

    out = mod._per_dataset_dict(preds, correct)
    rho = out["D4_affinity_diagnostic"]["summary"]["affinity_delta_spearman"]

    aff = np.asarray([0.9, 0.5, 0.1])
    w = np.asarray([0.9, 0.5, 0.3])
    w_gamma_0 = np.asarray([0.7, 0.2, 0.6])
    true_delta = w - w_gamma_0
    # The true counterfactual correlation (computed with the same
    # tie-averaged rank method D4 uses).
    expected = _spearman_rho(aff, true_delta)
    assert rho == pytest.approx(expected, abs=1e-9)
    # The counterfactual is genuinely used: the old constant-0.5 array would
    # have yielded a different (here: trivially perfect) correlation.
    old_delta = w - np.full_like(w, 0.5)
    old_rho = _spearman_rho(aff, old_delta)
    assert rho != pytest.approx(old_rho, abs=1e-9)
    assert abs(rho - 0.5) < 1e-9 and abs(old_rho - 1.0) < 1e-9


def test_04_coco_gt_remaps_image_ids_to_stems():
    """COCO GT with sequential int image ids (mask->box D-Fire style) must be
    remapped to the image file stems that cache predictions use; without the
    remap every proposal-vs-GT match fails and scripted-path mAP50 is 0.0
    (change_log 2026-08-07). LADD-style stem ids pass through unchanged."""
    mod = _load_04()
    coco = {
        "categories": [{"id": 1, "name": "smoke"}, {"id": 2, "name": "fire"}],
        "images": [
            {"id": 44, "file_name": "AoF07774.jpg"},
            {"id": 65, "file_name": "AoF07731.jpg"},
            {"id": 1000, "file_name": "1000"},  # LADD style: stem == id
        ],
        "annotations": [
            {"image_id": 44, "category_id": 2, "bbox": [0, 0, 10, 10]},
            {"image_id": 65, "category_id": 1, "bbox": [0, 0, 20, 20]},
            {"image_id": 1000, "category_id": 1, "bbox": [0, 0, 30, 30]},
        ],
    }
    gts = mod._coco_to_gt(coco)
    ids = [g["image_id"] for g in gts]
    assert ids == ["AoF07774", "AoF07731", "1000"]
    # bbox is converted xywh -> xyxy.
    assert gts[0]["bbox"] == [0, 0, 10, 10]
    assert gts[1]["class"] == "smoke" and gts[2]["class"] == "smoke"


def test_04_d4_missing_w_gamma_0_falls_back_to_constant():
    """Stale predictions files without w_gamma_0 keep the old constant-0.5
    counterfactual (backward compatible; 03_run_fusion.py has emitted the
    field since 2026-08-07)."""
    mod = _load_04()
    preds = [
        {"image_id": "a", "class": "x", "bbox": [0, 0, 10, 10], "score": 0.9,
         "affinity": 0.9, "gate_weight": 0.9},
        {"image_id": "b", "class": "x", "bbox": [0, 0, 10, 10], "score": 0.8,
         "affinity": 0.5, "gate_weight": 0.5},
    ]
    gt = [
        {"image_id": "a", "class": "x", "bbox": [0, 0, 10, 10]},
        {"image_id": "b", "class": "x", "bbox": [0, 0, 10, 10]},
    ]
    correct = mod._proposal_correct(preds, gt)
    arrays = mod._diag_arrays(preds, correct)
    assert arrays["w_gamma_0"].tolist() == pytest.approx([0.5, 0.5])
    out = mod._per_dataset_dict(preds, correct)
    assert "D4_affinity_diagnostic" in out
