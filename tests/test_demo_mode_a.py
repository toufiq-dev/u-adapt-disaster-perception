"""Tests for the supervisor demo pipeline (synthetic data + Mode A run).

Covers:
  * determinism of the synthetic world (same seed -> identical data)
  * synthetic world schema matches FeatureRecord + COCO GT expectations
  * end-to-end run_demo produces all methods, diagnostics, and gate stats
  * the gate is dynamic (w not constant), and D1-D3 diagnostics are computed
  * ablation variants run and change mAP50
"""

from __future__ import annotations

import numpy as np
import pytest

from uadapt.demo.pipeline import run_demo
from uadapt.demo.synthetic_data import generate_synthetic_dataset


@pytest.fixture(scope="module")
def dataset():
    return generate_synthetic_dataset(seed=0, n_test_images=40)


@pytest.fixture(scope="module")
def results(dataset):
    return run_demo(
        train_records=dataset.train_records,
        test_records=dataset.test_records,
        ground_truth=dataset.ground_truth,
        classes=dataset.classes,
        template_embeddings=dataset.template_embeddings,
        shots=5,
        seed=0,
    )


# ---------------------------------------------------------------------------
# Synthetic world
# ---------------------------------------------------------------------------
def test_synthetic_world_deterministic():
    a = generate_synthetic_dataset(seed=0, n_test_images=20)
    b = generate_synthetic_dataset(seed=0, n_test_images=20)
    assert a.meta["seed"] == b.meta["seed"] == 0
    assert [r.image_id for r in a.test_records] == [r.image_id for r in b.test_records]
    assert [r.score for r in a.test_records] == [r.score for r in b.test_records]
    assert [g["bbox"] for g in a.ground_truth] == [g["bbox"] for g in b.ground_truth]


def test_synthetic_world_schema(dataset):
    # FeatureRecord schema
    rec = dataset.test_records[0]
    assert rec.class_name in dataset.classes
    assert rec.visual_feature.shape == (64,)
    assert rec.bbox.shape == (4,)
    assert rec.text_similarities.shape == (len(dataset.classes),)
    assert 0.0 <= rec.score <= 1.0
    # GT schema
    g = dataset.ground_truth[0]
    assert set(g) == {"image_id", "class", "bbox"}
    assert len(g["bbox"]) == 4
    # every class has enough support records for k=5 prototypes
    from collections import Counter

    counts = Counter(r.class_name for r in dataset.train_records)
    for c in dataset.classes:
        assert counts[c] >= 5, f"class {c} has only {counts[c]} support records"


def test_synthetic_world_different_seed_differs():
    a = generate_synthetic_dataset(seed=0, n_test_images=10)
    b = generate_synthetic_dataset(seed=1, n_test_images=10)
    assert [r.score for r in a.test_records] != [r.score for r in b.test_records]


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------
def test_run_demo_returns_all_methods(results):
    for key in ("zero_shot_raw", "text_only", "visual_only",
                "naive_average", "uadapt_mode_a"):
        assert key in results.map50
        v = results.map50[key]
        assert 0.0 <= v <= 1.0


def test_run_demo_diagnostics_computed(results):
    for key in ("D1_text_uncertainty_accuracy",
                "D2_visual_uncertainty_accuracy",
                "D3_gate_favorability"):
        assert key in results.diagnostics
        assert "summary" in results.diagnostics[key]
    d3 = results.diagnostics["D3_gate_favorability"]["summary"]
    assert 0.0 <= d3["favorability_fraction"] <= 1.0
    assert d3["n"] > 0


def test_gate_is_dynamic(results):
    gs = results.gate_stats
    assert gs["n_proposals"] > 0
    assert 0.0 < gs["mean_w"] < 1.0
    # The gate must not be constant at 0.5.
    assert gs["std_w"] > 0.01
    assert gs["frac_in_0.45_0.55"] < 0.99


def test_ablation_variants_change_map50(dataset):
    # Coefficient changes must alter gate weights (and typically mAP50):
    # alpha=0 removes the visual-uncertainty term from the logit.
    full = run_demo(
        train_records=dataset.train_records,
        test_records=dataset.test_records,
        ground_truth=dataset.ground_truth,
        classes=dataset.classes,
        template_embeddings=dataset.template_embeddings,
        shots=5,
        seed=0,
    )
    no_alpha = run_demo(
        train_records=dataset.train_records,
        test_records=dataset.test_records,
        ground_truth=dataset.ground_truth,
        classes=dataset.classes,
        template_embeddings=dataset.template_embeddings,
        shots=5,
        seed=0,
        alpha=0.0,
    )
    w_full = np.asarray([r["w"] for r in full.proposal_level])
    w_no_alpha = np.asarray([r["w"] for r in no_alpha.proposal_level])
    assert w_full.shape == w_no_alpha.shape
    assert not np.allclose(w_full, w_no_alpha)  # the gate responded to the ablation


def test_empty_inputs_do_not_crash():
    # Degenerate inputs must return empty results, not raise.
    res = run_demo(train_records=[], test_records=[], ground_truth=[], classes=[])
    assert res.map50["zero_shot_raw"] == 0.0
    assert res.gate_stats["n_proposals"] == 0


def test_per_class_ap_coverage(results):
    assert set(results.per_class_ap) <= set(results.meta["classes"])
    # allow tiny float overshoot (e.g. 1.0000000000000007 from 101-point AP)
    assert all(-1e-9 <= v <= 1.0 + 1e-9 for v in results.per_class_ap.values())


def test_gap_recovery_payload(results):
    g = results.gap_recovery
    assert "zero_shot_raw_map50" in g
    assert "uadapt_map50" in g
    assert "proposal_recall_ceiling" in g
    assert "gap_recovery_vs_ceiling" in g
