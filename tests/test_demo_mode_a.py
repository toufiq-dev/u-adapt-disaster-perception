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
from uadapt.uncertainty.variance_estimators import (
    absolute_normalize,
    min_max_normalize,
)


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


# ---------------------------------------------------------------------------
# Absolute scaling normalization (2-class degeneracy fix, change_log 2026-08-03)
# ---------------------------------------------------------------------------
def test_absolute_normalize_maps_cosine_range_to_unit_interval():
    # Raw mean pairwise cosine distance ranges over [0, 2] (1 - cos, cos in
    # [-1, 1]); absolute scaling x/2.0 must map the full range to [0, 1].
    out = absolute_normalize(np.array([0.0, 1.0, 2.0]))
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(0.5)
    assert out[2] == pytest.approx(1.0)
    # Slight numerical over/undershoot outside [0, 2] is clipped.
    clipped = absolute_normalize(np.array([-0.5, 2.5]))
    assert clipped[0] == pytest.approx(0.0)
    assert clipped[1] == pytest.approx(1.0)


def test_absolute_normalize_class_count_invariant():
    # The normalized value of a class must NOT depend on how many classes are
    # in the set (min-max violates this: with C=2 the terms collapse to
    # {0, 1}, which is the 2-class degeneracy this function fixes).
    two = absolute_normalize(np.array([0.4, 1.6]))
    six = absolute_normalize(np.array([0.4, 1.6, 0.9, 0.2, 1.1, 1.8]))
    assert two[0] == pytest.approx(six[0])   # 0.4 -> 0.2 either way
    assert two[1] == pytest.approx(six[1])   # 1.6 -> 0.8 either way
    # For contrast: min-max is NOT class-count-invariant — the same values
    # normalize differently depending on the set they appear in.
    assert not np.allclose(
        min_max_normalize(np.array([0.4, 1.6])),
        min_max_normalize(np.array([0.4, 1.6, 0.9, 0.2, 1.1, 1.8]))[:2],
    )


def test_run_demo_absolute_fixes_2class_degeneracy():
    # On the 2-class (fire/smoke) stand-in, min-max actively hurts the gate
    # (U-ADAPT < naive averaging), while absolute scaling restores it.
    ds = generate_synthetic_dataset(classes=["fire", "smoke"], seed=0,
                                    n_test_images=40)
    mm = run_demo(ds.train_records, ds.test_records, ds.ground_truth,
                  ds.classes, template_embeddings=ds.template_embeddings,
                  shots=5, seed=0, norm_strategy="min-max")
    ab = run_demo(ds.train_records, ds.test_records, ds.ground_truth,
                  ds.classes, template_embeddings=ds.template_embeddings,
                  shots=5, seed=0, norm_strategy="absolute")
    # The fix: under absolute scaling U-ADAPT no longer underperforms naive
    # averaging, and it beats the (degenerate) min-max run.
    assert ab.map50["uadapt_mode_a"] >= ab.map50["naive_average"] - 1e-9
    assert ab.map50["uadapt_mode_a"] > mm.map50["uadapt_mode_a"]
    # The two strategies produce genuinely different gate behavior.
    w_ab = np.asarray([p["w"] for p in ab.proposal_level])
    w_mm = np.asarray([p["w"] for p in mm.proposal_level])
    assert w_ab.shape == w_mm.shape
    assert not np.allclose(w_ab, w_mm)


def test_default_norm_strategy_is_min_max(results):
    # Backward compatibility: the default stays min-max and is recorded.
    assert results.meta["norm_strategy"] == "min-max"


def test_invalid_norm_strategy_raises():
    ds = generate_synthetic_dataset(seed=0, n_test_images=10)
    with pytest.raises(ValueError, match="norm_strategy"):
        run_demo(ds.train_records, ds.test_records, ds.ground_truth,
                 ds.classes, template_embeddings=ds.template_embeddings,
                 shots=5, seed=0, norm_strategy="bogus")



def test_figure5_real_image_rendering(tmp_path):
    # A tiny real image + image_paths => Figure 5 must render real detections
    # (no exception) and report that real imagery was used.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from uadapt.demo.plotting import figure5_qualitative

    img = tmp_path / "img.png"
    plt.imsave(img, np.zeros((64, 64, 3), dtype=np.uint8))

    proposals = [
        {"image_id": "demo_0000", "class": "fire", "bbox": [10, 10, 30, 30],
         "w": 0.85, "s_text": 0.3, "s_visual": 0.8,
         "text_correct": False, "visual_correct": True},
        {"image_id": "demo_0001", "class": "smoke", "bbox": [5, 5, 20, 20],
         "w": 0.2, "s_text": 0.9, "s_visual": 0.2,
         "text_correct": True, "visual_correct": False},
        {"image_id": "demo_0002", "class": "vehicle", "bbox": [8, 8, 25, 25],
         "w": 0.1, "s_text": 0.7, "s_visual": 0.3,
         "text_correct": True, "visual_correct": False},
    ]
    ground_truth = [
        {"image_id": "demo_0000", "class": "fire", "bbox": [10, 10, 30, 30]},
        {"image_id": "demo_0001", "class": "smoke", "bbox": [5, 5, 20, 20]},
    ]
    image_paths = {k: str(img) for k in ("demo_0000", "demo_0001", "demo_0002")}

    fig, axes = plt.subplots(1, 3)
    try:
        note = figure5_qualitative(proposals, ground_truth, axes, seed=0,
                                   image_paths=image_paths)
        assert "real detections" in note
        assert "schematic" not in note
    finally:
        plt.close(fig)


def test_figure5_schematic_fallback_without_paths(results, dataset):
    # Without image_paths (synthetic mode) Figure 5 must fall back to the
    # schematic scene and still render without error.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from uadapt.demo.plotting import figure5_qualitative

    fig, axes = plt.subplots(1, 3)
    try:
        note = figure5_qualitative(results.proposal_level, dataset.ground_truth,
                                   axes, seed=0, image_paths=None)
        assert "schematic" in note
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# REAL-cache mode: per-proposal uncertainty estimators give D1/D2/D3 signal
# (change_log.md 2026-08-05 — replaces the 0.5 placeholder path)
# ---------------------------------------------------------------------------
def _real_mode_world(seed: int = 7):
    """Build a real-like world (FeatureRecords WITHOUT template embeddings).

    Four groups of proposals engineer the exact regime the per-proposal
    estimators must capture (sizes 40/40/20/20 so D1/D2 keep power despite
    the disagreeing groups):

      * text confident  -> sharp class similarities (low entropy)
      * text unreliable -> flat similarities (high entropy)
      * visual reliable -> box AT its class support (near feature, low
                           box-to-support distance)
      * visual weak     -> box OPPOSITE the class centroid (affinity ~ 0)

    Group layout (text_ok = gt_correct, visual_ok = affinity >= 0.65):
      0. text+visual ok      sharp, near,  GT present   (40)
      1. text+visual wrong   flat,  far,   no GT        (40)
      2. text ok, vis wrong  sharp, far,   GT present   (20)  -> text better
      3. text wrong, vis ok  flat,  near,  no GT        (20)  -> visual better

    With D1/D2 evaluated against the PRE-REGISTERED proposal correctness
    (gt_correct): low-entropy text and low box-to-support distance both
    predict correctness (D1 rho ~ 1, D2 rho ~ 0.33), while the disagreeing
    groups keep D3's subsets non-empty.
    """
    from uadapt.features.cache_engine import FeatureRecord

    rng = np.random.default_rng(seed)
    classes = ["person", "fire", "smoke"]
    D = 8
    dirs = np.zeros((3, D))
    dirs[0, :3] = 1.0
    dirs[1, 3:6] = 1.0
    dirs[2, 6:] = 1.0
    centroid = {c: d / np.linalg.norm(d) for c, d in zip(classes, dirs)}

    def near(c):
        f = centroid[c] + 0.01 * rng.normal(size=D)
        return f / np.linalg.norm(f)

    def far(c):
        # Opposite the class centroid: affinity ~ 0 (< 0.65 -> visual weak),
        # box-to-support distance near its max.
        return -centroid[c]

    def sharp(c):
        s = np.full(3, 0.07)
        s[classes.index(c)] = 0.85
        return s

    def flat():
        return np.full(3, 0.34)

    # Support pool (train) — 6 near records per class -> the prototype.
    train = []
    for c in classes:
        for j in range(6):
            train.append(
                FeatureRecord(
                    image_id=f"train_{c}_{j}", class_name=c, score=0.6,
                    bbox=np.array([0, 0, 10, 10], dtype=np.float32),
                    visual_feature=near(c).astype(np.float32),
                    text_similarities=sharp(c).astype(np.float32),
                    classes=classes,
                )
            )

    test, gt = [], []
    # group sizes: 0 -> 40, 1 -> 40, 2 -> 20, 3 -> 20
    for group, n_grp in ((0, 40), (1, 40), (2, 20), (3, 20)):
        for _ in range(n_grp):
            idx = len(test)
            c = classes[idx % 3]
            box = np.array([idx, idx, idx + 10, idx + 10], dtype=np.float32)
            if group in (0, 2):
                gt.append({"image_id": f"img_{idx}", "class": c, "bbox": box.tolist()})
            if group == 0:      # text+visual ok
                sims, feat = sharp(c), near(c)
            elif group == 1:    # both wrong
                sims, feat = flat(), far(c)
            elif group == 2:    # text ok, visual weak
                sims, feat = sharp(c), far(c)
            else:               # text wrong, visual ok
                sims, feat = flat(), near(c)
            test.append(
                FeatureRecord(
                    image_id=f"img_{idx}", class_name=c, score=0.6, bbox=box,
                    visual_feature=feat.astype(np.float32),
                    text_similarities=sims.astype(np.float32),
                    classes=classes,
                )
            )
    return train, test, gt, classes


def test_real_mode_per_proposal_variance_gives_diagnostics_signal():
    train, test, gt, classes = _real_mode_world()
    res = run_demo(
        train_records=train, test_records=test, ground_truth=gt,
        classes=classes, template_embeddings=None, shots=5, seed=0,
        norm_strategy="min-max",
    )
    rows = res.proposal_level
    assert len(rows) == 120

    tv = np.asarray([r["norm_text_var"] for r in rows], dtype=float)
    vv = np.asarray([r["norm_visual_var"] for r in rows], dtype=float)
    tc = np.asarray([r["text_correct"] for r in rows], dtype=bool)
    vc = np.asarray([r["visual_correct"] for r in rows], dtype=bool)

    # Per-proposal, continuous (NOT the class-constant 0.5 placeholder).
    assert np.ptp(tv) > 0.2
    assert np.ptp(vv) > 0.2
    assert np.all((tv >= 0.0) & (tv <= 1.0))
    assert np.all((vv >= 0.0) & (vv <= 1.0))

    # text_correct is NON-tautological: real misses exist and correlate with
    # entropy (high-entropy flat-sim proposals are the wrong ones).
    assert 0.4 < tc.mean() < 0.7
    assert vc.mean() > 0.0

    d1 = res.diagnostics["D1_text_uncertainty_accuracy"]["summary"]
    d2 = res.diagnostics["D2_visual_uncertainty_accuracy"]["summary"]
    d3 = res.diagnostics["D3_gate_favorability"]["summary"]
    # D1/D2 use the pre-registered proposal correctness (gt_correct).
    assert d1["spearman_rho"] > 0.3
    assert d2["spearman_rho"] > 0.2
    # Both disagreeing subsets are non-empty and the gate favors the better
    # modality (text-better -> w < 0.5, visual-better -> w > 0.5).
    assert d3["n"] == pytest.approx(40.0)
    assert d3["favorability_fraction"] > 0.6

    # meta records WHICH estimator produced the terms (auditability).
    assert res.meta["text_uncertainty_estimator"] == "class_similarity_entropy"
    assert res.meta["visual_uncertainty_estimator"] == "box_to_support_distance"


def test_real_mode_meta_estimators_on_synthetic_path(results):
    # The synthetic path keeps the template-ensemble / support-dispersion
    # estimators (backward compatible).
    assert results.meta["text_uncertainty_estimator"] == "template_ensemble_variance"
    assert results.meta["visual_uncertainty_estimator"] == "support_dispersion"


# ---------------------------------------------------------------------------
# Beta-regression fallback gate (pre-registered D5 contingency) — wiring
# ---------------------------------------------------------------------------
def test_run_demo_beta_fallback_gate(dataset):
    """run_demo(gate_type="beta_fallback") runs end-to-end and records the
    gate type in meta (auditability)."""
    results = run_demo(
        train_records=dataset.train_records,
        test_records=dataset.test_records,
        ground_truth=dataset.ground_truth,
        classes=dataset.classes,
        template_embeddings=dataset.template_embeddings,
        shots=5,
        seed=0,
        gate_type="beta_fallback",
    )
    for key in ("zero_shot_raw", "text_only", "visual_only",
                "naive_average", "uadapt_mode_a"):
        assert key in results.map50
        assert 0.0 <= results.map50[key] <= 1.0
    # The Beta gate must produce valid weights and a fused score in [0, 1].
    gs = results.gate_stats
    assert 0.0 <= gs["mean_w"] <= 1.0
    ws = np.asarray([r["w"] for r in results.proposal_level])
    fused = np.asarray([r["fused"] for r in results.proposal_level])
    assert np.all((ws >= 0.0) & (ws <= 1.0))
    assert np.all((fused >= 0.0) & (fused <= 1.0))
    assert results.meta["gate_type"] == "beta_fallback"


def test_run_demo_beta_fallback_differs_from_analytic(dataset):
    """The Beta fallback must produce genuinely different gate weights from
    the analytic gate (it is a real alternative, not a no-op)."""
    analytic = run_demo(
        train_records=dataset.train_records,
        test_records=dataset.test_records,
        ground_truth=dataset.ground_truth,
        classes=dataset.classes,
        template_embeddings=dataset.template_embeddings,
        shots=5,
        seed=0,
    )
    beta = run_demo(
        train_records=dataset.train_records,
        test_records=dataset.test_records,
        ground_truth=dataset.ground_truth,
        classes=dataset.classes,
        template_embeddings=dataset.template_embeddings,
        shots=5,
        seed=0,
        gate_type="beta_fallback",
    )
    w_a = np.asarray([r["w"] for r in analytic.proposal_level])
    w_b = np.asarray([r["w"] for r in beta.proposal_level])
    assert w_a.shape == w_b.shape
    assert not np.allclose(w_a, w_b)


def test_run_demo_default_gate_type_is_analytic(results):
    # Backward compatibility: the default gate is the pre-registered analytic
    # one and is recorded in meta.
    assert results.meta["gate_type"] == "analytic"


def test_run_demo_rejects_unknown_gate_type(dataset):
    with pytest.raises(ValueError, match="gate_type"):
        run_demo(
            train_records=dataset.train_records,
            test_records=dataset.test_records,
            ground_truth=dataset.ground_truth,
            classes=dataset.classes,
            template_embeddings=dataset.template_embeddings,
            shots=5,
            seed=0,
            gate_type="bogus",
        )
