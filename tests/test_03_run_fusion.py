"""Unit tests for scripts/03_run_fusion.py's Mode A fused-score output.

Regression tests for the review finding that ``_run_mode_a`` emitted the raw
cached detector ``score`` instead of the fused score
``S_final = (1 - w) * S_text + w * S_visual``. These tests pin the corrected
behavior: the emitted ``score`` IS the fused score and differs from the raw
detector score whenever the gate weight ``w != 0.5``.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "03_run_fusion.py"


def _load_03():
    """Import scripts/03_run_fusion.py as a module (no CLI side effects)."""
    spec = importlib.util.spec_from_file_location("run_fusion_03_script", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _make_record(score=0.2, sims=(0.9,), class_name="fire", classes=("fire",),
                 feature=(1.0, 0.0, 0.0)):
    from uadapt.features.cache_engine import FeatureRecord

    return FeatureRecord(
        image_id="img0",
        class_name=class_name,
        score=score,
        bbox=np.asarray([0, 0, 10, 10], dtype=np.float32),
        visual_feature=np.asarray(feature, dtype=np.float32),
        text_similarities=np.asarray(sims, dtype=np.float32),
        classes=list(classes),
    )


def _prototype_payload(centroid=(1.0, 0.0, 0.0), sigma_visual=0.0):
    return {
        "shots": 5,
        "prototypes": {
            "fire": {
                "centroid": list(centroid),
                "sigma_visual": sigma_visual,
                "n_kept": 5,
                "support_ids": ["s0"],
            }
        },
    }


def test_run_mode_a_emits_fused_score_not_raw_score():
    """With w != 0.5 the emitted score must be the fused score, not rec.score.

    Setup: box feature == prototype centroid -> affinity = 1.0; single class
    -> text variance 0.0; sigma_visual 0.0 -> visual variance 0.0; therefore
    w = sigma(1.0) ~= 0.731 != 0.5.
    """
    mod = _load_03()
    from uadapt.fusion.mode_a_analytic import ModeAGate

    rec = _make_record(score=0.2, sims=(0.9,))
    out = mod._run_mode_a([rec], _prototype_payload(), ModeAGate())
    assert len(out) == 1
    row = out[0]

    w = row["gate_weight"]
    assert w != pytest.approx(0.5)

    # S_text = raw similarity of the predicted class; S_visual = affinity.
    expected = (1.0 - w) * 0.9 + w * 1.0
    assert row["score"] == pytest.approx(expected)
    # The fused score must differ from the raw cached detector score.
    assert row["score"] != pytest.approx(rec.score)
    assert 0.0 <= row["score"] <= 1.0


def test_run_mode_a_fused_score_matches_fuse_scores():
    """The emitted score equals fuse_scores(s_text, s_visual, w) in general.

    Two classes so the text-variance term (entropy) is non-zero and the
    variance terms differ from the affinity term — the general formula path.
    """
    mod = _load_03()
    from uadapt.fusion.mode_a_analytic import ModeAGate, fuse_scores

    rec = _make_record(score=0.5, sims=(0.6, 0.3), class_name="fire",
                       classes=("fire", "smoke"))
    out = mod._run_mode_a([rec], _prototype_payload(sigma_visual=0.4), ModeAGate())
    assert len(out) == 1
    row = out[0]

    expected = fuse_scores(row["s_text"], row["s_visual"], row["gate_weight"])
    assert row["score"] == pytest.approx(expected)
    # S_text is the raw similarity of the predicted class.
    assert row["s_text"] == pytest.approx(0.6)
    # S_visual is the affinity proxy.
    assert row["s_visual"] == pytest.approx(row["affinity"])
    # All downstream-required keys are present (04_evaluate.py compatibility).
    assert set(["image_id", "class", "score", "bbox", "gate_weight",
                "affinity", "s_text", "s_visual",
                "norm_text_var", "norm_visual_var"]) <= set(row)


def test_run_mode_a_w_equals_half_still_uses_formula():
    """w == 0.5 must produce S_final = 0.5*S_text + 0.5*S_visual.

    Setup: text variance 0, visual variance 0, affinity 0 -> w = sigma(0) = 0.5.
    """
    mod = _load_03()
    from uadapt.fusion.mode_a_analytic import ModeAGate

    rec = _make_record(score=0.9, sims=(0.9,), feature=(1.0, 0.0, 0.0))
    payload = _prototype_payload(centroid=(-1.0, 0.0, 0.0))  # affinity = 0.0
    out = mod._run_mode_a([rec], payload, ModeAGate())
    row = out[0]
    assert row["gate_weight"] == pytest.approx(0.5)
    assert row["score"] == pytest.approx(0.5 * 0.9 + 0.5 * 0.0)


def test_run_mode_a_skips_records_without_prototype():
    """Proposals whose class has no prototype are skipped (Mode A flow)."""
    mod = _load_03()
    from uadapt.fusion.mode_a_analytic import ModeAGate

    rec = _make_record(class_name="smoke")  # payload has only 'fire'
    out = mod._run_mode_a([rec], _prototype_payload(), ModeAGate())
    assert out == []
