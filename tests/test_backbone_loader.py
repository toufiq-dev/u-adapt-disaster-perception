"""Tests for src/uadapt/models/backbone_loader.py (pure-python helpers).

These helpers power the real-data feature extraction path and are testable
without torch/transformers installed (the heavy model classes import lazily
inside __init__/predict, keeping unit tests dependency-free).

Covered:
* _class_token_spans  — per-class token spans in the Grounding DINO prompt
* _roi_mean_feature   — RoI mean-pooling over the encoder vision grid
* resolve_device      — CUDA -> MPS/CPU fallback when CUDA is unavailable
"""

from __future__ import annotations

import numpy as np
import pytest

from uadapt.models.backbone_loader import (
    _class_token_spans,
    _roi_mean_feature,
    resolve_device,
)


class _MockTokenizer:
    """Tiny stand-in for a BERT-style tokenizer with subword tokens."""

    def __init__(self, vocab: dict[str, str]):
        # id -> token (strings only; ids are positional in the prompt)
        self._tokens = vocab

    def convert_ids_to_tokens(self, ids) -> list[str]:
        return [self._tokens[i] for i in ids]

    def tokenize(self, text: str) -> list[str]:
        # crude split so classes can span multiple subwords
        return text.split()


# Grounding DINO prompt: [CLS] a <c1> . a <c2> . [SEP]
_CLS_TOKENS = {0: "[CLS]", 1: "a", 2: "person", 3: ".", 4: "a", 5: "car", 6: ".", 7: "[SEP]"}


def _prompt_ids(tokens: dict[int, str]) -> list[int]:
    return list(tokens)


def test_class_token_spans_single_word_classes():
    tok = _MockTokenizer(_CLS_TOKENS)
    spans = _class_token_spans(tok, _prompt_ids(_CLS_TOKENS), ["person", "car"])
    # "a person" spans ids [1, 3); "a car" spans ids [4, 6)
    assert spans == [(1, 3), (4, 6)]


def test_class_token_spans_multitoken_class():
    vocab = {0: "[CLS]", 1: "a", 2: "search", 3: "and", 4: "rescue", 5: "person",
             6: ".", 7: "a", 8: "fire", 9: ".", 10: "[SEP]"}
    tok = _MockTokenizer(vocab)
    # "search and rescue person" tokenizes into 4 subwords at ids 2..5,
    # preceded by the 'a' article at id 1 -> span [1, 6)
    ids = list(vocab)
    spans = _class_token_spans(tok, ids, ["search and rescue person", "fire"])
    assert spans == [(1, 6), (7, 9)]  # includes the preceding 'a'


def test_class_token_spans_missing_class_raises():
    tok = _MockTokenizer(_CLS_TOKENS)
    with pytest.raises(ValueError, match="could not locate class"):
        _class_token_spans(tok, _prompt_ids(_CLS_TOKENS), ["dragon"])


def test_roi_mean_feature_pools_inside_box():
    # 4 tokens of dim 2 laid out in a 2x2 grid (row-major: 0 1 / 2 3);
    # the top-left quadrant of the image maps to token 0 only
    vision = np.arange(8, dtype=np.float64).reshape(4, 2)
    feat = _roi_mean_feature(vision, (0.0, 0.0, 0.5, 0.5), img_w=100.0, img_h=100.0)
    assert feat.shape == (2,)
    assert feat == pytest.approx(vision[0])  # tokens (0,0) only


def test_roi_mean_feature_full_box_pools_all_tokens():
    # box spanning the whole image pools every token in the grid
    vision = np.arange(8, dtype=np.float64).reshape(4, 2)
    feat = _roi_mean_feature(vision, (0.0, 0.0, 100.0, 100.0), img_w=100.0, img_h=100.0)
    assert feat == pytest.approx(vision.mean(axis=0))


def test_roi_mean_feature_uses_exact_grid_dims():
    # 9 tokens laid out 3x3, but caller knows the true grid is 3 rows x 3 cols.
    # Box covering the top-left 2x2 tokens -> mean of tokens 0,1,3,4.
    vision = np.arange(18, dtype=np.float64).reshape(9, 2)
    feat = _roi_mean_feature(
        vision, (0.0, 0.0, 100.0 / 3 * 2, 100.0 / 3 * 2),
        img_w=100.0, img_h=100.0, grid_h=3, grid_w=3,
    )
    expected = vision[[0, 1, 3, 4]].mean(axis=0)
    assert feat == pytest.approx(expected)


def test_roi_mean_feature_degenerate_box_returns_zeros():
    vision = np.ones((9, 3))
    feat = _roi_mean_feature(vision, (10.0, 10.0, 5.0, 5.0), img_w=100.0, img_h=100.0)
    assert np.all(feat == 0.0)


def test_roi_mean_feature_clamps_to_image():
    vision = np.ones((9, 3))
    # box mostly outside the image -> clamped region still yields a feature
    feat = _roi_mean_feature(vision, (-50.0, -50.0, 25.0, 25.0), img_w=100.0, img_h=100.0)
    assert feat.shape == (3,)
    assert np.all(feat == 1.0)


def test_resolve_device_falls_back_when_cuda_unavailable(monkeypatch):
    import uadapt.models.backbone_loader as bl

    class _NoCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _NoMps:
        @staticmethod
        def is_available() -> bool:
            return False

    fake_torch = type("torch", (), {"cuda": _NoCuda, "backends": type("b", (), {"mps": _NoMps})})
    monkeypatch.setattr(bl, "torch", fake_torch, raising=False)
    # resolve_device imports torch inside; patch sys.modules instead
    import sys

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert resolve_device("cuda") == "cpu"
    assert resolve_device("mps") == "mps"  # explicit request kept
