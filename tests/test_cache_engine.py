"""Cache-engine tests: extract_and_cache sequence path + RAM-safe streaming.

The streaming path (lazy iterable of (image, image_id) pairs) was added
2026-08-05 so 01_extract_and_cache.py can process whole splits without
decoding them all into RAM at once (a full LADD train split would need tens
of GB decoded; see change_log.md).
"""

import numpy as np
import pytest

from uadapt.features.cache_engine import FeatureCacheEngine, load_cache
from uadapt.models.backbone_loader import Proposal


class _StubBackbone:
    """Deterministic backbone stub (no torch/transformers needed)."""

    name = "stub"

    def predict(self, image, classes, image_id=None):
        return [
            Proposal(
                image_id=image_id,
                bbox=np.zeros((4,), dtype=float),
                score=0.5,
                class_name=classes[0],
                visual_feature=np.ones((4,), dtype=float),
                text_similarities=np.full((len(classes),), 0.5, dtype=float),
                classes=list(classes),
            )
        ]


def test_extract_and_cache_sequence_path(tmp_path):
    engine = FeatureCacheEngine(_StubBackbone(), cache_dir=tmp_path, top_k=10)
    imgs = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]
    engine.extract_and_cache(
        imgs, ["a", "b"], split="test", image_ids=["i0", "i1", "i2"]
    )
    recs = load_cache(tmp_path, split="test")
    assert len(recs) == 3
    assert {r.image_id for r in recs} == {"i0", "i1", "i2"}
    assert all(r.classes == ["a", "b"] for r in recs)


def test_extract_and_cache_streaming_pairs(tmp_path):
    engine = FeatureCacheEngine(_StubBackbone(), cache_dir=tmp_path, top_k=10)

    def pairs():
        for i in range(4):
            yield np.zeros((4, 4, 3), dtype=np.uint8), f"s{i}"

    engine.extract_and_cache(pairs(), ["a", "b"], split="test", image_ids=None)
    recs = load_cache(tmp_path, split="test")
    assert len(recs) == 4
    assert {r.image_id for r in recs} == {"s0", "s1", "s2", "s3"}


def test_extract_and_cache_streaming_rejects_image_ids(tmp_path):
    engine = FeatureCacheEngine(_StubBackbone(), cache_dir=tmp_path)

    def pairs():
        yield np.zeros((4, 4, 3), dtype=np.uint8), "s0"

    with pytest.raises(ValueError):
        engine.extract_and_cache(pairs(), ["a"], split="test", image_ids=["x"])


def test_extract_and_cache_streaming_resume(tmp_path):
    engine = FeatureCacheEngine(_StubBackbone(), cache_dir=tmp_path, top_k=10)

    def pairs_all():
        for i in range(4):
            yield np.zeros((4, 4, 3), dtype=np.uint8), f"r{i}"

    # First pass caches all four; a resumed second pass must not duplicate.
    engine.extract_and_cache(pairs_all(), ["a"], split="test")
    engine.extract_and_cache(pairs_all(), ["a"], split="test")
    recs = load_cache(tmp_path, split="test")
    assert len(recs) == 4
