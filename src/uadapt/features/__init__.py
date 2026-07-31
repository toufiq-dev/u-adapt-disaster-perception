"""Feature extraction and caching (frozen backbone, one pass per image)."""

from .cache_engine import FeatureCacheEngine, FeatureRecord, load_cache

__all__ = ["FeatureCacheEngine", "FeatureRecord", "load_cache"]
