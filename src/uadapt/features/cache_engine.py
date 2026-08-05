"""Feature extraction and caching engine.

The frozen backbone runs **once per image**; candidate proposals are limited
to **top-k** (default k=100; k=300 only as an ablation) and the resulting box
features / image features / text similarities are cached to disk **outside the
repository** (issue #4). Downstream stages (prototypes, fusion, metrics) read
only from the cache — no backbone re-runs.

Cache format: one compressed ``.npz`` per split directory, with per-proposal
feature arrays, plus a small ``manifest.json`` describing the pipeline
settings. The cache directory must live outside the git repository
(default: ``cached_features/`` which is gitignored).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from uadapt.models.backbone_loader import Proposal, limit_top_k

logger = logging.getLogger(__name__)

# Default top-k for primary experiments. 300 is reserved for the upper-bound
# ablation only (see docs/pre_registration.md).
DEFAULT_TOP_K = 100
ABLATION_TOP_K = 300


@dataclass
class FeatureRecord:
    """Cached per-proposal features (immutable-ish, plain data)."""

    image_id: str
    class_name: str
    score: float
    bbox: np.ndarray                       # (4,)
    visual_feature: np.ndarray             # (D,) box feature
    text_similarities: np.ndarray          # (C,) per-class text similarities
    classes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "class_name": self.class_name,
            "score": float(self.score),
            "bbox": self.bbox.astype(float).tolist(),
            "visual_feature": self.visual_feature.astype(float).tolist(),
            "text_similarities": self.text_similarities.astype(float).tolist(),
            "classes": list(self.classes),
        }


class FeatureCacheEngine:
    """Runs the frozen backbone once per image, limits proposals to top-k, and
    caches the extracted features to disk."""

    def __init__(
        self,
        backbone: Any,
        cache_dir: Path | str = "cached_features",
        top_k: int = DEFAULT_TOP_K,
        feature_fn: Optional[Callable[[Proposal], np.ndarray]] = None,
    ) -> None:
        self.backbone = backbone
        self.cache_dir = Path(cache_dir)
        self.top_k = top_k
        self.feature_fn = feature_fn  # optional per-box feature extractor

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------
    def extract_and_cache(
        self,
        images: Sequence[Any],
        classes: List[str],
        split: str = "val",
        image_ids: Optional[Sequence[str]] = None,
        resume: bool = True,
    ) -> Path:
        """Extract features for ``images`` and cache them for ``split``.

        Args:
            images: sequence of images (np.ndarray HxWx3), OR a lazy iterable
                of ``(image, image_id)`` pairs — the RAM-safe streaming mode
                used by 01_extract_and_cache.py: each image is decoded,
                processed and its reference dropped before the next is read,
                so peak image RAM is ~1 frame instead of the whole split
                (a full LADD train split of ~1,200 aerial images would
                otherwise need tens of GB decoded).
            classes: open-vocabulary class list.
            split: cache key (train / val / test).
            image_ids: optional parallel ids (defaults to indices). Must be
                None when ``images`` is a lazy pairs iterable.
            resume: skip images already cached (Colab-friendly).

        Returns the cache directory.
        """
        split_dir = self.cache_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = split_dir / "manifest.json"

        records: Dict[str, List[FeatureRecord]] = {}
        if resume and manifest_path.exists():
            records = _load_manifest_records(manifest_path)
            logger.info("resuming split %s with %d images cached", split, len(records))

        # Streaming mode: ``images`` has no __len__ -> it is a lazy iterable
        # of (image, image_id) pairs (each image freed right after use).
        try:
            len(images)  # type: ignore[arg-type]
            is_stream = False
        except TypeError:
            is_stream = True
        if is_stream:
            if image_ids is not None:
                raise ValueError(
                    "image_ids must be None when images is a lazy (image, id) "
                    "pairs iterable"
                )
            # Pairs are yielded as (image, image_id); normalize to (id, image)
            # so the shared loop below handles both modes identically.
            items = ((iid, img) for img, iid in images)  # type: ignore[assignment]
        else:
            ids = image_ids or [f"img_{i:06d}" for i in range(len(images))]  # type: ignore[arg-type]
            items = zip(ids, images)  # type: ignore[arg-type]

        for image_id, image in items:
            if resume and image_id in records:
                continue
            proposals = self.backbone.predict(image, classes, image_id=image_id)
            proposals = limit_top_k(proposals, self.top_k)
            records[image_id] = [self._to_record(p, classes) for p in proposals]
            if len(records) % 50 == 0:
                _write_records(split_dir, records)
                logger.info("cached %d images for split %s", len(records), split)

        _write_records(split_dir, records)
        with open(manifest_path, "w") as fh:
            json.dump(
                {
                    "backbone": getattr(self.backbone, "name", "unknown"),
                    "top_k": self.top_k,
                    "classes": classes,
                    "n_images": len(records),
                },
                fh,
                indent=2,
            )
        logger.info("finished caching split %s (%d images)", split, len(records))
        return split_dir

    def _to_record(self, proposal: Proposal, classes: List[str]) -> FeatureRecord:
        visual = proposal.visual_feature
        if visual is None:
            visual = (
                self.feature_fn(proposal)
                if self.feature_fn is not None
                else np.zeros((0,), dtype=np.float32)
            )
        text_sim = proposal.text_similarities
        if text_sim is None:
            text_sim = np.zeros((len(classes),), dtype=np.float32)
        return FeatureRecord(
            image_id=proposal.image_id,
            class_name=proposal.class_name,
            score=proposal.score,
            bbox=np.asarray(proposal.bbox, dtype=np.float32),
            visual_feature=np.asarray(visual, dtype=np.float32),
            text_similarities=np.asarray(text_sim, dtype=np.float32),
            classes=list(classes),
        )


def load_cache(cache_dir: Path | str, split: str = "val") -> List[FeatureRecord]:
    """Load cached feature records for a split (shared by all downstream stages).

    Returns a FLAT list of :class:`FeatureRecord` (the on-disk format is a
    dict keyed by image_id, but every consumer — prototype builder, fusion,
    demo eval — iterates records as a plain sequence).
    """
    split_dir = Path(cache_dir) / split
    manifest = split_dir / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(
            f"no cache found at {split_dir} (run 01_extract_and_cache.py first)"
        )
    by_image = _load_manifest_records(manifest)
    return [r for recs in by_image.values() for r in recs]


# ----------------------------------------------------------------------
# Disk helpers (npz per image batch + manifest with embedded records)
# ----------------------------------------------------------------------
def _records_to_arrays(records: Dict[str, List[FeatureRecord]]) -> Dict[str, np.ndarray]:
    """Flatten records into parallel arrays for npz storage."""
    flat = [r for recs in records.values() for r in recs]
    dim = max((r.visual_feature.size for r in flat), default=0)
    n_class = max((r.text_similarities.size for r in flat), default=0)
    arr_visual = np.zeros((len(flat), dim), dtype=np.float32)
    arr_text = np.zeros((len(flat), n_class), dtype=np.float32)
    arr_bbox = np.zeros((len(flat), 4), dtype=np.float32)
    arr_score = np.zeros((len(flat),), dtype=np.float32)
    ids: List[str] = []
    for i, r in enumerate(flat):
        arr_visual[i] = r.visual_feature
        arr_text[i] = r.text_similarities
        arr_bbox[i] = r.bbox
        arr_score[i] = r.score
        ids.append(r.image_id)
    return {
        "image_ids": np.asarray(ids),
        "class_names": np.asarray([r.class_name for r in flat]),
        "scores": arr_score,
        "bboxes": arr_bbox,
        "visual_features": arr_visual,
        "text_similarities": arr_text,
    }


def _write_records(split_dir: Path, records: Dict[str, List[FeatureRecord]]) -> None:
    arrays = _records_to_arrays(records)
    np.savez_compressed(split_dir / "features.npz", **arrays)
    # Small json sidecar so records survive without torch installed.
    with open(split_dir / "records.json", "w") as fh:
        json.dump(
            {"records": {k: [r.to_dict() for r in v] for k, v in records.items()}},
            fh,
        )


def _load_manifest_records(manifest_path: Path) -> Dict[str, List[FeatureRecord]]:
    records_json = manifest_path.parent / "records.json"
    if not records_json.exists():
        return {}
    with open(records_json) as fh:
        payload = json.load(fh)
    out: Dict[str, List[FeatureRecord]] = {}
    for image_id, recs in payload["records"].items():
        out[image_id] = [
            FeatureRecord(
                image_id=r["image_id"],
                class_name=r["class_name"],
                score=float(r["score"]),
                bbox=np.asarray(r["bbox"], dtype=np.float32),
                visual_feature=np.asarray(r["visual_feature"], dtype=np.float32),
                text_similarities=np.asarray(r["text_similarities"], dtype=np.float32),
                classes=list(r["classes"]),
            )
            for r in recs
        ]
    return out
