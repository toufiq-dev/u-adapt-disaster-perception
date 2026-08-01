"""Deterministic synthetic demo world (supervisor demo fallback data).

The full U-ADAPT pipeline needs cached backbone features and COCO ground
truth; both are gitignored and unavailable until the raw datasets are
downloaded (Milestone 1). This module synthesizes a small, reproducible world
that matches the REAL pipeline schemas exactly:

  * ``FeatureRecord`` lists for train and test splits (the same schema the
    frozen-backbone cache produces),
  * per-class prompt-template embeddings (for ``sigma^2_text`` via the real
    ``normalized_text_variance`` estimator),
  * COCO-style ground-truth boxes (for mAP evaluation).

The world is *engineered* so the U-ADAPT mechanism is visible — this is a
demo, not a result:

  * per-class "noise" profiles control how unreliable the text vs visual
    modality is (the exact regime the analytic gate is designed for);
  * raw detector scores overlap between true hits and distractors (an
    imperfect zero-shot baseline — otherwise there is no gap to recover);
  * hit proposals carry a strong signal in whichever modality is reliable
    for their class, so the gate can route to the better modality;
  * distractor proposals are weak in BOTH modalities.

.. warning::
    Results on synthetic data carry NO scientific weight. They demonstrate
    the pipeline wiring and the diagnostic machinery only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from uadapt.features.cache_engine import FeatureRecord

DEFAULT_CLASSES = ["fire", "smoke", "person", "vehicle", "debris", "roof"]
FEATURE_DIM = 64
N_TEMPLATES = 20          # prompt-template ensemble size (matches pre-registration)
N_SUPPORT_POOL = 10       # train records per class (prototype-building pool)
IMG_SIZE = 512            # square canvas for schematic rendering

# Per-class modality-noise profiles. High ``visual_noise`` => unreliable
# visual modality (high sigma^2_visual + high visual error rate); high
# ``text_noise`` => unreliable text modality. The classes are chosen to span
# the regimes: visual-unreliable (fire/smoke), text-unreliable (person/
# vehicle), and balanced (debris/roof).
PROFILES: Dict[str, Dict[str, float]] = {
    "fire":    {"text_noise": 0.10, "visual_noise": 0.45},
    "smoke":   {"text_noise": 0.15, "visual_noise": 0.38},
    "person":  {"text_noise": 0.45, "visual_noise": 0.15},
    "vehicle": {"text_noise": 0.38, "visual_noise": 0.20},
    "debris":  {"text_noise": 0.30, "visual_noise": 0.30},
    "roof":    {"text_noise": 0.22, "visual_noise": 0.26},
}
_FALLBACK_PROFILE = {"text_noise": 0.25, "visual_noise": 0.25}


@dataclass
class SyntheticDataset:
    """Bundle returned by :func:`generate_synthetic_dataset`."""

    classes: List[str]
    template_embeddings: Dict[str, np.ndarray]   # class -> (N_TEMPLATES, D)
    train_records: List[FeatureRecord]           # prototype-building pool
    test_records: List[FeatureRecord]            # evaluated proposals
    ground_truth: List[Dict]                     # COCO-style GT boxes
    meta: Dict = field(default_factory=dict)


def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _l2_rows(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def _noise_scale_for_affinity(target_affinity: float) -> float:
    """Feature-noise scale t such that cos(centroid, centroid + t*u) ~= 2a-1.

    cos = 1 / sqrt(1 + t^2)  =>  t = sqrt(1 / cos^2 - 1).
    """
    cos = float(np.clip(2.0 * target_affinity - 1.0, 0.05, 0.99))
    return float(np.sqrt(max(1.0 / (cos * cos) - 1.0, 0.0)))


def generate_synthetic_dataset(
    classes: Optional[Sequence[str]] = None,
    seed: int = 0,
    n_test_images: int = 80,
    feature_dim: int = FEATURE_DIM,
    n_templates: int = N_TEMPLATES,
    n_support_pool: int = N_SUPPORT_POOL,
) -> SyntheticDataset:
    """Generate the deterministic synthetic demo world (seed=0 default).

    Args:
        classes: class vocabulary (defaults to the 6-class demo set).
        seed: RNG seed for full reproducibility.
        n_test_images: number of test images (50-100 recommended).
        feature_dim: box-feature dimensionality.
        n_templates: prompt-template ensemble size per class.
        n_support_pool: train records per class for prototype building.
    """
    rng = np.random.default_rng(seed)
    classes = list(classes or DEFAULT_CLASSES)
    profiles = {c: dict(PROFILES.get(c, _FALLBACK_PROFILE)) for c in classes}

    # --- visual class centroids (unit vectors in feature space) -------------
    centroids = _l2_rows(rng.normal(size=(len(classes), feature_dim)))
    centroid = {c: centroids[i] for i, c in enumerate(classes)}

    # --- per-class text template embeddings (sigma^2_text signal) -----------
    text_centroids = _l2_rows(rng.normal(size=(len(classes), feature_dim)))
    template_embeddings: Dict[str, np.ndarray] = {}
    for i, c in enumerate(classes):
        emb = text_centroids[i] + rng.normal(
            scale=0.5 * profiles[c]["text_noise"], size=(n_templates, feature_dim)
        )
        template_embeddings[c] = _l2_rows(emb)

    # --- train records: per-class support pool (prototype builder input) ----
    train_records: List[FeatureRecord] = []
    for c in classes:
        aff_c = _hit_affinity(profiles[c]["visual_noise"], 0.0)
        t = _noise_scale_for_affinity(aff_c)
        # unit-direction noise: |u| = 1, so cos = 1/sqrt(1 + t^2) as intended
        noise = t * _l2_rows(rng.normal(size=(n_support_pool, feature_dim)))
        feats = _l2_rows(centroid[c][None, :] + noise)
        for j in range(n_support_pool):
            train_records.append(
                _make_record(
                    rng=rng,
                    image_id=f"train_{c}_{j:02d}",
                    class_name=c,
                    visual_feature=feats[j],
                    classes=classes,
                    is_hit=True,
                    profiles=profiles,
                    centroid=centroid[c],
                )
            )

    # --- test images: GT boxes + hit proposals + distractors -----------------
    test_records: List[FeatureRecord] = []
    ground_truth: List[Dict] = []
    for img_i in range(n_test_images):
        image_id = f"demo_{img_i:04d}"
        n_gt = int(rng.integers(1, 4))
        for _ in range(n_gt):
            cls = classes[int(rng.integers(len(classes)))]
            box = _random_box(rng)
            ground_truth.append(
                {"image_id": image_id, "class": cls, "bbox": box.tolist()}
            )
            # hit proposal for this GT object
            test_records.append(
                _make_record(
                    rng=rng,
                    image_id=image_id,
                    class_name=cls,
                    visual_feature=None,
                    classes=classes,
                    is_hit=True,
                    profiles=profiles,
                    centroid=centroid[cls],
                    bbox=_jitter_box(rng, box),
                )
            )
        # distractors (random boxes/classes; weak in both modalities)
        for _ in range(int(rng.integers(2, 5))):
            cls = classes[int(rng.integers(len(classes)))]
            test_records.append(
                _make_record(
                    rng=rng,
                    image_id=image_id,
                    class_name=cls,
                    visual_feature=None,
                    classes=classes,
                    is_hit=False,
                    profiles=profiles,
                    centroid=centroid[cls],
                    bbox=_random_box(rng),
                )
            )

    meta = {
        "data_source": "synthetic",
        "seed": seed,
        "classes": classes,
        "feature_dim": feature_dim,
        "n_templates": n_templates,
        "n_support_pool": n_support_pool,
        "n_test_images": n_test_images,
        "n_train_records": len(train_records),
        "n_test_records": len(test_records),
        "n_ground_truth": len(ground_truth),
        "profiles": {c: dict(p) for c, p in profiles.items()},
    }
    return SyntheticDataset(
        classes=classes,
        template_embeddings=template_embeddings,
        train_records=train_records,
        test_records=test_records,
        ground_truth=ground_truth,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Record construction
# ---------------------------------------------------------------------------
def _hit_affinity(visual_noise: float, noise: float) -> float:
    """Target affinity for a HIT proposal of a class with ``visual_noise``.

    High visual noise => low affinity (the box feature drifts from the
    prototype) => the gate should down-weight the visual branch.
    """
    return float(np.clip(0.95 - 0.9 * visual_noise + 0.03 * noise, 0.42, 0.97))


def _make_record(
    rng: np.random.Generator,
    image_id: str,
    class_name: str,
    classes: Sequence[str],
    is_hit: bool,
    profiles: Dict[str, Dict[str, float]],
    centroid: np.ndarray,
    visual_feature: Optional[np.ndarray] = None,
    bbox: Optional[np.ndarray] = None,
) -> FeatureRecord:
    """Build one FeatureRecord with engineered text/visual signals.

    * text_similarities: for HITS the proposal's own class is boosted (high
      when the class's text modality is reliable, low text_noise);
      distractors get weak similarities for ALL classes (spurious boxes).
    * visual_feature: drawn near the class centroid with a noise scale that
      encodes the class's visual reliability (hits) or far from it
      (distractors).
    * score: raw detector confidence overlaps between hits and distractors so
      the zero-shot baseline is imperfect (there is a gap to recover).
    """
    profile = profiles[class_name]
    c_idx = list(classes).index(class_name)

    # --- text signal --------------------------------------------------------
    sims = rng.uniform(0.06, 0.34, size=len(classes))
    if is_hit:
        base = float(
            np.clip(0.90 - 1.6 * profile["text_noise"] + 0.08 * rng.normal(), 0.05, 0.95)
        )
        sims[c_idx] = base
    sims = np.clip(sims, 0.0, 1.0).astype(np.float32)

    # --- visual feature ------------------------------------------------------
    if visual_feature is None:
        if is_hit:
            target_aff = _hit_affinity(profile["visual_noise"], rng.normal())
        else:
            target_aff = float(np.clip(rng.uniform(0.30, 0.55), 0.30, 0.60))
        t = _noise_scale_for_affinity(target_aff)
        # unit-direction noise: |u| = 1, so the realized affinity matches the
        # target (raw Gaussian noise would scale with sqrt(D) and make the
        # feature random, collapsing all affinities toward 0.5).
        u = _l2(rng.normal(size=(centroid.size,)))
        visual_feature = _l2(centroid + t * u).astype(np.float32)

    score = float(rng.uniform(0.30, 0.75)) if is_hit else float(rng.uniform(0.10, 0.65))
    return FeatureRecord(
        image_id=image_id,
        class_name=class_name,
        score=score,
        bbox=np.asarray(bbox if bbox is not None else _random_box(rng), dtype=np.float32),
        visual_feature=np.asarray(visual_feature, dtype=np.float32),
        text_similarities=sims,
        classes=list(classes),
    )


def _random_box(rng: np.random.Generator) -> np.ndarray:
    x1 = float(rng.uniform(10, IMG_SIZE - 90))
    y1 = float(rng.uniform(10, IMG_SIZE - 90))
    w = float(rng.uniform(30, 120))
    h = float(rng.uniform(30, 120))
    return np.asarray([x1, y1, min(x1 + w, IMG_SIZE - 1), min(y1 + h, IMG_SIZE - 1)])


def _jitter_box(rng: np.random.Generator, box: np.ndarray, jitter: float = 0.22) -> np.ndarray:
    w = box[2] - box[0]
    h = box[3] - box[1]
    dx = rng.uniform(-jitter, jitter) * w
    dy = rng.uniform(-jitter, jitter) * h
    return np.clip(
        np.asarray([box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]),
        0,
        IMG_SIZE - 1,
    )
