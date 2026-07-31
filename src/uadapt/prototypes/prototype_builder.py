"""Prototype construction (Phase 3 of the U-ADAPT pipeline).

For each target class we build:

* **Text prototype** — the mean of M=20 CLIP text-encoder embeddings over a
  structured prompt-template ensemble, plus the mean pairwise cosine distance
  across templates as the raw text uncertainty signal.
* **Visual prototype** — the L2-normalized centroid of the k support box
  features (k in {1,3,5}) sampled from the target-domain training split,
  plus the mean pairwise cosine distance across supports as the raw visual
  uncertainty signal (0 for k=1 — no observed dispersion).

Outlier rejection is pre-registered: Mahalanobis distance with shrinkage
covariance for k >= 5 (2-sigma cutoff); cosine threshold 0.5 relative to the
centroid for k < 5.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from uadapt.features.cache_engine import FeatureRecord

logger = logging.getLogger(__name__)

# Pre-registered constants (see docs/pre_registration.md)
DEFAULT_M_TEMPLATES = 20
COSINE_REJECT_THRESHOLD = 0.5
MAHALANOBIS_SIGMA_CUTOFF = 2.0
SHRINKAGE_DEFAULT = 0.1


@dataclass
class TextPrototype:
    """Class text prototype built from a prompt-template ensemble."""

    class_name: str
    embeddings: np.ndarray           # (M, D) template embeddings
    mean_embedding: np.ndarray       # (D,)
    sigma_text: float                # mean pairwise cosine distance (raw, [0,2])

    def cosine_similarity(self, feature: np.ndarray) -> float:
        return float(
            _cosine(np.asarray(feature, dtype=np.float64), self.mean_embedding)
        )


@dataclass
class VisualPrototype:
    """Class visual prototype built from k support box features."""

    class_name: str
    support_features: np.ndarray     # (k, D)
    centroid: np.ndarray             # L2-normalized mean of kept supports
    sigma_visual: float              # mean pairwise cosine distance (0 for k=1)
    support_ids: List[str] = field(default_factory=list)
    n_kept: int = 0

    def cosine_similarity(self, feature: np.ndarray) -> float:
        return float(
            _cosine(np.asarray(feature, dtype=np.float64), self.centroid)
        )


# ----------------------------------------------------------------------
# Text prototypes
# ----------------------------------------------------------------------
def build_text_prototypes(
    classes: Sequence[str],
    text_encoder: Callable[[Sequence[str]], np.ndarray],
    m_templates: int = DEFAULT_M_TEMPLATES,
    prompt_templates: Optional[Sequence[str]] = None,
) -> Dict[str, TextPrototype]:
    """Build text prototypes from a prompt-template ensemble.

    ``text_encoder`` maps a batch of prompt strings to an (N, D) embedding
    matrix (e.g., CLIP text encoder). ``prompt_templates`` must contain the
    placeholder ``{class}``; defaults to the structured set from the proposal.
    """
    templates = list(prompt_templates or _DEFAULT_TEMPLATES)
    templates = templates[:m_templates] if m_templates else templates
    out: Dict[str, TextPrototype] = {}
    for cls in classes:
        prompts = [t.format(class_name=cls) for t in templates]
        embeddings = np.asarray(text_encoder(prompts), dtype=np.float64)
        embeddings = _l2_normalize_rows(embeddings)
        mean = _l2_normalize(embeddings.mean(axis=0))
        sigma = mean_pairwise_cosine_distance(embeddings)
        out[cls] = TextPrototype(
            class_name=cls,
            embeddings=embeddings,
            mean_embedding=mean,
            sigma_text=sigma,
        )
    return out


# ----------------------------------------------------------------------
# Visual prototypes
# ----------------------------------------------------------------------
def build_visual_prototypes(
    records: Sequence[FeatureRecord],
    classes: Sequence[str],
    shots: int = 5,
    rng: Optional[np.random.Generator] = None,
    reject_outliers_flag: bool = True,
) -> Dict[str, VisualPrototype]:
    """Build visual prototypes from cached training-split features.

    For each class, sample ``shots`` support boxes uniformly at random, apply
    pre-registered outlier rejection, and take the L2-normalized centroid.

    Args:
        records: cached training features (from FeatureCacheEngine).
        classes: target class list.
        shots: k in {1, 3, 5} support examples per class.
        rng: RNG for reproducible support sampling (seeded across 10 seeds).
        reject_outliers_flag: enable pre-registered outlier rejection.

    Raises:
        ValueError: if a class has fewer than ``shots`` cached proposals.
    """
    rng = rng or np.random.default_rng(0)
    by_class: Dict[str, List[FeatureRecord]] = {}
    for r in records:
        by_class.setdefault(r.class_name, []).append(r)

    out: Dict[str, VisualPrototype] = {}
    for cls in classes:
        pool = by_class.get(cls, [])
        if len(pool) < shots:
            raise ValueError(
                f"class '{cls}' has {len(pool)} cached proposals < shots={shots}"
            )
        idx = rng.choice(len(pool), size=shots, replace=False)
        supports = [pool[i] for i in idx]
        features = _l2_normalize_rows(
            np.stack([s.visual_feature for s in supports], axis=0).astype(np.float64)
        )
        ids = [s.image_id for s in supports]

        if reject_outliers_flag and len(features) > 1:
            features, ids, kept = reject_outliers(features, ids=ids)
        else:
            kept = len(features)

        centroid = _l2_normalize(features.mean(axis=0))
        sigma = mean_pairwise_cosine_distance(features) if len(features) > 1 else 0.0
        out[cls] = VisualPrototype(
            class_name=cls,
            support_features=features,
            centroid=centroid,
            sigma_visual=sigma,
            support_ids=list(ids),
            n_kept=kept,
        )
    return out


def reject_outliers(
    features: np.ndarray,
    ids: Optional[Sequence[str]] = None,
    cosine_threshold: float = COSINE_REJECT_THRESHOLD,
    sigma_cutoff: float = MAHALANOBIS_SIGMA_CUTOFF,
) -> tuple[np.ndarray, List[str], int]:
    """Pre-registered outlier rejection on support features.

    * k >= 5: Mahalanobis distance with shrinkage covariance; exclude supports
      farther than ``sigma_cutoff`` standard deviations from the centroid.
    * k < 5: exclude supports with cosine similarity < ``cosine_threshold``
      relative to the centroid (stable from few samples).
    """
    features = np.asarray(features, dtype=np.float64)
    n = len(features)
    ids = list(ids or [f"support_{i}" for i in range(n)])
    if n <= 1:
        return features, ids, n

    centroid = features.mean(axis=0)
    if n >= 5:
        cov = np.cov(features, rowvar=False) + SHRINKAGE_DEFAULT * np.eye(
            features.shape[1]
        )
        inv_cov = np.linalg.pinv(cov)
        diff = features - centroid
        mahal = np.sqrt(np.einsum("ij,jk,ik->i", diff, inv_cov, diff))
        keep = mahal <= sigma_cutoff
    else:
        sims = features @ _l2_normalize(centroid)
        keep = sims >= cosine_threshold

    if keep.sum() == 0:  # never return an empty prototype
        keep[0] = True
    return features[keep], [ids[i] for i in range(n) if keep[i]], int(keep.sum())


# ----------------------------------------------------------------------
# Shared helpers (mirrored in uadapt.uncertainty.variance_estimators; kept
# local to avoid circular imports)
# ----------------------------------------------------------------------
_DEFAULT_TEMPLATES = (
    "a photo of a {class_name}",
    "an aerial view of a {class_name}",
    "a disaster scene with {class_name}",
    "a search and rescue image of {class_name}",
    "a high-resolution aerial image of a {class_name}",
    "a small {class_name} in an aerial photo",
    "a damaged {class_name} in a disaster zone",
    "an overhead view of a {class_name}",
    "a satellite view showing a {class_name}",
    "a drone image of a {class_name}",
    "a {class_name} seen from above",
    "a blurred {class_name} in a smoky scene",
    "a clear view of a {class_name}",
    "a distant {class_name} in a cluttered scene",
    "a {class_name} surrounded by debris",
    "an aerial photograph containing a {class_name}",
    "a rescue operation scene with a {class_name}",
    "a {class_name} in flooded terrain",
    "a top-down view of a {class_name}",
    "a {class_name} partially occluded by smoke",
)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _l2_normalize_rows(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def mean_pairwise_cosine_distance(features: np.ndarray) -> float:
    """2/(M(M-1)) * sum_{i<j} (1 - cos(e_i, e_j)); 0 for a single sample."""
    n = len(features)
    if n < 2:
        return 0.0
    feats = _l2_normalize_rows(np.asarray(features, dtype=np.float64))
    sim = feats @ feats.T
    iu = np.triu_indices(n, k=1)
    return float(np.mean(1.0 - sim[iu]))
