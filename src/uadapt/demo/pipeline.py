"""End-to-end Mode A demo pipeline (shared by script, notebook, and tests).

Runs the REAL U-ADAPT code paths on a data source (synthetic by default,
real cache when available):

    1. visual prototypes        -> build_visual_prototypes (real code)
    2. text uncertainty         -> normalized_text_variance on template
                                   embeddings (synthetic) or per-proposal
                                   class-similarity entropy (real cache)
    3. per-proposal signals     -> s_text, affinity (s_visual), normalized
                                   variances (min-max or absolute scaling,
                                   per ``norm_strategy``), gate weight
                                   w = sigmoid(-a*vv + b*tv + g*aff)
    4. fused scores             -> S = (1-w)*S_text + w*S_visual
    5. baselines                -> raw zero-shot (detector score), text-only
                                   (w=0), visual-only (w=1), naive (w=0.5)
    6. metrics                  -> mAP50 per method, per-class AP, proposal
                                   recall ceiling, gap recovery
    7. diagnostics              -> D1/D2/D3 (real pre-registered functions)

Everything downstream of the data source uses the production modules; the
only synthetic part is the fake world in :mod:`synthetic_data` (a demo, not
a result).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from uadapt.features.cache_engine import (
    FeatureRecord,
    class_index,
    class_text_score,
)
from uadapt.fusion.mode_a_analytic import BetaGate, ModeAGate, fuse_scores
from uadapt.metrics.detection_metrics import (
    _iou,
    compute_map50,
    compute_per_class_ap,
    gap_recovery,
    proposal_recall,
)
from uadapt.metrics.diagnostics import (
    d1_text_uncertainty_accuracy,
    d2_visual_uncertainty_accuracy,
    d3_gate_favorability,
)
from uadapt.prototypes.prototype_builder import build_visual_prototypes
from uadapt.uncertainty.variance_estimators import (
    COSINE_DISTANCE_MAX,
    absolute_normalize,
    min_max_normalize,
    normalized_text_variance,
    proposal_text_variance,
    proposal_visual_variance,
    visual_affinity,
)

# Visual-affinity threshold below which a proposal counts as "visual weak".
VISUAL_CORRECT_AFFINITY = 0.65


@dataclass
class DemoResults:
    """Everything the demo computes (json-serializable via :meth:`to_dict`)."""

    meta: Dict
    map50: Dict[str, float]          # method -> mAP50
    per_class_ap: Dict[str, float]   # class -> AP (U-ADAPT fused)
    diagnostics: Dict                # D1/D2/D3 summaries
    gate_stats: Dict                 # w distribution summary
    gap_recovery: Dict               # self-contained + literature numbers
    ablation: Dict[str, float]       # coefficient ablation mAP50
    proposal_level: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "meta": self.meta,
            "map50": self.map50,
            "per_class_ap": self.per_class_ap,
            "diagnostics": self.diagnostics,
            "gate_stats": self.gate_stats,
            "gap_recovery": self.gap_recovery,
            "ablation": self.ablation,
        }


def run_demo(
    train_records: Sequence[FeatureRecord],
    test_records: Sequence[FeatureRecord],
    ground_truth: Sequence[Dict],
    classes: Sequence[str],
    template_embeddings: Optional[Dict[str, np.ndarray]] = None,
    shots: int = 5,
    seed: int = 0,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    zero_shot_reference: Optional[float] = None,
    transfer_reference: Optional[float] = None,
    norm_strategy: str = "min-max",
    gate_type: str = "analytic",
) -> DemoResults:
    """Run the full Mode A demo on a data source. Returns :class:`DemoResults`.

    Args:
        train_records: prototype-building pool (train split).
        test_records: proposals to score (test split).
        ground_truth: COCO-style GT boxes: {image_id, class, bbox}.
        classes: class vocabulary.
        template_embeddings: class -> (M, D) prompt embeddings; if None the
            per-class text variance defaults to 0.5 (real-cache placeholder).
        shots: k support examples per class (k in {1, 3, 5}).
        seed: RNG seed (prototype sampling).
        alpha/beta/gamma: analytic gate coefficients (ablation support).
        zero_shot_reference / transfer_reference: optional literature mAP50
            numbers used for the gap-recovery figure (clearly labelled).
        norm_strategy: how per-class variance terms are scaled to [0, 1]:
            ``min-max`` (default, support-set statistics; collapses to {0, 1}
            for C=2 classes) or ``absolute`` (x / 2.0, class-count-
            independent — fixes the 2-class degeneracy, change_log 2026-08-03).
        gate_type: ``analytic`` (default, pre-registered Mode A gate) or
            ``beta_fallback`` (pre-registered D5 Beta-regression variant —
            hedges the gate toward the neutral 0.5 weight when the variance
            terms are extreme).
    """
    classes = list(classes)
    if norm_strategy not in ("min-max", "absolute"):
        raise ValueError(
            f"unknown norm_strategy {norm_strategy!r} (choices: min-max, absolute)"
        )
    if gate_type not in ("analytic", "beta_fallback"):
        raise ValueError(
            f"unknown gate_type {gate_type!r} (choices: analytic, beta_fallback)"
        )
    rng = np.random.default_rng(seed)

    # --- prototypes + per-class normalized uncertainties --------------------
    protos = build_visual_prototypes(
        list(train_records), classes, shots=shots, rng=rng
    )
    if template_embeddings is not None:
        sigma_text = {
            c: normalized_text_variance(np.asarray(template_embeddings[c]))
            for c in classes
        }
        norm_text = _normalize_class_values(sigma_text, classes, norm_strategy)
        sigma_visual = {c: protos[c].sigma_visual for c in classes}
        norm_visual = _normalize_class_values(sigma_visual, classes, norm_strategy)
    else:
        # REAL-cache mode: no prompt-template ensemble is cached, and the old
        # class-constant 0.5 placeholder left D1 with ZERO input variance
        # (pooled D1/D2/D3 = 0.000 on the n=10 pilot). We use the per-proposal
        # estimators (``proposal_text_variance`` entropy / box-to-support
        # distance) — continuous signals — see change_log.md 2026-08-05.
        norm_text = None
        norm_visual = None

    # REAL-cache mode per-proposal visual variance: mean (1 - cos) between
    # the box feature and the class support set (continuous in [0, 2]),
    # normalized once over the scored proposals with the configured strategy.
    # Like the text term, this replaces the class-constant value that
    # underpowered D2 at C=3 distinct values. The RAW values are kept too:
    # D5 (Taylor-validity sentinel) must be computed on the absolute scale
    # (raw / 2.0) — min-max normalization would spread any distribution
    # across [0, 1] by construction and defeat the clustering flag.
    real_prop_text: Optional[Dict[int, float]] = None
    real_prop_visual: Optional[Dict[int, float]] = None
    real_prop_visual_raw: Optional[Dict[int, float]] = None
    if template_embeddings is None:
        scored = [rec for rec in test_records if rec.class_name in protos]
        real_prop_text = {
            id(r): proposal_text_variance(r.text_similarities) for r in scored
        }
        raw_visual = np.asarray(
            [
                proposal_visual_variance(
                    r.visual_feature, protos[r.class_name].support_features
                )
                for r in scored
            ],
            dtype=float,
        )
        norm_raw_visual = _normalize_proposal_values(raw_visual, norm_strategy)
        real_prop_visual = {
            id(r): float(v) for r, v in zip(scored, norm_raw_visual)
        }
        real_prop_visual_raw = {
            id(r): float(v) for r, v in zip(scored, raw_visual)
        }

    if gate_type == "beta_fallback":
        gate = BetaGate(alpha=alpha, beta=beta, gamma=gamma)
    else:
        gate = ModeAGate(alpha=alpha, beta=beta, gamma=gamma)
    gt_by = _index_gt(ground_truth)

    rows: List[Dict] = []
    for rec in test_records:
        if rec.class_name not in protos:
            continue
        proto = protos[rec.class_name]
        affinity = visual_affinity(rec.visual_feature, proto.centroid)
        s_text = class_text_score(rec)
        s_visual = affinity
        gt_correct = _gt_match(rec, gt_by)
        raw_text = None
        raw_visual_dist = None
        if template_embeddings is not None:
            nt = float(norm_text[rec.class_name])
            nv = float(norm_visual[rec.class_name])
            text_correct = bool(
                np.argmax(rec.text_similarities) == class_index(rec)
            )
        else:
            # REAL-cache mode: per-proposal uncertainties + a NON-tautological
            # text-correctness label. The backbone assigns
            # class_name = argmax(text_similarities), so comparing the argmax
            # to class_name is always True; "text correct" instead means the
            # text modality's top-1 class matches a GT box (IoU >= 0.5) — for
            # the cached proposals this is exactly gt_correct.
            nt = float(real_prop_text[id(rec)])          # type: ignore[index]
            nv = float(real_prop_visual[id(rec)])        # type: ignore[index]
            text_correct = bool(gt_correct)
            # Raw per-proposal values for the honest D5 sentinel (see above).
            raw_text = nt  # entropy is self-normalized to [0, 1]
            raw_visual_dist = float(real_prop_visual_raw[id(rec)])  # type: ignore[index]
        w = gate.weight(nt, nv, affinity)
        fused = fuse_scores(s_text, s_visual, w)
        visual_correct = affinity >= VISUAL_CORRECT_AFFINITY
        rows.append(
            {
                "image_id": rec.image_id,
                "class": rec.class_name,
                "bbox": rec.bbox.tolist(),
                "score": float(rec.score),
                "s_text": float(s_text),
                "s_visual": float(s_visual),
                "affinity": float(affinity),
                "norm_text_var": nt,
                "norm_visual_var": nv,
                "text_entropy": raw_text,       # None in synthetic mode
                "visual_distance_raw": raw_visual_dist,  # None in synthetic mode
                "w": float(w),
                "fused": float(fused),
                "naive": float(fuse_scores(s_text, s_visual, 0.5)),
                "gt_correct": bool(gt_correct),
                "text_correct": text_correct,
                "visual_correct": visual_correct,
            }
        )

    # --- per-method predictions ---------------------------------------------
    preds = {
        "zero_shot_raw": _rank(rows, "score"),
        "text_only": _rank(rows, "s_text"),
        "visual_only": _rank(rows, "s_visual"),
        "naive_average": _rank(rows, "naive"),
        "uadapt_mode_a": _rank(rows, "fused"),
    }
    map50 = {k: compute_map50(v, list(ground_truth)) for k, v in preds.items()}
    per_class_ap = compute_per_class_ap(preds["uadapt_mode_a"], list(ground_truth))
    ceiling = proposal_recall(preds["zero_shot_raw"], list(ground_truth))
    # Oracle re-rank ceiling: rank every GT-correct proposal above every
    # incorrect one. This is the maximum mAP any re-scoring method (incl.
    # U-ADAPT) can reach on this proposal set — the honest transfer bound.
    oracle_preds = [
        {
            "image_id": r["image_id"],
            "class": r["class"],
            "score": 1.0 if r["gt_correct"] else 0.0,
            "bbox": r["bbox"],
        }
        for r in rows
    ]
    oracle_map50 = compute_map50(oracle_preds, list(ground_truth))

    # --- gate weight statistics ----------------------------------------------
    ws = np.asarray([r["w"] for r in rows], dtype=float)
    gate_stats = {
        "n_proposals": int(len(ws)),
        "mean_w": float(ws.mean()) if len(ws) else 0.0,
        "std_w": float(ws.std()) if len(ws) else 0.0,
        "frac_below_0.45": float(np.mean(ws < 0.45)) if len(ws) else 0.0,
        "frac_above_0.55": float(np.mean(ws > 0.55)) if len(ws) else 0.0,
        "frac_in_0.45_0.55": (
            float(np.mean((ws >= 0.45) & (ws <= 0.55))) if len(ws) else 0.0
        ),
    }

    # --- diagnostics D1-D3 (real functions) ----------------------------------
    # D1/D2 correlate each modality's uncertainty with THAT modality's error
    # rate (text_correct / visual_correct), per the demo spec. gt_correct is
    # the IoU-based proposal correctness used for mAP and gap recovery.
    tv = np.asarray([r["norm_text_var"] for r in rows], dtype=float)
    vv = np.asarray([r["norm_visual_var"] for r in rows], dtype=float)
    text_ok = np.asarray([r["text_correct"] for r in rows], dtype=bool)
    visual_ok = np.asarray([r["visual_correct"] for r in rows], dtype=bool)
    correct = np.asarray([r["gt_correct"] for r in rows], dtype=bool)
    w_arr = ws

    text_better = text_ok & ~visual_ok
    visual_better = visual_ok & ~text_ok

    # D1/D2 correctness: the PRE-REGISTERED proposal correctness
    # (IoU >= 0.5 with same-class GT = gt_correct), matching the diagnostics
    # module spec and scripts/04_evaluate.py. The synthetic demo keeps the
    # per-modality flags it was engineered around; on the REAL cache the
    # per-modality labels are degenerate — text_ok == gt_correct by
    # construction (the backbone's class_name is the text argmax) and the
    # affinity threshold (0.65) saturates on RoI-pooled features (every
    # n=10-pilot proposal had affinity >= 0.87), which would leave D2 with a
    # constant correctness array (rho = 0). D3 keeps the per-modality
    # disagreeing subsets either way (change_log.md 2026-08-05).
    if template_embeddings is not None:
        d1 = d1_text_uncertainty_accuracy(tv, text_ok)
        d2 = d2_visual_uncertainty_accuracy(vv, visual_ok)
    else:
        d1 = d1_text_uncertainty_accuracy(tv, correct)
        d2 = d2_visual_uncertainty_accuracy(vv, correct)
    d3 = d3_gate_favorability(w_arr[text_better], w_arr[visual_better])
    diagnostics = {
        "D1_text_uncertainty_accuracy": {
            "summary": d1.summary, "flag": d1.flag, "raw": d1.raw,
        },
        "D2_visual_uncertainty_accuracy": {
            "summary": d2.summary, "flag": d2.flag, "raw": d2.raw,
        },
        "D3_gate_favorability": {"summary": d3.summary, "flag": d3.flag},
    }

    # --- gap recovery ----------------------------------------------------------
    gap = {
        "zero_shot_raw_map50": map50["zero_shot_raw"],
        "uadapt_map50": map50["uadapt_mode_a"],
        "proposal_recall_ceiling": float(ceiling),
        "oracle_rerank_map50": float(oracle_map50),
        "gap_recovery_vs_ceiling": gap_recovery(
            map50["uadapt_mode_a"], map50["zero_shot_raw"], float(ceiling)
        ),
        "gap_recovery_vs_oracle": gap_recovery(
            map50["uadapt_mode_a"], map50["zero_shot_raw"], float(oracle_map50)
        ),
    }
    if zero_shot_reference is not None and transfer_reference is not None:
        # Literature mAP is on the 0-100 scale; our mAP is a 0-1 fraction.
        # Store both, and compute recovery on the percent scale.
        gap["zero_shot_literature"] = float(zero_shot_reference)
        gap["transfer_literature"] = float(transfer_reference)
        gap["gap_recovery_vs_literature"] = gap_recovery(
            map50["uadapt_mode_a"] * 100.0,
            float(zero_shot_reference),
            float(transfer_reference),
        )

    meta = {
        "seed": seed,
        "shots": shots,
        "coefficients": {"alpha": alpha, "beta": beta, "gamma": gamma},
        "n_train_records": len(train_records),
        "n_test_records": len(test_records),
        "n_ground_truth": len(ground_truth),
        "n_scored_proposals": len(rows),
        "classes": classes,
        "norm_strategy": norm_strategy,
        "gate_type": gate_type,
        # Which uncertainty estimators produced the per-proposal variance
        # terms (auditability; change_log.md 2026-08-05).
        "text_uncertainty_estimator": (
            "template_ensemble_variance"
            if template_embeddings is not None
            else "class_similarity_entropy"
        ),
        "visual_uncertainty_estimator": (
            "support_dispersion"
            if template_embeddings is not None
            else "box_to_support_distance"
        ),
    }

    return DemoResults(
        meta=meta,
        map50=map50,
        per_class_ap=per_class_ap,
        diagnostics=diagnostics,
        gate_stats=gate_stats,
        gap_recovery=gap,
        ablation={},
        proposal_level=rows,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalize_class_values(
    values: Dict[str, float], classes: Sequence[str], norm_strategy: str = "min-max"
) -> Dict[str, float]:
    """Scale per-class variance values to [0, 1].

    ``min-max`` (default) uses support-set statistics and maps C classes to
    C distinct values (degenerate for C=2: collapse to {0, 1}); ``absolute``
    divides the raw cosine distance by its fixed max
    (``COSINE_DISTANCE_MAX`` = 2.0) and is invariant to the class count
    (fixes the 2-class degeneracy). Note: under absolute scaling a raw
    ``k1_prior`` of 0.5 (max-entropy ablation) normalizes to 0.25, whereas
    min-max's constant-value special case maps any uniform input to 0.5 —
    consistent with the absolute philosophy (0.5 is a real magnitude, not a
    midpoint), but a behavior difference for k=1 runs.
    """
    classes = list(classes)
    if not classes:
        return {}
    arr = np.asarray([values[c] for c in classes], dtype=float)
    if norm_strategy == "absolute":
        norm = absolute_normalize(arr, scale=COSINE_DISTANCE_MAX)
    elif norm_strategy == "min-max":
        if arr.max() - arr.min() < 1e-12:
            norm = np.full_like(arr, 0.5)
        else:
            norm = min_max_normalize(arr)
    else:
        raise ValueError(
            f"unknown norm_strategy {norm_strategy!r} (choices: min-max, absolute)"
        )
    return {c: float(norm[i]) for i, c in enumerate(classes)}


def _normalize_proposal_values(
    raw: np.ndarray, norm_strategy: str = "min-max"
) -> np.ndarray:
    """Scale per-proposal raw variance values (cosine distances in [0, 2]) to
    [0, 1] with the configured strategy.

    ``absolute`` divides by the fixed cosine-distance max
    (``COSINE_DISTANCE_MAX`` = 2.0, class-count-independent); ``min-max`` uses
    the proposal-level statistics of THIS dataset. Constant or empty inputs
    map to the neutral 0.5 / stay empty so degenerate runs stay finite. The
    per-proposal TEXT estimator (``proposal_text_variance``) is
    self-normalizing (entropy / ln C, already in [0, 1]) and is therefore not
    routed through here — this function exists for the cosine-distance
    terms.
    """
    raw = np.asarray(raw, dtype=float)
    if raw.size == 0:
        return raw
    if norm_strategy == "absolute":
        return absolute_normalize(raw, scale=COSINE_DISTANCE_MAX)
    if norm_strategy == "min-max":
        if np.ptp(raw) < 1e-12:
            return np.full_like(raw, 0.5)
        return min_max_normalize(raw)
    raise ValueError(
        f"unknown norm_strategy {norm_strategy!r} (choices: min-max, absolute)"
    )


def _rank(rows: Sequence[Dict], score_key: str) -> List[Dict]:
    return [
        {"image_id": r["image_id"], "class": r["class"], "score": r[score_key], "bbox": r["bbox"]}
        for r in rows
    ]


def _index_gt(ground_truth: Sequence[Dict]) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    for g in ground_truth:
        out.setdefault(g["image_id"], []).append(g)
    return out


def _gt_match(rec: FeatureRecord, gt_by: Dict[str, List[Dict]]) -> bool:
    for g in gt_by.get(rec.image_id, []):
        if g["class"] == rec.class_name and _iou(rec.bbox, np.asarray(g["bbox"])) >= 0.5:
            return True
    return False
