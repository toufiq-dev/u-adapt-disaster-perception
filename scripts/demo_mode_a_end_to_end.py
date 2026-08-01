#!/usr/bin/env python
"""demo_mode_a_end_to_end.py — supervisor demo: Mode A end-to-end on a subset.

Runs the full U-ADAPT Mode A pipeline on a small subset (50-100 images) and
compares against the pre-registered baselines:

    * zero-shot raw detector scores (rec.score)
    * text-only        (w = 0)
    * visual-only      (w = 1)
    * naive averaging  (w = 0.5, T-Rex2-style surrogate)
    * U-ADAPT Mode A   (analytic gate, alpha=beta=gamma=1, T=1)

Outputs (JSON + console table):
    * outputs/supervisor_demo/results.json          (mAP50, per-class AP,
                                                     gap recovery, D1-D3)
    * outputs/supervisor_demo/proposal_level.json   (per-proposal rows for
                                                     the notebook figures)
    * outputs/supervisor_demo/figures/*.png         (optional --figures)

Data source (automatic): if a real feature cache exists at
``--cache-dir`` (train + test splits) AND ``--ground-truth`` is provided, the
real cached features are used. Otherwise a deterministic SYNTHETIC world
(seed=0) is generated — a mechanism demo, not a research result.

Usage:
    # synthetic demo (default, no data required)
    python scripts/demo_mode_a_end_to_end.py --out outputs/supervisor_demo/results.json

    # real cached features (when available after Milestone 1)
    python scripts/demo_mode_a_end_to_end.py \
        --cache-dir cached_features \
        --ground-truth data/annotations/dfire_test.json \
        --dataset-config configs/datasets/dfire.yaml

Everything is seeded (--seed 0) for full reproducibility.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("demo_mode_a")

# Pre-registered coefficient ablations for Figure 6 (proposal §8).
ABLATION_COEFFICIENTS = {
    "full": {"alpha": 1.0, "beta": 1.0, "gamma": 1.0},
    "alpha0": {"alpha": 0.0, "beta": 1.0, "gamma": 1.0},
    "beta0": {"alpha": 1.0, "beta": 0.0, "gamma": 1.0},
    "gamma0": {"alpha": 1.0, "beta": 1.0, "gamma": 0.0},
}


def load_yaml(path: Path) -> dict:
    import yaml

    with open(path) as fh:
        return yaml.safe_load(fh)


def load_json(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _coco_to_gt(coco: dict) -> list:
    """COCO annotations -> shared GT schema {image_id, class, bbox xyxy}."""
    cat_name = {c["id"]: c["name"] for c in coco.get("categories", [])}
    out = []
    for ann in coco.get("annotations", []):
        b = ann["bbox"]
        out.append(
            {
                "image_id": str(ann["image_id"]),
                "class": cat_name.get(ann["category_id"], "?"),
                "bbox": [b[0], b[1], b[0] + b[2], b[1] + b[3]],
            }
        )
    return out


def _subset_by_image(records, n_images: int, seed: int):
    """Keep only proposals from the first ``n_images`` distinct image ids."""
    seen: list = []
    idx = []
    for i, r in enumerate(records):
        if r.image_id not in seen:
            seen.append(r.image_id)
        if len(seen) > n_images:
            break
        idx.append(i)
    return [records[i] for i in idx], seen[:n_images]


def _load_real_data(cache_dir: Path, dataset_config: Path, ground_truth: Path,
                    n_images: int, seed: int):
    """Load real cached features + COCO GT (when available after Milestone 1)."""
    from uadapt.features.cache_engine import load_cache

    train = load_cache(cache_dir, split="train")
    test = load_cache(cache_dir, split="test")
    cfg = load_yaml(dataset_config)
    classes = list(cfg.get("retained_classes") or cfg["classes"])
    gt = _coco_to_gt(load_json(ground_truth))
    train, _ = _subset_by_image(train, n_images, seed)
    test, _ = _subset_by_image(test, n_images, seed)
    # Restrict GT to the subset's images.
    test_ids = {r.image_id for r in test}
    gt = [g for g in gt if g["image_id"] in test_ids]
    logger.info("real cache: %d train / %d test records, %d GT boxes",
                len(train), len(test), len(gt))
    return train, test, gt, classes, None


def _load_synthetic(n_images: int, seed: int, classes):
    from uadapt.demo.synthetic_data import generate_synthetic_dataset

    ds = generate_synthetic_dataset(classes=classes, seed=seed, n_test_images=n_images)
    logger.info("synthetic world: %d train / %d test records, %d GT boxes (seed=%d)",
                len(ds.train_records), len(ds.test_records), len(ds.ground_truth), seed)
    return ds.train_records, ds.test_records, ds.ground_truth, ds.classes, ds.template_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervisor demo: Mode A end-to-end.")
    parser.add_argument("--cache-dir", default="cached_features", type=Path)
    parser.add_argument("--dataset-config", default="configs/datasets/dfire.yaml", type=Path)
    parser.add_argument("--ground-truth", type=Path, help="COCO annotations JSON (real mode)")
    parser.add_argument("--shots", type=int, default=5, help="k support examples (1/3/5)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-test-images", type=int, default=80, help="subset size (50-100)")
    parser.add_argument("--classes", nargs="*", default=None, help="override class list")
    parser.add_argument("--out", default="outputs/supervisor_demo/results.json", type=Path)
    parser.add_argument("--proposal-out", default=None, type=Path)
    parser.add_argument("--figures-dir", default=None, type=Path,
                        help="also render Figures 1-6 here")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    from uadapt.demo.pipeline import run_demo

    use_real = (
        (args.cache_dir / "train" / "manifest.json").exists()
        and (args.cache_dir / "test" / "manifest.json").exists()
        and args.ground_truth is not None
    )
    if use_real:
        train, test, gt, classes, embeddings = _load_real_data(
            args.cache_dir, args.dataset_config, args.ground_truth,
            args.n_test_images, args.seed,
        )
    else:
        train, test, gt, classes, embeddings = _load_synthetic(
            args.n_test_images, args.seed, args.classes
        )

    # Literature gap-recovery references from the dataset config (if present),
    # in BOTH synthetic and real modes.
    cfg = load_yaml(args.dataset_config)
    zero_ref = cfg.get("meta", {}).get("zero_shot_map50_gdino")
    transfer_ref = cfg.get("meta", {}).get("transfer_map50_gdino")

    results = run_demo(
        train_records=train,
        test_records=test,
        ground_truth=gt,
        classes=classes,
        template_embeddings=embeddings,
        shots=args.shots,
        seed=args.seed,
        zero_shot_reference=zero_ref,
        transfer_reference=transfer_ref,
    )

    # Coefficient ablation (Figure 6).
    for key, coeff in ABLATION_COEFFICIENTS.items():
        ab = run_demo(
            train_records=train,
            test_records=test,
            ground_truth=gt,
            classes=classes,
            template_embeddings=embeddings,
            shots=args.shots,
            seed=args.seed,
            **coeff,
        )
        results.ablation[key] = ab.map50["uadapt_mode_a"]

    results.meta["data_source"] = "synthetic" if embeddings is not None else "real"
    results.meta["n_test_images"] = args.n_test_images

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results.to_dict(), fh, indent=2)
    logger.info("wrote %s", args.out)

    proposal_out = args.proposal_out or (args.out.parent / "proposal_level.json")
    with open(proposal_out, "w") as fh:
        json.dump(
            {"proposals": results.proposal_level, "ground_truth": list(gt)},
            fh,
            indent=2,
        )
    logger.info("wrote %s", proposal_out)

    _print_table(results)

    if not args.no_figures:
        from uadapt.demo.plotting import render_all_figures

        fig_dir = args.figures_dir or (args.out.parent / "figures")
        saved = render_all_figures(
            results.to_dict(), results.proposal_level, list(gt), str(fig_dir)
        )
        logger.info("figures -> %s", ", ".join(saved))


def _print_table(results) -> None:
    """Console table of mAP50 for every method + diagnostics summary."""
    from uadapt.demo.pipeline import DemoResults

    if not isinstance(results, DemoResults):
        return
    rows = [
        ("zero_shot_raw", "Zero-shot (raw detector scores)"),
        ("text_only", "Text-only        (w = 0)"),
        ("visual_only", "Visual-only      (w = 1)"),
        ("naive_average", "Naive averaging  (w = 0.5)"),
        ("uadapt_mode_a", "U-ADAPT Mode A   (analytic gate)"),
    ]
    print("\n" + "=" * 62)
    print("U-ADAPT supervisor demo — mAP50 (subset)")
    print("=" * 62)
    base = results.map50.get("zero_shot_raw", 0.0)
    for key, label in rows:
        v = results.map50.get(key, float("nan"))
        delta = v - base if key != "zero_shot_raw" else 0.0
        print(f"  {label:32s} {v:6.3f}   (d{delta:+6.3f})")
    print("-" * 62)
    gs = results.gate_stats
    print(f"  gate w: mean={gs['mean_w']:.3f} std={gs['std_w']:.3f} "
          f"| {gs['frac_below_0.45'] * 100:.0f}% < 0.45, "
          f"{gs['frac_above_0.55'] * 100:.0f}% > 0.55, "
          f"{gs['frac_in_0.45_0.55'] * 100:.0f}% near 0.5")
    for key in ("D1_text_uncertainty_accuracy", "D2_visual_uncertainty_accuracy",
                "D3_gate_favorability"):
        d = results.diagnostics.get(key, {})
        s = d.get("summary", {})
        if "spearman_rho" in s:
            print(f"  {key}: rho={s['spearman_rho']:+.3f}")
        else:
            print(f"  {key}: {s}")
    print("=" * 62)


if __name__ == "__main__":
    main()
