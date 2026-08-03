"""Publication-quality figures for the supervisor demo (Figure 1-6).

Each function draws one figure onto a matplotlib ``axes`` and returns a short
annotation string; the notebook calls these and saves the PNGs. All figures
work from the JSON produced by scripts/demo_mode_a_end_to_end.py
(results.json + proposal_level.json).

Figure list (proposal §supervisor demo):
  Fig 1  gate weight distribution (is the gate dynamic?)
  Fig 2  D1/D2 uncertainty-accuracy correlation (core assumption)
  Fig 3  D3 gate favorability (is the gate useful?)
  Fig 4  gap recovery analysis (zero-shot | U-ADAPT | transfer)
  Fig 5  qualitative examples (schematic; no real imagery in the demo)
  Fig 6  coefficient ablation (alpha/beta/gamma contributions)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def figure1_gate_weights(proposals: Sequence[Dict], ax) -> str:
    """Histogram of gate weights w, color-coded by which modality was correct."""
    ws = np.asarray([p["w"] for p in proposals], dtype=float)
    text_ok = np.asarray([p["text_correct"] for p in proposals], dtype=bool)
    visual_ok = np.asarray([p["visual_correct"] for p in proposals], dtype=bool)

    ax.hist(ws, bins=24, range=(0, 1), color="#cfd8dc", edgecolor="white", label="all")
    ax.hist(ws[text_ok], bins=24, range=(0, 1), histtype="step", lw=2,
            color="#1f77b4", label="text correct")
    ax.hist(ws[visual_ok], bins=24, range=(0, 1), histtype="step", lw=2,
            color="#ff7f0e", label="visual correct")
    ax.axvline(0.5, color="black", ls="--", lw=1, label="w = 0.5 (naive)")
    ax.set_xlabel("gate weight w  (1 => trust visual, 0 => trust text)")
    ax.set_ylabel("proposals")
    ax.set_title("Figure 1 — Gate weight distribution (dynamic, not stuck at 0.5)")
    ax.legend(fontsize=8, loc="upper right")

    frac_stuck = float(np.mean((ws >= 0.45) & (ws <= 0.55))) if len(ws) else 0.0
    return (
        f"mean w = {ws.mean():.3f}, std = {ws.std():.3f}; "
        f"{frac_stuck * 100:.1f}% of proposals within ±0.05 of 0.5"
    )


def figure2_d1_d2(diagnostics: Dict, ax) -> str:
    """D1/D2: binned uncertainty vs error rate scatter + Spearman rho."""
    for pos, key, xlab, color in (
        (1, "D1_text_uncertainty_accuracy", "normalized text variance", "#1f77b4"),
        (2, "D2_visual_uncertainty_accuracy", "normalized visual variance", "#ff7f0e"),
    ):
        sub = ax[pos - 1] if hasattr(ax, "__len__") else ax
        d = diagnostics[key]
        mids = np.asarray(d["raw"]["bin_mid"], dtype=float)
        err = np.asarray(d["raw"]["error_rate"], dtype=float)
        rho = d["summary"].get("spearman_rho", float("nan"))
        sub.scatter(mids, err, s=60, color=color, edgecolor="white", zorder=3)
        sub.plot(mids, err, color=color, lw=1, alpha=0.5)
        sub.axhline(0.5, color="grey", ls=":", lw=1)
        sub.set_xlabel(xlab)
        sub.set_ylabel("proposal error rate")
        sub.set_ylim(0, 1.05)
        sub.set_title(f"{key}  (Spearman rho = {rho:+.3f})", fontsize=10)
    ax[0].set_title("Figure 2 — Uncertainty–accuracy correlation (D1/D2)", fontsize=11)
    return "D1 rho + D2 rho reported on each subplot"


def figure3_gate_favorability(diagnostics: Dict, ax) -> str:
    """D3: % cases where the gate favored the more accurate modality."""
    d = diagnostics["D3_gate_favorability"]["summary"]
    frac = d.get("favorability_fraction", float("nan"))
    p = d.get("binomial_pvalue", float("nan"))
    n = int(d.get("n", 0))

    ax.bar(["gate favorability"], [frac * 100], color="#2ca02c", width=0.45)
    ax.axhline(50, color="black", ls="--", lw=1, label="chance (50%)")
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of disagreeing cases")
    ax.set_title(f"Figure 3 — Gate favorability (D3)  n={n}")
    ax.legend(fontsize=8)
    ax.text(0, frac * 100 + 2, f"{frac * 100:.1f}%  (p = {p:.3g})",
            ha="center", fontsize=10)
    return f"favorability {frac * 100:.1f}% over n={n} disagreeing cases (binomial p={p:.3g})"


def figure4_gap_recovery(gap: Dict, ax) -> str:
    """Fig 4: zero-shot | U-ADAPT | transfer ceiling bars + gap recovery %.

    All bars are on the 0-1 mAP scale. Literature references are stored on
    the percent scale (e.g. 27.5 / 65.6); they are divided by 100 here so the
    dashed reference lines sit correctly on the same axis.
    """
    zero = gap.get("zero_shot_raw_map50", 0.0)
    adapt = gap.get("uadapt_map50", 0.0)
    ceiling = gap.get("oracle_rerank_map50", gap.get("proposal_recall_ceiling", 0.0))
    lit_ceiling = gap.get("transfer_literature")
    lit_zero = gap.get("zero_shot_literature")

    labels = ["Zero-shot\n(raw scores)", "U-ADAPT\nMode A", "Transfer\n(oracle re-rank)"]
    heights = [zero, adapt, ceiling]
    colors = ["#d62728", "#1f77b4", "#2ca02c"]
    bars = ax.bar(labels, heights, color=colors, width=0.5, alpha=0.9)
    for bar, h in zip(bars, heights):
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}",
                ha="center", fontsize=10)

    rec = gap.get("gap_recovery_vs_oracle", gap.get("gap_recovery_vs_ceiling"))
    if rec is not None:
        ax.annotate(
            f"gap recovery\n{rec * 100:.0f}%",
            xy=(1, adapt), xytext=(1.4, (ceiling + zero) / 2),
            fontsize=10, color="#1f77b4",
            arrowprops=dict(arrowstyle="->", color="#1f77b4"),
        )
    # Literature references (percent scale -> fraction on the 0-1 axis).
    lit_ceiling_f = lit_ceiling / 100.0 if lit_ceiling is not None else None
    lit_zero_f = lit_zero / 100.0 if lit_zero is not None else None
    if lit_ceiling_f is not None and lit_zero_f is not None:
        ax.axhline(lit_ceiling_f, color="#2ca02c", ls=":", lw=1.5)
        ax.text(2.35, lit_ceiling_f + 0.01, f"literature transfer {lit_ceiling:.1f}%",
                fontsize=8, color="#2ca02c", ha="right")
        ax.axhline(lit_zero_f, color="#d62728", ls=":", lw=1.5)
        ax.text(2.35, lit_zero_f + 0.01, f"literature zero-shot {lit_zero:.1f}%",
                fontsize=8, color="#d62728", ha="right")
    top_vals = list(heights)
    if lit_ceiling_f is not None:
        top_vals.append(lit_ceiling_f)
    ax.set_ylim(0, max(top_vals) * 1.25 + 0.05)
    ax.set_ylabel("mAP50")
    ax.set_title("Figure 4 — Gap recovery analysis")
    return f"gap recovery vs oracle = {rec * 100:.1f}%" if rec is not None else ""


def figure5_qualitative(
    proposals: Sequence[Dict],
    ground_truth: Sequence[Dict],
    ax,
    seed: int = 0,
    image_paths: Optional[Dict[str, str]] = None,
) -> str:
    """Fig 5: qualitative examples — high w, low w, and a gate-corrected case.

    When ``image_paths`` maps a proposal's ``image_id`` to an existing image
    file (real cached mode), the panel shows the REAL detection image with GT
    (green) and proposal (blue) boxes. Otherwise a deterministic stylized
    scene is drawn (synthetic-mode fallback), so the figure always renders.
    """
    rng = np.random.default_rng(seed)
    gt_by: Dict[str, List[Dict]] = {}
    for g in ground_truth:
        gt_by.setdefault(g["image_id"], []).append(g)

    resolved = _resolve_image_paths(image_paths)

    def _pick(cond) -> Optional[Dict]:
        # Prefer a matching proposal whose image is resolvable (real
        # detections); fall back to any matching proposal (schematic).
        for p in proposals:
            if cond(p) and p["image_id"] in resolved:
                return p
        for p in proposals:
            if cond(p):
                return p
        return None

    cases = [
        ("High w — gate trusts visual (low visual uncertainty / high affinity)",
         lambda p: p["w"] > 0.7 and p["visual_correct"]),
        ("Low w — gate trusts text (high visual uncertainty / low affinity)",
         lambda p: p["w"] < 0.3 and p["text_correct"]),
        ("Gate corrected naive averaging (modality disagreement)",
         lambda p: abs(p["w"] - 0.5) > 0.25 and p["text_correct"] != p["visual_correct"]),
    ]

    used_real = False
    axes = ax if hasattr(ax, "__len__") else [ax]
    for panel, (title, cond) in zip(axes, cases):
        p = _pick(cond)
        if p is None:
            panel.set_title(title + "\n(no example found)", fontsize=8)
            panel.axis("off")
            continue
        img_path = resolved.get(p["image_id"])
        gt = gt_by.get(p["image_id"], [])
        if img_path is not None:
            try:
                _draw_real_detection(panel, img_path, p, gt)
                used_real = True
            except Exception:
                # Corrupt/unreadable image: fall back to the schematic panel
                # rather than killing the whole figure render.
                panel.cla()
                _draw_schematic(panel, rng, p, gt)
        else:
            _draw_schematic(panel, rng, p, gt)
        panel.set_title(title, fontsize=8)
    axes[0].set_title(
        "Figure 5 — Qualitative examples (real detections)" if used_real
        else "Figure 5 — Qualitative examples (schematic)",
        fontsize=11,
    )
    if used_real:
        return "real detections shown for high-w, low-w, and gate-corrected cases"
    return "high-w, low-w, and gate-corrected cases shown schematically"


def _resolve_image_paths(image_paths: Optional[Dict[str, str]]) -> Dict[str, "Path"]:
    """Return {image_id: Path} for entries whose file actually exists."""
    from pathlib import Path

    if not image_paths:
        return {}
    out: Dict[str, Path] = {}
    for image_id, p in image_paths.items():
        path = Path(p)
        if path.exists():
            out[image_id] = path
    return out


def _draw_real_detection(panel, img_path, p: Dict, gt: Sequence[Dict]) -> None:
    """Render the real image with GT + proposal boxes (native pixel coords)."""
    img = plt.imread(str(img_path))
    panel.imshow(img)
    panel.set_xlim(0, img.shape[1])
    panel.set_ylim(img.shape[0], 0)
    panel.set_xticks([])
    panel.set_yticks([])
    _draw_boxes_and_text(panel, p, gt)


def _draw_schematic(panel, rng: np.random.Generator, p: Dict, gt: Sequence[Dict]) -> None:
    """Deterministic stylized scene (synthetic-mode fallback)."""
    panel.set_facecolor("#f5f5f5")
    panel.set_xlim(0, 512)
    panel.set_ylim(512, 0)
    panel.set_xticks([])
    panel.set_yticks([])
    for _ in range(8):  # deterministic stylized scene blocks
        x, y = rng.uniform(0, 430, 2)
        w, h = rng.uniform(20, 90, 2)
        panel.add_patch(plt.Rectangle((x, y), w, h, facecolor="#d7d7d7",
                                      edgecolor="none"))
    _draw_boxes_and_text(panel, p, gt)


def _draw_boxes_and_text(panel, p: Dict, gt: Sequence[Dict]) -> None:
    """GT boxes (green) + proposal box (blue) + per-proposal gate annotation."""
    for g in gt:
        b = np.asarray(g["bbox"])
        panel.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                      fill=False, edgecolor="#2ca02c", lw=2.5))
    b = np.asarray(p["bbox"])
    panel.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                  fill=False, edgecolor="#1f77b4", lw=2.5))
    panel.text(
        5, 20,
        f"{p['class']}  w={p['w']:.2f}  s_text={p['s_text']:.2f} "
        f"s_vis={p['s_visual']:.2f}\n"
        f"text_ok={p['text_correct']}  visual_ok={p['visual_correct']}",
        fontsize=8, va="top",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
    )


def figure6_ablation(ablation: Dict[str, float], ax) -> str:
    """Fig 6: full gate vs alpha=0 / beta=0 / gamma=0 (mAP50)."""
    labels = ["Full\n(1,1,1)", "alpha=0\n(no vis. unc.)", "beta=0\n(no text unc.)",
              "gamma=0\n(no affinity)"]
    keys = ["full", "alpha0", "beta0", "gamma0"]
    vals = [ablation.get(k, 0.0) for k in keys]
    colors = ["#1f77b4", "#d62728", "#d62728", "#d62728"]
    bars = ax.bar(labels, vals, color=colors, width=0.55)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.3f}",
                ha="center", fontsize=9)
    ax.set_ylim(0, max(vals + [0.1]) * 1.15 + 0.02)
    ax.set_ylabel("mAP50")
    ax.set_title("Figure 6 — Coefficient ablation (each component's contribution)")
    return "ablation mAP50: " + ", ".join(f"{k}={v:.3f}" for k, v in zip(keys, vals))


def render_all_figures(
    results: Dict,
    proposals: Sequence[Dict],
    ground_truth: Sequence[Dict],
    out_dir: str,
    image_paths: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Render and save Figures 1-6 to ``out_dir``. Returns saved paths.

    ``image_paths`` (optional {image_id: path}) enables real-detection panels
    in Figure 5; missing/unresolvable files fall back to schematic panels.
    """
    import os

    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    saved: List[str] = []

    specs = [
        ("figure1_gate_weights.png", lambda ax: figure1_gate_weights(proposals, ax)),
        ("figure2_d1_d2.png", lambda ax: figure2_d1_d2(results["diagnostics"], ax)),
        ("figure3_gate_favorability.png", lambda ax: figure3_gate_favorability(results["diagnostics"], ax)),
        ("figure4_gap_recovery.png", lambda ax: figure4_gap_recovery(results["gap_recovery"], ax)),
        ("figure5_qualitative.png", lambda ax: figure5_qualitative(
            proposals, ground_truth, ax, image_paths=image_paths)),
        ("figure6_ablation.png", lambda ax: figure6_ablation(results["ablation"], ax)),
    ]
    for fname, fn in specs:
        if fname == "figure2_d1_d2.png":
            fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        elif fname == "figure5_qualitative.png":
            fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        else:
            fig, ax = plt.subplots(figsize=(7, 4.2))
            axes = ax
        fn(axes)
        fig.tight_layout()
        path = os.path.join(out_dir, fname)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
    return saved
