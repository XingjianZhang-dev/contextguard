#!/usr/bin/env python3
"""Figures 2-5 of the revised paper, drawn from ``results/`` only (no run records needed)."""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import NAMES, RESULTS, read_json  # noqa: E402
from style import (  # noqa: E402
    CG,
    FORECASTER,
    INK2,
    LOSS_STYLE,
    LW,
    PDF_METADATA,
    RIDGE,
    WIDTH,
    ZERO,
    grid,  # noqa: E402
    panel,
    plt,
    zero,
)

OUT = RESULTS / "figures"


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", metadata=PDF_METADATA)
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


def effect(summary: dict, loss: str = "mse"):
    m = summary["metrics"][loss]
    return m["geometric_relative_change_percent"], m["dataset_cluster_bootstrap_ci95_percent"]


def read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def interval(ax, y, e, lo, hi, color, marker="o", size=5.0, lw=1.2, open_marker=False):
    ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle="butt", zorder=2)
    if open_marker:
        ax.plot(e, y, marker=marker, ms=size, mfc="white", mec="#5f5e5a", mew=1.0, zorder=3)
    else:
        ax.plot(e, y, marker=marker, color=color, ms=size, mec="white", mew=0.7, zorder=3)


def figure2() -> None:
    primary = read_json("estimators.json")["primary"]["contextguard_adaptive__vs__topology"]
    cells = read_csv(RESULTS / "cells" / "primary.csv")
    mse = {(r["dataset"], r["model"], r["severity"], r["seed"], r["method"]): float(r["mse"]) for r in cells}
    fig, (a1, a2) = plt.subplots(
        1, 2, figsize=(WIDTH, 2.35), gridspec_kw={"width_ratios": [0.85, 1.35], "wspace": 0.55}
    )
    for i, (loss, name, color, marker) in enumerate(LOSS_STYLE):
        e, (lo, hi) = effect(primary, loss)
        interval(a1, i, e, lo, hi, color, marker, 5.5, 1.4)
        a1.text(hi + 1.5, i, f"{e:.1f}%", va="center", fontsize=7, color=INK2)
    a1.set_yticks(range(3))
    a1.set_yticklabels([x[1] for x in LOSS_STYLE])
    a1.set_ylim(2.6, -0.6)
    a1.set_xlim(-58, 8)
    a1.set_xlabel("Change vs topology (%)")
    zero(a1)
    grid(a1)
    panel(a1, "A", "Overall (160 paired cells, 95% CI)", x=-0.30)
    per = defaultdict(list)
    for (ds, mod, sev, seed, m), v in mse.items():
        if m == "contextguard_adaptive":
            per[(ds, mod)].append(math.log(v / mse[(ds, mod, sev, seed, "topology")]))
    datasets = sorted({d for d, _ in per}, key=lambda d: np.mean(per[(d, "chronos2")] + per[(d, "timesfm25")]))
    for i, ds in enumerate(datasets):
        vals = [100 * (math.exp(np.mean(per[(ds, m)])) - 1) for m in ("chronos2", "timesfm25")]
        a2.plot(vals, [i - 0.13, i + 0.13], color="#c9c8c3", lw=1.0, zorder=1)
        for v, mod, dy in zip(vals, ("chronos2", "timesfm25"), (-0.13, 0.13)):
            c, mk, _ = FORECASTER[mod]
            a2.plot(v, i + dy, marker=mk, color=c, ms=5, mec="white", mew=0.7, zorder=3)
    a2.set_yticks(range(len(datasets)))
    a2.set_yticklabels([NAMES[d] for d in datasets])
    a2.set_ylim(len(datasets) - 0.4, -0.6)
    a2.set_xlim(-75, 2)
    a2.set_xlabel("MSE change vs topology (%)")
    zero(a2)
    grid(a2)
    panel(a2, "B", "By dataset and forecaster", x=-0.36)
    a2.legend(
        handles=[
            Line2D(
                [],
                [],
                marker=FORECASTER[m][1],
                color=FORECASTER[m][0],
                ls="",
                ms=5,
                mec="white",
                label=FORECASTER[m][2],
            )
            for m in ("chronos2", "timesfm25")
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, 0.0),
        handletextpad=0.3,
    )
    save(fig, "fig2_primary")


def figure3() -> None:
    c = read_json("estimators.json")["primary"]
    est = (
        ("contextguard", "contextguard_adaptive", "ContextGuard", CG),
        ("ridge_ungated", "ridge_gated", "Ridge", RIDGE),
        ("lw_ungated", "lw_gated", "Ledoit–Wolf", LW),
    )
    fig, axs = plt.subplots(1, 3, figsize=(WIDTH, 2.3), gridspec_kw={"width_ratios": [1.15, 1.0, 0.9], "wspace": 0.62})
    a = axs[0]
    for i, (u, g, name, col) in enumerate(est):
        eu, (lu, hu) = effect(c[f"{u}__vs__topology"])
        eg, (lg, hg) = effect(c[f"{g}__vs__topology"])
        a.plot([lu, hu], [i - 0.13] * 2, color=col, alpha=0.35, lw=1.2, solid_capstyle="butt")
        a.plot([lg, hg], [i + 0.13] * 2, color=col, lw=1.2, solid_capstyle="butt")
        a.plot(eu, i - 0.13, "o", ms=5, mfc="white", mec=col, mew=1.1, zorder=3)
        a.plot(eg, i + 0.13, "o", ms=5, color=col, mec="white", mew=0.7, zorder=3)
        a.text(-62, i, name, ha="right", va="center", fontsize=7)
    a.set_yticks([])
    a.spines["left"].set_visible(False)
    a.set_ylim(2.55, -0.55)
    a.set_xlim(-60, 0)
    a.set_xlabel("MSE change vs topology (%)")
    grid(a)
    panel(a, "A", "Ungated vs gated", x=-0.33)
    a = axs[1]
    for i, (u, g, name, col) in enumerate(est):
        for k, (loss, _, _, mk) in enumerate(LOSS_STYLE):
            e, (lo, hi) = effect(c[f"{g}__vs__{u}"], loss)
            interval(a, i + (k - 1) * 0.22, e, lo, hi, col, mk, 4.2, 1.0)
    a.set_yticks(range(3))
    a.set_yticklabels([e[2] for e in est])
    a.set_ylim(2.55, -0.55)
    zero(a)
    grid(a)
    a.set_xlabel("Gated vs ungated (%)")
    panel(a, "B", "Effect of the gate", x=-0.48)
    a.legend(
        handles=[Line2D([], [], marker=mk, color="#6f6e6a", ls="", ms=4.2, label=ln) for (_, ln, _, mk) in LOSS_STYLE],
        loc="upper center",
        bbox_to_anchor=(0.45, -0.3),
        ncol=3,
        handletextpad=0.1,
        columnspacing=0.8,
        fontsize=6.5,
    )
    a = axs[2]
    for i, (u, g, name, col) in enumerate(est):
        lu = c[f"{u}__vs__topology"]["metrics"]["mse"]["losses"]
        lg = c[f"{g}__vs__topology"]["metrics"]["mse"]["losses"]
        a.barh(i - 0.17, lu, height=0.3, color=col, alpha=0.35)
        a.barh(i + 0.17, lg, height=0.3, color=col)
        a.text(lu + 0.15, i - 0.17, str(lu), va="center", fontsize=6.5, color=INK2)
        a.text(lg + 0.15, i + 0.17, str(lg), va="center", fontsize=6.5, color=INK2)
    a.set_yticks(range(3))
    a.set_yticklabels(["CG", "Ridge", "LW"])
    a.set_ylim(2.55, -0.55)
    a.set_xlim(0, 7.5)
    grid(a)
    a.set_xlabel("Cells worse than topology")
    panel(a, "C", "Negative transfer", x=-0.25)
    save(fig, "fig3_estimators_gate")


def figure4() -> None:
    d = read_json("determinants.json")
    primary_ch = [u for u in d["primary_channel_level_timesfm25"]["units"] if u["eligible"]]
    held_ch = d["heldout_channel_level"]["units"]
    lo_, hi_ = -0.3, 1.1
    clip = lambda v: float(np.clip(v, lo_, hi_))  # noqa: E731
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 2.75), gridspec_kw={"wspace": 0.32})
    a1.plot([0, 1], [0, 1], color="#a8a7a2", lw=0.7, ls=(0, (3, 2)), zorder=1)
    for u in primary_ch:
        a1.plot(
            clip(u["R2"]),
            clip(u["recovered"]),
            marker="s",
            color=FORECASTER["timesfm25"][0],
            ms=3.6,
            mec="white",
            mew=0.4,
            ls="",
            zorder=3,
        )
    for u in held_ch:
        c, mk, _ = FORECASTER[u["model"]]
        a1.plot(clip(u["R2"]), clip(u["recovered"]), marker=mk, mfc="none", mec=c, mew=0.7, ms=3.6, ls="", zorder=2)
    r_p = d["primary_channel_level_timesfm25"]["spearman_recovered_vs"]["R2"]
    r_h = d["heldout_channel_level"]["spearman_recovered_vs"]["R2"]
    a1.set_xlim(lo_ - 0.03, hi_ - 0.05)
    a1.set_ylim(lo_ - 0.03, hi_ + 0.03)
    grid(a1, "both")
    a1.set_xlabel("Test-context reconstruction $R^2$ of the failed channel")
    a1.set_ylabel("Fraction of dropout loss recovered")
    a1.text(
        0.97,
        0.04,
        f"Spearman ρ = {r_p:.2f} (primary)\nρ = {r_h:.2f} (held-out)",
        transform=a1.transAxes,
        va="bottom",
        ha="right",
        fontsize=7,
        color=INK2,
    )
    panel(a1, "A", "Channel level", x=-0.15)
    a1.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="s",
                color=FORECASTER["timesfm25"][0],
                ls="",
                ms=4,
                mec="white",
                label="Primary, TimesFM 2.5",
            ),
            Line2D(
                [],
                [],
                marker="s",
                mfc="none",
                mec=FORECASTER["timesfm25"][0],
                ls="",
                ms=4,
                label="Held-out, TimesFM 2.5",
            ),
            Line2D(
                [],
                [],
                marker="^",
                mfc="none",
                mec=FORECASTER["tirex"][0],
                ls="",
                ms=4,
                label="Held-out / primary, TiRex",
            ),
        ],
        loc="upper left",
        handletextpad=0.2,
    )
    prim = [u for u in d["primary_dataset_level"]["units"] if u["metric"] == "mse"]
    for u in prim:
        c, mk, _ = FORECASTER[u["model"]]
        a2.plot(u["mean_I"], u["recovered"], marker=mk, color=c, ms=5, mec="white", mew=0.6, ls="", zorder=3)
    held = d["heldout_dataset_level"]["units"] + d["tirex_primary_dataset_level"]["units"]
    for u in held:
        if u["loss_increase"] >= 0.05:
            c, mk, _ = FORECASTER[u["model"]]
            a2.plot(u["mean_I"], clip(u["recovered"]), marker=mk, mfc="none", mec=c, mew=0.8, ms=5, ls="", zorder=2)
    label_at = {("ett2_15T", "chronos2"): "ETT2", ("bizitobs_service", "chronos2"): "BizITObs Service"}
    for u in prim:
        if (u["dataset"], u["model"]) in label_at:
            a2.annotate(
                label_at[(u["dataset"], u["model"])],
                (u["mean_I"], u["recovered"]),
                xytext=(6, -2),
                textcoords="offset points",
                fontsize=6.5,
                color=INK2,
                va="center",
            )
    ett2h = [u for u in d["heldout_dataset_level"]["units"] if u["dataset"] == "ett2_H"]
    a2.annotate(
        "ETT2 (H)",
        (ett2h[0]["mean_I"], np.mean([u["recovered"] for u in ett2h])),
        xytext=(6, -2),
        textcoords="offset points",
        fontsize=6.5,
        color=INK2,
        va="center",
    )
    rp = d["primary_dataset_level"]["spearman_recovered_vs_mean_I"]["mse"]
    rh = d["heldout_dataset_level"]["spearman_recovered_vs_mean_I"]
    a2.set_xlim(-0.02, 1.05)
    a2.set_ylim(-0.05, 1.08)
    grid(a2, "both")
    a2.set_xlabel("Training-only calibration reliability (mean $I_j$)")
    a2.set_ylabel("Fraction of dropout loss recovered")
    a2.text(
        0.03,
        0.97,
        f"Spearman ρ = {rp:.2f} (primary)\nρ = {rh:.2f} (held-out)",
        transform=a2.transAxes,
        va="top",
        fontsize=7,
        color=INK2,
    )
    panel(a2, "B", "Dataset level (filled: primary; open: held-out)", x=-0.15)
    a2.legend(
        handles=[
            Line2D(
                [],
                [],
                marker=FORECASTER[m][1],
                color=FORECASTER[m][0],
                ls="",
                ms=5,
                mec="white",
                label=FORECASTER[m][2],
            )
            for m in FORECASTER
        ],
        loc="lower right",
        handletextpad=0.2,
    )
    save(fig, "fig4_determinants")


def figure5() -> None:
    sev = [
        r for r in read_csv(RESULTS / "summaries" / "severity_and_boundary.csv") if r["corruption"] == "sensor_dropout"
    ]
    ctx = read_csv(RESULTS / "summaries" / "context_sensitivity.csv")
    b = read_json("breadth.json")
    fig, axs = plt.subplots(1, 3, figsize=(WIDTH, 2.3), gridspec_kw={"width_ratios": [1, 1, 1.15], "wspace": 0.45})
    for k, (rows, key, xlabel, title) in enumerate(
        (
            (sev, "missing_rate_percent", "Channels dropped (%)", "Dropout severity"),
            (ctx, "context_length", "Context length", "Context length"),
        )
    ):
        a = axs[k]
        xs = sorted({int(r[key]) for r in rows})
        for mi, (loss, _, col, mk) in enumerate(LOSS_STYLE):
            for xi, x in enumerate(xs):
                r = next(q for q in rows if int(q[key]) == x and q["metric"] == loss)
                e, lo, hi = (
                    float(r[c]) for c in ("geometric_relative_change_percent", "cluster_ci_low", "cluster_ci_high")
                )
                xx = xi + (mi - 1) * 0.2
                a.plot([xx, xx], [lo, hi], color=col, lw=1.0, solid_capstyle="butt")
                a.plot(xx, e, marker=mk, color=col, ms=4.2, mec="white", mew=0.6, zorder=3)
        a.set_xticks(range(len(xs)))
        a.set_xticklabels(xs)
        a.set_xlabel(xlabel)
        a.set_ylim(-72, 5)
        zero(a, vertical=False)
        grid(a, "y")
        panel(a, "AB"[k], title, x=-0.2)
    axs[0].set_ylabel("Change vs topology (%)")
    axs[1].legend(
        handles=[
            Line2D([], [], marker=mk, color=col, ls="", ms=4.2, mec="white", label=nm)
            for (_, nm, col, mk) in LOSS_STYLE
        ],
        loc="upper center",
        bbox_to_anchor=(-0.2, -0.3),
        ncol=3,
        handletextpad=0.1,
        columnspacing=0.8,
        fontsize=6.5,
    )
    a = axs[2]
    native = {"chronos2": b["native_missing"]["heldout_chronos2"], "tirex": b["native_missing"]["heldout_tirex"]}
    for i, mod in enumerate(("chronos2", "timesfm25", "tirex")):
        e, (lo, hi) = effect(b["heldout"][mod])
        interval(a, i - 0.15, e, lo, hi, CG, "o", 4.5, 1.1)
        if mod in native:
            e2, (lo2, hi2) = effect(native[mod]["native_vs_topology"])
            interval(a, i + 0.17, e2, lo2, hi2, ZERO, "o", 4.5, 1.1, open_marker=True)
        else:
            a.text(-68, i + 0.17, "native path raises an error", va="center", fontsize=6.3, color=INK2, style="italic")
    a.set_yticks(range(3))
    a.set_yticklabels([FORECASTER[m][2] for m in ("chronos2", "timesfm25", "tirex")])
    a.set_ylim(2.6, -0.6)
    a.set_xlim(-70, 25)
    zero(a)
    grid(a)
    a.set_xlabel("MSE change vs topology (%)")
    panel(a, "C", "Native missing-value input", x=-0.42)
    a.legend(
        handles=[
            Line2D([], [], marker="o", color=CG, ls="", ms=4.5, mec="white", label="ContextGuard"),
            Line2D([], [], marker="o", mfc="white", mec="#5f5e5a", ls="", ms=4.5, label="Native NaN input"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.45, -0.3),
        ncol=2,
        handletextpad=0.1,
        columnspacing=0.8,
        fontsize=6.5,
    )
    save(fig, "fig5_robustness_native")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (figure2, figure3, figure4, figure5):
        f()
        print(f"{f.__name__} written")


if __name__ == "__main__":
    main()
