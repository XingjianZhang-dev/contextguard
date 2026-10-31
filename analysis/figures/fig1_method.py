#!/usr/bin/env python3
"""Figure 1: ContextGuard's training and inference paths.

Panel B shows the absolute correlation matrix of the ETT1 training covariance
(``results/summaries/training_correlations.json``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import RESULTS  # noqa: E402
from style import CG, INK, INK2, PDF_METADATA, plt  # noqa: E402

CGF, GR = "#eef4fc", "#8a8984"
Rm = np.asarray(json.loads((RESULTS / "summaries" / "training_correlations.json").read_text())["ett1_15T"])
plt.rcParams.update({"savefig.pad_inches": 0.03})
fig = plt.figure(figsize=(6.85, 3.15))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 137)
ax.set_ylim(0, 63)
ax.axis("off")


def box(x, y, w, h, fc=CGF, ec=CG, lw=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.4", fc=fc, ec=ec, lw=lw))


def arrow(p0, p1, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            p0, p1, arrowstyle="-|>", mutation_scale=9, lw=1.0, color="#6f6e6a", connectionstyle=f"arc3,rad={rad}"
        )
    )


def head(x, y, letter, title):
    ax.text(x, y, letter, fontsize=8.5, fontweight="bold", color=INK, va="top")
    ax.text(x + 3.2, y, title, fontsize=8, fontweight="bold", color=INK, va="top")


# row tags
for y, t in ((47, "Training\nhistory"), (16, "Frozen\ninference")):
    ax.add_patch(
        FancyBboxPatch((0.6, y - 12), 9.4, 24, boxstyle="round,pad=0.3,rounding_size=1.4", fc="#2f3b52", ec="none")
    )
    ax.text(5.3, y, t, color="white", fontsize=7.5, fontweight="bold", ha="center", va="center", linespacing=1.15)
# ---- row 1
box(12.5, 35.5, 33, 23.5)
head(14, 58, "A", "Chronological split")
ax.add_patch(Rectangle((15, 45), 22, 4, fc=CG, ec="none"))
ax.add_patch(Rectangle((37.3, 45), 5.5, 4, fc="#86b6ef", ec="none"))
ax.text(26, 43.6, "fit (80%)", fontsize=6.8, ha="center", va="top", color=INK2)
ax.text(40, 43.6, "calibrate", fontsize=6.8, ha="center", va="top", color=INK2)
ax.text(15, 38.3, "Test windows are never used", fontsize=6.8, color=INK2)
arrow((46.3, 47), (49.2, 47))
box(50, 35.5, 33, 23.5)
head(51.5, 58, "B", "Cross-channel structure")
iax = fig.add_axes([52.0 / 137, 38.0 / 63, 10.0 / 137, 13.5 / 63])
iax.imshow(np.abs(Rm), cmap="Blues", vmin=0, vmax=1)
iax.set_xticks([])
iax.set_yticks([])
for s_ in iax.spines.values():
    s_.set_visible(False)
ax.text(64.0, 49.5, r"pooled mean $\mu$", fontsize=6.9, color=INK)
ax.text(64.0, 45.3, r"pairwise cov. $\Sigma$", fontsize=6.9, color=INK)
ax.text(64.0, 39.5, "estimated once for\nall forecasters", fontsize=6.5, color=INK2, linespacing=1.1)
arrow((83.8, 47), (86.7, 47))
box(87.5, 35.5, 33, 23.5)
head(89, 58, "C", "Calibration gate")
ax.text(89.5, 50.5, r"$\hat a_j=\mathrm{clip}_{[0,1]}\,(p_j^{\top}y_j\,/\,p_j^{\top}p_j)$", fontsize=7.2, color=INK)
ax.text(89.5, 45.3, r"$g_j=\hat a_j$  if  $I_j\geq\tau$, else $0$", fontsize=7.2, color=INK)
ax.text(89.5, 39.0, "zero weight is always feasible", fontsize=6.6, color=INK2)
box(124, 39, 11.5, 16, fc="#f3f2ef", ec=GR)
ax.text(129.75, 51.5, "Stored", fontsize=7, ha="center", fontweight="bold", color=INK)
ax.text(129.75, 47.0, r"$\mu,\ \Sigma,\ g$", fontsize=7.2, ha="center", color=INK)
ax.text(129.75, 42.3, "0 trainable\nparameters", fontsize=6.2, ha="center", va="center", color=INK2, linespacing=1.05)
arrow((121.3, 47), (123.4, 47))
ax.add_patch(
    FancyArrowPatch(
        (129.75, 38.4),
        (78.0, 28.4),
        arrowstyle="-|>",
        mutation_scale=9,
        lw=0.9,
        color="#8a8984",
        ls=(0, (3, 2)),
        connectionstyle="arc3,rad=-0.12",
    )
)
# ---- row 2
box(12.5, 4.5, 33, 23.5)
head(14, 27, "D", "Missingness geometry")
rows = [("clean context", "identity"), ("short gaps", "temporal fill"), ("channel absent", "gated repair")]
for k, (a, c) in enumerate(rows):
    y = 20 - k * 5.4
    ax.text(15, y, a, fontsize=6.9, color=INK, va="center")
    ax.text(30.5, y, r"$\rightarrow$", fontsize=7, color=GR, va="center")
    ax.text(
        33.2, y, c, fontsize=6.9, color=CG if k == 2 else INK, va="center", fontweight="bold" if k == 2 else "normal"
    )
arrow((46.3, 16), (49.2, 16))
box(50, 4.5, 33, 23.5)
head(51.5, 27, "E", "Gated conditional repair")
ax.text(51.8, 19.6, r"$p_j=\mu_j+\Sigma_{j,O}(\Sigma_{O,O}+\lambda I)^{-1}(x_O-\mu_O)$", fontsize=6.6, color=INK)
ax.text(51.8, 14.2, r"$\tilde x_j=(1-g_j)\,x_j^{\mathrm{ffill}}+g_j\,p_j$", fontsize=7.2, color=INK)
ax.text(51.8, 8.0, "observed entries restored exactly", fontsize=6.6, color=INK2)
arrow((83.8, 16), (86.7, 16))
box(87.5, 4.5, 33, 23.5, fc="#f3f2ef", ec=GR)
head(89, 27, "F", "Unchanged forecaster")
ax.text(104, 18.0, "Chronos-2  ·  TimesFM 2.5  ·  TiRex", fontsize=7, ha="center", color=INK)
ax.text(104, 12.5, "weights and interface frozen", fontsize=6.6, ha="center", color=INK2)
xs = np.linspace(91, 117, 60)
base = 7.6 + 0.9 * np.sin((xs - 91) / 3.2)
ax.plot(xs[:35], base[:35], color=INK, lw=0.9)
ax.fill_between(
    xs[34:], base[34:] - np.linspace(0, 1.6, 26), base[34:] + np.linspace(0, 1.6, 26), color="#86b6ef", alpha=0.6, lw=0
)
ax.plot(xs[34:], base[34:], color=CG, lw=1.0)
ax.text(129.75, 16, "probabilistic\nforecast", fontsize=6.8, ha="center", va="center", color=INK, linespacing=1.1)
arrow((121.3, 16), (124.6, 16))
OUT = RESULTS / "figures"
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig1_method.pdf", metadata=PDF_METADATA)
fig.savefig(OUT / "fig1_method.png", dpi=200)
print("fig1 written")
