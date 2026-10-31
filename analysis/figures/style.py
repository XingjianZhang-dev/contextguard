"""Shared figure style: 6.85 in text width, Liberation Sans 7-8 pt, validated categorical palette.

ContextGuard is drawn in one blue family throughout; the three forecasters use distinct hues and
markers so that every figure stays readable in grayscale and for color-vision deficiencies.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

WIDTH = 6.85
INK, INK2, GRID, ZERO = "#0b0b0b", "#52514e", "#e6e5e1", "#8a8984"
CG, CG_DARK, CG_LIGHT = "#2a78d6", "#184f95", "#86b6ef"
RIDGE, LW = "#eb6834", "#1baf7a"
FORECASTER = {
    "chronos2": ("#6250d6", "o", "Chronos-2"),
    "timesfm25": ("#e87ba4", "s", "TimesFM 2.5"),
    "tirex": ("#008300", "^", "TiRex"),
}
PDF_METADATA = {"CreationDate": None}  # byte-reproducible PDFs
LOSS_STYLE = (("mse", "MSE", CG_DARK, "o"), ("mae", "MAE", CG, "s"), ("quantile_loss", "Quantile loss", CG_LIGHT, "D"))

plt.rcParams.update(
    {
        "font.family": "Liberation Sans",
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#3d3d3a",
        "xtick.color": "#3d3d3a",
        "ytick.color": "#3d3d3a",
        "axes.labelcolor": INK,
        "text.color": INK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "legend.frameon": False,
        "lines.solid_capstyle": "round",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Liberation Sans",
        "mathtext.it": "Liberation Sans:italic",
        "mathtext.bf": "Liberation Sans:bold",
        "mathtext.sf": "Liberation Sans",
        "mathtext.cal": "Liberation Sans",
    }
)


def panel(ax, letter: str, title: str | None = None, x: float = -0.02) -> None:
    ax.text(x, 1.04, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")
    if title:
        ax.text(0.0, 1.04, title, transform=ax.transAxes, fontsize=8, va="bottom", ha="left", color=INK)


def grid(ax, axis: str = "x") -> None:
    ax.grid(axis=axis, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def zero(ax, vertical: bool = True) -> None:
    (ax.axvline if vertical else ax.axhline)(0, color=ZERO, lw=0.6, zorder=1)
