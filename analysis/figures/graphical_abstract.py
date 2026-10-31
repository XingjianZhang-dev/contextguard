#!/usr/bin/env python3
"""Graphical abstract on a real Jena Weather test window, forecast by Chronos-2.

The window is chosen by a rule fixed in advance: among all test windows with channel 2 failed, the one
whose Chronos-2 MSE improvement (topology -> ContextGuard) is the median. Requires the GIFT-Eval data and
the Chronos-2 checkpoint (``experiments/download_assets.py``) and a CUDA GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments" / "original" / "scripts"))
import run_gift_fault_eval as b  # noqa: E402
from style import CG, INK, INK2, PDF_METADATA, plt  # noqa: E402
from style import CG_LIGHT as CGL  # noqa: E402

plt.rcParams.update({"savefig.pad_inches": 0.04, "axes.spines.top": True, "axes.spines.right": True})
FAIL = "#e34948"
ds = load_from_disk(str(ROOT / "data/GiftEval/jena_weather/10T")).with_format("numpy")
recs = [b.as_target(ds[i]) for i in range(len(ds))]
H, L = 48, 512
W = b.split_windows(min(r.shape[1] for r in recs), H)
means, scales, pm, cov, rel, _ = b.training_statistics(recs, H, W, 0.05)
desc = b.descriptors_for_split(recs, H, W, L, "test", 256)
j = 2
CTX = np.asarray([((recs[a][:, f - L : f].T - means[a]) / scales[a]) for a, f in desc], dtype=np.float32)
FUT = np.asarray([((recs[a][:, f : f + H].T - means[a]) / scales[a]) for a, f in desc], dtype=np.float32)
MASK = np.zeros_like(CTX, bool)
MASK[:, :, j] = True
COR = CTX.copy()
COR[MASK] = 0
RCG = b.repair_batch(COR, MASK, "contextguard_adaptive", pm, cov, rel, torch.device("cuda"))
RTP = b.repair_batch(COR, MASK, "topology", pm, cov, rel, torch.device("cuda"))
P = b.ChronosPredictor(str(ROOT / "models/amazon-chronos-2"))
PCG, QCG = P.predict(RCG, H)
PTP, QTP = P.predict(RTP, H)
imp = np.mean((PTP[:, :, j] - FUT[:, :, j]) ** 2, 1) - np.mean((PCG[:, :, j] - FUT[:, :, j]) ** 2, 1)
w = int(np.argsort(imp)[len(imp) // 2])  # rule fixed in advance: the window with the median improvement
ctx, fut, rep_cg, rep_tp = CTX[w : w + 1], FUT[w], RCG[w : w + 1], RTP[w : w + 1]
pt_cg, q_cg, pt_tp, q_tp = PCG[w : w + 1], QCG[w : w + 1], PTP[w : w + 1], QTP[w : w + 1]
ri, fs = desc[w]
fig = plt.figure(figsize=(7.5, 3.0))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 40)
ax.axis("off")


def box(x, y, w, h, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.6", fc=fc, ec=ec, lw=0.9))


def arrow(x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=11, lw=1.2, color="#6f6e6a"))


# panel 1: sensor stream
ax.text(1, 37.2, "Sensor stream with a failed channel", fontsize=8.5, color=INK, fontweight="bold")
sub = [0, 1, 2, 3]
win = slice(L - 260, L)
for k, ch in enumerate(sub):
    y0 = 29 - k * 7.2
    y = ctx[0, win, ch]
    y = (y - y.min()) / (np.ptp(y) + 1e-9) * 5 + y0
    xs = np.linspace(1, 24, y.size)
    if ch == j:
        ax.plot(xs, np.full_like(xs, y0 + 2.5), color=FAIL, lw=1.2, ls=(0, (2.5, 1.5)))
        ax.text(24.6, y0 + 2.5, "failed", fontsize=7, color=FAIL, va="center")
    else:
        ax.plot(xs, y, color="#7d7c77", lw=0.9)
ax.text(1, 1.2, "Peers stay synchronized", fontsize=7, color=INK2)
arrow(29.0, 32.5, 18)
# panel 2: ContextGuard
box(33.5, 4.5, 26, 30, "#eef4fc", CG)
ax.text(46.5, 31.6, "ContextGuard", fontsize=10, color=CG, fontweight="bold", ha="center")
steps = [
    "Calibrate cross-channel structure\non training history only",
    "Reliability gate: repair only when\ncalibrated gain clears a threshold",
    "Rebuild the failed channel from\npeers; observed values unchanged",
]
for k, s_ in enumerate(steps):
    yy = 25.5 - k * 7.6
    ax.add_patch(plt.Circle((36.6, yy), 1.35, color=CG))
    ax.text(36.6, yy, str(k + 1), color="white", fontsize=7.5, ha="center", va="center", fontweight="bold")
    ax.text(38.8, yy, s_, fontsize=7, color=INK, va="center", linespacing=1.15)
ax.text(46.5, 5.6, "0 trainable parameters  ·  +1% runtime", fontsize=7, color=INK2, ha="center")
arrow(60.6, 63.6, 18)
# panel 3: frozen forecaster
box(64.5, 12.5, 11.5, 11, "#f3f2ef", "#8a8984")
ax.text(
    70.25,
    20.6,
    "Frozen\nforecaster",
    fontsize=8,
    ha="center",
    va="center",
    color=INK,
    fontweight="bold",
    linespacing=1.1,
)
ax.text(70.25, 15.0, "unchanged", fontsize=6.8, ha="center", color=INK2)
ax.text(70.25, 9.3, "Chronos-2\nTimesFM 2.5\nTiRex", fontsize=6.5, ha="center", va="top", color=INK2, linespacing=1.15)
arrow(77.0, 79.6, 18)
# panel 4: forecast for the failed channel (inset axes)
ia = fig.add_axes([0.805, 0.17, 0.185, 0.62])
hist = 96
xh = np.arange(-hist, 0)
xf = np.arange(0, H)
ia.plot(xh, ctx[0, -hist:, j], color="#b5b4af", lw=0.8)
ia.plot(xh, rep_tp[0, -hist:, j], color="#8a8984", lw=0.9, ls=(0, (2, 1.5)))
ia.plot(xh, rep_cg[0, -hist:, j], color=CG, lw=1.0)
ia.fill_between(xf, q_tp[0, :, j, 0], q_tp[0, :, j, -1], color="#d9d8d3", lw=0)
ia.plot(xf, pt_tp[0, :, j], color="#8a8984", lw=0.9)
ia.fill_between(xf, q_cg[0, :, j, 0], q_cg[0, :, j, -1], color=CGL, alpha=0.6, lw=0)
ia.plot(xf, pt_cg[0, :, j], color=CG, lw=1.1)
ia.plot(xf, fut[:, j], color=INK, lw=0.9)
ia.axvline(0, color="#a8a7a2", lw=0.6)
ia.set_xticks([])
ia.set_yticks([])
for s_ in ia.spines.values():
    s_.set_color("#c9c8c3")
    s_.set_linewidth(0.6)
fig.text(0.805, 0.835, "Forecast of the failed channel", fontsize=8.5, fontweight="bold", color=INK)
ia.legend(
    handles=[
        Line2D([], [], color=CG, lw=1.2, label="With ContextGuard"),
        Line2D([], [], color="#8a8984", lw=1.0, ls=(0, (2, 1.5)), label="Temporal fill"),
        Line2D([], [], color=INK, lw=1.0, label="Actual"),
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=1,
    fontsize=6.6,
    frameon=False,
    handlelength=1.6,
    borderaxespad=0.2,
)
out = ROOT / "results" / "figures"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "graphical_abstract.pdf", metadata=PDF_METADATA)
fig.savefig(out / "graphical_abstract.png", dpi=400)
print(
    "GA mse topo",
    float(np.mean((pt_tp[0, :, j] - fut[:, j]) ** 2)),
    "mse cg",
    float(np.mean((pt_cg[0, :, j] - fut[:, j]) ** 2)),
    "window",
    ri,
    fs,
    "gate",
    float(rel[j]),
)
