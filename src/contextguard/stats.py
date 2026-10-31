"""Paired statistics used throughout the paper.

For each paired cell (dataset, forecaster, severity, seed) the effect is the log ratio of a loss under
the method and under the baseline. The summary reports the geometric relative change, a two-level
dataset-cluster bootstrap interval (datasets, then cells within datasets; 20,000 replicates, seed
260827), exact wins/ties/losses, and one-sided Wilcoxon and sign tests with Holm adjustment across the
three losses. The implementation is identical to ``audit_formal_pairs.py`` of the original release.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import binomtest, t, wilcoxon

LOSSES = ("mse", "mae", "quantile_loss")
BOOTSTRAP_REPETITIONS = 20_000
BOOTSTRAP_SEED = 260_827


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Monotone Holm-Bonferroni adjusted p-values."""
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running, total = 0.0, len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _interval(values: np.ndarray) -> tuple[float, float]:
    if values.size < 2:
        return float(values.mean()), float(values.mean())
    half = float(t.ppf(0.975, values.size - 1) * values.std(ddof=1) / math.sqrt(values.size))
    return float(values.mean()) - half, float(values.mean()) + half


def cluster_bootstrap(rows: Sequence[Mapping], metric: str, repetitions: int, seed: int) -> tuple[float, float]:
    """95% interval of the geometric change: resample datasets, then cells within each dataset."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(float(row[f"log_{metric}_ratio"]))
    datasets = sorted(grouped)
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions)
    for i in range(repetitions):
        means = []
        for d in rng.choice(datasets, size=len(datasets), replace=True):
            values = np.asarray(grouped[str(d)])
            means.append(float(rng.choice(values, size=values.size, replace=True).mean()))
        estimates[i] = float(np.mean(means))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return (math.exp(float(low)) - 1.0) * 100.0, (math.exp(float(high)) - 1.0) * 100.0


def summarize(rows: Sequence[Mapping], repetitions: int = BOOTSTRAP_REPETITIONS, seed: int = BOOTSTRAP_SEED) -> dict:
    """Summarize paired rows (see :func:`paired_rows`)."""
    out: dict = {"paired_cells": len(rows), "metrics": {}}
    for k, metric in enumerate(LOSSES):
        lr = np.asarray([r[f"log_{metric}_ratio"] for r in rows], dtype=np.float64)
        diff = np.asarray([r[f"{metric}_difference"] for r in rows], dtype=np.float64)
        low, high = _interval(lr)
        wins = int(np.sum(diff < -1e-12))
        ties = int(np.sum(np.abs(diff) <= 1e-12))
        losses = int(np.sum(diff > 1e-12))
        nonzero = diff[np.abs(diff) > 1e-12]
        if nonzero.size:
            w_p = float(wilcoxon(nonzero, alternative="less").pvalue)
            s_p = float(binomtest(wins, wins + losses, 0.5, alternative="greater").pvalue)
        else:
            w_p = s_p = 1.0
        sd = lr.std(ddof=1) if lr.size > 1 else 0.0
        out["metrics"][metric] = {
            "geometric_relative_change_percent": (math.exp(float(lr.mean())) - 1.0) * 100.0,
            "t_ci95_percent": [(math.exp(low) - 1.0) * 100.0, (math.exp(high) - 1.0) * 100.0],
            "dataset_cluster_bootstrap_ci95_percent": list(cluster_bootstrap(rows, metric, repetitions, seed + k)),
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "wilcoxon_less_p": w_p,
            "sign_test_greater_p": s_p,
            "paired_log_ratio_effect_dz": float(lr.mean() / sd) if sd > 0 else None,
        }
    for name in ("wilcoxon_less", "sign_test_greater"):
        adj = holm_adjust([out["metrics"][m][f"{name}_p"] for m in LOSSES])
        for i, m in enumerate(LOSSES):
            out["metrics"][m][f"{name}_holm_p"] = adj[i]
    for label in ("baseline", "method"):
        cov_err = np.asarray([abs(float(r[f"{label}_coverage80"]) - 0.8) for r in rows])
        widths = np.asarray([r[f"{label}_width80"] for r in rows], dtype=np.float64)
        out[label] = {"mean_absolute_coverage80_error": float(cov_err.mean()), "mean_width80": float(widths.mean())}
    return out


def paired_rows(metrics: Mapping[tuple, Mapping], method: str, baseline: str, where=lambda cell: True) -> list[dict]:
    """Build paired rows from ``{(dataset, model, severity, seed, method): metrics}``."""
    rows = []
    for key, x in sorted(metrics.items(), key=lambda kv: tuple(map(str, kv[0]))):
        cell = key[:-1]
        if key[-1] != method or cell + (baseline,) not in metrics or not where(cell):
            continue
        y = metrics[cell + (baseline,)]
        row = {"dataset": cell[0]}
        for m in LOSSES:
            row[f"log_{m}_ratio"] = math.log(x[m] / y[m])
            row[f"{m}_difference"] = x[m] - y[m]
        for label, d in (("baseline", y), ("method", x)):
            row[f"{label}_coverage80"] = d["coverage80"]
            row[f"{label}_width80"] = d["width80"]
        rows.append(row)
    return rows


def compare(metrics: Mapping[tuple, Mapping], method: str, baseline: str, where=lambda cell: True) -> dict:
    """Paired summary of ``method`` against ``baseline``."""
    return summarize(paired_rows(metrics, method, baseline, where))
