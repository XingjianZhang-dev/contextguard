#!/usr/bin/env python3
"""Audit paired formal-test cells with deterministic inferential statistics."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, t, wilcoxon


LOSSES = ("mse", "mae", "quantile_loss")


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return monotone Holm--Bonferroni adjusted p-values."""
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def key_for(run: dict[str, object]) -> tuple[object, ...]:
    args = run["args"]
    return (
        args["dataset_path"],
        args["model"],
        args["corruption"],
        args["severity"],
        args["corruption_seed"],
    )


def interval(values: np.ndarray) -> tuple[float, float]:
    if values.size < 2:
        value = float(values.mean())
        return value, value
    half = float(t.ppf(0.975, values.size - 1) * values.std(ddof=1) / math.sqrt(values.size))
    mean = float(values.mean())
    return mean - half, mean + half


def cluster_bootstrap(
    rows: list[dict[str, object]], metric: str, repetitions: int, seed: int
) -> tuple[float, float]:
    """Resample datasets, then model-seed cells within sampled datasets."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(float(row[f"log_{metric}_ratio"]))
    datasets = sorted(grouped)
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        selected = rng.choice(datasets, size=len(datasets), replace=True)
        dataset_means = []
        for dataset in selected:
            values = np.asarray(grouped[str(dataset)], dtype=np.float64)
            sampled = rng.choice(values, size=values.size, replace=True)
            dataset_means.append(float(sampled.mean()))
        estimates[index] = float(np.mean(dataset_means))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return (math.exp(float(low)) - 1.0) * 100.0, (math.exp(float(high)) - 1.0) * 100.0


def summarize(rows: list[dict[str, object]], repetitions: int, seed: int) -> dict[str, object]:
    output: dict[str, object] = {"paired_cells": len(rows), "metrics": {}}
    for metric_index, metric in enumerate(LOSSES):
        log_ratios = np.asarray([row[f"log_{metric}_ratio"] for row in rows], dtype=np.float64)
        differences = np.asarray([row[f"{metric}_difference"] for row in rows], dtype=np.float64)
        low, high = interval(log_ratios)
        wins = int(np.sum(differences < -1e-12))
        ties = int(np.sum(np.abs(differences) <= 1e-12))
        losses = int(np.sum(differences > 1e-12))
        nonzero = differences[np.abs(differences) > 1e-12]
        if nonzero.size:
            wilcoxon_p = float(wilcoxon(nonzero, alternative="less").pvalue)
            sign_p = float(binomtest(wins, wins + losses, 0.5, alternative="greater").pvalue)
        else:
            wilcoxon_p = 1.0
            sign_p = 1.0
        effect_dz = (
            float(log_ratios.mean() / log_ratios.std(ddof=1))
            if log_ratios.size > 1 and log_ratios.std(ddof=1) > 0
            else None
        )
        output["metrics"][metric] = {
            "geometric_relative_change_percent": (math.exp(float(log_ratios.mean())) - 1.0) * 100.0,
            "t_ci95_percent": [
                (math.exp(low) - 1.0) * 100.0,
                (math.exp(high) - 1.0) * 100.0,
            ],
            "dataset_cluster_bootstrap_ci95_percent": list(
                cluster_bootstrap(rows, metric, repetitions, seed + metric_index)
            ),
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "wilcoxon_less_p": wilcoxon_p,
            "sign_test_greater_p": sign_p,
            "paired_log_ratio_effect_dz": effect_dz,
        }
    wilcoxon_adjusted = holm_adjust(
        [float(output["metrics"][metric]["wilcoxon_less_p"]) for metric in LOSSES]
    )
    sign_adjusted = holm_adjust(
        [float(output["metrics"][metric]["sign_test_greater_p"]) for metric in LOSSES]
    )
    for index, metric in enumerate(LOSSES):
        output["metrics"][metric]["wilcoxon_less_holm_p"] = wilcoxon_adjusted[index]
        output["metrics"][metric]["sign_test_greater_holm_p"] = sign_adjusted[index]
    for label, prefix in (("baseline", "baseline"), ("method", "method")):
        coverage_error = np.asarray(
            [abs(float(row[f"{prefix}_coverage80"]) - 0.8) for row in rows]
        )
        widths = np.asarray([row[f"{prefix}_width80"] for row in rows], dtype=np.float64)
        output[label] = {
            "mean_absolute_coverage80_error": float(coverage_error.mean()),
            "mean_width80": float(widths.mean()),
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", default="topology")
    parser.add_argument("--method", default="contextguard_adaptive")
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=260_827)
    args = parser.parse_args()

    runs = json.loads(args.summary.read_text(encoding="utf-8"))["runs"]
    paired: dict[tuple[object, ...], dict[str, dict[str, object]]] = defaultdict(dict)
    for run in runs:
        if run.get("status") == "complete":
            paired[key_for(run)][run["args"]["repair"]] = run

    rows: list[dict[str, object]] = []
    for key, methods in paired.items():
        if args.baseline not in methods or args.method not in methods:
            continue
        baseline = methods[args.baseline]["metrics"]
        method = methods[args.method]["metrics"]
        row: dict[str, object] = {
            "dataset": key[0],
            "model": key[1],
            "corruption": key[2],
            "severity": key[3],
            "seed": key[4],
            "baseline_coverage80": baseline["coverage80"],
            "method_coverage80": method["coverage80"],
            "baseline_width80": baseline["width80"],
            "method_width80": method["width80"],
        }
        for metric in LOSSES:
            row[f"log_{metric}_ratio"] = math.log(method[metric] / baseline[metric])
            row[f"{metric}_difference"] = method[metric] - baseline[metric]
        rows.append(row)

    severities = sorted({int(row["severity"]) for row in rows})
    report = {
        "baseline": args.baseline,
        "method": args.method,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": args.seed,
        "overall": summarize(rows, args.bootstrap_repetitions, args.seed),
        "by_severity": {
            str(severity): summarize(
                [row for row in rows if row["severity"] == severity],
                args.bootstrap_repetitions,
                args.seed + severity * 10,
            )
            for severity in severities
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
