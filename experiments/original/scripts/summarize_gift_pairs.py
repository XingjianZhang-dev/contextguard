#!/usr/bin/env python3
"""Summarize paired GIFT-Eval repair runs with seed-level uncertainty."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

from scipy.stats import t


METRICS = ("mse", "mae", "quantile_loss", "coverage80", "width80")


def confidence_interval(values: list[float]) -> list[float]:
    if len(values) < 2:
        return [values[0], values[0]]
    mean = statistics.mean(values)
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    margin = float(t.ppf(0.975, len(values) - 1)) * standard_error
    return [mean - margin, mean + margin]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--baseline", default="topology")
    parser.add_argument("--method", default="contextguard_adaptive")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    runs = json.loads(args.summary.read_text(encoding="utf-8"))["runs"]
    complete = [run for run in runs if run["status"] == "complete"]
    indexed: dict[tuple[str, str, str, int, int], dict[str, dict]] = {}
    for run in complete:
        run_args = run["args"]
        key = (
            run_args["dataset_path"],
            run_args["model"],
            run_args["corruption"],
            int(run_args["severity"]),
            int(run_args["corruption_seed"]),
        )
        indexed.setdefault(key, {})[run_args["repair"]] = run

    paired = {
        key: pair
        for key, pair in indexed.items()
        if args.baseline in pair and args.method in pair
    }
    groups: dict[tuple[str, str, str, int], list[tuple[int, dict[str, dict]]]] = {}
    for key, pair in paired.items():
        dataset, model, corruption, severity, seed = key
        groups.setdefault((dataset, model, corruption, severity), []).append((seed, pair))

    group_summaries = []
    all_log_ratios = {metric: [] for metric in METRICS[:3]}
    for key, seed_pairs in sorted(groups.items()):
        dataset, model, corruption, severity = key
        seed_pairs.sort()
        metric_summary = {}
        for metric in METRICS:
            baseline = [pair[args.baseline]["metrics"][metric] for _, pair in seed_pairs]
            method = [pair[args.method]["metrics"][metric] for _, pair in seed_pairs]
            differences = [right - left for left, right in zip(baseline, method, strict=True)]
            current = {
                "baseline_mean": statistics.mean(baseline),
                "method_mean": statistics.mean(method),
                "paired_difference_mean": statistics.mean(differences),
                "paired_difference_ci95": confidence_interval(differences),
                "wins": sum(right < left for left, right in zip(baseline, method, strict=True)),
                "ties": sum(right == left for left, right in zip(baseline, method, strict=True)),
            }
            if metric in all_log_ratios:
                ratios = [right / left for left, right in zip(baseline, method, strict=True)]
                log_ratios = [math.log(value) for value in ratios]
                log_ci = confidence_interval(log_ratios)
                current["geometric_relative_change_percent"] = 100.0 * (
                    math.exp(statistics.mean(log_ratios)) - 1.0
                )
                current["geometric_relative_change_ci95_percent"] = [
                    100.0 * (math.exp(bound) - 1.0) for bound in log_ci
                ]
                all_log_ratios[metric].extend(log_ratios)
            metric_summary[metric] = current
        group_summaries.append(
            {
                "dataset_path": dataset,
                "model": model,
                "corruption": corruption,
                "severity": severity,
                "seeds": [seed for seed, _ in seed_pairs],
                "metrics": metric_summary,
            }
        )

    overall = {}
    for metric, log_ratios in all_log_ratios.items():
        log_ci = confidence_interval(log_ratios)
        overall[metric] = {
            "geometric_relative_change_percent": 100.0
            * (math.exp(statistics.mean(log_ratios)) - 1.0),
            "geometric_relative_change_ci95_percent": [
                100.0 * (math.exp(bound) - 1.0) for bound in log_ci
            ],
            "paired_cells": len(log_ratios),
        }

    result = {
        "baseline": args.baseline,
        "method": args.method,
        "complete_runs": len(complete),
        "paired_cells": len(paired),
        "overall": overall,
        "groups": group_summaries,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
