#!/usr/bin/env python3
"""Audit a one-axis post-formal sensitivity sweep without selecting on test."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path


LOSSES = ("mse", "mae", "quantile_loss")
REPORTED_METRICS = LOSSES + ("coverage80", "width80")


def completed_runs(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [run for run in payload["runs"] if run.get("status") == "complete"]


def cell_key(run: dict[str, object]) -> tuple[object, ...]:
    args = run["args"]
    return (
        args["dataset_path"],
        args["model"],
        args["corruption"],
        args["severity"],
        args["corruption_seed"],
    )


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def relative_percent(numerator: float, denominator: float) -> float:
    return (numerator / denominator - 1.0) * 100.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_summary", type=Path)
    parser.add_argument("formal_summary", type=Path)
    parser.add_argument("--axis", required=True)
    parser.add_argument("--default", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", default="topology")
    parser.add_argument("--method", default="contextguard_adaptive")
    args = parser.parse_args()

    sweep_runs = completed_runs(args.sweep_summary)
    if not sweep_runs:
        raise ValueError("Sensitivity summary contains no complete runs")
    for run in sweep_runs:
        if args.axis not in run["args"]:
            raise ValueError(f"Sensitivity axis {args.axis!r} is absent")
        if run["args"]["repair"] != args.method:
            raise ValueError("Sweep contains a repair method other than --method")

    settings: dict[float, dict[tuple[object, ...], dict[str, object]]] = defaultdict(dict)
    for run in sweep_runs:
        setting = float(run["args"][args.axis])
        key = cell_key(run)
        if key in settings[setting]:
            raise ValueError(f"Duplicate sensitivity cell at {setting}: {key}")
        settings[setting][key] = run
    ordered_settings = sorted(settings)
    default_setting = min(ordered_settings, key=lambda value: abs(value - args.default))
    if abs(default_setting - args.default) > 1e-12:
        raise ValueError(f"Default {args.default} is absent from the sweep")
    expected_cells = set(settings[default_setting])
    if any(set(cells) != expected_cells for cells in settings.values()):
        raise ValueError("Sensitivity settings do not contain identical cells")

    formal_baseline: dict[tuple[object, ...], dict[str, object]] = {}
    formal_method: dict[tuple[object, ...], dict[str, object]] = {}
    for run in completed_runs(args.formal_summary):
        key = cell_key(run)
        if key not in expected_cells:
            continue
        repair = run["args"]["repair"]
        if repair == args.baseline:
            formal_baseline[key] = run
        elif repair == args.method:
            formal_method[key] = run
    if set(formal_baseline) != expected_cells or set(formal_method) != expected_cells:
        raise ValueError("Formal summary is missing matching baseline or method cells")

    default_differences = []
    for key, run in settings[default_setting].items():
        for metric in REPORTED_METRICS:
            default_differences.append(
                abs(float(run["metrics"][metric]) - float(formal_method[key]["metrics"][metric]))
            )

    setting_reports: dict[str, object] = {}
    for setting in ordered_settings:
        cells = settings[setting]
        report: dict[str, object] = {
            "cells": len(cells),
            "mean_absolute_coverage80_error": sum(
                abs(float(run["metrics"]["coverage80"]) - 0.8) for run in cells.values()
            ) / len(cells),
            "mean_width80": sum(
                float(run["metrics"]["width80"]) for run in cells.values()
            ) / len(cells),
            "metrics": {},
        }
        for metric in LOSSES:
            raw = [float(run["metrics"][metric]) for run in cells.values()]
            baseline_ratios = [
                float(run["metrics"][metric]) / float(formal_baseline[key]["metrics"][metric])
                for key, run in cells.items()
            ]
            default_ratios = [
                float(run["metrics"][metric])
                / float(settings[default_setting][key]["metrics"][metric])
                for key, run in cells.items()
            ]
            wins = sum(ratio < 1.0 - 1e-12 for ratio in baseline_ratios)
            ties = sum(abs(ratio - 1.0) <= 1e-12 for ratio in baseline_ratios)
            report["metrics"][metric] = {
                "geometric_mean": geometric_mean(raw),
                "geometric_change_vs_formal_baseline_percent": (
                    geometric_mean(baseline_ratios) - 1.0
                ) * 100.0,
                "geometric_change_vs_frozen_default_percent": (
                    geometric_mean(default_ratios) - 1.0
                ) * 100.0,
                "minimum_cell_change_vs_formal_baseline_percent": relative_percent(
                    min(baseline_ratios), 1.0
                ),
                "maximum_cell_change_vs_formal_baseline_percent": relative_percent(
                    max(baseline_ratios), 1.0
                ),
                "wins_vs_formal_baseline": wins,
                "ties_vs_formal_baseline": ties,
                "losses_vs_formal_baseline": len(baseline_ratios) - wins - ties,
            }
        setting_reports[str(setting)] = report

    output = {
        "axis": args.axis,
        "baseline": args.baseline,
        "formal_summary": str(args.formal_summary),
        "frozen_default": default_setting,
        "frozen_default_exact_formal_match": max(default_differences) == 0.0,
        "frozen_default_max_absolute_formal_difference": max(default_differences),
        "method": args.method,
        "paired_cells_per_setting": len(expected_cells),
        "settings": setting_reports,
        "sweep_summary": str(args.sweep_summary),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
