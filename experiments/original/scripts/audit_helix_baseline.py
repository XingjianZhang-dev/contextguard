#!/usr/bin/env python3
"""Compare HELIX, topology, and frozen ContextGuard on identical test cells."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

from audit_formal_pairs import LOSSES, key_for, summarize


def load(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [run for run in payload["runs"] if run.get("status") == "complete"]


def method_map(
    runs: list[dict[str, object]], method: str
) -> dict[tuple[object, ...], dict[str, object]]:
    output = {}
    for run in runs:
        if run["args"]["repair"] == method:
            key = key_for(run)
            if key in output:
                raise ValueError(f"Duplicate {method} cell: {key}")
            output[key] = run
    return output


def paired_rows(
    baseline: dict[tuple[object, ...], dict[str, object]],
    method: dict[tuple[object, ...], dict[str, object]],
) -> list[dict[str, object]]:
    if set(baseline) != set(method):
        missing_baseline = sorted(set(method) - set(baseline))
        missing_method = sorted(set(baseline) - set(method))
        raise ValueError(
            f"Non-identical cells; missing baseline={missing_baseline[:3]}, "
            f"missing method={missing_method[:3]}"
        )
    rows = []
    for key in sorted(baseline):
        baseline_metrics = baseline[key]["metrics"]
        method_metrics = method[key]["metrics"]
        row: dict[str, object] = {
            "dataset": key[0],
            "model": key[1],
            "corruption": key[2],
            "severity": key[3],
            "seed": key[4],
            "baseline_coverage80": baseline_metrics["coverage80"],
            "method_coverage80": method_metrics["coverage80"],
            "baseline_width80": baseline_metrics["width80"],
            "method_width80": method_metrics["width80"],
        }
        for metric in LOSSES:
            row[f"log_{metric}_ratio"] = math.log(
                method_metrics[metric] / baseline_metrics[metric]
            )
            row[f"{metric}_difference"] = method_metrics[metric] - baseline_metrics[metric]
        rows.append(row)
    return rows


def comparison_report(
    rows: list[dict[str, object]], repetitions: int, seed: int
) -> dict[str, object]:
    severities = sorted({int(row["severity"]) for row in rows})
    return {
        "overall": summarize(rows, repetitions, seed),
        "by_severity": {
            str(severity): summarize(
                [row for row in rows if row["severity"] == severity],
                repetitions,
                seed + severity * 10,
            )
            for severity in severities
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("formal_summary", type=Path)
    parser.add_argument("helix_summary", type=Path)
    parser.add_argument("--training-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=260_827)
    args = parser.parse_args()

    formal = load(args.formal_summary)
    helix_runs = load(args.helix_summary)
    topology = method_map(formal, "topology")
    contextguard = method_map(formal, "contextguard_adaptive")
    helix = method_map(helix_runs, "helix")
    comparisons = {
        "helix_vs_topology": (topology, helix),
        "contextguard_vs_helix": (helix, contextguard),
        "contextguard_vs_topology": (topology, contextguard),
    }
    report: dict[str, object] = {
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": args.seed,
        "comparisons": {
            name: comparison_report(
                paired_rows(baseline, method),
                args.bootstrap_repetitions,
                args.seed + index * 100,
            )
            for index, (name, (baseline, method)) in enumerate(comparisons.items())
        },
        "formal_summary": str(args.formal_summary),
        "helix_summary": str(args.helix_summary),
        "helix_runtime": {
            "runs": len(helix_runs),
            "median_elapsed_seconds": statistics.median(
                float(run["elapsed_seconds"]) for run in helix_runs
            ),
            "median_repair_seconds": statistics.median(
                float(run["repair_seconds"]) for run in helix_runs
            ),
            "median_forecast_seconds": statistics.median(
                float(run["forecast_seconds"]) for run in helix_runs
            ),
            "median_peak_cuda_memory_bytes": statistics.median(
                int(run["peak_cuda_memory_bytes"]) for run in helix_runs
            ),
            "imputer_parameters": sorted(
                {int(run["imputer_parameters"]) for run in helix_runs}
            ),
        },
    }
    if args.training_summary is not None:
        training = json.loads(args.training_summary.read_text(encoding="utf-8"))["runs"]
        complete = [run for run in training if run.get("status") == "complete"]
        report["helix_training"] = {
            "datasets": len(complete),
            "total_elapsed_seconds": sum(float(run["elapsed_seconds"]) for run in complete),
            "median_elapsed_seconds": statistics.median(
                float(run["elapsed_seconds"]) for run in complete
            ),
            "parameter_counts": sorted({int(run["parameters"]) for run in complete}),
            "training_summary": str(args.training_summary),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
