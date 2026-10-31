#!/usr/bin/env python3
"""Audit paired ContextGuard results and runtime across context lengths."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from audit_formal_pairs import key_for
from audit_helix_baseline import comparison_report, paired_rows


REPORTED_METRICS = ("mse", "mae", "quantile_loss", "coverage80", "width80")


def load(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [run for run in payload["runs"] if run.get("status") == "complete"]


def runtime(runs: list[dict[str, object]]) -> dict[str, float]:
    return {
        "median_elapsed_seconds": statistics.median(
            float(run["elapsed_seconds"]) for run in runs
        ),
        "median_forecast_seconds": statistics.median(
            float(run["forecast_seconds"]) for run in runs
        ),
        "median_peak_cuda_memory_bytes": statistics.median(
            int(run["peak_cuda_memory_bytes"]) for run in runs
        ),
        "median_repair_seconds": statistics.median(
            float(run["repair_seconds"]) for run in runs
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_summary", type=Path)
    parser.add_argument("formal_summary", type=Path)
    parser.add_argument("--default", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=260_827)
    args = parser.parse_args()

    sweep = load(args.sweep_summary)
    grouped: dict[int, dict[str, dict[tuple[object, ...], dict[str, object]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for run in sweep:
        length = int(run["args"]["context_length"])
        repair = str(run["args"]["repair"])
        key = key_for(run)
        if key in grouped[length][repair]:
            raise ValueError(f"Duplicate context-length cell: {length}, {repair}, {key}")
        grouped[length][repair][key] = run
    if args.default not in grouped:
        raise ValueError("Frozen default context length is absent")

    default_keys = set(grouped[args.default]["topology"])
    formal = load(args.formal_summary)
    formal_maps: dict[str, dict[tuple[object, ...], dict[str, object]]] = defaultdict(dict)
    for run in formal:
        key = key_for(run)
        repair = str(run["args"]["repair"])
        if key in default_keys and repair in {"topology", "contextguard_adaptive"}:
            formal_maps[repair][key] = run
    if any(set(formal_maps[method]) != default_keys for method in formal_maps):
        raise ValueError("Formal summary does not contain all frozen-default cells")

    differences = []
    for method in ("topology", "contextguard_adaptive"):
        for key, run in grouped[args.default][method].items():
            for metric in REPORTED_METRICS:
                differences.append(
                    abs(float(run["metrics"][metric]) - float(formal_maps[method][key]["metrics"][metric]))
                )

    setting_reports: dict[str, object] = {}
    for index, length in enumerate(sorted(grouped)):
        methods = grouped[length]
        topology = methods["topology"]
        contextguard = methods["contextguard_adaptive"]
        rows = paired_rows(topology, contextguard)
        setting_reports[str(length)] = {
            "comparison": comparison_report(
                rows,
                args.bootstrap_repetitions,
                args.seed + index * 100,
            ),
            "runtime": {
                "topology": runtime(list(topology.values())),
                "contextguard_adaptive": runtime(list(contextguard.values())),
            },
        }

    output = {
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": args.seed,
        "formal_summary": str(args.formal_summary),
        "frozen_default": args.default,
        "frozen_default_exact_formal_match": max(differences) == 0.0,
        "frozen_default_max_absolute_formal_difference": max(differences),
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
