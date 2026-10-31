#!/usr/bin/env python3
"""Audit fingerprints, model hashes, status, and duplicate cells in summaries."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


def coordinate_key(run: dict[str, object]) -> str:
    args = run["args"]
    coordinates = {
        "context_length": args.get("context_length"),
        "corruption": args.get("corruption"),
        "corruption_seed": args.get("corruption_seed"),
        "dataset_path": args.get("dataset_path"),
        "model": args.get("model"),
        "repair": args.get("repair"),
        "severity": args.get("severity"),
        "split": args.get("split"),
    }
    for optional in (
        "reliability_threshold",
        "short_blend",
        "missing_activation",
        "long_gap_fraction_threshold",
        "short_gap_max",
        "topology_gap_max",
        "imputer_path",
    ):
        if optional in args:
            coordinates[optional] = args[optional]
    return json.dumps(coordinates, sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fingerprints: dict[str, set[str]] = defaultdict(set)
    model_hashes: dict[str, set[str]] = defaultdict(set)
    reports = {}
    all_passed = True
    for path in args.summaries:
        payload = json.loads(path.read_text(encoding="utf-8"))
        runs = payload["runs"]
        statuses: dict[str, int] = defaultdict(int)
        keys = []
        for run in runs:
            statuses[str(run.get("status"))] += 1
            if run.get("status") != "complete":
                continue
            key = coordinate_key(run)
            keys.append(key)
            run_args = run["args"]
            fingerprints[str(run_args["dataset_path"])].add(str(run["dataset_fingerprint"]))
            model_hashes[str(run_args["model"])].add(str(run["model_sha256"]))
        duplicates = len(keys) - len(set(keys))
        passed = statuses.get("complete", 0) == len(runs) and duplicates == 0
        all_passed &= passed
        reports[str(path)] = {
            "coordinate_duplicates": duplicates,
            "passed": passed,
            "runs": len(runs),
            "statuses": dict(statuses),
        }

    fingerprint_conflicts = {
        path: sorted(values) for path, values in fingerprints.items() if len(values) != 1
    }
    model_hash_conflicts = {
        model: sorted(values) for model, values in model_hashes.items() if len(values) != 1
    }
    all_passed &= not fingerprint_conflicts and not model_hash_conflicts
    output = {
        "all_passed": all_passed,
        "dataset_fingerprint_conflicts": fingerprint_conflicts,
        "dataset_fingerprints": {path: sorted(values)[0] for path, values in fingerprints.items()},
        "model_hash_conflicts": model_hash_conflicts,
        "model_hashes": {model: sorted(values)[0] for model, values in model_hashes.items()},
        "summaries": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
