#!/usr/bin/env python3
"""Compare a clean-environment rerun against the frozen formal matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = ("mse", "mae", "raw_mse", "quantile_loss", "coverage80", "width80")


def key(run: dict[str, object]) -> tuple[object, ...]:
    args = run["args"]
    return (
        args["dataset_path"],
        args["model"],
        args["corruption"],
        args["severity"],
        args["corruption_seed"],
        args["repair"],
    )


def load(path: Path) -> dict[tuple[object, ...], dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    output = {}
    for run in payload["runs"]:
        if run.get("status") != "complete":
            raise ValueError(f"Incomplete run in {path}: {run.get('run_id')}")
        coordinate = key(run)
        if coordinate in output:
            raise ValueError(f"Duplicate coordinate in {path}: {coordinate}")
        output[coordinate] = run
    return output


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def array_archive_path(summary: Path, run: dict[str, object]) -> Path:
    """Locate archived window arrays on the original node or after relocation."""
    embedded = Path(str(run["args"]["output"])) / "window_metrics.npz"
    relocated = summary.resolve().parent / str(run["run_id"]) / "window_metrics.npz"
    for candidate in (embedded, relocated):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate window_metrics.npz for "
        f"{run['run_id']}; checked {embedded} and {relocated}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("reproduction", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = load(args.reference)
    reproduction = load(args.reproduction)
    if set(reference) != set(reproduction):
        raise ValueError("Reference and reproduction coordinates differ")

    metric_maxima = {metric: 0.0 for metric in METRICS}
    array_maxima: dict[str, float] = {}
    byte_exact_array_runs = 0
    exact_metric_runs = 0
    exact_array_runs = 0
    fingerprint_matches = 0
    model_hash_matches = 0
    for coordinate in sorted(reference):
        old = reference[coordinate]
        new = reproduction[coordinate]
        metric_exact = True
        for metric in METRICS:
            difference = abs(float(old["metrics"][metric]) - float(new["metrics"][metric]))
            metric_maxima[metric] = max(metric_maxima[metric], difference)
            metric_exact &= difference == 0.0
        exact_metric_runs += int(metric_exact)
        fingerprint_matches += int(old["dataset_fingerprint"] == new["dataset_fingerprint"])
        model_hash_matches += int(old["model_sha256"] == new["model_sha256"])

        old_arrays = np.load(array_archive_path(args.reference, old))
        new_arrays = np.load(array_archive_path(args.reproduction, new))
        if set(old_arrays.files) != set(new_arrays.files):
            raise ValueError(f"Array fields differ for {coordinate}")
        arrays_exact = True
        arrays_byte_exact = True
        for field in old_arrays.files:
            left = old_arrays[field]
            right = new_arrays[field]
            if left.shape != right.shape:
                raise ValueError(f"Array shape differs for {coordinate}, {field}")
            if left.dtype != right.dtype:
                raise ValueError(f"Array dtype differs for {coordinate}, {field}")
            arrays_byte_exact &= left.tobytes(order="C") == right.tobytes(order="C")
            if np.issubdtype(left.dtype, np.number):
                finite = np.isfinite(left) & np.isfinite(right)
                maximum = float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else 0.0
                nan_pattern_equal = bool(np.array_equal(np.isnan(left), np.isnan(right)))
                exact = bool(np.array_equal(left, right, equal_nan=True))
                array_maxima[field] = max(array_maxima.get(field, 0.0), maximum)
                arrays_exact &= exact and nan_pattern_equal
            else:
                arrays_exact &= bool(np.array_equal(left, right))
        exact_array_runs += int(arrays_exact)
        byte_exact_array_runs += int(arrays_byte_exact)

    runs = len(reference)
    passed = (
        exact_metric_runs == runs
        and exact_array_runs == runs
        and byte_exact_array_runs == runs
        and fingerprint_matches == runs
        and model_hash_matches == runs
    )
    report = {
        "array_max_absolute_difference": array_maxima,
        "byte_exact_array_runs": byte_exact_array_runs,
        "dataset_fingerprint_matches": fingerprint_matches,
        "exact_array_runs": exact_array_runs,
        "exact_metric_runs": exact_metric_runs,
        "metric_max_absolute_difference": metric_maxima,
        "model_hash_matches": model_hash_matches,
        "passed": passed,
        "reference": str(args.reference),
        "reproduction": str(args.reproduction),
        "runs": runs,
    }
    write_json(args.output, report)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
