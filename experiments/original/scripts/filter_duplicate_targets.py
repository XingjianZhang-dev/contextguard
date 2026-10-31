#!/usr/bin/env python3
"""Recompute run summaries after excluding training-duplicate test targets."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from datasets import load_from_disk

from audit_data_splits import array_hash
from run_gift_fault_eval import as_target, descriptors_for_split, split_windows


METRICS = ("mse", "mae", "raw_mse", "quantile_loss", "coverage80", "width80")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if "extends" not in config:
        return config
    base = load_config((path.parent / config["extends"]).resolve())
    overrides = {key: value for key, value in config.items() if key != "extends"}
    matrix_overrides = overrides.pop("matrix", {})
    base.update(overrides)
    base.setdefault("matrix", {}).update(matrix_overrides)
    return base


def duplicate_descriptors(
    dataset_path: Path,
    prediction_length: int,
    context_length: int,
    max_windows: int,
) -> tuple[set[tuple[int, int]], int]:
    dataset = load_from_disk(str(dataset_path)).with_format("numpy")
    records = [as_target(dataset[index]) for index in range(len(dataset))]
    windows = split_windows(min(record.shape[1] for record in records), prediction_length)
    descriptors = descriptors_for_split(
        records,
        prediction_length,
        windows,
        context_length,
        "test",
        max_windows,
    )
    selected_by_record: dict[int, list[int]] = {}
    for record_index, forecast_start in descriptors:
        selected_by_record.setdefault(record_index, []).append(forecast_start)
    duplicates: set[tuple[int, int]] = set()
    for record_index, starts in selected_by_record.items():
        record = records[record_index]
        train_end = record.shape[1] - prediction_length * (windows + 1)
        train_hashes = {
            array_hash(record[:, start:start + prediction_length])
            for start in range(0, train_end - prediction_length + 1)
        }
        for forecast_start in starts:
            target_hash = array_hash(
                record[:, forecast_start:forecast_start + prediction_length]
            )
            if target_hash in train_hashes:
                duplicates.add((record_index, forecast_start))
    return duplicates, len(descriptors)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    duplicate_maps: dict[str, set[tuple[int, int]]] = {}
    duplicate_report: dict[str, object] = {}
    for name, specification in config["datasets"].items():
        path = str(specification["path"])
        duplicates, selected = duplicate_descriptors(
            Path(path),
            int(specification["prediction_length"]),
            int(config["context_length"]),
            int(config["max_windows"]),
        )
        duplicate_maps[path] = duplicates
        duplicate_report[name] = {
            "duplicate_selected_test_targets": len(duplicates),
            "retained_selected_test_targets": selected - len(duplicates),
            "selected_test_targets": selected,
        }
        print(f"{name}: exclude={len(duplicates)}/{selected}", flush=True)

    outputs = []
    for summary_path in args.summaries:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        filtered_runs = []
        for original in payload["runs"]:
            if original.get("status") != "complete":
                raise ValueError(f"Incomplete run in {summary_path}: {original.get('run_id')}")
            run = copy.deepcopy(original)
            run_dir = Path(str(run["args"]["output"]))
            arrays = np.load(run_dir / "window_metrics.npz")
            descriptors = list(
                zip(
                    arrays["record_index"].astype(int).tolist(),
                    arrays["forecast_start"].astype(int).tolist(),
                    strict=True,
                )
            )
            duplicates = duplicate_maps[str(run["args"]["dataset_path"])]
            retained = np.asarray(
                [descriptor not in duplicates for descriptor in descriptors], dtype=bool
            )
            if not retained.any():
                raise ValueError(f"Duplicate exclusion removed every window: {run['run_id']}")
            for metric in METRICS:
                run["metrics"][metric] = float(np.nanmean(arrays[metric][retained]))
            run["duplicate_target_windows_excluded"] = int((~retained).sum())
            run["windows_evaluated_before_duplicate_exclusion"] = len(retained)
            run["windows_evaluated"] = int(retained.sum())
            filtered_runs.append(run)
        output_path = args.output_dir / (
            summary_path.parent.name + "_duplicate_excluded.json"
        )
        write_json(
            output_path,
            {
                "duplicate_exclusion": duplicate_report,
                "experiment": payload["experiment"] + "_duplicate_excluded",
                "source_summary": str(summary_path),
                "runs": filtered_runs,
            },
        )
        outputs.append(str(output_path))
    write_json(
        args.output_dir / "duplicate_exclusion_manifest.json",
        {
            "config": str(args.config),
            "datasets": duplicate_report,
            "outputs": outputs,
            "rule": "Exclude selected test targets exactly matching any same-record, all-start training target window",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
