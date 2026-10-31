#!/usr/bin/env python3
"""Audit GIFT temporal splits and exact target-window duplication."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from datasets import load_from_disk

from run_gift_fault_eval import as_target, dataset_fingerprint, split_windows


def array_hash(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values, dtype=np.float32)
    finite = np.isfinite(contiguous)
    normalized = np.where(finite, contiguous, 0.0).astype(np.float32, copy=False)
    digest = hashlib.sha256()
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(finite.tobytes())
    digest.update(normalized.tobytes())
    return digest.hexdigest()


def temporally_constant(values: np.ndarray) -> bool:
    left = values[:, :-1]
    right = values[:, 1:]
    equal = (left == right) | (~np.isfinite(left) & ~np.isfinite(right))
    return bool(np.all(equal))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    datasets = config["datasets"]
    reports: dict[str, object] = {}
    all_temporal_splits_passed = True
    all_strict_duplicate_checks_passed = True
    for name, specification in datasets.items():
        path = Path(specification["path"])
        horizon = int(specification["prediction_length"])
        dataset = load_from_disk(str(path)).with_format("numpy")
        records = [as_target(dataset[index]) for index in range(len(dataset))]
        channels = records[0].shape[0]
        if any(record.shape[0] != channels for record in records):
            raise ValueError(f"Non-constant channel count in {name}")
        windows = split_windows(min(record.shape[1] for record in records), horizon)

        train_hash_origin: dict[str, int] = {}
        val_items: list[tuple[int, str, bool]] = []
        test_items: list[tuple[int, str, bool]] = []
        boundaries = []
        candidate_train_windows = 0
        temporal_disjoint = True
        same_record_val_train_matches = 0
        same_record_test_train_matches = 0
        same_record_test_val_matches = 0
        same_record_duplicate_val_windows = 0
        same_record_duplicate_test_windows = 0
        constant_same_record_val_train_matches = 0
        constant_same_record_test_train_matches = 0
        constant_same_record_test_val_matches = 0
        for record_index, record in enumerate(records):
            length = record.shape[1]
            train_end = length - horizon * (windows + 1)
            validation_start = train_end
            validation_end = validation_start + horizon
            test_start = length - horizon * windows
            test_end = length
            current_disjoint = (
                0 < train_end
                and train_end == validation_start
                and validation_end == test_start
                and test_start < test_end
            )
            temporal_disjoint &= current_disjoint
            boundaries.append(
                {
                    "length": length,
                    "record_index": record_index,
                    "test": [test_start, test_end],
                    "train": [0, train_end],
                    "validation": [validation_start, validation_end],
                    "temporally_disjoint": current_disjoint,
                }
            )
            local_train_hashes: set[str] = set()
            for start in range(0, train_end - horizon + 1):
                local_train_hashes.add(array_hash(record[:, start:start + horizon]))
                candidate_train_windows += 1
            val_values = record[:, validation_start:validation_end]
            val_hash = array_hash(val_values)
            val_constant = temporally_constant(val_values)
            local_test_items = [
                (
                    array_hash(record[:, start:start + horizon]),
                    temporally_constant(record[:, start:start + horizon]),
                )
                for start in range(test_start, test_end, horizon)
            ]
            local_test_hashes = [value for value, _ in local_test_items]
            same_record_val_train_matches += int(val_hash in local_train_hashes)
            constant_same_record_val_train_matches += int(
                val_hash in local_train_hashes and val_constant
            )
            same_record_test_train_matches += sum(
                value in local_train_hashes for value in local_test_hashes
            )
            constant_same_record_test_train_matches += sum(
                value in local_train_hashes and is_constant
                for value, is_constant in local_test_items
            )
            same_record_test_val_matches += sum(
                value == val_hash for value in local_test_hashes
            )
            constant_same_record_test_val_matches += sum(
                value == val_hash and is_constant and val_constant
                for value, is_constant in local_test_items
            )
            same_record_duplicate_test_windows += len(local_test_hashes) - len(
                set(local_test_hashes)
            )
            val_items.append((record_index, val_hash, val_constant))
            test_items.extend(
                (record_index, value, is_constant)
                for value, is_constant in local_test_items
            )
            for value in local_train_hashes:
                previous = train_hash_origin.get(value)
                if previous is None:
                    train_hash_origin[value] = record_index
                elif previous != record_index:
                    train_hash_origin[value] = -1

        val_hashes = [value for _, value, _ in val_items]
        test_hashes = [value for _, value, _ in test_items]
        cross_record_val_train_matches = sum(
            (origin := train_hash_origin.get(value)) is not None
            and (origin == -1 or origin != record_index)
            for record_index, value, _ in val_items
        )
        cross_record_test_train_matches = sum(
            (origin := train_hash_origin.get(value)) is not None
            and (origin == -1 or origin != record_index)
            for record_index, value, _ in test_items
        )
        val_records_by_hash: dict[str, set[int]] = {}
        for record_index, value, _ in val_items:
            val_records_by_hash.setdefault(value, set()).add(record_index)
        cross_record_test_val_matches = sum(
            any(other != record_index for other in val_records_by_hash.get(value, set()))
            for record_index, value, _ in test_items
        )
        global_duplicate_val_windows = len(val_hashes) - len(set(val_hashes))
        global_duplicate_test_windows = len(test_hashes) - len(set(test_hashes))
        strict_duplicate_free = (
            same_record_val_train_matches == 0
            and same_record_test_train_matches == 0
            and same_record_test_val_matches == 0
            and same_record_duplicate_val_windows == 0
            and same_record_duplicate_test_windows == 0
            and cross_record_val_train_matches == 0
            and cross_record_test_train_matches == 0
            and cross_record_test_val_matches == 0
            and global_duplicate_val_windows == 0
            and global_duplicate_test_windows == 0
        )
        all_temporal_splits_passed &= temporal_disjoint
        all_strict_duplicate_checks_passed &= strict_duplicate_free
        reports[name] = {
            "boundaries": boundaries,
            "candidate_training_target_windows_hashed": candidate_train_windows,
            "channels": channels,
            "constant_same_record_test_exact_matches_in_training": constant_same_record_test_train_matches,
            "constant_same_record_test_exact_matches_in_validation": constant_same_record_test_val_matches,
            "constant_same_record_validation_exact_matches_in_training": constant_same_record_val_train_matches,
            "cross_record_test_exact_matches_in_training": cross_record_test_train_matches,
            "cross_record_test_exact_matches_in_validation": cross_record_test_val_matches,
            "cross_record_validation_exact_matches_in_training": cross_record_val_train_matches,
            "dataset_fingerprint": dataset_fingerprint(path),
            "global_duplicate_test_windows": global_duplicate_test_windows,
            "global_duplicate_validation_windows": global_duplicate_val_windows,
            "horizon": horizon,
            "passed_temporal_split": temporal_disjoint,
            "record_count": len(records),
            "same_record_duplicate_test_windows": same_record_duplicate_test_windows,
            "same_record_duplicate_validation_windows": same_record_duplicate_val_windows,
            "same_record_test_exact_matches_in_training": same_record_test_train_matches,
            "same_record_test_exact_matches_in_validation": same_record_test_val_matches,
            "same_record_validation_exact_matches_in_training": same_record_val_train_matches,
            "nonconstant_same_record_test_exact_matches_in_training": (
                same_record_test_train_matches - constant_same_record_test_train_matches
            ),
            "nonconstant_same_record_test_exact_matches_in_validation": (
                same_record_test_val_matches - constant_same_record_test_val_matches
            ),
            "nonconstant_same_record_validation_exact_matches_in_training": (
                same_record_val_train_matches - constant_same_record_val_train_matches
            ),
            "split_windows": windows,
            "strict_exact_duplicate_free": strict_duplicate_free,
            "test_target_windows": len(test_hashes),
            "training_statistics_last_exclusive_index": [
                boundary["train"][1] for boundary in boundaries
            ],
            "validation_target_windows": len(val_hashes),
        }
        print(
            f"{name}: temporal={temporal_disjoint} strict_duplicate_free={strict_duplicate_free} "
            f"same_record_test_train={same_record_test_train_matches} "
            f"cross_record_test_train={cross_record_test_train_matches}",
            flush=True,
        )

    output = {
        "all_strict_duplicate_checks_passed": all_strict_duplicate_checks_passed,
        "all_temporal_splits_passed": all_temporal_splits_passed,
        "config": str(args.config),
        "datasets": reports,
        "hash_definition": "SHA-256(shape, finite mask, float32 finite values; all sliding training starts)",
    }
    write_json(args.output, output)
    return 0 if all_temporal_splits_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
