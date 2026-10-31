#!/usr/bin/env python3
"""Train a leakage-free HELIX sensor-dropout imputer on GIFT training data."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from datasets import load_from_disk

from run_gift_fault_eval import (
    as_target,
    dataset_fingerprint,
    sha256,
    split_windows,
    training_statistics,
)


def select_evenly(items: list[tuple[int, int]], maximum: int) -> list[tuple[int, int]]:
    if len(items) <= maximum:
        return items
    indices = np.linspace(0, len(items) - 1, maximum, dtype=np.int64)
    return [items[index] for index in indices]


def build_windows(
    records: list[np.ndarray],
    means: list[np.ndarray],
    scales: list[np.ndarray],
    train_ends: list[int],
    context_length: int,
    maximum: int,
    validation: bool,
) -> np.ndarray:
    descriptors: list[tuple[int, int]] = []
    stride = max(1, context_length // 4)
    for record_index, train_end in enumerate(train_ends):
        split = max(context_length, int(0.8 * train_end))
        first = split + context_length if validation else context_length
        last = train_end if validation else split
        descriptors.extend(
            (record_index, end)
            for end in range(first, last + 1, stride)
            if end - context_length >= (split if validation else 0)
        )
    descriptors = select_evenly(descriptors, maximum)
    if not descriptors:
        raise ValueError("No leakage-free HELIX windows could be constructed")
    windows = []
    for record_index, end in descriptors:
        window = records[record_index][:, end - context_length:end].T
        windows.append((window - means[record_index]) / scales[record_index])
    return np.asarray(windows, dtype=np.float32)


def corrupt_full_channels(values: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    output = values.copy()
    injected = np.zeros_like(output, dtype=bool)
    channels = output.shape[-1]
    for index in range(output.shape[0]):
        rng = np.random.default_rng(seed + 1_000_003 * (index + 1))
        rate = 0.10 if index % 2 == 0 else 0.50
        count = min(max(1, math.ceil(rate * channels)), max(1, channels - 1))
        dropped = rng.choice(channels, size=count, replace=False)
        injected[index, :, dropped] = np.isfinite(output[index, :, dropped])
        output[index, :, dropped] = np.nan
    return output, injected


class SensorDropoutMasker:
    """PyPOTS MIT masker using complete-channel failures at mixed severities."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def __call__(self, values, p: float = 0.2):
        del p
        is_tensor = torch.is_tensor(values)
        output = values.clone() if is_tensor else np.array(values, copy=True)
        channels = output.shape[-1]
        rate = float(self.rng.choice([0.10, 0.30, 0.50]))
        count = min(max(1, math.ceil(rate * channels)), max(1, channels - 1))
        dropped = self.rng.choice(channels, size=count, replace=False)
        if is_tensor:
            output[..., torch.as_tensor(dropped, device=output.device)] = float("nan")
        else:
            output[..., dropped] = np.nan
        return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--prediction-length", type=int, required=True)
    parser.add_argument("--max-train-windows", type=int, default=2048)
    parser.add_argument("--max-val-windows", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2021)
    args = parser.parse_args()

    started = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    dataset = load_from_disk(str(args.dataset_path)).with_format("numpy")
    records = [as_target(dataset[index]) for index in range(len(dataset))]
    channels = records[0].shape[0]
    minimum_length = min(record.shape[1] for record in records)
    forecast_windows = split_windows(minimum_length, args.prediction_length)
    means, scales, _, _, _, train_ends = training_statistics(
        records, args.prediction_length, forecast_windows
    )
    train_x = build_windows(
        records,
        means,
        scales,
        train_ends,
        args.context_length,
        args.max_train_windows,
        validation=False,
    )
    val_original = build_windows(
        records,
        means,
        scales,
        train_ends,
        args.context_length,
        args.max_val_windows,
        validation=True,
    )
    val_x, injected = corrupt_full_channels(val_original, args.seed + 10_000)

    from pypots.imputation import HELIX
    import pypots.imputation.saits.data as saits_data

    saits_data.mcar = SensorDropoutMasker(args.seed + 20_000)
    args.output.mkdir(parents=True, exist_ok=True)
    training_log = args.output / "training"
    model = HELIX(
        n_steps=args.context_length,
        n_features=channels,
        d_pe=16,
        d_feature_embed=1,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=0.1,
        ORT_weight=1.0,
        MIT_weight=1.0,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        num_workers=0,
        device="cuda",
        saving_path=str(training_log),
        model_saving_strategy="best",
        verbose=True,
    )
    model.fit(
        {"X": train_x},
        {"X": val_x, "X_ori": val_original},
    )
    imputed = model.impute({"X": val_x})
    valid_injected = injected & np.isfinite(val_original)
    errors = imputed[valid_injected] - val_original[valid_injected]
    artifact = args.output / "helix.pypots"
    model.save(str(artifact), overwrite=True)
    parameters = sum(parameter.numel() for parameter in model.model.parameters())
    manifest = {
        "architecture": {
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "d_feature_embed": 1,
            "d_model": args.d_model,
            "d_pe": 16,
            "dropout": 0.1,
            "n_features": channels,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
        },
        "artifact": artifact.name,
        "artifact_sha256": sha256(artifact),
        "dataset_fingerprint": dataset_fingerprint(args.dataset_path),
        "dataset_path": str(args.dataset_path),
        "elapsed_seconds": time.time() - started,
        "epochs": args.epochs,
        "hostname": platform.node(),
        "injected_validation_values": int(valid_injected.sum()),
        "model": "HELIX (ICML 2026 Spotlight; PyPOTS 1.5)",
        "parameters": int(parameters),
        "patience": args.patience,
        "seed": args.seed,
        "train_ends": train_ends,
        "train_windows": int(train_x.shape[0]),
        "training_mask": "complete-channel dropout; rates sampled from 0.10/0.30/0.50",
        "val_mae": float(np.mean(np.abs(errors))),
        "val_mse": float(np.mean(np.square(errors))),
        "val_windows": int(val_x.shape[0]),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
