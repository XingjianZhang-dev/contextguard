#!/usr/bin/env python3
"""Evaluate clean-safe context repair on multivariate GIFT-Eval windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk


MISSING_RATE = {1: 0.10, 2: 0.30, 3: 0.50}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(file_path.relative_to(path)).encode())
        digest.update(bytes.fromhex(sha256(file_path)))
    return digest.hexdigest()


def as_target(entry: dict) -> np.ndarray:
    target = np.asarray(entry["target"], dtype=np.float32)
    if target.ndim == 1:
        target = target[None, :]
    if target.ndim != 2:
        raise ValueError(f"Expected target shape (channels, time), got {target.shape}")
    return target


def split_windows(minimum_length: int, prediction_length: int) -> int:
    count = math.ceil(0.10 * minimum_length / prediction_length)
    return min(max(1, count), 20)


def training_statistics(
    records: list[np.ndarray],
    prediction_length: int,
    windows: int,
    reliability_threshold: float = 0.05,
    max_rows_per_record: int = 10_000,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[int],
]:
    means: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    train_ends: list[int] = []
    training_arrays: list[np.ndarray] = []

    channels = records[0].shape[0]
    global_sum = np.zeros(channels, dtype=np.float64)
    global_squared_sum = np.zeros(channels, dtype=np.float64)
    global_count = np.zeros(channels, dtype=np.int64)

    for target in records:
        train_end = target.shape[1] - prediction_length * (windows + 1)
        if train_end < 2:
            raise ValueError("Training portion is too short for leakage-free statistics")
        train = target[:, :train_end].T.astype(np.float64, copy=False)
        finite = np.isfinite(train)
        global_sum += np.where(finite, train, 0.0).sum(axis=0)
        global_squared_sum += np.where(finite, train * train, 0.0).sum(axis=0)
        global_count += finite.sum(axis=0)
        training_arrays.append(train)
        train_ends.append(train_end)

    if np.any(global_count < 2):
        raise ValueError("A channel has fewer than two finite training observations")
    global_mean = global_sum / global_count
    global_variance = global_squared_sum / global_count - global_mean**2
    global_scale = np.sqrt(np.maximum(global_variance, 1e-12))

    fit_rows: list[np.ndarray] = []
    calibration_rows: list[np.ndarray] = []
    for train in training_arrays:
        finite = np.isfinite(train)
        count = finite.sum(axis=0)
        finite_sum = np.where(finite, train, 0.0).sum(axis=0)
        mean = np.divide(
            finite_sum,
            count,
            out=np.full(channels, np.nan, dtype=np.float64),
            where=count > 0,
        )
        squared_deviation = np.where(finite, np.square(train - mean), 0.0).sum(axis=0)
        variance = np.divide(
            squared_deviation,
            count,
            out=np.full(channels, np.nan, dtype=np.float64),
            where=count > 0,
        )
        scale = np.sqrt(variance)
        mean = np.where(np.isfinite(mean), mean, global_mean)
        scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, global_scale)
        scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
        standardized = (train - mean) / scale
        stride = max(1, math.ceil(standardized.shape[0] / max_rows_per_record))
        standardized = standardized[::stride]
        split = max(1, int(0.8 * standardized.shape[0]))
        fit_rows.append(standardized[:split])
        if split < standardized.shape[0]:
            calibration_rows.append(standardized[split:])
        means.append(mean.astype(np.float32))
        scales.append(scale.astype(np.float32))

    marginal_sum = np.zeros(channels, dtype=np.float64)
    marginal_count = np.zeros(channels, dtype=np.int64)
    for rows in fit_rows:
        finite = np.isfinite(rows)
        marginal_sum += np.where(finite, rows, 0.0).sum(axis=0)
        marginal_count += finite.sum(axis=0)
    pooled_mean = np.divide(
        marginal_sum,
        marginal_count,
        out=np.zeros_like(marginal_sum),
        where=marginal_count > 0,
    )

    pooled_outer = np.zeros((channels, channels), dtype=np.float64)
    pair_count = np.zeros((channels, channels), dtype=np.int64)
    for rows in fit_rows:
        finite = np.isfinite(rows)
        centered = np.where(finite, rows - pooled_mean, 0.0)
        pooled_outer += centered.T @ centered
        pair_count += finite.astype(np.int64).T @ finite.astype(np.int64)
    covariance = np.divide(
        pooled_outer,
        np.maximum(pair_count - 1, 1),
        out=np.zeros_like(pooled_outer),
        where=pair_count > 1,
    )
    covariance = 0.5 * (covariance + covariance.T)
    diagonal = np.diag(covariance).copy()
    diagonal = np.where(np.isfinite(diagonal) & (diagonal > 1e-6), diagonal, 1.0)
    np.fill_diagonal(covariance, diagonal)
    covariance += 1e-6 * np.eye(covariance.shape[0])

    reliability = np.zeros(channels, dtype=np.float64)
    if calibration_rows:
        calibration = np.concatenate(calibration_rows, axis=0)
        for target_channel in range(channels):
            observed_channels = np.arange(channels) != target_channel
            valid = np.isfinite(calibration[:, target_channel]) & np.isfinite(
                calibration[:, observed_channels]
            ).all(axis=1)
            if valid.sum() < max(32, 4 * channels):
                continue
            x_other = calibration[valid][:, observed_channels]
            y = calibration[valid, target_channel] - pooled_mean[target_channel]
            cov_other = covariance[np.ix_(observed_channels, observed_channels)]
            cross = covariance[target_channel, observed_channels]
            coefficient = np.linalg.solve(
                cov_other + 1e-3 * np.eye(cov_other.shape[0]), cross
            )
            prediction = (x_other - pooled_mean[observed_channels]) @ coefficient
            denominator = float(prediction @ prediction)
            if denominator <= 1e-12:
                continue
            weight = float(np.clip((prediction @ y) / denominator, 0.0, 1.0))
            baseline_error = float(y @ y)
            blended_error = float(np.square(y - weight * prediction).sum())
            improvement = (baseline_error - blended_error) / max(baseline_error, 1e-12)
            if improvement >= reliability_threshold:
                reliability[target_channel] = weight

    return (
        means,
        scales,
        pooled_mean.astype(np.float32),
        covariance.astype(np.float32),
        reliability.astype(np.float32),
        train_ends,
    )


def descriptors_for_split(
    records: list[np.ndarray],
    prediction_length: int,
    windows: int,
    context_length: int,
    split: str,
    max_windows: int,
) -> list[tuple[int, int]]:
    descriptors: list[tuple[int, int]] = []
    for record_index, target in enumerate(records):
        length = target.shape[1]
        if split == "val":
            starts = [length - prediction_length * (windows + 1)]
        else:
            first = length - prediction_length * windows
            starts = [first + index * prediction_length for index in range(windows)]
        descriptors.extend(
            (record_index, start)
            for start in starts
            if start >= context_length and start + prediction_length <= length
        )
    if not descriptors:
        raise ValueError("No evaluation windows satisfy the requested context length")
    if max_windows > 0 and len(descriptors) > max_windows:
        selected = np.linspace(0, len(descriptors) - 1, max_windows, dtype=np.int64)
        descriptors = [descriptors[index] for index in selected]
    return descriptors


def corrupt_batch(
    context: np.ndarray,
    corruption: str,
    severity: int,
    seed: int,
    sample_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    output = context.copy()
    mask = ~np.isfinite(output)
    output[mask] = 0.0
    if corruption == "clean":
        return output, mask
    rate = MISSING_RATE[severity]
    batch, length, channels = output.shape
    for batch_index in range(batch):
        rng = np.random.default_rng(seed + 1_000_003 * (sample_offset + batch_index + 1))
        if corruption == "random_missing":
            current = rng.random((length, channels)) < rate
        elif corruption == "block_missing":
            current = np.zeros((length, channels), dtype=bool)
            block_length = max(1, round(length * rate))
            for channel in range(channels):
                start = int(rng.integers(0, length - block_length + 1))
                current[start:start + block_length, channel] = True
        elif corruption == "sensor_dropout":
            current = np.zeros((length, channels), dtype=bool)
            count = max(1, math.ceil(rate * channels))
            count = min(count, max(1, channels - 1))
            dropped = rng.choice(channels, size=count, replace=False)
            current[:, dropped] = True
        else:
            raise ValueError(f"Unknown corruption: {corruption}")
        mask[batch_index] |= current
        output[batch_index][current] = 0.0
    return output, mask


def forward_fill(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    output = x.clone()
    for index in range(1, output.shape[1]):
        output[:, index, :] = torch.where(
            mask[:, index, :], output[:, index - 1, :], output[:, index, :]
        )
    return output


def gap_lengths(mask: torch.Tensor) -> torch.Tensor:
    batch, length, channels = mask.shape
    forward = torch.zeros_like(mask, dtype=torch.int32)
    backward = torch.zeros_like(mask, dtype=torch.int32)
    running = torch.zeros((batch, channels), dtype=torch.int32, device=mask.device)
    for index in range(length):
        running = torch.where(mask[:, index, :], running + 1, 0)
        forward[:, index, :] = running
    running.zero_()
    for index in range(length - 1, -1, -1):
        running = torch.where(mask[:, index, :], running + 1, 0)
        backward[:, index, :] = running
    return torch.where(mask, forward + backward - 1, torch.zeros_like(forward))


def conditional_reconstruct(
    x: torch.Tensor,
    mask: torch.Tensor,
    mean: torch.Tensor,
    covariance: torch.Tensor,
) -> torch.Tensor:
    channels = x.shape[-1]
    flat_x = x.reshape(-1, channels)
    flat_mask = mask.reshape(-1, channels)
    output = flat_x.clone()
    for start in range(0, flat_x.shape[0], 4096):
        stop = min(start + 4096, flat_x.shape[0])
        current_x = flat_x[start:stop]
        current_mask = flat_mask[start:stop]
        observed = (~current_mask).to(x.dtype)
        system = covariance.view(1, channels, channels)
        system = system * observed.unsqueeze(2) * observed.unsqueeze(1)
        diagonal = torch.where(
            current_mask,
            torch.ones_like(observed),
            torch.full_like(observed, 1e-3),
        )
        system = system + torch.diag_embed(diagonal)
        right_hand_side = (current_x - mean) * observed
        weights = torch.linalg.solve(system, right_hand_side.unsqueeze(-1)).squeeze(-1)
        predictions = torch.matmul(weights, covariance.transpose(0, 1)) + mean
        output[start:stop] = torch.where(current_mask, predictions, current_x)
    return output.reshape_as(x)


def repair_batch(
    x: np.ndarray,
    mask: np.ndarray,
    method: str,
    pooled_mean: np.ndarray,
    covariance: np.ndarray,
    channel_reliability: np.ndarray,
    device: torch.device,
    short_blend: float = 0.25,
    missing_activation: float = 0.20,
    long_gap_fraction_threshold: float = 0.05,
    short_gap_max: int = 8,
    topology_gap_max: int = 32,
) -> np.ndarray:
    if method == "none" or not mask.any():
        return x
    tensor = torch.as_tensor(x, device=device)
    tensor_mask = torch.as_tensor(mask, device=device)
    forward = forward_fill(tensor, tensor_mask)
    if method == "ffill":
        return forward.cpu().numpy()
    backward = forward_fill(tensor.flip(1), tensor_mask.flip(1)).flip(1)
    temporal = torch.where(tensor_mask, 0.5 * (forward + backward), tensor)
    if method == "bidirectional":
        return temporal.cpu().numpy()

    lengths = gap_lengths(tensor_mask)
    topology = torch.where(tensor_mask & (lengths <= topology_gap_max), temporal, forward)
    if method == "topology":
        return torch.where(tensor_mask, topology, tensor).cpu().numpy()
    if method not in {"contextguard", "contextguard_adaptive"}:
        raise ValueError(f"Unknown repair method: {method}")

    mean = torch.as_tensor(pooled_mean, device=device)
    cov = torch.as_tensor(covariance, device=device)
    conditional = conditional_reconstruct(temporal, tensor_mask, mean, cov)
    reliability = torch.ones_like(mean)
    if method == "contextguard_adaptive":
        reliability = torch.as_tensor(channel_reliability, device=device)
    missing_fraction = tensor_mask.float().mean(dim=(1, 2), keepdim=True)
    long_gap_fraction = (lengths > topology_gap_max).float().mean(dim=(1, 2), keepdim=True)
    active_weight = (
        short_blend
        * (missing_fraction >= missing_activation).to(tensor.dtype)
        * (long_gap_fraction < long_gap_fraction_threshold).to(tensor.dtype)
        * reliability.view(1, 1, -1)
    )
    short_repair = (1.0 - active_weight) * temporal + active_weight * conditional
    repaired = torch.where(tensor_mask & (lengths <= short_gap_max), short_repair, forward)
    full_weight = reliability.view(1, 1, -1)
    full_repair = (1.0 - full_weight) * forward + full_weight * conditional
    repaired = torch.where(tensor_mask.all(dim=1).unsqueeze(1), full_repair, repaired)
    return torch.where(tensor_mask, repaired, tensor).cpu().numpy()


class ChronosPredictor:
    levels = np.asarray([0.1, 0.5, 0.9], dtype=np.float32)

    def __init__(self, model_path: str):
        from chronos import BaseChronosPipeline

        self.pipeline = BaseChronosPipeline.from_pretrained(
            model_path, device_map="cuda", local_files_only=True
        )

    def predict(self, inputs: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        # Chronos-2 natively shares information across the channels within
        # each multivariate task, so preserve (batch, channels, history).
        native_inputs = torch.as_tensor(inputs.transpose(0, 2, 1))
        quantiles, point = self.pipeline.predict_quantiles(
            native_inputs,
            prediction_length=horizon,
            quantile_levels=self.levels.tolist(),
            batch_size=256,
            context_length=inputs.shape[1],
        )
        point_array = torch.stack(point).numpy().transpose(0, 2, 1)
        quantile_array = torch.stack(quantiles).numpy().transpose(0, 2, 1, 3)
        return point_array, quantile_array


class TimesFMPredictor:
    levels = np.arange(1, 10, dtype=np.float32) / 10.0

    def __init__(self, model_path: str, context_length: int, horizon: int):
        import timesfm

        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            model_path, torch_compile=False, local_files_only=True
        )
        self.model.compile(
            timesfm.ForecastConfig(
                max_context=context_length,
                max_horizon=horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
                per_core_batch_size=128,
            )
        )

    def predict(self, inputs: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        batch, length, channels = inputs.shape
        flattened = inputs.transpose(0, 2, 1).reshape(batch * channels, length)
        point, quantiles = self.model.forecast(
            horizon=horizon, inputs=[row for row in flattened]
        )
        point = point.reshape(batch, channels, horizon).transpose(0, 2, 1)
        quantiles = quantiles[..., 1:].reshape(
            batch, channels, horizon, -1
        ).transpose(0, 2, 1, 3)
        return point, quantiles


class HelixRepairer:
    """Load a training-only HELIX checkpoint for a strong learned baseline."""

    def __init__(
        self,
        artifact_dir: Path,
        expected_fingerprint: str,
        expected_context_length: int,
        expected_channels: int,
    ):
        self.manifest = json.loads(
            (artifact_dir / "manifest.json").read_text(encoding="utf-8")
        )
        architecture = self.manifest["architecture"]
        if self.manifest["dataset_fingerprint"] != expected_fingerprint:
            raise ValueError("HELIX artifact was trained on a different dataset")
        if architecture["context_length"] != expected_context_length:
            raise ValueError("HELIX artifact context length does not match evaluation")
        if architecture["n_features"] != expected_channels:
            raise ValueError("HELIX artifact channel count does not match evaluation")
        from pypots.imputation import HELIX

        self.model = HELIX(
            n_steps=architecture["context_length"],
            n_features=architecture["n_features"],
            d_pe=architecture["d_pe"],
            d_feature_embed=architecture["d_feature_embed"],
            d_model=architecture["d_model"],
            n_heads=architecture["n_heads"],
            n_layers=architecture["n_layers"],
            dropout=architecture["dropout"],
            ORT_weight=1.0,
            MIT_weight=1.0,
            batch_size=architecture["batch_size"],
            epochs=1,
            patience=None,
            num_workers=0,
            device="cuda",
            saving_path=None,
            model_saving_strategy=None,
            verbose=False,
        )
        self.artifact = Path(self.manifest["artifact"])
        if not self.artifact.is_absolute():
            self.artifact = artifact_dir / self.artifact
        recorded_hash = self.manifest.get("artifact_sha256")
        if recorded_hash is not None and sha256(self.artifact) != recorded_hash:
            raise ValueError("HELIX artifact hash does not match its manifest")
        self.model.load(str(self.artifact))

    def impute(self, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        incomplete = np.where(mask, np.nan, values)
        imputed = self.model.impute({"X": incomplete})
        return np.where(mask, imputed, values).astype(np.float32, copy=False)


def predictor_for(args: argparse.Namespace):
    if args.model == "chronos2":
        return ChronosPredictor(args.model_path)
    if args.model == "timesfm25":
        return TimesFMPredictor(args.model_path, args.context_length, args.prediction_length)
    raise ValueError(args.model)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--model", choices=["chronos2", "timesfm25"], required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--prediction-length", type=int, required=True)
    parser.add_argument("--max-windows", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--corruption",
        choices=["clean", "random_missing", "block_missing", "sensor_dropout"],
        default="clean",
    )
    parser.add_argument("--severity", type=int, choices=[0, 1, 2, 3], default=0)
    parser.add_argument("--corruption-seed", type=int, default=101)
    parser.add_argument("--reliability-threshold", type=float, default=0.05)
    parser.add_argument("--short-blend", type=float, default=0.25)
    parser.add_argument("--missing-activation", type=float, default=0.20)
    parser.add_argument("--long-gap-fraction-threshold", type=float, default=0.05)
    parser.add_argument("--short-gap-max", type=int, default=8)
    parser.add_argument("--topology-gap-max", type=int, default=32)
    parser.add_argument(
        "--repair",
        choices=[
            "none",
            "ffill",
            "bidirectional",
            "topology",
            "contextguard",
            "contextguard_adaptive",
            "helix",
        ],
        default="none",
    )
    parser.add_argument("--imputer-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.corruption == "clean" and args.severity != 0:
        raise ValueError("Clean evaluation requires severity 0")
    if args.corruption != "clean" and args.severity == 0:
        raise ValueError("Corrupted evaluation requires severity 1, 2, or 3")
    if not 0.0 <= args.reliability_threshold <= 1.0:
        raise ValueError("Reliability threshold must be in [0, 1]")
    if not 0.0 <= args.short_blend <= 1.0:
        raise ValueError("Short blend must be in [0, 1]")
    if not 0.0 <= args.missing_activation <= 1.0:
        raise ValueError("Missing activation must be in [0, 1]")
    if not 0.0 <= args.long_gap_fraction_threshold <= 1.0:
        raise ValueError("Long-gap fraction threshold must be in [0, 1]")
    if args.short_gap_max < 1 or args.topology_gap_max < args.short_gap_max:
        raise ValueError("Gap maxima must satisfy 1 <= short <= topology")
    if args.repair == "helix" and args.imputer_path is None:
        raise ValueError("HELIX repair requires --imputer-path")

    started = time.time()
    torch.manual_seed(2021)
    np.random.seed(2021)
    dataset = load_from_disk(str(args.dataset_path)).with_format("numpy")
    records = [as_target(dataset[index]) for index in range(len(dataset))]
    fingerprint = dataset_fingerprint(args.dataset_path)
    channels = records[0].shape[0]
    if any(record.shape[0] != channels for record in records):
        raise ValueError("All records must have the same channel count")
    minimum_length = min(record.shape[1] for record in records)
    windows = split_windows(minimum_length, args.prediction_length)
    means, scales, pooled_mean, covariance, channel_reliability, train_ends = training_statistics(
        records, args.prediction_length, windows, args.reliability_threshold
    )
    descriptors = descriptors_for_split(
        records,
        args.prediction_length,
        windows,
        args.context_length,
        args.split,
        args.max_windows,
    )
    predictor = predictor_for(args)
    imputer = (
        HelixRepairer(
            args.imputer_path,
            fingerprint,
            args.context_length,
            channels,
        )
        if args.repair == "helix"
        else None
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    mse_values = []
    mae_values = []
    raw_mse_values = []
    quantile_loss_values = []
    coverage_values = []
    width_values = []
    processed_descriptors = []
    repair_seconds = 0.0
    forecast_seconds = 0.0

    for batch_start in range(0, len(descriptors), args.batch_size):
        current = descriptors[batch_start:batch_start + args.batch_size]
        contexts = []
        targets = []
        current_means = []
        current_scales = []
        for record_index, forecast_start in current:
            target = records[record_index]
            mean = means[record_index]
            scale = scales[record_index]
            context = target[
                :, forecast_start - args.context_length:forecast_start
            ].T
            future = target[
                :, forecast_start:forecast_start + args.prediction_length
            ].T
            contexts.append((context - mean) / scale)
            targets.append((future - mean) / scale)
            current_means.append(mean)
            current_scales.append(scale)
        contexts = np.asarray(contexts, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        current_means = np.asarray(current_means, dtype=np.float32)
        current_scales = np.asarray(current_scales, dtype=np.float32)
        corrupted, mask = corrupt_batch(
            contexts,
            args.corruption,
            args.severity,
            args.corruption_seed,
            batch_start,
        )
        torch.cuda.synchronize()
        operation_started = time.perf_counter()
        if imputer is not None:
            repaired = imputer.impute(corrupted, mask)
        else:
            repaired = repair_batch(
                corrupted,
                mask,
                args.repair,
                pooled_mean,
                covariance,
                channel_reliability,
                torch.device("cuda"),
                args.short_blend,
                args.missing_activation,
                args.long_gap_fraction_threshold,
                args.short_gap_max,
                args.topology_gap_max,
            )
        torch.cuda.synchronize()
        repair_seconds += time.perf_counter() - operation_started
        operation_started = time.perf_counter()
        point, quantiles = predictor.predict(repaired, args.prediction_length)
        torch.cuda.synchronize()
        forecast_seconds += time.perf_counter() - operation_started
        valid = np.isfinite(targets)
        squared = np.where(valid, (point - targets) ** 2, np.nan)
        absolute = np.where(valid, np.abs(point - targets), np.nan)
        mse_values.extend(np.nanmean(squared, axis=(1, 2)).tolist())
        mae_values.extend(np.nanmean(absolute, axis=(1, 2)).tolist())
        raw_point = point * current_scales[:, None, :] + current_means[:, None, :]
        raw_target = targets * current_scales[:, None, :] + current_means[:, None, :]
        raw_mse_values.extend(
            np.nanmean((raw_point - raw_target) ** 2, axis=(1, 2)).tolist()
        )
        error = targets[..., None] - quantiles
        levels = predictor.levels.reshape(1, 1, 1, -1)
        pinball = np.maximum(levels * error, (levels - 1.0) * error)
        quantile_loss_values.extend((2.0 * np.nanmean(pinball, axis=(1, 2, 3))).tolist())
        lower_index = int(np.argmin(np.abs(predictor.levels - 0.1)))
        upper_index = int(np.argmin(np.abs(predictor.levels - 0.9)))
        lower = quantiles[..., lower_index]
        upper = quantiles[..., upper_index]
        covered = np.where(
            valid, ((targets >= lower) & (targets <= upper)).astype(np.float32), np.nan
        )
        coverage_values.extend(
            np.nanmean(covered, axis=(1, 2)).tolist()
        )
        width_values.extend(
            np.nanmean(np.where(valid, upper - lower, np.nan), axis=(1, 2)).tolist()
        )
        processed_descriptors.extend(current)

    arrays = {
        "mse": np.asarray(mse_values, dtype=np.float32),
        "mae": np.asarray(mae_values, dtype=np.float32),
        "raw_mse": np.asarray(raw_mse_values, dtype=np.float32),
        "quantile_loss": np.asarray(quantile_loss_values, dtype=np.float32),
        "coverage80": np.asarray(coverage_values, dtype=np.float32),
        "width80": np.asarray(width_values, dtype=np.float32),
        "record_index": np.asarray([item[0] for item in processed_descriptors]),
        "forecast_start": np.asarray([item[1] for item in processed_descriptors]),
        "pooled_mean": pooled_mean,
        "pooled_covariance": covariance,
        "channel_reliability": channel_reliability,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "window_metrics.npz", **arrays)
    metrics = {
        name: float(np.nanmean(value))
        for name, value in arrays.items()
        if name in {"mse", "mae", "raw_mse", "quantile_loss", "coverage80", "width80"}
    }
    manifest = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "channels": channels,
        "dataset_examples": len(records),
        "dataset_fingerprint": fingerprint,
        "elapsed_seconds": time.time() - started,
        "repair_seconds": repair_seconds,
        "forecast_seconds": forecast_seconds,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "wrapper_trainable_parameters": 0,
        "imputer_parameters": int(imputer.manifest["parameters"]) if imputer else 0,
        "imputer_sha256": sha256(imputer.artifact) if imputer else None,
        "hostname": platform.node(),
        "metrics": metrics,
        "channel_reliability": channel_reliability.tolist(),
        "model_sha256": sha256(Path(args.model_path) / "model.safetensors"),
        "test_windows_in_protocol": windows,
        "train_ends": train_ends,
        "windows_evaluated": len(processed_descriptors),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
