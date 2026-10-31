#!/usr/bin/env python3
"""Revision evaluator: classical estimators, cross-fitted gate, breadth and native missing-value input.

One forecaster load per (dataset, model). For every severity x seed x method the script writes one run
directory (``manifest.json`` + ``window_metrics.npz``) with the same schema as the original evaluator,
plus per-channel losses, the dropped-channel mask, reconstruction errors on the dropped coordinates
(method, zero fallback, pooled-mean fallback) and the gate quantities of every estimator.

Data handling, the corruption generator, the repair path and the Chronos-2 / TimesFM 2.5 wrappers are
imported unchanged from the original evaluator (``original/scripts/run_gift_fault_eval.py``), so the
topology and ContextGuard runs reproduce the original records. All statistics come from the training
prefix that precedes every evaluation window.

Methods
    topology                 topology-aware temporal repair (reference)
    contextguard_adaptive    ContextGuard (gated), the method of the paper
    contextguard             ContextGuard without the gate (weight one)
    contextguard_crossfit    ContextGuard with the cross-fitted gate
    ridge_ungated / ridge_gated            ridge-regression reconstruction, without / with the gate
    lw_ungated / lw_gated                  Ledoit-Wolf reconstruction, without / with the gate
    nan_native               failed channel passed to the forecaster as missing values
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "original" / "scripts"))
sys.path.insert(0, str(HERE.parent / "src"))
import run_gift_fault_eval as base  # noqa: E402

from contextguard.estimators import (  # noqa: E402
    as_repair_inputs,
    crossfit_calibrate,
    gated,
    ledoit_wolf_estimator,
    ridge_estimator,
)
from contextguard.statistics import TAU, calibrate, conditional_predictor, training_segments  # noqa: E402

METHODS = (
    "topology",
    "contextguard_adaptive",
    "contextguard",
    "contextguard_crossfit",
    "ridge_ungated",
    "ridge_gated",
    "lw_ungated",
    "lw_gated",
    "nan_native",
)
SCALAR_METRICS = ("mse", "mae", "raw_mse", "quantile_loss", "coverage80", "width80")


class TiRexPredictor:
    """NX-AI TiRex, applied channel-independently with nine output quantiles (0.1, ..., 0.9)."""

    levels = np.arange(1, 10, dtype=np.float32) / 10.0
    REVISION = "63c740922493f5fbe60b277609ec62babfba2762"

    def __init__(self, repo_id: str):
        from huggingface_hub import hf_hub_download
        from tirex import load_model

        self.checkpoint = hf_hub_download(repo_id, "model.ckpt", revision=self.REVISION)
        self.model = load_model(repo_id, device="cuda", backend="torch", hf_kwargs={"revision": self.REVISION})

    def predict(self, inputs: np.ndarray, horizon: int):
        b, length, c = inputs.shape
        flat = inputs.transpose(0, 2, 1).reshape(b * c, length)
        q, mean = self.model.forecast(flat, output_type="numpy", prediction_length=horizon, batch_size=1024)
        q = np.asarray(q, dtype=np.float32)[:, :horizon]
        mean = np.asarray(mean, dtype=np.float32)[:, :horizon]
        return mean.reshape(b, c, horizon).transpose(0, 2, 1), q.reshape(b, c, horizon, -1).transpose(0, 2, 1, 3)


def estimator_states(records, prediction_length, windows):
    """Training statistics of every estimator and its gate (training prefix only)."""
    means, scales, pooled_mean, covariance, reliability, train_ends = base.training_statistics(
        records, prediction_length, windows, TAU
    )
    fit_rows, cal_rows = training_segments([r[:, :e].T for r, e in zip(records, train_ends)])
    pm, cov = pooled_mean.astype(np.float64), covariance.astype(np.float64)
    cg = calibrate(cal_rows, conditional_predictor(pm, cov), pm)
    cf = crossfit_calibrate(cal_rows, conditional_predictor(pm, cov), pm)
    ridge = ridge_estimator(fit_rows)
    lw = ledoit_wolf_estimator(fit_rows)
    ridge_gate, lw_gate = gated(ridge, cal_rows), gated(lw, cal_rows)
    rm, rc = as_repair_inputs(ridge)
    lm, lc = as_repair_inputs(lw)
    f32 = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731
    states = {
        "topology": ("topology", pooled_mean, covariance, reliability),
        "contextguard_adaptive": ("contextguard_adaptive", pooled_mean, covariance, reliability),
        "contextguard": ("contextguard", pooled_mean, covariance, reliability),
        "contextguard_crossfit": ("contextguard_adaptive", pooled_mean, covariance, f32(cf["g"])),
        "ridge_ungated": ("contextguard", rm, rc, reliability),
        "ridge_gated": ("contextguard_adaptive", rm, rc, f32(ridge_gate["g"])),
        "lw_ungated": ("contextguard", lm, lc, reliability),
        "lw_gated": ("contextguard_adaptive", lm, lc, f32(lw_gate["g"])),
    }
    listify = lambda d: {k: np.asarray(v).tolist() for k, v in d.items()}  # noqa: E731
    shared = {
        "gate0_replica_matches_archive": bool(np.allclose(cg["g"], reliability.astype(np.float64), atol=1e-5)),
        "gates": {
            "contextguard": listify(cg),
            "contextguard_crossfit": listify(cf),
            "ridge": listify(ridge_gate),
            "lw": listify(lw_gate),
        },
        "ridge": {
            "lambda": ridge["ridge"],
            "inner_scores": {str(k): v for k, v in ridge["scores"].items()},
            "n_complete_fit": ridge["n_complete_fit"],
            "n_inner_val": ridge["n_inner_val"],
        },
        "lw": {"shrinkage": lw["shrinkage"], "n_complete_fit": lw["n_complete_fit"]},
        "channel_reliability_archive": reliability.tolist(),
    }
    return means, scales, pooled_mean, train_ends, states, shared


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-key", required=True)
    ap.add_argument("--dataset-path", type=Path, required=True)
    ap.add_argument("--prediction-length", type=int, required=True)
    ap.add_argument("--model", choices=["chronos2", "timesfm25", "tirex"], required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--context-length", type=int, default=512)
    ap.add_argument("--max-windows", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--corruption", default="sensor_dropout", choices=["sensor_dropout", "clean"])
    ap.add_argument("--severities", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[201, 202, 203, 204, 205])
    ap.add_argument("--repairs", nargs="+", required=True, choices=METHODS)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--resume", action="store_true", help="skip runs whose manifest already exists")
    args = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(2021)
    np.random.seed(2021)
    ds = load_from_disk(str(args.dataset_path)).with_format("numpy")
    records = [base.as_target(ds[i]) for i in range(len(ds))]
    fingerprint = base.dataset_fingerprint(args.dataset_path)
    channels = records[0].shape[0]
    windows = base.split_windows(min(r.shape[1] for r in records), args.prediction_length)
    means, scales, pooled_mean, train_ends, states, shared = estimator_states(records, args.prediction_length, windows)
    descriptors = base.descriptors_for_split(
        records, args.prediction_length, windows, args.context_length, args.split, args.max_windows
    )
    if args.model == "tirex":
        predictor = TiRexPredictor(args.model_path)
        model_sha = base.sha256(Path(predictor.checkpoint))
    else:
        ns = argparse.Namespace(
            model=args.model,
            model_path=args.model_path,
            context_length=args.context_length,
            prediction_length=args.prediction_length,
        )
        predictor = base.predictor_for(ns)
        model_sha = base.sha256(Path(args.model_path) / "model.safetensors")
    device = torch.device("cuda")
    units = [(0, 101)] if args.corruption == "clean" else [(s, d) for s in args.severities for d in args.seeds]

    for severity, seed in units:
        for repair in args.repairs:
            name = f"{args.dataset_key}_{args.model}_{args.corruption}_{severity}_{repair}_{seed}"
            out = args.output_root / name
            if args.resume and (out / "manifest.json").exists():
                continue
            started = time.time()
            acc = {
                k: []
                for k in SCALAR_METRICS
                + ("pc_mse", "pc_mae", "pc_ql", "dropped", "natural_any", "rec_method", "rec_zero", "rec_pooled")
            }
            processed, repair_s, forecast_s = [], 0.0, 0.0
            for bstart in range(0, len(descriptors), args.batch_size):
                cur = descriptors[bstart : bstart + args.batch_size]
                ctx = np.asarray(
                    [(records[ri][:, fs - args.context_length : fs].T - means[ri]) / scales[ri] for ri, fs in cur],
                    np.float32,
                )
                tgt = np.asarray(
                    [(records[ri][:, fs : fs + args.prediction_length].T - means[ri]) / scales[ri] for ri, fs in cur],
                    np.float32,
                )
                cm = np.asarray([means[ri] for ri, _ in cur], np.float32)
                cs = np.asarray([scales[ri] for ri, _ in cur], np.float32)
                corrupted, mask = base.corrupt_batch(ctx, args.corruption, severity, seed, bstart)

                torch.cuda.synchronize()
                ts = time.perf_counter()
                if repair == "nan_native":
                    repaired = corrupted.copy()
                    repaired[mask] = np.nan
                else:
                    meth, mu, cv, gate = states[repair]
                    repaired = base.repair_batch(corrupted, mask, meth, mu, cv, gate, device)
                torch.cuda.synchronize()
                repair_s += time.perf_counter() - ts
                ts = time.perf_counter()
                point, quantiles = predictor.predict(repaired, args.prediction_length)
                torch.cuda.synchronize()
                forecast_s += time.perf_counter() - ts

                valid = np.isfinite(tgt)
                sq = np.where(valid, (point - tgt) ** 2, np.nan)
                ab = np.where(valid, np.abs(point - tgt), np.nan)
                acc["mse"].extend(np.nanmean(sq, axis=(1, 2)).tolist())
                acc["mae"].extend(np.nanmean(ab, axis=(1, 2)).tolist())
                raw_p = point * cs[:, None, :] + cm[:, None, :]
                raw_t = tgt * cs[:, None, :] + cm[:, None, :]
                acc["raw_mse"].extend(np.nanmean((raw_p - raw_t) ** 2, axis=(1, 2)).tolist())
                err = tgt[..., None] - quantiles
                lv = predictor.levels.reshape(1, 1, 1, -1)
                pinball = np.maximum(lv * err, (lv - 1.0) * err)
                acc["quantile_loss"].extend((2.0 * np.nanmean(pinball, axis=(1, 2, 3))).tolist())
                lo = quantiles[..., int(np.argmin(np.abs(predictor.levels - 0.1)))]
                hi = quantiles[..., int(np.argmin(np.abs(predictor.levels - 0.9)))]
                inside = ((tgt >= lo) & (tgt <= hi)).astype(np.float32)
                acc["coverage80"].extend(np.nanmean(np.where(valid, inside, np.nan), axis=(1, 2)).tolist())
                acc["width80"].extend(np.nanmean(np.where(valid, hi - lo, np.nan), axis=(1, 2)).tolist())
                with np.errstate(all="ignore"):
                    acc["pc_mse"].append(np.nanmean(sq, axis=1))
                    acc["pc_mae"].append(np.nanmean(ab, axis=1))
                    acc["pc_ql"].append(2.0 * np.nanmean(pinball, axis=(1, 3)))
                    natural = ~np.isfinite(ctx)
                    # a failed channel is masked over the whole context; channels that are entirely
                    # naturally missing are not counted as failed
                    if args.corruption == "clean":
                        dropped = np.zeros((mask.shape[0], mask.shape[2]), dtype=bool)
                    else:
                        dropped = mask.all(axis=1) & ~natural.all(axis=1)
                    acc["dropped"].append(dropped)
                    acc["natural_any"].append(natural.any(axis=1))
                    truth = mask & ~natural

                    def reconstruction_error(values):
                        e = np.where(truth, (values - np.nan_to_num(ctx)) ** 2, np.nan)
                        r = np.nanmean(e, axis=1)
                        r[~dropped] = np.nan
                        return r

                    acc["rec_method"].append(reconstruction_error(np.nan_to_num(repaired)))
                    acc["rec_zero"].append(reconstruction_error(np.zeros_like(ctx)))
                    acc["rec_pooled"].append(reconstruction_error(np.broadcast_to(pooled_mean, ctx.shape)))
                processed.extend(cur)

            arrays = {k: np.asarray(acc[k], dtype=np.float32) for k in SCALAR_METRICS}
            for k in ("pc_mse", "pc_mae", "pc_ql", "rec_method", "rec_zero", "rec_pooled"):
                arrays[k] = np.concatenate(acc[k], 0).astype(np.float32)
            arrays["dropped"] = np.concatenate(acc["dropped"], 0)
            arrays["natural_any"] = np.concatenate(acc["natural_any"], 0)
            arrays["record_index"] = np.asarray([p[0] for p in processed])
            arrays["forecast_start"] = np.asarray([p[1] for p in processed])
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / "window_metrics.npz", **arrays)
            metrics = {k: float(np.nanmean(arrays[k])) for k in SCALAR_METRICS}
            manifest = {
                "args": {
                    "dataset_path": str(args.dataset_path),
                    "dataset_key": args.dataset_key,
                    "model": args.model,
                    "model_path": args.model_path,
                    "split": args.split,
                    "context_length": args.context_length,
                    "prediction_length": args.prediction_length,
                    "max_windows": args.max_windows,
                    "batch_size": args.batch_size,
                    "corruption": args.corruption,
                    "severity": severity,
                    "corruption_seed": seed,
                    "repair": repair,
                },
                "dataset_fingerprint": fingerprint,
                "model_sha256": model_sha,
                "channels": channels,
                "metrics": metrics,
                "windows_evaluated": len(processed),
                "train_ends": train_ends,
                "repair_seconds": repair_s,
                "forecast_seconds": forecast_s,
                "elapsed_seconds": time.time() - started,
                "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
                "hostname": platform.node(),
                **shared,
            }
            (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
            print(f"DONE {name} {json.dumps({k: round(v, 5) for k, v in metrics.items()})}", flush=True)
    print(f"ALL_DONE {args.dataset_key} {args.model} {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
