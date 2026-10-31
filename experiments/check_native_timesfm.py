#!/usr/bin/env python3
"""Native missing-value input for TimesFM 2.5: pass the failed channel as NaN and record the outcome.

For each held-out configuration, the first batch of test contexts is corrupted by sensor dropout
(severity 1, fault seed 201), the failed coordinates are set to NaN, and the contexts are given to the
unchanged TimesFM 2.5 interface of the original evaluator. The outcome (forecast or exception) is written
to ``--output``.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
from datasets import load_from_disk

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "original" / "scripts"))
import run_gift_fault_eval as base  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default="models/google-timesfm-2.5-200m-pytorch")
    ap.add_argument("--configs", type=Path, default=HERE / "configs" / "heldout.tsv")
    ap.add_argument("--output", type=Path, default=Path("logs/revision/native_timesfm25.json"))
    args = ap.parse_args()
    model_sha = base.sha256(Path(args.model_path) / "model.safetensors")
    results = []
    for line in open(args.configs):
        key, path, horizon = line.split()
        horizon = int(horizon)
        ds = load_from_disk(f"data/GiftEval/{path}").with_format("numpy")
        records = [base.as_target(ds[i]) for i in range(len(ds))]
        windows = base.split_windows(min(r.shape[1] for r in records), horizon)
        means, scales, *_ = base.training_statistics(records, horizon, windows, 0.05)
        desc = base.descriptors_for_split(records, horizon, windows, 512, "test", 256)[:32]
        ctx = np.asarray([(records[r][:, f - 512 : f].T - means[r]) / scales[r] for r, f in desc], dtype=np.float32)
        corrupted, mask = base.corrupt_batch(ctx, "sensor_dropout", 1, 201, 0)
        native = corrupted.copy()
        native[mask] = np.nan
        predictor = base.TimesFMPredictor(args.model_path, 512, horizon)
        try:
            point, _ = predictor.predict(native, horizon)
            outcome = {"status": "forecast", "finite_fraction": float(np.isfinite(point).mean())}
        except Exception as exc:  # the outcome itself is the measurement
            outcome = {
                "status": "error",
                "exception": type(exc).__name__,
                "message": str(exc).splitlines()[0][:300],
                "location": traceback.extract_tb(exc.__traceback__)[-1].name,
            }
        results.append(
            {
                "dataset": key,
                "dataset_fingerprint": base.dataset_fingerprint(Path(f"data/GiftEval/{path}")),
                "windows": len(desc),
                "failed_channels_per_window": int(mask.all(axis=1).sum(axis=1).max()),
                **outcome,
            }
        )
        print(key, outcome, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "model": "timesfm25",
                "model_sha256": model_sha,
                "corruption": "sensor_dropout",
                "severity": 1,
                "corruption_seed": 201,
                "results": results,
            },
            indent=1,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
