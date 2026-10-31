#!/usr/bin/env python3
"""Stability of cross-channel relations after training (Section 5.7): test-context consistency.

For every test window and dropout pattern of the primary design, each observed channel is reconstructed
from the other observed channels with the training statistics (Eq. 2) inside the current context, and
scored by R^2 against the pooled-mean fallback. The consistency of a dataset is the mean over windows
and failed channels; it is defined for datasets with at least three channels (two observed channels are
needed). No forecaster and no label is involved. Requires the GIFT-Eval data (``experiments/download_assets.py``).

Output: ``results/summaries/test_context_consistency.json``.
"""

from __future__ import annotations

import sys

import numpy as np
from common import RESULTS, ROOT, write_json
from datasets import load_from_disk

sys.path.insert(0, str(ROOT / "experiments" / "original" / "scripts"))
import run_gift_fault_eval as base  # noqa: E402

CONTEXT, MAX_WINDOWS, SEVERITIES, SEEDS = 512, 256, (1, 3), (201, 202, 203, 204, 205)


def conditional(mean, cov, j, observed, x):
    coef = np.linalg.solve(cov[np.ix_(observed, observed)] + 1e-3 * np.eye(len(observed)), cov[j, observed])
    return mean[j] + (x[:, observed] - mean[observed]) @ coef


def dataset_consistency(path: str, horizon: int) -> dict:
    ds = load_from_disk(str(ROOT / "data" / "GiftEval" / path)).with_format("numpy")
    records = [base.as_target(ds[i]) for i in range(len(ds))]
    channels = records[0].shape[0]
    windows = base.split_windows(min(r.shape[1] for r in records), horizon)
    means, scales, pm, cov, _, _ = base.training_statistics(records, horizon, windows, 0.05)
    pm, cov = pm.astype(float), cov.astype(float)
    desc = base.descriptors_for_split(records, horizon, windows, CONTEXT, "test", MAX_WINDOWS)
    ctx = np.asarray([(records[ri][:, fs - CONTEXT : fs].T - means[ri]) / scales[ri] for ri, fs in desc], np.float32)
    scores = []
    for severity in SEVERITIES:
        for seed in SEEDS:
            for start in range(0, len(desc), 32):
                batch = ctx[start : start + 32]
                _, mask = base.corrupt_batch(batch, "sensor_dropout", severity, seed, start)
                for w in range(batch.shape[0]):
                    x = batch[w].astype(float)
                    finite = np.isfinite(x)
                    failed = [j for j in range(channels) if mask[w, :, j].all() and finite[:, j].any()]
                    observed = [k for k in range(channels) if k not in failed]
                    if len(observed) < 2 or not failed:
                        continue
                    rows = finite[:, observed].all(1)
                    if rows.sum() < 16:
                        continue
                    xo = x[rows]
                    sse = sst = 0.0
                    for k in observed:
                        others = np.array([o for o in observed if o != k])
                        sse += np.sum((xo[:, k] - conditional(pm, cov, k, others, xo)) ** 2)
                        sst += np.sum((xo[:, k] - pm[k]) ** 2)
                    score = 1 - sse / max(sst, 1e-12)
                    # one entry per failed channel with enough finite truth rows in the window
                    scores += [score for j in failed if (rows & finite[:, j]).sum() >= 16]
    return {"channels": channels, "windows_scored": len(scores), "consistency": float(np.mean(scores))}


def main() -> None:
    out = {}
    for line in open(ROOT / "experiments" / "configs" / "primary.tsv"):
        key, path, horizon = line.split()
        ds = load_from_disk(str(ROOT / "data" / "GiftEval" / path))
        if base.as_target(ds[0]).shape[0] < 3:
            continue
        out[key] = dataset_consistency(path, int(horizon))
        print(f"{key:22s} channels {out[key]['channels']:3d}  consistency {out[key]['consistency']:+.3f}", flush=True)
    write_json(out, RESULTS / "summaries" / "test_context_consistency.json")


if __name__ == "__main__":
    main()
