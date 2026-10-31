#!/usr/bin/env python3
"""Out-of-sample validation of the calibration gate (Section 5.6, Supplementary Table S12b).

1. Cross-fitting: activation decisions of the cross-fitted gate (weight on the first half of each
   calibration segment, improvement on the second half) against the published gate, and the effect of
   the cross-fitted gate on the forecasts.
2. Test-context reconstruction: because dropout is applied to observed test contexts, the true values of
   a failed channel are known. For every active channel, the mean squared reconstruction error of gated
   ContextGuard is compared with that of the pooled-mean fallback over all windows where the channel
   failed. Reconstruction does not depend on the forecaster; the Chronos-2 runs are used.

Output: ``results/summaries/gate_validation.json``.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from common import RESULTS, compare, effect, metrics, runs, write_json

from contextguard.stats import BOOTSTRAP_REPETITIONS, BOOTSTRAP_SEED


def crossfit_agreement(name: str) -> dict:
    per_dataset = {}
    for key, r in runs(name).items():
        if key[0] in per_dataset:
            continue
        published = (np.asarray(r.gates["contextguard"]["g"]) > 0).astype(int)
        crossfit = (np.asarray(r.gates["contextguard_crossfit"]["g"]) > 0).astype(int)
        per_dataset[key[0]] = {
            "I": r.gates["contextguard"]["I"],
            "g": r.gates["contextguard"]["g"],
            "I_crossfit": r.gates["contextguard_crossfit"]["I"],
            "g_crossfit": r.gates["contextguard_crossfit"]["g"],
            "agree": int((published == crossfit).sum()),
            "channels": int(published.size),
        }
    return {
        "datasets": per_dataset,
        "agree": sum(d["agree"] for d in per_dataset.values()),
        "channels": sum(d["channels"] for d in per_dataset.values()),
    }


def reconstruction(name: str, model: str = "chronos2") -> dict:
    method_err, pooled_err = defaultdict(list), defaultdict(list)
    for key, r in runs(name).items():
        if key[1] != model or key[4] != "contextguard_adaptive":
            continue
        z = r.arrays()
        gate = np.asarray(r.gates["contextguard"]["g"])
        for ch in np.flatnonzero(gate > 0):
            sel = z["dropped"][:, ch] & np.isfinite(z["rec_method"][:, ch])
            if sel.any():
                method_err[(key[0], int(ch))].extend(z["rec_method"][sel, ch].tolist())
                pooled_err[(key[0], int(ch))].extend(z["rec_pooled"][sel, ch].tolist())
    units = sorted(method_err)
    ratio = {u: float(np.mean(method_err[u]) / np.mean(pooled_err[u])) for u in units}
    by_dataset = defaultdict(list)
    for (ds, _), v in ratio.items():
        by_dataset[ds].append(math.log(v))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    datasets = sorted(by_dataset)
    estimates = []
    for _ in range(BOOTSTRAP_REPETITIONS):
        vals = []
        for d in rng.choice(datasets, size=len(datasets)):
            v = np.asarray(by_dataset[d])
            vals.append(rng.choice(v, size=v.size).mean())
        estimates.append(np.mean(vals))
    lo, hi = np.exp(np.quantile(estimates, [0.025, 0.975]))
    return {
        "active_channels": len(units),
        "better_than_fallback": int(sum(v < 1 for v in ratio.values())),
        "geometric_mean_ratio": float(math.exp(np.mean([math.log(v) for v in ratio.values()]))),
        "dataset_cluster_ci95": [float(lo), float(hi)],
        "ratios": {f"{d}|{c}": v for (d, c), v in ratio.items()},
    }


def main() -> None:
    out = {}
    for name in ("primary", "heldout"):
        cf = crossfit_agreement(name)
        rec = reconstruction(name)
        out[name] = {"crossfit_agreement": cf, "test_context_reconstruction": rec}
        print(
            f"[{name}] cross-fitted gate agrees on {cf['agree']}/{cf['channels']} channels; "
            f"gated reconstruction beats the fallback on {rec['better_than_fallback']}/{rec['active_channels']} "
            f"active channels, geometric ratio {rec['geometric_mean_ratio']:.3f} "
            f"[{rec['dataset_cluster_ci95'][0]:.3f}, {rec['dataset_cluster_ci95'][1]:.3f}]"
        )
    s = compare(metrics("primary"), "contextguard_crossfit", "contextguard_adaptive")
    out["primary"]["crossfit_forecast_effect"] = s
    print(
        "[primary] cross-fitted vs published gate: "
        + ", ".join(f"{loss} {effect(s, loss)[0]:+.2f}%" for loss in ("mse", "mae", "quantile_loss"))
    )
    write_json(out, RESULTS / "summaries" / "gate_validation.json")


if __name__ == "__main__":
    main()
