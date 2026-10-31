#!/usr/bin/env python3
"""What determines the gain (Section 5.7, Table 5, Fig. 4, Supplementary Table S12c).

The forecast improvement equals the loss caused by dropout times the fraction of that loss which repair
recovers. With the clean-input forecast of the same windows as reference,

    dropout loss  = L_topology / L_clean - 1
    recovered     = (L_topology - L_ContextGuard) / (L_topology - L_clean)

Dataset level: means over the paired cells of each dataset and forecaster. Channel level: means over all
windows in which the channel failed, for the channel-independent forecasters (TimesFM 2.5, TiRex), whose
forecast of a channel depends on that channel's input alone. Channel-level predictors are the mean and
maximum absolute correlation with the other channels and the calibration improvement ``I_j`` (all from
training data), and the test-context reconstruction R^2 of the failed channel.

Outputs: ``results/summaries/determinants.json`` and ``results/summaries/training_correlations.json``
(correlation matrices of the training covariance, unit diagonal).
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from common import ORIGINAL, RESULTS, ROOT, runs, write_json
from scipy.stats import spearmanr

MIN_WINDOWS = 10  # channel units need at least ten failed windows
MIN_LOSS = 0.05  # and a dropout loss of at least 5% (otherwise the recovered fraction is undefined)


def mean_improvement(gates: dict) -> float:
    return float(np.nanmean(np.clip(np.asarray(gates["contextguard"]["I"], float), 0, 1)))


def dataset_units(set_name: str, clean_name: str, models, metric: str = "mse") -> list[dict]:
    R, C = runs(set_name), runs(clean_name)
    out = []
    for ds in sorted({k[0] for k in R}):
        for mod in models:
            cs = sorted(
                k[:4]
                for k in R
                if k[0] == ds and k[1] == mod and k[4] == "topology" and k[:4] + ("contextguard_adaptive",) in R
            )
            if not cs or (ds, mod, "topology") not in C:
                continue
            t = np.mean([R[c + ("topology",)].metrics[metric] for c in cs])
            g = np.mean([R[c + ("contextguard_adaptive",)].metrics[metric] for c in cs])
            l0 = C[(ds, mod, "topology")].metrics[metric]
            out.append(
                {
                    "dataset": ds,
                    "model": mod,
                    "metric": metric,
                    "loss_increase": float(t / l0 - 1),
                    "recovered": float((t - g) / (t - l0)),
                    "mean_I": mean_improvement(R[cs[0] + ("topology",)].gates),
                }
            )
    return out


def channel_units(set_name: str, clean_name: str, models) -> list[dict]:
    R, C = runs(set_name), runs(clean_name)
    out = []
    for ds in sorted({k[0] for k in R}):
        for mod in models:
            cs = sorted(
                k[:4]
                for k in R
                if k[0] == ds and k[1] == mod and k[4] == "topology" and k[:4] + ("contextguard_adaptive",) in R
            )
            if not cs or (ds, mod, "topology") not in C:
                continue
            zc = C[(ds, mod, "topology")].arrays()
            improvement = np.asarray(R[cs[0] + ("topology",)].gates["contextguard"]["I"], float)
            acc = defaultdict(lambda: defaultdict(list))
            for c in cs:
                zt = R[c + ("topology",)].arrays()
                zg = R[c + ("contextguard_adaptive",)].arrays()
                for j in range(zt["dropped"].shape[1]):
                    sel = zt["dropped"][:, j]
                    if not sel.any():
                        continue
                    for k, z, field in (
                        ("t", zt, "pc_mse"),
                        ("g", zg, "pc_mse"),
                        ("c", zc, "pc_mse"),
                        ("rm", zg, "rec_method"),
                        ("rp", zg, "rec_pooled"),
                    ):
                        acc[j][k].extend(z[field][sel, j])
            for j, a in sorted(acc.items()):
                t, g, c0 = (np.nanmean(a[k]) for k in ("t", "g", "c"))
                out.append(
                    {
                        "dataset": ds,
                        "model": mod,
                        "channel": int(j),
                        "windows": len(a["t"]),
                        "loss_increase": float(t / c0 - 1),
                        "recovered": float((t - g) / (t - c0)) if t > c0 else float("nan"),
                        "R2": float(1 - np.nanmean(a["rm"]) / np.nanmean(a["rp"])),
                        "I": float(np.clip(improvement[j], 0, 1)) if np.isfinite(improvement[j]) else 0.0,
                    }
                )
    return out


def eligible(units: list[dict]) -> list[dict]:
    return [
        u
        for u in units
        if u["windows"] >= MIN_WINDOWS
        and u["loss_increase"] >= MIN_LOSS
        and np.isfinite(u["recovered"])
        and np.isfinite(u["R2"])
    ]


def rho(units, x, y="recovered") -> float:
    return float(spearmanr([u[y] for u in units], [u[x] for u in units]).correlation)


def training_correlations() -> dict[str, np.ndarray]:
    """Correlation matrices of the training covariance, read from the original run records."""
    paths = {line.split()[1]: line.split()[0] for line in open(ROOT / "experiments/configs/primary.tsv")}
    out = {}
    for npz in sorted((ORIGINAL / "gift_sensor_v4_formal_test").glob("*contextguard_adaptive*/window_metrics.npz")):
        args = json.loads((npz.parent / "manifest.json").read_text())["args"]
        key = paths[args["dataset_path"].removeprefix("data/GiftEval/")]
        if key in out:
            continue
        cov = np.load(npz)["pooled_covariance"].astype(np.float64)
        sd = np.sqrt(np.diag(cov))
        r = cov / np.outer(sd, sd)
        np.fill_diagonal(r, np.nan)
        out[key] = r
    return out


def main() -> None:
    # dataset level, primary configurations
    primary = dataset_units("primary", "primary_clean", ("chronos2", "timesfm25"), "mse") + dataset_units(
        "primary", "primary_clean", ("chronos2", "timesfm25"), "quantile_loss"
    )
    rho_primary = {m: rho([u for u in primary if u["metric"] == m], "mean_I") for m in ("mse", "quantile_loss")}
    # channel level, TimesFM 2.5 on the primary configurations
    corr = training_correlations()
    all_chans = channel_units("primary", "primary_clean", ("timesfm25",))
    for u in all_chans:
        r = corr[u["dataset"]][u["channel"]]
        u["mean_abs_r"], u["max_abs_r"] = float(np.nanmean(np.abs(r))), float(np.nanmax(np.abs(r)))
    chans = eligible(all_chans)
    for u in all_chans:
        u["eligible"] = u in chans
    rho_channels = {k: rho(chans, k) for k in ("mean_abs_r", "max_abs_r", "I", "R2")}
    # held-out configurations and TiRex
    held_a = dataset_units("heldout", "heldout_clean", ("chronos2", "timesfm25", "tirex"))
    held_b = dataset_units("tirex", "tirex_clean", ("tirex",))
    big = lambda us: [u for u in us if u["loss_increase"] >= MIN_LOSS]  # noqa: E731
    held_chans = eligible(
        channel_units("heldout", "heldout_clean", ("timesfm25", "tirex"))
        + channel_units("tirex", "tirex_clean", ("tirex",))
    )
    ett = {
        mod: {
            d: next(u["recovered"] for u in held_a if u["dataset"] == d and u["model"] == mod)
            for d in ("ett1_H", "ett2_H")
        }
        for mod in ("chronos2", "timesfm25", "tirex")
    }
    out = {
        "primary_dataset_level": {"units": primary, "spearman_recovered_vs_mean_I": rho_primary},
        "primary_channel_level_timesfm25": {"units": all_chans, "n": len(chans), "spearman_recovered_vs": rho_channels},
        "heldout_dataset_level": {
            "units": held_a,
            "n": len(big(held_a)),
            "spearman_recovered_vs_mean_I": rho(big(held_a), "mean_I"),
        },
        "tirex_primary_dataset_level": {
            "units": held_b,
            "n": len(big(held_b)),
            "spearman_recovered_vs_mean_I": rho(big(held_b), "mean_I"),
        },
        "heldout_channel_level": {
            "units": held_chans,
            "n": len(held_chans),
            "spearman_recovered_vs": {"R2": rho(held_chans, "R2"), "I": rho(held_chans, "I")},
        },
        "ett_hourly_recovered": ett,
    }
    print(
        "primary, dataset level (16 pairs): Spearman(recovered, mean I_j) "
        f"MSE {rho_primary['mse']:.3f}, QL {rho_primary['quantile_loss']:.3f}"
    )
    print(
        f"primary, channel level (TimesFM 2.5, n={len(chans)}): "
        + ", ".join(f"{k} {v:.3f}" for k, v in rho_channels.items())
    )
    held, tirex = out["heldout_dataset_level"], out["tirex_primary_dataset_level"]
    print(
        f"held-out, dataset level: rho {held['spearman_recovered_vs_mean_I']:.3f} (n={held['n']}); "
        f"TiRex primary: rho {tirex['spearman_recovered_vs_mean_I']:.3f} (n={tirex['n']})"
    )
    print(
        f"held-out + TiRex, channel level (n={len(held_chans)}): "
        f"R2 {rho(held_chans, 'R2'):.3f}, I {rho(held_chans, 'I'):.3f}"
    )
    print(
        "hourly ETT recovered fraction: "
        + ", ".join(f"{m} ETT1 {v['ett1_H']:.3f} ETT2 {v['ett2_H']:.3f}" for m, v in ett.items())
    )
    write_json(out, RESULTS / "summaries" / "determinants.json")
    write_json(
        {k: np.where(np.isnan(v), 1.0, v) for k, v in corr.items()},
        RESULTS / "summaries" / "training_correlations.json",
    )


if __name__ == "__main__":
    main()
