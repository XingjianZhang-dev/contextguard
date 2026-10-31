"""The released summaries contain the numbers reported in the paper."""

import json

import pytest
from conftest import ROOT

SUMMARIES = ROOT / "results" / "summaries"


def load(name):
    return json.loads((SUMMARIES / name).read_text())


def change(summary, loss="mse"):
    return round(summary["metrics"][loss]["geometric_relative_change_percent"], 2)


def test_primary_effect_and_estimators():
    c = load("estimators.json")["primary"]
    assert [change(c["contextguard_adaptive__vs__topology"], loss) for loss in ("mse", "mae", "quantile_loss")] == [
        -37.89,
        -31.07,
        -40.01,
    ]
    assert change(c["ridge_ungated__vs__topology"]) == -34.58
    assert change(c["lw_ungated__vs__topology"]) == -35.57
    assert change(c["contextguard__vs__topology"]) == -36.43
    assert [
        change(c[k])
        for k in (
            "contextguard_adaptive__vs__contextguard",
            "ridge_gated__vs__ridge_ungated",
            "lw_gated__vs__lw_ungated",
        )
    ] == [-2.30, -2.02, -2.27]
    assert [
        c[f"{m}__vs__topology"]["metrics"]["mse"]["losses"]
        for m in ("contextguard", "contextguard_adaptive", "ridge_ungated", "ridge_gated", "lw_ungated", "lw_gated")
    ] == [6, 3, 6, 4, 6, 3]
    assert change(c["contextguard_adaptive__vs__ridge_gated"]) == -3.10
    assert change(c["contextguard_adaptive__vs__lw_gated"]) == -1.36
    for key in (
        "contextguard_adaptive__vs__contextguard",
        "ridge_gated__vs__ridge_ungated",
        "lw_gated__vs__lw_ungated",
    ):
        for loss in ("mse", "mae", "quantile_loss"):
            assert c[key]["metrics"][loss]["dataset_cluster_bootstrap_ci95_percent"][1] < 0


def test_gate_validation():
    g = load("gate_validation.json")
    p, h = g["primary"], g["heldout"]
    assert (p["crossfit_agreement"]["agree"], p["crossfit_agreement"]["channels"]) == (49, 50)
    rec = p["test_context_reconstruction"]
    assert (rec["better_than_fallback"], rec["active_channels"]) == (44, 45)
    assert round(rec["geometric_mean_ratio"], 3) == 0.108
    assert [round(v, 3) for v in rec["dataset_cluster_ci95"]] == [0.013, 0.367]
    assert (h["crossfit_agreement"]["agree"], h["crossfit_agreement"]["channels"]) == (44, 46)
    assert (
        h["test_context_reconstruction"]["better_than_fallback"],
        h["test_context_reconstruction"]["active_channels"],
    ) == (37, 41)


def test_determinants():
    d = load("determinants.json")
    rho = d["primary_channel_level_timesfm25"]["spearman_recovered_vs"]
    assert d["primary_channel_level_timesfm25"]["n"] == 49
    assert [round(rho[k], 2) for k in ("mean_abs_r", "max_abs_r", "I", "R2")] == [0.70, 0.79, 0.78, 0.96]
    assert round(d["primary_dataset_level"]["spearman_recovered_vs_mean_I"]["mse"], 2) == 0.95
    assert round(d["heldout_dataset_level"]["spearman_recovered_vs_mean_I"], 2) == 0.90
    assert d["heldout_channel_level"]["n"] == 131
    assert round(d["heldout_channel_level"]["spearman_recovered_vs"]["R2"], 2) == 0.93


def test_breadth_and_native_input():
    b = load("breadth.json")
    assert [change(b["heldout"]["all"], loss) for loss in ("mse", "mae", "quantile_loss")] == [-27.86, -23.31, -31.69]
    assert [change(b["tirex_primary"], loss) for loss in ("mse", "mae", "quantile_loss")] == [-35.15, -30.26, -36.66]
    n = b["native_missing"]
    assert (n["heldout_chronos2"]["contextguard_lower_mse"], n["heldout_chronos2"]["native_width_at_least_10x"]) == (
        60,
        60,
    )
    assert n["heldout_tirex"]["contextguard_lower_mse"] == 56


@pytest.mark.parametrize("name", ["severity_and_boundary.csv", "context_sensitivity.csv"])
def test_original_study_tables_exist(name):
    assert (SUMMARIES / name).stat().st_size > 0
