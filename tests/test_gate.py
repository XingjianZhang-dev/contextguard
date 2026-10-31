"""Proposition 2 and the behaviour of the calibration gate on channels with known structure."""

import numpy as np

from contextguard import ContextGuard, fit_statistics
from contextguard.statistics import conditional_predictor, training_segments


def test_gated_reconstruction_never_loses_to_the_fallback_on_calibration(sensors):
    """With the clipped weight, the calibration error is at most the pooled-mean error (I_j >= 0)."""
    x, split = sensors
    s = fit_statistics(x[:split])
    _, cal = training_segments(x[:split])
    cal = np.concatenate(cal)
    predict = conditional_predictor(s.pooled_mean.astype(float), s.covariance.astype(float))
    for j in range(s.channels):
        others = np.arange(s.channels) != j
        y = cal[:, j] - s.pooled_mean[j]
        p = predict(j, cal[:, others], others) - s.pooled_mean[j]
        assert np.sum((y - s.weight[j] * p) ** 2) <= np.sum(y**2) + 1e-9
        assert s.improvement[j] >= 0


def test_gate_activates_peers_and_abstains_on_independent_channel(sensors):
    x, split = sensors
    s = fit_statistics(x[:split])
    assert s.gate[0] > 0.9 and s.gate[1] > 0.9
    assert s.gate[2] == 0  # independent channel: no recovery
    assert s.improvement[2] < 0.05


def test_recovery_of_a_failed_peer_beats_temporal_repair(sensors):
    x, split = sensors
    guard = ContextGuard().fit(x[:split])
    errors = {"contextguard": [], "topology": []}
    for start in range(split + 200, len(x) - 512, 97):
        ctx = x[start : start + 512].copy()
        truth = ctx[:, 0].copy()
        ctx[:, 0] = np.nan
        for method in errors:
            errors[method].append(np.mean((guard.repair(ctx, method=method)[:, 0] - truth) ** 2))
    assert np.mean(errors["contextguard"]) < 0.1 * np.mean(errors["topology"])
