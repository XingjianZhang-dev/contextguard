"""Proposition 1: observed coordinates and clean contexts are invariant under every repair method."""

import numpy as np
import pytest

from contextguard import ContextGuard, fit_statistics, repair
from contextguard.repair import METHODS


def contexts(rng, batch=8, length=256, channels=5):
    x = rng.normal(size=(batch, length, channels)).astype(np.float32)
    mask = np.zeros_like(x, dtype=bool)
    mask[:, :, 1] = True  # one failed channel
    mask[:, 40:60, 3] = True  # a short gap
    mask[:, 100:180, 4] = True  # a long gap
    mask |= rng.random(x.shape) < 0.05  # scattered missing values
    return x, mask


@pytest.mark.parametrize("method", METHODS)
def test_observed_coordinates_are_unchanged(method):
    rng = np.random.default_rng(1)
    train = rng.normal(size=(4000, 5)).astype(np.float32)
    s = fit_statistics(train)
    x, mask = contexts(rng)
    out = repair(x, mask, method, s.pooled_mean, s.covariance, s.gate)
    assert np.array_equal(out[~mask], x[~mask])
    assert np.isfinite(out).all()


@pytest.mark.parametrize("method", METHODS)
def test_clean_context_is_returned_bitwise(method):
    rng = np.random.default_rng(2)
    s = fit_statistics(rng.normal(size=(4000, 3)).astype(np.float32))
    x = rng.normal(size=(4, 128, 3)).astype(np.float32)
    out = repair(x, np.zeros_like(x, dtype=bool), method, s.pooled_mean, s.covariance, s.gate)
    assert np.array_equal(out, x)


def test_closed_gate_falls_back_to_temporal_repair():
    rng = np.random.default_rng(3)
    s = fit_statistics(rng.normal(size=(4000, 5)).astype(np.float32))  # independent channels
    assert np.all(s.gate == 0)
    x, mask = contexts(rng)
    gated = repair(x, mask, "contextguard", s.pooled_mean, s.covariance, s.gate)
    fallback = repair(x, mask, "ffill")
    failed = mask.all(axis=1, keepdims=True) & mask
    assert np.array_equal(gated[failed], fallback[failed])


def test_guard_preserves_original_units(sensors):
    x, split = sensors
    guard = ContextGuard().fit(x[:split])
    ctx = x[split + 100 : split + 612].copy()
    ctx[:, 0] = np.nan
    out = guard.repair(ctx)
    observed = np.isfinite(ctx)
    assert out.shape == ctx.shape
    assert np.array_equal(out[observed], ctx[observed])
    assert np.isfinite(out).all()
