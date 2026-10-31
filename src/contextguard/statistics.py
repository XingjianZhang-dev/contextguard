"""Training-only statistics: pooled pairwise covariance and the calibration gate.

Everything in this module is computed from the training prefix of each series. The prefix is
standardized per series, thinned to at most ``max_rows`` rows, and split chronologically into a
fit segment (first 80%) and a calibration segment (last 20%). The fit segment supplies the pooled
mean and the pairwise covariance (Eq. 1); the calibration segment supplies the clipped weight
``a_j``, the improvement ``I_j`` over the pooled-mean fallback, and the gate ``g_j`` (Eqs. 3-5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

#: Activation threshold on the calibrated improvement ``I_j``.
TAU = 0.05
#: Ridge stabilization of the observed-block system in Eq. (2).
RIDGE = 1e-3


@dataclass(frozen=True)
class TrainingStatistics:
    """Statistics estimated from training data only.

    Attributes:
        means, scales: per-series standardization (one ``(C,)`` array per series).
        pooled_mean: ``(C,)`` mean of the standardized fit segment.
        covariance: ``(C, C)`` pairwise-complete covariance of the standardized fit segment.
        weight: ``(C,)`` clipped calibration weight ``a_j`` in [0, 1].
        improvement: ``(C,)`` calibrated improvement ``I_j`` (NaN when the count rule abstains).
        gate: ``(C,)`` gate ``g_j = a_j`` if ``I_j >= tau`` else 0.
        count: ``(C,)`` number of complete calibration rows available for channel ``j``.
    """

    means: list[np.ndarray]
    scales: list[np.ndarray]
    pooled_mean: np.ndarray
    covariance: np.ndarray
    weight: np.ndarray
    improvement: np.ndarray
    gate: np.ndarray
    count: np.ndarray

    @property
    def channels(self) -> int:
        return int(self.pooled_mean.shape[0])


def _segments(train: Sequence[np.ndarray], max_rows: int, fit_fraction: float):
    """Standardize each training prefix and split it into fit and calibration rows.

    ``train`` holds one ``(time, channels)`` array per series; NaN marks natural missingness.
    """
    channels = train[0].shape[1]
    gsum = np.zeros(channels)
    gsq = np.zeros(channels)
    gcnt = np.zeros(channels, dtype=np.int64)
    arrays = []
    for series in train:
        x = np.asarray(series, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != channels:
            raise ValueError("every training series must have shape (time, channels)")
        if x.shape[0] < 2:
            raise ValueError("training prefix is too short")
        finite = np.isfinite(x)
        gsum += np.where(finite, x, 0.0).sum(0)
        gsq += np.where(finite, x * x, 0.0).sum(0)
        gcnt += finite.sum(0)
        arrays.append(x)
    if np.any(gcnt < 2):
        raise ValueError("a channel has fewer than two finite training observations")
    gmean = gsum / gcnt
    gscale = np.sqrt(np.maximum(gsq / gcnt - gmean**2, 1e-12))

    means, scales, fit_rows, cal_rows = [], [], [], []
    for x in arrays:
        finite = np.isfinite(x)
        count = finite.sum(0)
        mean = np.divide(np.where(finite, x, 0.0).sum(0), count, out=np.full(channels, np.nan), where=count > 0)
        var = np.divide(
            np.where(finite, np.square(x - mean), 0.0).sum(0), count, out=np.full(channels, np.nan), where=count > 0
        )
        scale = np.sqrt(var)
        mean = np.where(np.isfinite(mean), mean, gmean)
        scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, gscale)
        scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
        z = (x - mean) / scale
        z = z[:: max(1, math.ceil(z.shape[0] / max_rows))]
        split = max(1, int(fit_fraction * z.shape[0]))
        fit_rows.append(z[:split])
        cal_rows.append(z[split:])
        means.append(mean.astype(np.float32))
        scales.append(scale.astype(np.float32))
    return means, scales, fit_rows, cal_rows


def pooled_moments(fit_rows: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Pooled mean and pairwise-complete covariance of the fit segment (Eq. 1)."""
    channels = fit_rows[0].shape[1]
    msum = np.zeros(channels)
    mcnt = np.zeros(channels, dtype=np.int64)
    for rows in fit_rows:
        finite = np.isfinite(rows)
        msum += np.where(finite, rows, 0.0).sum(0)
        mcnt += finite.sum(0)
    mean = np.divide(msum, mcnt, out=np.zeros_like(msum), where=mcnt > 0)
    outer = np.zeros((channels, channels))
    pairs = np.zeros((channels, channels), dtype=np.int64)
    for rows in fit_rows:
        finite = np.isfinite(rows)
        centered = np.where(finite, rows - mean, 0.0)
        outer += centered.T @ centered
        f = finite.astype(np.int64)
        pairs += f.T @ f
    cov = np.divide(outer, np.maximum(pairs - 1, 1), out=np.zeros_like(outer), where=pairs > 1)
    cov = 0.5 * (cov + cov.T)
    diag = np.diag(cov).copy()
    np.fill_diagonal(cov, np.where(np.isfinite(diag) & (diag > 1e-6), diag, 1.0))
    cov += 1e-6 * np.eye(channels)
    return mean, cov


def conditional_predictor(mean: np.ndarray, cov: np.ndarray, ridge: float = RIDGE):
    """Return ``f(j, x_obs, observed) -> E[x_j | x_obs]`` under the ridge-stabilized Gaussian model."""

    def predict(j: int, x_obs: np.ndarray, observed: np.ndarray) -> np.ndarray:
        block = cov[np.ix_(observed, observed)] + ridge * np.eye(int(observed.sum()))
        coef = np.linalg.solve(block, cov[j, observed])
        return mean[j] + (x_obs - mean[observed]) @ coef

    return predict


def calibrate(cal_rows: Sequence[np.ndarray], predict, mean: np.ndarray, tau: float = TAU) -> dict:
    """Calibration gate (Eqs. 3-5) for any conditional predictor.

    For each channel ``j`` the predictor reconstructs ``x_j`` from all other channels on the complete
    calibration rows. The clipped least-squares weight ``a_j`` and the improvement ``I_j`` over the
    pooled-mean fallback are computed; the channel is activated (``g_j = a_j``) only if
    ``I_j >= tau`` and at least ``max(32, 4C)`` complete rows are available.
    """
    channels = mean.shape[0]
    out = {
        "a": np.zeros(channels),
        "I": np.full(channels, np.nan),
        "g": np.zeros(channels),
        "n": np.zeros(channels, dtype=np.int64),
    }
    rows = [r for r in cal_rows if r.shape[0]]
    if not rows:
        return out
    cal = np.concatenate(rows, 0)
    for j in range(channels):
        others = np.arange(channels) != j
        valid = np.isfinite(cal[:, j]) & np.isfinite(cal[:, others]).all(1)
        out["n"][j] = int(valid.sum())
        if valid.sum() < max(32, 4 * channels):
            continue
        y = cal[valid, j] - mean[j]
        p = predict(j, cal[valid][:, others], others) - mean[j]
        den = float(p @ p)
        if den <= 1e-12:
            continue
        a = float(np.clip((p @ y) / den, 0.0, 1.0))
        base = float(y @ y)
        imp = (base - float(np.square(y - a * p).sum())) / max(base, 1e-12)
        out["a"][j], out["I"][j] = a, imp
        if imp >= tau:
            out["g"][j] = a
    return out


def fit_statistics(
    train: np.ndarray | Sequence[np.ndarray], tau: float = TAU, max_rows: int = 10_000, fit_fraction: float = 0.8
) -> TrainingStatistics:
    """Estimate ContextGuard's statistics from training prefixes.

    Args:
        train: one ``(time, channels)`` array, or a list of them (one per series). Only data that
            precede every evaluation window may be passed. NaN marks natural missingness.
        tau: activation threshold on the calibrated improvement.
        max_rows: rows kept per series after uniform thinning.
        fit_fraction: chronological share of each series used as the fit segment.
    """
    if isinstance(train, np.ndarray):
        train = [train]
    means, scales, fit_rows, cal_rows = _segments(train, max_rows, fit_fraction)
    mean, cov = pooled_moments(fit_rows)
    gate = calibrate(cal_rows, conditional_predictor(mean, cov), mean, tau)
    return TrainingStatistics(
        means=means,
        scales=scales,
        pooled_mean=mean.astype(np.float32),
        covariance=cov.astype(np.float32),
        weight=gate["a"],
        improvement=gate["I"],
        gate=gate["g"].astype(np.float32),
        count=gate["n"],
    )


def training_segments(train: np.ndarray | Sequence[np.ndarray], max_rows: int = 10_000, fit_fraction: float = 0.8):
    """Standardized fit and calibration rows, exposed for alternative estimators."""
    if isinstance(train, np.ndarray):
        train = [train]
    _, _, fit_rows, cal_rows = _segments(train, max_rows, fit_fraction)
    return fit_rows, cal_rows
