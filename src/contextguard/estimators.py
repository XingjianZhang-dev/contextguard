"""Alternative cross-channel estimators evaluated under the same calibration gate (Section 5.5)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .statistics import RIDGE, TAU, calibrate, conditional_predictor

#: Candidate ridge penalties, selected by chronological validation inside the fit segment.
RIDGE_GRID = (1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0)


def complete_rows(rows: Sequence[np.ndarray]) -> list[np.ndarray]:
    return [r[np.isfinite(r).all(1)] for r in rows]


def ridge_estimator(fit_rows: Sequence[np.ndarray], grid: Sequence[float] = RIDGE_GRID) -> dict:
    """Ridge regression of each channel on all others, fitted on complete fit-segment rows.

    One penalty per dataset is chosen by chronological validation: in every series the first 80% of
    the complete fit rows form the inner fit set and the last 20% the inner validation set; the score
    is the mean normalized validation MSE over channels.
    """
    cc = complete_rows(fit_rows)
    channels = cc[0].shape[1]
    inner_fit = np.concatenate([r[: int(0.8 * r.shape[0])] for r in cc], 0)
    inner_val = np.concatenate([r[int(0.8 * r.shape[0]) :] for r in cc], 0)
    m = inner_fit.mean(0)
    s = np.cov(inner_fit, rowvar=False).reshape(channels, channels)
    scores = {}
    for lam in grid:
        f = conditional_predictor(m, s, lam)
        errs = []
        for j in range(channels):
            o = np.arange(channels) != j
            errs.append(
                np.mean((inner_val[:, j] - f(j, inner_val[:, o], o)) ** 2) / max(np.var(inner_val[:, j]), 1e-12)
            )
        scores[lam] = float(np.mean(errs))
    lam = min(scores, key=scores.get)
    full = np.concatenate(cc, 0)
    return {
        "mean": full.mean(0),
        "cov": np.cov(full, rowvar=False).reshape(channels, channels),
        "ridge": lam,
        "scores": scores,
        "n_complete_fit": int(full.shape[0]),
        "n_inner_val": int(inner_val.shape[0]),
    }


def ledoit_wolf_estimator(fit_rows: Sequence[np.ndarray]) -> dict:
    """Ledoit-Wolf shrinkage covariance on complete fit-segment rows (used in Eq. 2)."""
    from sklearn.covariance import LedoitWolf

    full = np.concatenate(complete_rows(fit_rows), 0)
    lw = LedoitWolf().fit(full)
    return {
        "mean": lw.location_,
        "cov": lw.covariance_,
        "ridge": RIDGE,
        "shrinkage": float(lw.shrinkage_),
        "n_complete_fit": int(full.shape[0]),
    }


def gated(estimator: dict, cal_rows: Sequence[np.ndarray], tau: float = TAU) -> dict:
    """Apply ContextGuard's calibration rule to an estimator's own calibration predictions."""
    f = conditional_predictor(estimator["mean"], estimator["cov"], estimator["ridge"])
    return calibrate(cal_rows, f, estimator["mean"], tau)


def as_repair_inputs(estimator: dict) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance to pass to :func:`contextguard.repair.repair`.

    The repair solver stabilizes the observed block with ``RIDGE``; adding ``(ridge - RIDGE) I`` to the
    covariance makes the solve use the estimator's own penalty exactly.
    """
    c = estimator["cov"].shape[0]
    cov = estimator["cov"] + (estimator["ridge"] - RIDGE) * np.eye(c)
    return estimator["mean"].astype(np.float32), cov.astype(np.float32)


def crossfit_calibrate(cal_rows: Sequence[np.ndarray], predict, mean: np.ndarray, tau: float = TAU) -> dict:
    """Cross-fitted gate: ``a_j`` from the first chronological half of each series' calibration
    segment, ``I_j`` from the second half; the count rule is applied to each half."""
    channels = mean.shape[0]
    first = [r[: r.shape[0] // 2] for r in cal_rows]
    second = [r[r.shape[0] // 2 :] for r in cal_rows]
    weights = calibrate(first, predict, mean, tau=-np.inf)
    out = {
        "a": weights["a"].copy(),
        "I": np.full(channels, np.nan),
        "g": np.zeros(channels),
        "n_first": weights["n"],
        "n_second": np.zeros(channels, dtype=np.int64),
    }
    cal = np.concatenate([r for r in second if r.shape[0]], 0)
    need = max(32, 4 * channels)
    for j in range(channels):
        others = np.arange(channels) != j
        valid = np.isfinite(cal[:, j]) & np.isfinite(cal[:, others]).all(1)
        out["n_second"][j] = int(valid.sum())
        if valid.sum() < need or weights["n"][j] < need:
            continue
        y = cal[valid, j] - mean[j]
        p = predict(j, cal[valid][:, others], others) - mean[j]
        base = float(y @ y)
        out["I"][j] = (base - float(np.square(y - out["a"][j] * p).sum())) / max(base, 1e-12)
        if out["I"][j] >= tau:
            out["g"][j] = out["a"][j]
    return out
