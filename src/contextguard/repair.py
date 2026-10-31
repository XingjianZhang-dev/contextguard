"""Context repair: topology-aware temporal repair and ContextGuard's gated conditional recovery.

All functions operate on standardized contexts of shape ``(batch, length, channels)`` with a boolean
mask that is True where a value is missing. Observed coordinates are never modified (Proposition 1).
"""

from __future__ import annotations

import numpy as np

from .statistics import RIDGE

SHORT_BLEND = 0.25  # beta: blend weight for short gaps under heavy missingness
MISSING_ACTIVATION = 0.20  # missing fraction above which short gaps use the blend
LONG_GAP_FRACTION = 0.05  # the blend is disabled when long gaps dominate the context
SHORT_GAP_MAX = 8  # gaps up to this length are short
TOPOLOGY_GAP_MAX = 32  # gaps up to this length are interpolated; longer gaps are carried forward

METHODS = ("ffill", "bidirectional", "topology", "contextguard", "contextguard_ungated")


def forward_fill(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = x.copy()
    for t in range(1, out.shape[1]):
        out[:, t] = np.where(mask[:, t], out[:, t - 1], out[:, t])
    return out


def gap_lengths(mask: np.ndarray) -> np.ndarray:
    """Length of the contiguous missing run that contains each masked coordinate (0 if observed)."""
    b, length, c = mask.shape
    fwd = np.zeros(mask.shape, dtype=np.int32)
    bwd = np.zeros(mask.shape, dtype=np.int32)
    run = np.zeros((b, c), dtype=np.int32)
    for t in range(length):
        run = np.where(mask[:, t], run + 1, 0)
        fwd[:, t] = run
    run = np.zeros((b, c), dtype=np.int32)
    for t in range(length - 1, -1, -1):
        run = np.where(mask[:, t], run + 1, 0)
        bwd[:, t] = run
    return np.where(mask, fwd + bwd - 1, 0)


def conditional_reconstruct(
    x: np.ndarray, mask: np.ndarray, mean: np.ndarray, cov: np.ndarray, ridge: float = RIDGE, chunk: int = 4096
) -> np.ndarray:
    """Eq. (2): replace each missing coordinate by its ridge-stabilized conditional expectation
    given the coordinates observed at the same time step."""
    c = x.shape[-1]
    fx = x.reshape(-1, c)
    fm = mask.reshape(-1, c)
    out = fx.copy()
    for start in range(0, fx.shape[0], chunk):
        cx, cm = fx[start : start + chunk], fm[start : start + chunk]
        obs = (~cm).astype(x.dtype)
        system = cov[None] * obs[:, :, None] * obs[:, None, :]
        system = system + np.einsum("ni,ij->nij", np.where(cm, 1.0, ridge).astype(x.dtype), np.eye(c, dtype=x.dtype))
        rhs = (cx - mean) * obs
        w = np.linalg.solve(system, rhs[..., None])[..., 0]
        pred = w @ cov.T + mean
        out[start : start + chunk] = np.where(cm, pred, cx)
    return out.reshape(x.shape)


def repair(
    x: np.ndarray,
    mask: np.ndarray,
    method: str = "contextguard",
    pooled_mean: np.ndarray | None = None,
    covariance: np.ndarray | None = None,
    gate: np.ndarray | None = None,
) -> np.ndarray:
    """Repair a batch of standardized contexts.

    Args:
        x: ``(batch, length, channels)`` contexts; values under ``mask`` are ignored.
        mask: boolean array of the same shape, True where a value is missing.
        method: one of ``METHODS``. ``contextguard`` is the gated method of the paper;
            ``contextguard_ungated`` uses weight one for every channel.
        pooled_mean, covariance, gate: training statistics (required for the ContextGuard methods).
    """
    x = np.asarray(x, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    x = np.where(mask, 0.0, x).astype(np.float32)
    if not mask.any():
        return x
    fwd = forward_fill(x, mask)
    if method == "ffill":
        return fwd
    bwd = forward_fill(x[:, ::-1], mask[:, ::-1])[:, ::-1]
    temporal = np.where(mask, 0.5 * (fwd + bwd), x)
    if method == "bidirectional":
        return temporal
    lengths = gap_lengths(mask)
    if method == "topology":
        return np.where(mask & (lengths <= TOPOLOGY_GAP_MAX), temporal, fwd).astype(np.float32)

    mean = np.asarray(pooled_mean, dtype=np.float32)
    cov = np.asarray(covariance, dtype=np.float32)
    conditional = conditional_reconstruct(temporal, mask, mean, cov)
    weight = np.ones_like(mean) if method == "contextguard_ungated" else np.asarray(gate, dtype=np.float32)
    missing_fraction = mask.mean(axis=(1, 2), keepdims=True, dtype=np.float32)
    long_fraction = (lengths > TOPOLOGY_GAP_MAX).mean(axis=(1, 2), keepdims=True, dtype=np.float32)
    active = (
        SHORT_BLEND
        * (missing_fraction >= MISSING_ACTIVATION)
        * (long_fraction < LONG_GAP_FRACTION)
        * weight.reshape(1, 1, -1)
    ).astype(np.float32)
    short = (1.0 - active) * temporal + active * conditional
    out = np.where(mask & (lengths <= SHORT_GAP_MAX), short, fwd)
    w = weight.reshape(1, 1, -1)
    full = (1.0 - w) * fwd + w * conditional
    out = np.where(mask.all(axis=1, keepdims=True), full, out)
    return np.where(mask, out, x).astype(np.float32)
