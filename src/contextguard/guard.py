"""High-level interface: fit on training data, repair contexts before any frozen forecaster."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .repair import repair
from .statistics import TAU, TrainingStatistics, fit_statistics


class ContextGuard:
    """Training-history-only repair of failed sensor channels.

    ``fit`` estimates the pooled cross-channel structure and the calibration gate from training
    prefixes; ``repair`` fills missing coordinates of a context in the original units. Observed values
    are returned unchanged, and a channel whose calibrated improvement is below ``tau`` falls back to
    topology-aware temporal repair.

    Example::

        guard = ContextGuard().fit(train)          # train: (time, channels), NaN allowed
        fixed = guard.repair(context)              # context: (length, channels) with NaN for failures
        forecast = model.predict(fixed)            # any frozen forecaster
    """

    def __init__(self, tau: float = TAU, gated: bool = True, max_rows: int = 10_000, fit_fraction: float = 0.8):
        self.tau = tau
        self.gated = gated
        self.max_rows = max_rows
        self.fit_fraction = fit_fraction
        self.stats_: TrainingStatistics | None = None

    # ------------------------------------------------------------------ fitting
    def fit(self, train: np.ndarray | Sequence[np.ndarray]) -> "ContextGuard":
        """Fit on one ``(time, channels)`` training array or a list of them (one per series)."""
        self.stats_ = fit_statistics(train, self.tau, self.max_rows, self.fit_fraction)
        return self

    @property
    def gate_(self) -> dict[str, np.ndarray]:
        """Per-channel calibration quantities: weight ``a``, improvement ``I``, gate ``g``, count ``n``."""
        s = self._check()
        return {"a": s.weight, "I": s.improvement, "g": s.gate, "n": s.count}

    # ------------------------------------------------------------------ repair
    def repair(
        self, context: np.ndarray, mask: np.ndarray | None = None, series: int = 0, method: str | None = None
    ) -> np.ndarray:
        """Repair ``context`` of shape ``(length, channels)`` or ``(batch, length, channels)``.

        Args:
            context: values in the original units; missing entries may be NaN.
            mask: True where a value is missing (defaults to ``~isfinite(context)``).
            series: index of the training series whose standardization applies.
            method: override the repair method (see ``contextguard.repair.METHODS``).
        """
        s = self._check()
        x = np.asarray(context, dtype=np.float32)
        single = x.ndim == 2
        if single:
            x = x[None]
        m = ~np.isfinite(x) if mask is None else np.asarray(mask, dtype=bool).reshape(x.shape)
        m = m | ~np.isfinite(x)
        mu, sd = s.means[series], s.scales[series]
        z = np.where(m, 0.0, (x - mu) / sd).astype(np.float32)
        method = method or ("contextguard" if self.gated else "contextguard_ungated")
        out = repair(z, m, method, s.pooled_mean, s.covariance, s.gate) * sd + mu
        out = np.where(m, out, x).astype(np.float32)
        return out[0] if single else out

    def _check(self) -> TrainingStatistics:
        if self.stats_ is None:
            raise RuntimeError("call fit() before using the guard")
        return self.stats_
