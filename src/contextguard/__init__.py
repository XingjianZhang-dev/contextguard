"""ContextGuard: calibrated cross-channel repair of sensor dropout for frozen time-series foundation models."""

from .guard import ContextGuard
from .repair import repair
from .statistics import TAU, TrainingStatistics, fit_statistics

__all__ = ["ContextGuard", "TrainingStatistics", "fit_statistics", "repair", "TAU"]
__version__ = "3.0.0"
