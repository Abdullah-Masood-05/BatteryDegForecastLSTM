"""Smartphone battery degradation forecasting with an LSTM.

Public API::

    from battery_forecast import TrainingConfig, load_dataset, train
    from battery_forecast.forecast import recursive_forecast
"""

from .config import FEATURE_COLUMNS, RAW_COLUMN_NAMES, TrainingConfig
from .data import load_dataset

__version__ = "1.0.0"

__all__ = [
    "FEATURE_COLUMNS",
    "RAW_COLUMN_NAMES",
    "TrainingConfig",
    "load_dataset",
    "__version__",
]
