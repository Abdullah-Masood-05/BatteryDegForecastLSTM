"""Central configuration for the battery degradation forecasting pipeline.

Every tunable of the pipeline lives here so that the notebook, the CLI and any
downstream experiment share one source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Column names assigned to the raw Kaggle CSV (its own header row is unusable).
RAW_COLUMN_NAMES: tuple[str, ...] = (
    "imei_number",
    "phone_model",
    "android_version",
    "battery_technology",
    "battery_capacity",
    "timestamp",
    "screen_status",
    "battery_percentage",
    "app_running",
    "cpu_usage",
    "battery_temperature",
    "battery_voltage",
    "battery_current",
    "network_connected",
    "plugged_in",
)

#: Features fed to the LSTM. The first entry is also the forecast target.
FEATURE_COLUMNS: tuple[str, ...] = (
    "battery_percentage",
    "app_running",
    "cpu_usage",
    "battery_voltage",
)

#: Timezone the device recordings are localised to.
LOCAL_TIMEZONE = "Asia/Karachi"

#: Repository-level default paths (resolved relative to the repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "battery_dataset.csv"
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "battery_lstm_model.h5"


@dataclass(frozen=True)
class TrainingConfig:
    """Hyper-parameters for training and forecasting.

    Defaults reproduce the reference run reported in the README and the
    dashboard (30 epochs, 24-step sequences, single 50-unit LSTM layer).
    """

    sequence_length: int = 24
    train_fraction: float = 0.8
    validation_split: float = 0.1
    epochs: int = 30
    batch_size: int = 32
    lstm_units: int = 50
    forecast_horizon: int = 24
