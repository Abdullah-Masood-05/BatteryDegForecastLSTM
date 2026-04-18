"""Loading and cleaning of the raw Kaggle battery telemetry export.

The raw CSV ships without a usable header and with several columns that are
entirely empty. :func:`load_dataset` reproduces the cleaning applied in the
exploration notebook and returns an analysis-ready frame.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import LOCAL_TIMEZONE, RAW_COLUMN_NAMES

_NUMERIC_COLUMNS = (
    "battery_capacity",
    "timestamp",
    "battery_percentage",
    "cpu_usage",
    "battery_temperature",
    "battery_voltage",
    "battery_current",
)

_BOOLEAN_COLUMNS = ("plugged_in", "screen_status")


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the raw CSV, assigning descriptive column names.

    The file's first row is a data row misparsed as a header, so it is
    skipped and :data:`~battery_forecast.config.RAW_COLUMN_NAMES` is applied
    instead.
    """
    return pd.read_csv(path, header=None, names=list(RAW_COLUMN_NAMES), skiprows=1)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the notebook's cleaning steps to a raw frame.

    - drop rows without a device identifier,
    - coerce numeric sensor channels (unparseable values become ``NaN``),
    - map textual booleans to real booleans,
    - convert the epoch-millisecond timestamp to :data:`LOCAL_TIMEZONE`,
    - drop columns that end up entirely empty.
    """
    df = df.dropna(subset=["imei_number"]).copy()

    for column in _NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["network_connected"] = df["network_connected"].astype(str)
    for column in _BOOLEAN_COLUMNS:
        df[column] = (
            df[column].astype(str).str.lower().map({"true": True, "false": False})
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], unit="ms", utc=True, errors="coerce"
    ).dt.tz_convert(LOCAL_TIMEZONE)

    return df.drop(columns=df.columns[df.isnull().all()])


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and clean the dataset in one step."""
    return clean(load_raw(path))
