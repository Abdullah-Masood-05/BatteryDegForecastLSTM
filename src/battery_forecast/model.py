"""LSTM model definition and persistence helpers.

TensorFlow is imported lazily inside the functions so the rest of the
package (data loading, feature engineering) stays importable in
environments without TensorFlow installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_lstm(
    sequence_length: int, n_features: int, lstm_units: int = 50
) -> Any:
    """Build and compile the forecasting network.

    Architecture: a single LSTM layer (ReLU) followed by a one-unit dense
    head that regresses the next scaled battery percentage. Compiled with
    Adam and mean-squared-error loss.
    """
    from tensorflow.keras.layers import LSTM, Dense, Input
    from tensorflow.keras.models import Sequential

    model = Sequential(
        [
            Input(shape=(sequence_length, n_features)),
            LSTM(lstm_units, activation="relu", return_sequences=False),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def save_model(model: Any, path: str | Path) -> None:
    """Save a trained model, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)


def load_model(path: str | Path) -> Any:
    """Load a previously trained model from disk."""
    from tensorflow.keras.models import load_model as keras_load_model

    return keras_load_model(path)
