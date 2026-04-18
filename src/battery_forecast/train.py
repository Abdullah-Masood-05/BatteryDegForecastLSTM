"""End-to-end training: load, clean, scale, window, fit, evaluate."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import FEATURE_COLUMNS, TrainingConfig
from .data import load_dataset
from .features import chronological_split, make_sequences, scale_features
from .model import build_lstm, save_model


@dataclass
class TrainingResult:
    """Artifacts of a completed training run."""

    model: Any
    scaler: Any
    scaled_data: np.ndarray
    history: dict[str, list[float]] = field(default_factory=dict)
    test_mse: float = float("nan")

    def save_history(self, path: str | Path) -> None:
        """Write the loss curves and test MSE to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"history": self.history, "test_mse": self.test_mse}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train(
    data_path: str | Path,
    model_path: str | Path | None = None,
    config: TrainingConfig = TrainingConfig(),
    verbose: int = 1,
) -> TrainingResult:
    """Train the LSTM on the telemetry CSV and evaluate on the held-out tail.

    The split is chronological (default 80/20) with the last 10% of the
    training window used for validation. If ``model_path`` is given the
    fitted model is saved there.
    """
    df = load_dataset(data_path)
    if df.empty:
        raise ValueError(f"No usable rows found in {data_path!s} after cleaning.")

    scaled, scaler = scale_features(df)
    X, y = make_sequences(scaled, config.sequence_length)
    X_train, X_test, y_train, y_test = chronological_split(
        X, y, config.train_fraction
    )

    model = build_lstm(
        config.sequence_length, len(FEATURE_COLUMNS), config.lstm_units
    )
    history = model.fit(
        X_train,
        y_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_split=config.validation_split,
        verbose=verbose,
    )
    test_mse = float(model.evaluate(X_test, y_test, verbose=0))

    if model_path is not None:
        save_model(model, model_path)

    return TrainingResult(
        model=model,
        scaler=scaler,
        scaled_data=scaled,
        history={k: [float(v) for v in vals] for k, vals in history.history.items()},
        test_mse=test_mse,
    )
