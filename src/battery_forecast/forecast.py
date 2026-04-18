"""Recursive multi-step forecasting with a trained model."""

from __future__ import annotations

from typing import Any

import numpy as np

from .features import inverse_transform_target


def recursive_forecast(
    model: Any,
    scaled_data: np.ndarray,
    horizon: int,
    scaler: Any | None = None,
) -> np.ndarray:
    """Forecast ``horizon`` steps beyond the end of ``scaled_data``.

    The model predicts one step at a time; each prediction is rolled into
    the input window before predicting the next step. Exogenous channels
    (everything but the target) are held at zero for future steps, matching
    the notebook's reference procedure — a documented simplification, see
    the README's limitations section.

    Returns battery percentages if ``scaler`` is provided, otherwise the
    raw scaled predictions.
    """
    sequence_length = model.input_shape[1]
    n_features = scaled_data.shape[1]
    window = scaled_data[-sequence_length:].copy()

    predictions: list[float] = []
    for _ in range(horizon):
        next_scaled = model.predict(
            window.reshape(1, sequence_length, n_features), verbose=0
        )
        predictions.append(float(next_scaled[0, 0]))
        window = np.roll(window, -1, axis=0)
        window[-1] = np.concatenate(
            [next_scaled, np.zeros((1, n_features - 1))], axis=1
        )[0]

    forecast = np.array(predictions)
    if scaler is not None:
        forecast = inverse_transform_target(scaler, forecast)
    return forecast
