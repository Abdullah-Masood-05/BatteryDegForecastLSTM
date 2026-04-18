"""Feature scaling and sequence construction for the LSTM."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .config import FEATURE_COLUMNS


def scale_features(df: pd.DataFrame) -> tuple[np.ndarray, MinMaxScaler]:
    """Min-max scale the model features to ``[0, 1]``.

    Returns the scaled matrix (rows in original order) together with the
    fitted scaler, which is needed later to invert predictions.
    """
    features = df[list(FEATURE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(features), scaler


def make_sequences(
    data: np.ndarray, sequence_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build supervised (X, y) pairs from a scaled feature matrix.

    Each sample is a window of ``sequence_length`` consecutive rows; the
    label is the target channel (column 0, battery percentage) of the row
    immediately after the window.
    """
    windows, labels = [], []
    for start in range(len(data) - sequence_length):
        windows.append(data[start : start + sequence_length])
        labels.append(data[start + sequence_length, 0])
    return np.array(windows), np.array(labels)


def chronological_split(
    X: np.ndarray, y: np.ndarray, train_fraction: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split sequences into train/test sets without shuffling.

    Time order is preserved so the test set is strictly in the future
    relative to the training set.
    """
    cutoff = int(len(X) * train_fraction)
    return X[:cutoff], X[cutoff:], y[:cutoff], y[cutoff:]


def inverse_transform_target(
    scaler: MinMaxScaler, values: np.ndarray
) -> np.ndarray:
    """Map scaled target values back to battery percentage.

    The scaler was fitted on the full feature matrix, so the target column
    is padded with zeros for the remaining channels before inverting.
    """
    values = np.asarray(values).reshape(-1, 1)
    padding = np.zeros((len(values), scaler.n_features_in_ - 1))
    return scaler.inverse_transform(np.concatenate([values, padding], axis=1))[:, 0]
