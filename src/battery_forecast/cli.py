"""Command-line interface: ``battery-forecast train`` and ``... forecast``."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_DATA_PATH, DEFAULT_MODEL_PATH, TrainingConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="battery-forecast",
        description="Train and run the smartphone battery degradation LSTM.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train", help="Train the LSTM and report the held-out test MSE."
    )
    train_parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA_PATH,
        help="Path to battery_dataset.csv (default: %(default)s).",
    )
    train_parser.add_argument(
        "--model-out", type=Path, default=DEFAULT_MODEL_PATH,
        help="Where to save the trained model (default: %(default)s).",
    )
    train_parser.add_argument(
        "--history-out", type=Path, default=None,
        help="Optional JSON file for loss curves and test MSE.",
    )
    train_parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    train_parser.add_argument(
        "--batch-size", type=int, default=TrainingConfig.batch_size
    )
    train_parser.add_argument(
        "--sequence-length", type=int, default=TrainingConfig.sequence_length
    )

    forecast_parser = subparsers.add_parser(
        "forecast", help="Forecast future battery percentage with a saved model."
    )
    forecast_parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    forecast_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    forecast_parser.add_argument(
        "--horizon", type=int, default=TrainingConfig.forecast_horizon,
        help="Number of future steps to predict (default: %(default)s).",
    )

    return parser


def _run_train(args: argparse.Namespace) -> None:
    from .train import train

    config = TrainingConfig(
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    result = train(args.data, model_path=args.model_out, config=config)
    print(f"Test MSE (scaled): {result.test_mse:.3e}")
    print(f"Model saved to {args.model_out}")
    if args.history_out is not None:
        result.save_history(args.history_out)
        print(f"Training history written to {args.history_out}")


def _run_forecast(args: argparse.Namespace) -> None:
    from .data import load_dataset
    from .features import scale_features
    from .forecast import recursive_forecast
    from .model import load_model

    df = load_dataset(args.data)
    scaled, scaler = scale_features(df)
    model = load_model(args.model)
    forecast = recursive_forecast(model, scaled, args.horizon, scaler=scaler)

    print(f"Battery percentage forecast, next {args.horizon} steps:")
    for step, value in enumerate(forecast, start=1):
        print(f"  t+{step:<3d} {value:6.2f} %")


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``battery-forecast`` console script."""
    args = _build_parser().parse_args(argv)
    if args.command == "train":
        _run_train(args)
    elif args.command == "forecast":
        _run_forecast(args)


if __name__ == "__main__":
    main()
