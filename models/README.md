# Model card — `battery_lstm_model.h5`

Single-layer LSTM regressor that predicts the next battery-percentage reading
from a 24-step window of four scaled telemetry channels.

## Architecture

| Layer | Configuration |
|---|---|
| Input | (24 timesteps × 4 features) |
| LSTM | 50 units, ReLU, `return_sequences=False` |
| Dense | 1 unit (next scaled battery %) |

Optimizer **Adam** · loss **MSE** · ~**11K** trainable parameters.

## Training data

385,429 readings from a single Samsung SM-A910F (Android 8.0.0, 5,000 mAh
Li-ion). Features: `battery_percentage`, `app_running`, `cpu_usage`,
`battery_voltage`, min-max scaled to [0, 1]. Chronological 80/20 train/test
split, 10% of the training window used for validation.

## Reference metrics (30 epochs, batch 32)

| Metric (scaled units) | Value |
|---|---|
| Final training loss | 1.72 × 10⁻⁶ |
| Final validation loss | 5.55 × 10⁻⁷ |
| Held-out test MSE | < 10⁻⁴ (prints as 0.0000) |

**Read these numbers with care.** Battery percentage changes slowly relative
to the sampling rate, so a model that closely tracks the previous value
already achieves a very low one-step-ahead MSE. Compare against a persistence
baseline before drawing conclusions about long-horizon skill.

## Intended use & limitations

- Research/portfolio demonstration of an LSTM time-series pipeline — not a
  production battery-health estimator.
- Trained on **one device**; no claim of generalization to other hardware,
  OS versions, or usage patterns.
- Multi-step forecasts hold exogenous channels at zero (see
  `battery_forecast.forecast.recursive_forecast`), which limits long-horizon
  realism.

## Reproduce / reload

```bash
uv run battery-forecast train --data data/battery_dataset.csv --model-out models/battery_lstm_model.h5
```

```python
from battery_forecast.model import load_model
model = load_model("models/battery_lstm_model.h5")
```
