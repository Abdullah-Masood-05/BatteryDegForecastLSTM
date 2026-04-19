# Data

The raw dataset is **not** checked into git (≈385K rows). Download it from Kaggle
and place it in this directory as `battery_dataset.csv`:

- **Dataset:** [Mobile Battery with Time](https://www.kaggle.com/datasets/rahulgarg28/mobile-battery-with-time)

Using the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) (via `uvx`, no install needed):

```bash
uvx kaggle datasets download -d rahulgarg28/mobile-battery-with-time -p data --unzip
# rename the extracted CSV if needed:
#   data/battery_dataset.csv
```

## What's inside

| Property | Value |
|---|---|
| Records | 385,429 sensor readings |
| Device | Samsung SM-A910F (single device) |
| OS | Android 8.0.0 |
| Battery | 5,000 mAh Li-ion |
| Signals | battery %, voltage, current, temperature, CPU usage, screen state, running app, network, charging state |

The raw export has an unusable header row and several fully-empty columns;
`battery_forecast.data.load_dataset()` applies the documented cleaning steps
(descriptive column names, numeric coercion, timezone-aware timestamps,
empty-column removal).
