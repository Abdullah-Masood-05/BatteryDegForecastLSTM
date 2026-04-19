"""Export figures and training history from the executed notebook.

The dashboard (docs/) and README embed results of the reference notebook
run. After re-executing the notebook, run this script to refresh those
assets:

    python scripts/export_notebook_assets.py

Outputs:
- docs/assets/figures/<name>.png     (every embedded matplotlib figure)
- docs/assets/training_history.json  (per-epoch loss / val_loss, test MSE)
"""

from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = REPO_ROOT / "notebooks" / "battery_degradation_forecasting.ipynb"
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "assets"

#: Stable, descriptive file names keyed by a snippet of the producing cell.
FIGURE_NAMES = {
    "sns.scatterplot": "eda_cpu_vs_battery",
    "sns.boxplot": "eda_apps_boxplot",
    "sns.heatmap": "correlation_matrix",
    "sns.pairplot": "kmeans_clusters",
    'history.history["loss"]': "training_history",
    "Actual vs Predicted": "actual_vs_predicted",
    "future_predictions": "forecast",
}

EPOCH_LOSS_RE = re.compile(r"loss: ([0-9.e-]+) - val_loss: ([0-9.e-]+)")
TEST_MSE_RE = re.compile(r"Test Loss \(MSE\): ([0-9.e-]+)")


def figure_name(source: str, cell_index: int) -> str:
    for snippet, name in FIGURE_NAMES.items():
        if snippet in source:
            return name
    return f"figure_cell_{cell_index}"


def export(notebook_path: Path, out_dir: Path) -> None:
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    history: dict[str, list[float]] = {"loss": [], "val_loss": []}
    test_mse: float | None = None
    exported = 0

    for index, cell in enumerate(nb["cells"]):
        source = "".join(cell.get("source", []))
        for output in cell.get("outputs", []):
            png = output.get("data", {}).get("image/png")
            if png:
                target = figures_dir / f"{figure_name(source, index)}.png"
                target.write_bytes(base64.b64decode(png))
                exported += 1
            if output.get("output_type") == "stream":
                text = "".join(output.get("text", []))
                for train_loss, val_loss in EPOCH_LOSS_RE.findall(text):
                    history["loss"].append(float(train_loss))
                    history["val_loss"].append(float(val_loss))
                match = TEST_MSE_RE.search(text)
                if match:
                    test_mse = float(match.group(1))

    history_path = out_dir / "training_history.json"
    history_path.write_text(
        json.dumps({"history": history, "test_mse": test_mse}, indent=2),
        encoding="utf-8",
    )
    print(f"Exported {exported} figures to {figures_dir}")
    print(
        f"Wrote {history_path} "
        f"({len(history['loss'])} epochs, test_mse={test_mse})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    export(args.notebook, args.out_dir)


if __name__ == "__main__":
    main()
