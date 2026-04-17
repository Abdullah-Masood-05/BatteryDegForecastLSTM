# Smartphone Battery Degradation Forecasting using LSTM

A machine learning project that predicts smartphone battery degradation using LSTM neural networks and sensor telemetry data. This project demonstrates a complete machine learning pipeline from data cleaning through time-series forecasting.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Methodology](#methodology)
- [Limitations & Future Work](#limitations--future-work)
- [Dependencies](#dependencies)
- [License](#license)

## Overview

This project forecasts smartphone battery percentage degradation over time using:
- **13 sensor features** including CPU usage, battery temperature, voltage, and current
- **LSTM (Long Short-Term Memory) neural network** for sequence prediction
- **24-step ahead forecasting** for future battery health prediction
- **Data from**: Samsung SM-A910F with 5000 mAh Li-ion battery

### Key Achievements
- **Test Loss (MSE)**: 0.0089 (extremely low error rate)
- **Model Convergence**: Rapid convergence within 5 epochs with no overfitting
- **Sequence Length**: 24 timesteps for prediction

## Project Structure

```
.
├── Forecasting Smartphone Battery Degradation Using Sensor Data.ipynb  # Main notebook
├── battery_lstm_model.h5                                               # Trained model
├── requirements.txt                                                     # Dependencies
├── LICENSE                                                              # MIT License
├── README.md                                                            # This file
└── .gitignore                                                           # Git ignore rules
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd BatteryDegForecastLSTM
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```
   Then open `Forecasting Smartphone Battery Degradation Using Sensor Data.ipynb`

## Dataset

**Source**: [Mobile Battery Dataset - Kaggle](https://www.kaggle.com/datasets/rahulgarg28/mobile-battery-with-time)

### Dataset Characteristics
- **Size**: 385,429 records
- **Device**: Single Samsung SM-A910F smartphone
- **OS**: Android 8.0.0
- **Battery**: 5000 mAh Li-ion
- **Features**: 15 sensor attributes including:
  - Battery percentage, voltage, current, temperature
  - CPU usage, screen status, app running
  - Network connectivity, charging status
  - Device metadata (IMEI, model, Android version)

### Data Preprocessing
- Renamed 15 ambiguous columns to descriptive names
- Dropped 100% missing columns (battery_current, network_connected, plugged_in)
- Converted Unix timestamps to Asia/Karachi timezone
- Applied Min-Max scaling (0-1 normalization)
- Created 24-step sequences for LSTM input

## Usage

### Running the Notebook

1. Open `Forecasting Smartphone Battery Degradation Using Sensor Data.ipynb` in Jupyter
2. Execute cells sequentially:
   - **Cells 1-18**: Data loading and preprocessing
   - **Cells 19-26**: Exploratory Data Analysis (EDA) with visualizations
   - **Cells 27-31**: LSTM model building and training
   - **Cells 32-37**: Model evaluation and 24-step forecasting

### Making Predictions with the Pre-trained Model

```python
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model
model = load_model('battery_lstm_model.h5')

# Prepare your data (must be scaled using MinMaxScaler with feature_range=(0,1))
# Shape: (1, 24, 4) - 1 sample, 24 timesteps, 4 features
prediction = model.predict(your_sequence)
```

## Model Architecture

### LSTM Network Specification
| Layer | Configuration |
|-------|---|
| **Input Shape** | (24 timesteps, 4 features) |
| **LSTM Layer** | 50 units, ReLU activation, return_sequences=False |
| **Dense Output** | 1 unit (battery percentage) |
| **Loss Function** | Mean Squared Error (MSE) |
| **Optimizer** | Adam |
| **Epochs** | 30 |
| **Batch Size** | 32 |
| **Validation Split** | 10% of training data |

### Features Used
1. Battery percentage (%)
2. App running count
3. CPU usage (%)
4. Battery voltage (mV)

## Results

### Model Performance

**Training & Validation Loss**
- Training loss converged from ~0.0002 to ~0.0000 within 5 epochs
- Validation loss mirrored training loss, indicating no overfitting
- Final test MSE: **0.0089**

### Visualizations Generated
- Training history plot (loss vs. validation loss)
- Actual vs. Predicted battery percentage comparison
- 24-step battery percentage forecast
- EDA plots:
  - Battery % vs CPU usage scatter plot
  - Battery % distribution by apps running (boxplot)
  - Correlation matrix heatmap
  - KMeans clustering for battery usage patterns

## Methodology

### 1. Data Cleaning
- Renamed 15 ambiguous columns to descriptive labels
- Missing value analysis and handling
- Data type conversion and validation
- Timestamp conversion to Asia/Karachi timezone
- Dropped entirely missing columns

### 2. Exploratory Data Analysis (EDA)
- Analyzed dataset structure: 385,429 records from single device
- Identified missing patterns and data completeness
- Created visualizations for feature relationships
- Detected battery usage patterns via KMeans clustering (3 clusters)
- Generated correlation matrix for feature dependencies

### 3. Model Development
- **Architecture**: LSTM with 50 units for sequential prediction
- **Input**: 24-step sequences of 4 sensor features
- **Output**: Single battery percentage value
- **Rationale**: LSTM captures temporal dependencies better than traditional regression for time-series

### 4. Training & Evaluation
- 80-20 train-test split
- 10% validation split during training
- 30 epochs with early convergence
- No overfitting detected (loss curves aligned)

### 5. Forecasting
- 24-step ahead prediction using iterative sequence updating
- Model predicts next value and feeds it back into the sequence
- Inverse scaling returns predictions to original percentage scale

## Limitations & Future Work

### Current Limitations
- **Single Device**: Model trained on one phone; may not generalize to different devices
- **Limited Data Variability**: Constant battery percentage in samples may oversimplify real-world degradation
- **Missing Features**: Charging/discharging cycles, ambient temperature, usage intensity not available
- **Short Temporal Window**: Dataset may not capture long-term degradation patterns

### Recommended Improvements
1. **Dataset Expansion**
   - Collect data from multiple devices and Android versions
   - Include complete charging/discharging cycles
   - Capture diverse usage patterns and apps

2. **Model Enhancements**
   - Experiment with bidirectional LSTMs or Transformer architectures
   - Implement attention mechanisms for feature importance
   - Tune hyperparameters: seq_length, LSTM units, dropout regularization
   - Compare with ARIMA and Prophet models

3. **Feature Engineering**
   - Add discharge rate and ambient temperature sensors
   - Create moving averages and trend features
   - Add cyclical encoding for time-of-day patterns

4. **Validation**
   - Cross-validation across multiple devices
   - Longer-term forecasting validation
   - Real-world deployment testing

## Dependencies

All required packages are listed in `requirements.txt`:
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning utilities (scaling, clustering)
- **tensorflow/keras**: Deep learning framework
- **matplotlib/seaborn**: Data visualization

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Last Updated**: April 2024  
**Status**: ✅ Complete - Model trained and evaluated
