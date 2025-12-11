# Climate Extreme Event Prediction

Machine learning pipeline for predicting extreme weather events (coldwaves, heatwaves, and wind events) using CatBoost gradient boosting models with ECMWF ERA5 climate data.

## Overview

This project provides a complete workflow for:
- **Binary classification** of extreme weather events (e.g., coldwave detection)
- **Regression-based prediction** of temperature anomalies for event forecasting
- Extensive **feature engineering** from meteorological variables
- **Hyperparameter optimization** using Optuna
- Model evaluation with standard and relaxed temporal metrics

## Project Structure

```
.
├── configs/                          # YAML configuration files
│   ├── config_climate_coldwave.yaml  # Coldwave classification config
│   ├── config_climate_heatwave.yaml  # Heatwave classification config
│   ├── config_climate_wind.yaml      # Wind event classification config
│   ├── config_climate_reg_cold.yaml  # Coldwave regression config
│   └── config_climate_reg_wind.yaml  # Wind regression config
├── artifacts/                        # Trained models and predictions
│   ├── coldwave/                     # Coldwave classification artifacts
│   ├── coldwave_reg/                 # Coldwave regression artifacts
│   ├── heatwave/                     # Heatwave classification artifacts
│   └── wind/                         # Wind event artifacts
├── make_target/                      # Target variable generation scripts
│   ├── read_and_extract_target.py    # Wind gust target extraction
│   ├── read_and_extract_temp_targets.py  # Temperature target extraction
│   └── binarize_temp_targets.py      # Binary threshold application
├── eda_scripts/                      # Exploratory data analysis
│   ├── climate_eda.py
│   ├── eda_heatwaves.py
│   └── spatial_maps_single_day.py
├── train_climate.py                  # Binary classification training
├── train_climate_reg.py              # Regression model training
├── test_climate.py                   # Classification evaluation
├── test_climate_reg.py               # Regression evaluation
├── generate_reg_predictions.py       # Generate regression predictions
├── load_climate_data.py              # Data loading utilities
├── load_data.py                      # Core data loading functions
├── feature_engineering.py            # Feature computation module
├── prepare_land.py                   # Static land feature preparation
├── compute_climatology.py            # Climatology computation
├── optuna_optimize_climate.py        # Hyperparameter optimization
├── visualization.py                  # Plotting utilities
└── requirements.txt                  # Python dependencies
```

## Installation

### Prerequisites
- Python 3.9+
- Conda (recommended) or virtualenv

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd code
```

2. Create and activate a conda environment:
```bash
conda create -n climate python=3.10
conda activate climate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Requirements

### Climate Features
Place ECMWF ERA5 data in Zarr format in the features directory. Required variables include:
- `t2m` - 2m temperature
- `sd` - Snow depth
- `tp` - Total precipitation
- `u10`, `v10` - 10m wind components
-  `msl` - mean atmospheric pressure
- Pressure level variables (850hPa, 1000hPa): geopotential, temperature, wind components

### Target Data
Generate target files using scripts in `make_target/`:
```bash
# For temperature-based events (coldwaves/heatwaves)
python make_target/read_and_extract_temp_targets.py

# For wind events
python make_target/read_and_extract_target.py
```

### Static Land Data
Required files in `climate_data/land_data/`:
- `forest_data.nc` - Forest cover data
- `GMTED2010_15n015_00625deg.nc` - Elevation data
- `IMERG_land_sea_mask.nc` - Land-sea mask for distance calculations

## Configuration

Configurations are defined in YAML files under `configs/`. Key sections include:

```yaml
data:
  features_dir: /path/to/climate_data/ECMWF
  target_dir: climate_data/target/target_cold_waves
  latitude_range: [40.0, 60.0]
  longitude_range: [30.0, 120.0]
  train_years: [2017, 2018, 2019, 2020, 2021]
  test_years: [2022, 2023, 2024]
  sample_fraction: 0.06  # Subsampling for faster training

features:
  - t2m
  - sd
  - tp
  # ... additional variables

feature_engineering:
  temporal:
    month: true
    day_of_year: true
    sin_cos_annual: true
  ewm:
    variables: [t2m, tp]
    spans: [3, 7, 14]
  spatial:
    variables: [t2m]
    window_sizes: [5, 11]
    stats: [mean, std]
  # ... additional engineering options

model:
  iterations: 500
  learning_rate: 0.01
  depth: 8
  loss_function: Logloss
  eval_metric: F1
```

## Usage

### Training

#### Binary Classification
Train a classifier for extreme event detection:
```bash
python train_climate.py --config configs/config_climate_coldwave.yaml
```

#### Regression
Train a regressor for temperature anomaly prediction:
```bash
python train_climate_reg.py --config configs/config_climate_reg_cold.yaml
```

### Evaluation

#### Classification Evaluation
```bash
python test_climate.py --config configs/config_climate_coldwave.yaml
```

#### Regression Evaluation
```bash
python test_climate_reg.py --config configs/config_climate_reg_cold.yaml
```

### Hyperparameter Optimization
```bash
python optuna_optimize_climate.py --config configs/config_climate_coldwave.yaml --trials 50
```

### Generating Predictions
Generate regression predictions for use as features in classification:
```bash
python generate_reg_predictions.py --config configs/config_climate_reg_cold.yaml
```

## Feature Engineering

The pipeline supports extensive feature engineering:

| Feature Type | Description |
|-------------|-------------|
| **Temporal** | Month, day of year, sin/cos annual cycles |
| **Lag** | Historical values (1, 3, 7 days) |
| **EWM** | Exponential weighted moving averages |
| **Spatial** | Neighborhood statistics (mean, std, min, max) |
| **Gradients** | Spatial gradients (lat, lon, magnitude) |
| **Temporal Diff** | Rate of change over time periods |
| **Climatology** | Anomalies from daily climatology |
| **Wind Chill** | Wind chill index calculation |
| **Advection** | Temperature advection |
| **Regional** | Domain-wide aggregated statistics |

## Model Architecture

### Two-Stage Approach
1. **Regression Model**: Predicts temperature anomaly (deviation from climatology)
2. **Classification Model**: Uses regression predictions + meteorological features to classify extreme events

### Key Techniques
- **Label Smoothing**: Temporal and spatial smoothing of binary labels
- **Undersampling**: Balancing highly imbalanced extreme event datasets
- **Dynamic Threshold Tuning**: Optimizing decision threshold on validation set
- **Class Weighting**: Scale positive class weights for rare events
```

## Output Artifacts

After training, the following artifacts are generated:
- `*.cbm` - CatBoost model file
- `*.importance.csv` - Feature importance rankings
- `*.report.txt` - Training/validation metrics
- `predictions/` - Prediction files in Parquet format

## Metrics

### Standard Metrics
- Precision, Recall, F1-score
- PR AUC (Average Precision)
- Confusion matrix

### Relaxed Metrics
Relaxed precision/recall allowing ±1 day temporal tolerance for event matching.

## Dependencies

Core dependencies (see `requirements.txt` for versions):
- `xarray`, `netCDF4` - Climate data handling
- `catboost` - Gradient boosting models
- `scikit-learn` - ML utilities
- `optuna` - Hyperparameter optimization
- `polars`, `pandas` - Data manipulation
- `dask`, `zarr` - Parallel/chunked data processing
- `geopandas` - Geospatial operations
- `numba` (optional) - Accelerated feature engineering