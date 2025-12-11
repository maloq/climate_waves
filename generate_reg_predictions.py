"""
Generate regression model predictions for all years (train + test).
These predictions are saved and can be used as features in the classification model.

Usage:
    python generate_reg_predictions.py --config configs/config_climate_reg_cold.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np
import xarray as xr
from catboost import CatBoostRegressor

from load_climate_data import load_config, load_years
from prepare_land import prepare_land_data, landsea_distance


def load_daily_average(climatology_path: str, meta: pd.DataFrame) -> np.ndarray:
    """
    Load daily average climatology and map to metadata samples.
    
    Args:
        climatology_path: Path to the daily average NetCDF file
        meta: DataFrame with 'time', 'latitude', 'longitude' columns
        
    Returns:
        Array of daily average temperatures for each sample in meta
    """
    ds = xr.open_dataset(climatology_path)
    
    # Get the temperature variable (usually 't2m')
    temp_var = list(ds.data_vars)[0]
    
    # Extract day of year from metadata
    meta_time = pd.to_datetime(meta["time"])
    day_of_year = meta_time.dt.dayofyear
    
    # Climatology has 365 days, handle leap year day 366
    day_of_year = day_of_year.clip(upper=365)
    
    # Get unique coordinates for efficient lookup
    lats = meta["latitude"].values
    lons = meta["longitude"].values
    
    # Extract climatology data
    clim_data = ds[temp_var].values  # Shape: (365, lat, lon)
    clim_lats = ds["latitude"].values
    clim_lons = ds["longitude"].values
    
    # Build lookup indices
    lat_indices = np.abs(clim_lats[:, None] - lats[None, :]).argmin(axis=0)
    lon_indices = np.abs(clim_lons[:, None] - lons[None, :]).argmin(axis=0)
    time_indices = day_of_year.values - 1  # 0-indexed
    
    # Vectorized lookup
    daily_avg = clim_data[time_indices, lat_indices, lon_indices]
    
    ds.close()
    return daily_avg


def generate_predictions_for_year(
    year: int,
    model: CatBoostRegressor,
    config: Dict,
    climatology_path: str,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generate regression predictions for a single year.
    
    Returns:
        DataFrame with columns: time, latitude, longitude, reg_pred_anomaly
    """
    if verbose:
        print(f"\nProcessing Year {year}...")
    
    try:
        # Load data for this year
        # Force full sampling (sample_fraction=1.0) for generating predictions
        config_copy = config.copy()
        config_copy["data"] = config["data"].copy()
        config_copy["data"]["sample_fraction"] = 1.0
        
        X, y_raw, meta = load_years(
            [year], 
            config=config_copy, 
            return_metadata=True, 
            apply_label_smoothing_flag=False,
            return_hard_labels=False,
            verbose=verbose
        )
    except Exception as e:
        print(f"  Skipping {year} due to load error: {e}")
        return pd.DataFrame()
        
    if X.empty:
        print(f"  No data for {year}.")
        return pd.DataFrame()

    # Add static features (must match training!)
    # 1. Distance to Coast
    if verbose:
        print(f"  Adding static features...")
    
    meta_with_dist, dist_cols = landsea_distance(meta, lat_col="latitude", lon_col="longitude")
    for col in dist_cols:
        X[col] = meta_with_dist[col].values
        
    # 2. Land Data
    land_files = [
        "climate_data/land_data/forest_data.nc",
        "climate_data/land_data/GMTED2010_15n015_00625deg.nc"
    ]
    meta_with_land, land_cols = prepare_land_data(land_files, meta, lat_col="latitude", lon_col="longitude")
    for col in land_cols: 
        if col in meta_with_land.columns:
            X[col] = meta_with_land[col].values

    # Make predictions (model outputs anomaly)
    if verbose:
        print(f"  Making predictions...")
    preds_anomaly = model.predict(X)
    
    # Create output dataframe
    result = pd.DataFrame({
        "time": meta["time"].values,
        "latitude": meta["latitude"].values,
        "longitude": meta["longitude"].values,
        "reg_pred_anomaly": preds_anomaly
    })
    
    if verbose:
        print(f"  Generated {len(result):,} predictions")
        print(f"  Prediction stats: min={preds_anomaly.min():.2f}, max={preds_anomaly.max():.2f}, mean={preds_anomaly.mean():.2f}")
    
    return result


def generate_all_predictions(config: Dict) -> None:
    """Generate predictions for all years (train + test)."""
    
    data_cfg = config["data"]
    output_cfg = config["output"]
    bin_cfg = config.get("binarization", {})
    
    # Get all years to process
    train_years = data_cfg.get("train_years", [])
    test_years = data_cfg.get("test_years", [])
    all_years = sorted(set(train_years) | set(test_years))
    
    print(f"\n{'='*60}")
    print("GENERATING REGRESSION PREDICTIONS FOR ALL YEARS")
    print(f"{'='*60}")
    print(f"Train years: {train_years}")
    print(f"Test years: {test_years}")
    print(f"All years to process: {all_years}")
    
    # Load model
    model_path = output_cfg["model_path"]
    print(f"\nLoading model from: {model_path}")
    model = CatBoostRegressor()
    model.load_model(model_path)
    
    # Load climatology path
    climatology_path = bin_cfg.get("threshold_file", "climate_data/target_reg/daily_average_cold.nc")
    print(f"Climatology path: {climatology_path}")
    
    # Output directory for predictions
    pred_dir = Path(output_cfg.get("predictions_dir", "artifacts/coldwave_reg/predictions"))
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each year
    all_predictions = []
    
    for year in all_years:
        result = generate_predictions_for_year(
            year=year,
            model=model,
            config=config,
            climatology_path=climatology_path,
            verbose=True
        )
        
        if not result.empty:
            # Save individual year predictions
            year_path = pred_dir / f"reg_predictions_{year}.parquet"
            result.to_parquet(year_path, index=False)
            print(f"  Saved to: {year_path}")
            
            all_predictions.append(result)
    
    # Save combined predictions file
    if all_predictions:
        combined = pd.concat(all_predictions, ignore_index=True)
        combined_path = pred_dir / "reg_predictions_all.parquet"
        combined.to_parquet(combined_path, index=False)
        
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Total predictions: {len(combined):,}")
        print(f"Years processed: {len(all_predictions)}")
        print(f"Combined file saved to: {combined_path}")
        print(f"Individual year files saved to: {pred_dir}/reg_predictions_<year>.parquet")
        
        # Print distribution stats
        print(f"\nPrediction distribution:")
        print(f"  Min:  {combined['reg_pred_anomaly'].min():.2f}")
        print(f"  Max:  {combined['reg_pred_anomaly'].max():.2f}")
        print(f"  Mean: {combined['reg_pred_anomaly'].mean():.2f}")
        print(f"  Std:  {combined['reg_pred_anomaly'].std():.2f}")
        
        # Extreme event counts (using threshold from config)
        threshold = bin_cfg.get("threshold_val", 10.0)
        n_extreme = (combined['reg_pred_anomaly'] < -threshold).sum()
        print(f"\n  Predicted extreme events (anomaly < -{threshold}): {n_extreme:,} ({100*n_extreme/len(combined):.2f}%)")


def main() -> None:
    print("=" * 60)
    print("GENERATE REGRESSION PREDICTIONS FOR ALL YEARS")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(description="Generate regression predictions for use as features")
    parser.add_argument("--config", default="configs/config_climate_reg_cold.yaml", help="Path to regression config YAML")
    args = parser.parse_args()
    
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    print("Config loaded successfully.")
    
    generate_all_predictions(config)


if __name__ == "__main__":
    main()
