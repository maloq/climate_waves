from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import xarray as xr
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report

from load_climate_data import load_config, load_years
from prepare_land import prepare_land_data, landsea_distance

def compute_binarization_thresholds(
    meta_chunk: pd.DataFrame, 
    threshold_ds: xr.Dataset, 
    date_col: str = "time",
    lat_col: str = "latitude",
    lon_col: str = "longitude"
) -> np.ndarray:
    """
    Extracts the threshold baseline value (e.g. daily average temperature) for each point in meta_chunk.
    Assumes threshold_ds has a variable we want to subtract/add threshold to.
    """
    # Assuming threshold_ds has shape (dayofyear/time, lat, lon)
    # We need to map meta_chunk timestamps to dayofyear or nearest time in threshold_ds
    
    # 1. Identify variable
    # Assumption: Single variable file or 't2m' or similar. We take the first data var.
    var_name = list(threshold_ds.data_vars)[0]
    da = threshold_ds[var_name]
    
    # 2. Check alignment method
    vals = []
    
    # Optimize by vectorizing?
    # xarray selection is slow in a loop.
    # Advanced: Use xarray advanced indexing with DataArrays constructed from meta_chunk
    
    times = pd.to_datetime(meta_chunk[date_col])
    lats = xr.DataArray(meta_chunk[lat_col].values, dims="points")
    lons = xr.DataArray(meta_chunk[lon_col].values, dims="points")
    
    if "dayofyear" in da.coords:
        doy = xr.DataArray(times.dayofyear.values, dims="points")
        # sel with DataArrays performs pointwise selection
        selection = da.sel(dayofyear=doy, latitude=lats, longitude=lons, method="nearest")
    elif "time" in da.coords:
        # If it's a full time series, align by time
        target_times = xr.DataArray(times.values, dims="points")
        selection = da.sel(time=target_times, latitude=lats, longitude=lons, method="nearest")
    else:
        # Fallback: Maybe just spatial?
        selection = da.sel(latitude=lats, longitude=lons, method="nearest")
        
    return selection.values

def load_ground_truth_binary(
    data_cfg: Dict, 
    bin_cfg: Dict, 
    year: int, 
    meta_df: pd.DataFrame
) -> pd.Series:
    """
    Loads separate binary target file for ground truth comparison.
    """
    target_dir = data_cfg["target_dir"] # Usually same dir, or we can look in parent
    # User said: climate_data/target_reg/cold_waves/target_binary_p05_2010.nc
    # Our data_cfg["target_dir"] points to climate_data/target_reg/cold_waves
    
    # Use template from bin_cfg if provided, else guess
    template = bin_cfg.get("ground_truth_file_template", "target_binary_p05_{year}.nc")
    fname = template.format(year=year)
    fpath = Path(target_dir) / fname
    
    if not fpath.exists():
        print(f"  [Warning] Ground truth binary file not found: {fpath}. Skipping classification metrics.")
        return None
        
    ds = xr.open_dataset(fpath)
    var_name = bin_cfg.get("ground_truth_var", list(ds.data_vars)[0])
    
    # Strategy: Reconstruct DataArray using coordinates from temperature file
    # This avoids all issues with broken/missing time indices in the binary file.
    try:
        temp_template = data_cfg.get("target_file_template", "target_temperature_p05_{year}.nc")
        temp_path = Path(data_cfg["target_dir"]) / temp_template.format(year=year)
        
        if temp_path.exists():
            # Use open_mfdataset to match training loading behavior (might handle decoding differently)
            ds_temp = xr.open_mfdataset([str(temp_path)], chunks=None) 
            
            if "valid_time" in ds_temp.coords:
                ds_temp = ds_temp.rename({"valid_time": "time"})
            
            # Use dimensions/coords from temperature file
            # Verify size match implicitly by creation or explicitly
            vals = ds[var_name].values
            
            # Locate the relevant variable in ds_temp to get dims/coords
            temp_var_name = list(ds_temp.data_vars)[0]
            da_temp = ds_temp[temp_var_name]
            
            if vals.shape == da_temp.shape:
                 # Reconstruct!
                 da_bin = xr.DataArray(vals, coords=da_temp.coords, dims=da_temp.dims, name=var_name)
                 
                 # Now select
                 if "time" in da_bin.coords: 
                     da_bin = da_bin.sortby("time")
                 
                 # Ensure time is index (it should be from source)
                 times = pd.to_datetime(meta_df["time"])
                 lats = xr.DataArray(meta_df["latitude"].values, dims="points")
                 lons = xr.DataArray(meta_df["longitude"].values, dims="points")
                 target_times = xr.DataArray(times.values, dims="points")
                 
                 # Remove tolerance to avoid float/timedelta error. 'nearest' should work if types match.
                 selection = da_bin.sel(time=target_times, latitude=lats, longitude=lons, method="nearest")
                 return selection.values
            else:
                 print(f"  [Warning] Shape mismatch: Binary {vals.shape} vs Temp {da_temp.shape}. Cannot align.")
                 return None
    except Exception as e:
        print(f"  [Error] Failed to reconstruct/select ground truth: {e}")
        return None
        
    return None

def test_model(config: Dict) -> None:
    data_cfg = config["data"]
    output_cfg = config["output"]
    bin_cfg = config.get("binarization", {})
    
    print(f"\nTest years: {data_cfg['test_years']}")
    print(f"Loading model from: {output_cfg['model_path']}")
    
    model = CatBoostRegressor()
    model.load_model(output_cfg["model_path"])
    
    # Prepare Binarization Threshold Data
    threshold_ds = None
    if bin_cfg.get("enabled", False):
        thresh_path = bin_cfg["threshold_file"]
        print(f"Loading threshold file for binarization: {thresh_path}")
        try:
            threshold_ds = xr.open_dataset(thresh_path)
        except Exception as e:
            print(f"Error loading threshold file: {e}. Binarization disabled.")
            bin_cfg["enabled"] = False
            
    # Accumulate metrics
    all_y_true_reg = []
    all_y_pred_reg = []
    
    all_y_true_bin = []
    all_y_pred_bin = []
    
    # Process year by year to manage memory, similar to load_years internal but we use load_years helper
    # Actually load_years loads all. For test, we might want to loop manually if memory is tight, 
    # but let's stick to standard `load_years` for consistency with training unless test set is huge.
    # Given typical config, test years are few (3 years).
    
    # We load each year individually to perform binarization accurately per year
    for year in data_cfg["test_years"]:
        print(f"\nProcessing Year {year}...")
        try:
             # Load data for this year
            X, y, meta = load_years(
                [year], 
                config=config, 
                return_metadata=True, 
                apply_label_smoothing_flag=False,
                return_hard_labels=False,
                verbose=False
            )
        except Exception as e:
            print(f"Skipping {year} due to load error: {e}")
            continue
            
        if X.empty:
            print(f"No data for {year}.")
            continue

        # Feature Integration (Must match training!)
        # 1. Distance
        meta_with_dist, dist_cols = landsea_distance(meta, lat_col="latitude", lon_col="longitude")
        for col in dist_cols: X[col] = meta_with_dist[col].values
            
        # 2. Land Data
        land_files = [
            "climate_data/land_data/forest_data.nc",
            "climate_data/land_data/GMTED2010_15n015_00625deg.nc"
        ]
        meta_with_land, land_cols = prepare_land_data(land_files, meta, lat_col="latitude", lon_col="longitude")
        for col in land_cols: 
            if col in meta_with_land.columns: X[col] = meta_with_land[col].values
            
        # Predict
        preds = model.predict(X)
        
        all_y_true_reg.extend(y)
        all_y_pred_reg.extend(preds)
        
        # Binarization
        if bin_cfg.get("enabled", False) and threshold_ds is not None:
            # 1. Get Baseline
            baseline = compute_binarization_thresholds(meta, threshold_ds)
            
            # 2. Apply Logic
            thresh_val = bin_cfg.get("threshold_val", 10.0)
            operator = bin_cfg.get("operator", "less")
            
            if operator == "less":
                # Cold wave: Temp < Avg - Threshold
                binary_preds = (preds < (baseline - thresh_val)).astype(int)
            else:
                # Heat wave: Temp > Avg + Threshold
                binary_preds = (preds > (baseline + thresh_val)).astype(int)
                
            # 3. Get Ground Truth
            binary_truth = load_ground_truth_binary(data_cfg, bin_cfg, year, meta)
            
            if binary_truth is not None:
                all_y_pred_bin.extend(binary_preds)
                all_y_true_bin.extend(binary_truth)
                
                # Per-year stats
                yr_f1 = classification_report(binary_truth, binary_preds, output_dict=True, zero_division=0).get("1", {}).get("f1-score", 0)
                print(f"  Year {year} F1: {yr_f1:.4f} (Positives: {np.sum(binary_preds)} pred / {np.sum(binary_truth)} true)")
            else:
                print(f"  Skipping classification metrics for {year} (No ground truth).")

    # Final Metrics
    print("\n" + "="*60)
    print("OVERALL RESULTS")
    print("="*60)
    
    if all_y_true_reg:
        rmse = np.sqrt(mean_squared_error(all_y_true_reg, all_y_pred_reg))
        mae = mean_absolute_error(all_y_true_reg, all_y_pred_reg)
        r2 = r2_score(all_y_true_reg, all_y_pred_reg)
        print(f"REGRESSION:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R2:   {r2:.4f}")
    
    if all_y_true_bin:
        print(f"\nCLASSIFICATION (Binarized):")
        report = classification_report(all_y_true_bin, all_y_pred_bin, zero_division=0)
        print(report)
    
    print("="*60)

def main() -> None:
    print("TESTING CLIMATE REGRESSION")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_climate_reg_cold.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    test_model(config)

if __name__ == "__main__":
    main()
