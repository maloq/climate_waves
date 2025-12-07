from __future__ import annotations

import os
import glob
import re
import pathlib
import time as time_lib
from typing import Dict, Iterable, List, Tuple, Union
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from feature_engineering import engineer_features, get_feature_names

# Reuse some helper functions from load_data.py if possible, or reimplement
from load_data import _subset_region, _to_dataframe_fast, load_config, apply_label_smoothing

def find_climate_files(climate_data_dir: str, variable: str):
    """
    Return (file_list, variable_dir) for *variable*.
    """
    # Try <dir>/<variable>
    variable_dir = os.path.join(climate_data_dir, variable)
    # Search for .zarr directories
    pattern = os.path.join(variable_dir, f"{variable}_*.zarr")
    files = glob.glob(pattern)
    
    # Fallback usually not needed if structure is known, but keeping from example
    if not files:
        # Try one level deeper just in case
        pattern = os.path.join(climate_data_dir, "**", f"{variable}_*.zarr")
        files = glob.glob(pattern, recursive=True)
        if files:
            variable_dir = str(pathlib.Path(files[0]).parent)

    if not files:
        raise FileNotFoundError(
            f"No Zarr stores named '{variable}_*.zarr' found in {climate_data_dir}"
        )
    return sorted(files), variable_dir

def load_zarr_variable(
    climate_data_dir: str, 
    variable: str, 
    time_range: Tuple[pd.Timestamp, pd.Timestamp] = None,
    lat_range: Tuple[float, float] = None, 
    lon_range: Tuple[float, float] = None
) -> xr.Dataset:
    """Load a single variable from Zarr, lazily."""
    files, _ = find_climate_files(climate_data_dir, variable)
    
    # Assuming one Zarr store or multiple that need combining
    # xr.open_mfdataset works with zarr if using engine='zarr', but it's tricky with list of paths.
    # Usually you open one zarr store at a time.
    # If there are multiple, they might be split by time.
    
    datasets = []
    # Use open_mfdataset to match example logic, allowing auto-detect
    # The example uses open_mfdataset on the list of files.
    # We can try that directly.
    try:
        # Note: open_mfdataset with zarr stores in a list is not standard xarray behavior 
        # (usually it's for netcdf), but the user example does it.
        # It relies on the backend being able to handle it.
        # If engine is not specified, xarray tries to guess.
        
        # If there is only one file, open_dataset might be better, but let's try open_mfdataset if multiple.
        if len(files) == 1:
             ds = xr.open_dataset(files[0], chunks="auto", decode_timedelta=False)
        else:
             ds = xr.open_mfdataset(
                files, 
                combine="by_coords", 
                decode_timedelta=False,
                chunks="auto",
                parallel=True
             )
        datasets.append(ds)
    except Exception as e:
        print(f"Warning: Failed to open files for {variable}: {e}")
        # Fallback to trying individual files with open_dataset
        for f in files:
            try:
                # auto-detect engine
                ds = xr.open_dataset(f, chunks="auto", decode_timedelta=False)
                datasets.append(ds)
            except Exception as e2:
                print(f"  Failed to open {f}: {e2}")

    if not datasets:
        raise IOError(f"Could not open any datasets for {variable}")

    if len(datasets) > 1:
        # Concatenate if multiple files
        # Assuming they are split by time or something compatible
        ds = xr.concat(datasets, dim="valid_time") # Adjust dim if needed
        ds = ds.sortby("valid_time")
    else:
        ds = datasets[0]

    # Standardize coordinate names
    if "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    
    if "time" not in ds.coords:
        raise ValueError(f"Dataset for {variable} missing time coordinate (checked 'time' and 'valid_time')")

    # Spatial slicing
    if lat_range and ('latitude' in ds.coords or 'lat' in ds.coords):
        lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
        # Handle decreasing latitude if necessary
        lat_vals = ds[lat_name]
        start, end = lat_range
        if lat_vals[0] > lat_vals[-1]:
             ds = ds.sel({lat_name: slice(end, start)})
        else:
             ds = ds.sel({lat_name: slice(start, end)})
             
    if lon_range and ('longitude' in ds.coords or 'lon' in ds.coords):
        lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'
        ds = ds.sel({lon_name: slice(lon_range[0], lon_range[1])})

    # Time slicing
    if time_range:
        dataset_time = ds['time'] # already renamed
        # Convert pandas timestamps to numpy datetime64 compatible with xarray
        # Assuming dataset uses datetime64[ns]
        ds = ds.sel(time=slice(time_range[0], time_range[1]))

    return ds

def load_years(
    years: Iterable[int],
    config: Dict,
    return_metadata: bool = False,
    verbose: bool = True,
    apply_feature_engineering: bool = True,
    return_hard_labels: bool = False,
    apply_label_smoothing_flag: bool = True,
    **kwargs # consume extra args
) -> Tuple[pd.DataFrame, pd.Series] | Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load data for specified years from Zarr sources + Targets.
    """
    years = sorted(list(years))
    if verbose:
        print(f"Loading climate data for years: {years} from Zarr files...")
        
    data_cfg = config["data"]
    features_dir = data_cfg["features_dir"]
    variables = config["features"]
    
    # Determine full time range needed
    # We load slightly more to allow for lags
    start_year = min(years)
    end_year = max(years)
    
    # Add buffer for lags (Use 2 years for building features as requested)
    # The feature engineering needs raw data before the start date for lags/windows
    buffer_days = 730 
    start_date = pd.Timestamp(f"{start_year}-01-01") - pd.Timedelta(days=buffer_days)
    end_date = pd.Timestamp(f"{end_year}-12-31")
    
    lat_range = tuple(data_cfg["latitude_range"])
    lon_range = tuple(data_cfg["longitude_range"])

    # 1. Load all variables
    feature_datasets = []
    
    for var in variables:
        if verbose:
            print(f"  Loading variable: {var}...", end=" ", flush=True)
        try:
            ds = load_zarr_variable(
                features_dir, 
                var, 
                time_range=(start_date, end_date),
                lat_range=lat_range,
                lon_range=lon_range
            )
            # Ensure unified lat/lon names for merging
            ds = ds.rename({
                k: v for k, v in {'lat': 'latitude', 'lon': 'longitude'}.items() 
                if k in ds.coords and v not in ds.coords
            })
            feature_datasets.append(ds)
            if verbose:
                print("done.")
        except Exception as e:
            if verbose: print(f"failed! ({e})")
            raise

    # 2. Merge variables
    if verbose:
        print("  Merging variables...", end=" ", flush=True)
    
    # Use inner join to ensure we have data for all variables
    ds_features = xr.merge(feature_datasets, join="inner")
    
    # 3. Feature Engineering
    if apply_feature_engineering:
        if verbose:
            print("  Applying feature engineering...", end=" ", flush=True)
        
        # Load into memory for FE (might be heavy, but FE in dask can be slow too)
        # Assuming memory is sufficient as per original script behavior
        ds_features = ds_features.load()
        
        ds_features = engineer_features(
            ds_features, 
            config.get("feature_engineering", {}),
            time_dim="time",
            lat_dim="latitude",
            lon_dim="longitude"
        )
        if verbose:
            print("done.")

    # 4. Trim time range to requested years (remove buffer)
    # Be precise: start of start_year to end of end_year
    exact_start = pd.Timestamp(f"{start_year}-01-01")
    exact_end = pd.Timestamp(f"{end_year}-12-31")
    ds_features = ds_features.sel(time=slice(exact_start, exact_end))

    # --- NEW: Interpolate features to target resolution ---
    interpolate = kwargs.get("interpolate", config.get("data", {}).get("interpolate", True))
    if interpolate:
        if verbose:
            print("  Interpolating features to target grid...", end=" ", flush=True)

        # Find a sample target file to get the grid
        sample_target_path = None
        target_path_template = pathlib.Path(data_cfg["target_dir"]) / data_cfg["target_file_template"]
        for yr in years:
            p = pathlib.Path(str(target_path_template).format(year=yr))
            if p.exists():
                sample_target_path = p
                break
                
        if sample_target_path:
            with xr.open_dataset(sample_target_path) as ds_t_sample:
                # Standardize coords in sample
                if "date" in ds_t_sample.coords: ds_t_sample = ds_t_sample.rename({"date": "time"})
                # Rename lat/lon if needed to match features (latitude/longitude)
                rename_dict = {}
                if "lat" in ds_t_sample.coords and "latitude" not in ds_t_sample.coords: rename_dict["lat"] = "latitude"
                if "lon" in ds_t_sample.coords and "longitude" not in ds_t_sample.coords: rename_dict["lon"] = "longitude"
                if rename_dict:
                    ds_t_sample = ds_t_sample.rename(rename_dict)
                    
                # Subset to config range (same as we do for targets later)
                ds_t_sample = _subset_region(ds_t_sample, lat_range, lon_range)
                
                # Extract grid
                target_lats = ds_t_sample["latitude"]
                target_lons = ds_t_sample["longitude"]
                
                # Interpolate
                ds_features = ds_features.interp(
                    latitude=target_lats, 
                    longitude=target_lons, 
                    method="linear"
                )
            if verbose:
                print(f"done. (Grid: {len(target_lats)}x{len(target_lons)})")
        else:
            print("Warning: Could not find any target files to determine grid for interpolation!")
    elif verbose:
        print("  Skipping interpolation (interpolate=False)")
    # ------------------------------------------------------

    # 5. Load Targets for each year and merge
    if verbose:
        print("  Loading targets...", end=" ", flush=True)
        
    target_dfs = []
    for yr in years:
        target_path = pathlib.Path(data_cfg["target_dir"]) / data_cfg["target_file_template"].format(year=yr)
        if not target_path.exists():
             raise FileNotFoundError(f"Target file missing for year {yr}: {target_path}")
        
        with xr.open_dataset(target_path) as ds_t:
             # Standardize coords
             if "date" in ds_t.coords: ds_t = ds_t.rename({"date": "time"})
             ds_t = _subset_region(ds_t, lat_range, lon_range)
             ds_t = ds_t.load()
             
             # Convert to DF
             df_t, y_t, meta_t = _to_dataframe_fast(ds_t, data_cfg["target_var"])
             
             # Re-attach target to df for merging
             df_t[data_cfg["target_var"]] = y_t
             # Add metadata for safe merge
             df_t["time"] = meta_t["time"]
             df_t["latitude"] = meta_t["latitude"]
             df_t["longitude"] = meta_t["longitude"]
             
             target_dfs.append(df_t)

    full_target_df = pd.concat(target_dfs, axis=0, ignore_index=True)
    
    # 6. Merge Features with Target
    # Convert features to DF
    if verbose:
        print("  converting features to DataFrame...", end=" ", flush=True)
        
    # We need a dummy target var for _to_dataframe_fast or use a different method.
    # _to_dataframe_fast assumes a target var is present to separate X and y.
    # But here ds_features only has features.
    
    # Let's check _to_dataframe_fast again in load_data.py
    # It executes: y = df.pop(target_var).astype("int64")
    # So we must provide a variable.
    # Let's add a dummy target to ds_features matching the shape of existing vars
    first_var_name = list(ds_features.data_vars)[0]
    ds_features["__dummy__"] = (ds_features[first_var_name] * 0).astype(int)
    df_features, _, meta_features = _to_dataframe_fast(ds_features, "__dummy__")
    
    # Add meta to df_features for merging
    df_features["time"] = meta_features["time"]
    df_features["latitude"] = meta_features["latitude"]
    df_features["longitude"] = meta_features["longitude"]
    
    if verbose:
        print("  Merging features with target...", end=" ", flush=True)

    # Merge on coordinates
    # Ensure compatible types
    # xarray loading might result in slightly different datetime precision vs netcdf target
    # Often it's safer to coerce to proper datetime in pandas
    
    df_features["time"] = pd.to_datetime(df_features["time"])
    full_target_df["time"] = pd.to_datetime(full_target_df["time"])
    
    # Round coordinates to avoid float precision issues
    for col in ["latitude", "longitude"]:
        df_features[col] = df_features[col].round(2)
        full_target_df[col] = full_target_df[col].round(2)
        
    # Merge
    merged_df = pd.merge(
        df_features, 
        full_target_df, 
        on=["time", "latitude", "longitude"], 
        how="inner"
    )
    
    if verbose:
        print(f"done. (Final shape: {merged_df.shape})")

    # 7. Finalize
    target_var = data_cfg["target_var"]
    y = merged_df.pop(target_var)
    
    # Extract metadata
    meta = merged_df[["time", "latitude", "longitude"]].copy()
    merged_df = merged_df.drop(columns=["time", "latitude", "longitude"])
    
    # Label Smoothing
    y_hard = y.copy()
    ls_enabled = config.get("label_smoothing", {}).get("enabled", False)
    if ls_enabled and apply_label_smoothing_flag:
         if verbose: print("  Applying label smoothing...")
         ls_cfg = config["label_smoothing"]
         y = apply_label_smoothing(
             y, meta,
             temporal_sigma=ls_cfg.get("temporal_sigma", 1.0),
             temporal_radius=ls_cfg.get("temporal_radius", 3),
             spatial_sigma=ls_cfg.get("spatial_sigma", 1.0),
             spatial_radius=ls_cfg.get("spatial_radius", 1),
             max_smooth_value=ls_cfg.get("max_smooth_value", 1.0),
             min_smooth_value=ls_cfg.get("min_smooth_value", 0.0),
             verbose=verbose
         )

    # Return
    if return_hard_labels and return_metadata:
        return merged_df, y, meta, y_hard
    elif return_hard_labels:
        return merged_df, y, y_hard
    elif return_metadata:
        return merged_df, y, meta
    else:
        return merged_df, y
