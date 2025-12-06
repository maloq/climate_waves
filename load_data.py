from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from scipy.ndimage import gaussian_filter

from feature_engineering import engineer_features, get_feature_names
import sys

# =============================================================================
# FEATURE NAMING HELPERS
# =============================================================================

def get_expected_features(config: Dict) -> List[str]:
    """Generate list of all expected feature names based on config."""
    # We can use the utility from feature_engineering if it works for us.
    # But get_feature_names requires base_features list which we might simply get from config OR from the file.
    # Let's trust config["features"] as base.
    base_features = config.get("features", [])
    if not base_features:
        # Fallback: if FEATURES is not defined in config, we might have a problem detecting base vars for caching
        # But load_single_year usually fails if features are missing anyway.
        return []
        
    return get_feature_names(base_features, config.get("feature_engineering", {}))


def get_missing_features_config(full_config: Dict, existing_columns: List[str]) -> Dict:
    """Identify which features are missing and return a partial config to generate them."""
    full_fe = full_config.get("feature_engineering", {})
    if not full_fe.get("enabled", False):
        return {}

    existing_set = set(existing_columns)
    missing_fe = {"enabled": True}
    has_missing = False
    
    # print(f"DEBUG: checking for missing features against {len(existing_set)} columns") # Debug

    # Temporal
    if "temporal" in full_fe:
        t_cfg = full_fe["temporal"]
        miss_t = {}
        if t_cfg.get("month", True) and "month" not in existing_set:
            miss_t["month"] = True
        if t_cfg.get("day_of_year", True) and "day_of_year" not in existing_set:
            miss_t["day_of_year"] = True
        if t_cfg.get("sin_cos_annual", True) and ("sin_annual" not in existing_set or "cos_annual" not in existing_set):
            miss_t["sin_cos_annual"] = True
        if t_cfg.get("sin_cos_semiannual", False) and ("sin_semiannual" not in existing_set or "cos_semiannual" not in existing_set):
            miss_t["sin_cos_semiannual"] = True
        
        if miss_t:
            missing_fe["temporal"] = miss_t
            has_missing = True

    # Check function for list-based features
    # We need to replicate naming logic of feature_engineering exactly or import it?
    # feature_engineering doesn't expose individual namers easily.
    # Reimplementing basic naming logic here.

    def check_list_features(section_name, var_key, param_key, naming_fn):
        if section_name not in full_fe: return
        
        sec_cfg = full_fe[section_name]
        vars_list = sec_cfg.get(var_key, [])
        params_list = sec_cfg.get(param_key, [])
        
        needed_vars = set()
        needed_params = set()
        
        section_has_missing = False
        
        for var in vars_list:
            for param in params_list:
                name = naming_fn(var, param)
                # If checking multiple names (e.g. gradients), naming_fn returns list
                names = name if isinstance(name, list) else [name]
                if any(n not in existing_set for n in names):
                    needed_vars.add(var)
                    needed_params.add(param)
                    section_has_missing = True

        if section_has_missing:
            # We must include ALL params for the needed vars because engineer_features 
            # computes outer product of (vars x params). 
            # If we only pass missing params, we might miss combinations if we are not careful.
            # But engineer_features takes list of vars and list of params.
            # So if we pass [v1] and [p1, p2], it computes v1_p1 and v1_p2.
            # If v1_p1 exists but v1_p2 is missing, we need to compute v1_p2.
            # But engineer_features will assume we want both if we pass [v1] and [p1, p2].
            # This is fine, recomputing v1_p1 is acceptable cost for simplicity.
            
            missing_fe[section_name] = {
                var_key: list(needed_vars),
                param_key: list(needed_params)
            }
            # Copy other keys like 'stats' for spatial
            for k, v in sec_cfg.items():
                if k not in [var_key, param_key]:
                    missing_fe[section_name][k] = v
            return True
        return False

    # Lag
    if check_list_features("lag", "variables", "lags", lambda v, p: f"{v}_lag{p}"): has_missing = True
    # EWM
    if check_list_features("ewm", "variables", "spans", lambda v, p: f"{v}_ewm{p}"): has_missing = True
    # Temporal Diff
    if check_list_features("temporal_diff", "variables", "periods", lambda v, p: f"{v}_diff{p}"): has_missing = True
    
    # Spatial Stats
    spatial_vars = set()
    spatial_windows = set()
    if "spatial" in full_fe:
        s_cfg = full_fe["spatial"]
        stats = s_cfg.get("stats", ["mean", "std"])
        for var in s_cfg.get("variables", []):
            for win in s_cfg.get("window_sizes", []):
                # Check if all stats for this window exist
                if any(f"{var}_{stat}{win}" not in existing_set for stat in stats):
                    spatial_vars.add(var)
                    spatial_windows.add(win)
        
        if spatial_vars:
            missing_fe["spatial"] = {
                "variables": list(spatial_vars),
                "window_sizes": list(spatial_windows),
                "stats": stats
            }
            has_missing = True

    # Gradients
    if "gradients" in full_fe:
        g_cfg = full_fe["gradients"]
        g_vars = []
        for var in g_cfg.get("variables", []):
            if any(f"{var}_grad_{s}" not in existing_set for s in ["lat", "lon", "mag"]):
                g_vars.append(var)
        if g_vars:
            missing_fe["gradients"] = {"variables": g_vars}
            has_missing = True

    # Anomalies
    if "anomaly" in full_fe:
        a_cfg = full_fe["anomaly"]
        a_vars = [v for v in a_cfg.get("variables", []) if f"{v}_anom" not in existing_set]
        if a_vars:
            missing_fe["anomaly"] = {"variables": a_vars}
            has_missing = True

    if has_missing:
        return missing_fe
    return {}


# =============================================================================
# SPATIO-TEMPORAL LABEL SMOOTHING
# =============================================================================

def apply_label_smoothing(
    y: pd.Series,
    meta: pd.DataFrame,
    temporal_sigma: float = 1.0,
    temporal_radius: int = 3,
    spatial_sigma: float = 1.0,
    spatial_radius: int = 1,
    max_smooth_value: float = 1.0,
    min_smooth_value: float = 0.0,
    verbose: bool = True,
) -> pd.Series:
    """Apply spatio-temporal label smoothing to binary heatwave labels.
    
    Instead of hard 0/1 targets, this "smears" the heatwave label into 
    surrounding days and locations. A day next to a heatwave becomes 
    intermediate values (e.g., 0.3 or 0.5) instead of hard 0.
    
    The smoothing uses Gaussian kernels for both temporal and spatial dimensions.
    
    Args:
        y: Binary labels (0 or 1) as pandas Series
        meta: DataFrame with 'latitude', 'longitude', 'time' columns
        temporal_sigma: Standard deviation for temporal Gaussian smoothing (in days)
        temporal_radius: Number of days to consider for temporal smoothing
        spatial_sigma: Standard deviation for spatial Gaussian smoothing (in grid cells)
        spatial_radius: Number of grid cells to consider for spatial smoothing
        max_smooth_value: Maximum value for smoothed positive labels (default 1.0)
        min_smooth_value: Minimum value for background (default 0.0)
        verbose: Print progress information
    
    Returns:
        Smoothed labels as pandas Series with float values in [min_smooth_value, max_smooth_value]
    
    Example:
        >>> y_smooth = apply_label_smoothing(
        ...     y, meta, 
        ...     temporal_sigma=1.5, temporal_radius=3,
        ...     spatial_sigma=1.0, spatial_radius=2
        ... )
    """
    if verbose:
        print("Applying spatio-temporal label smoothing...", end=" ", flush=True)
    
    # Convert to numpy for faster operations
    y_arr = y.values.astype(np.float64)
    # Ensure times are numpy datetime64[ns]
    times = pd.to_datetime(meta['time'].values).values.astype("datetime64[ns]")
    lats = meta['latitude'].values
    lons = meta['longitude'].values
    
    # Get unique coordinates
    unique_lats = np.sort(np.unique(lats))
    unique_lons = np.sort(np.unique(lons))
    unique_times = np.sort(np.unique(times))
    
    # Create mappings from coordinates to indices
    lat_to_idx = {lat: i for i, lat in enumerate(unique_lats)}
    lon_to_idx = {lon: i for i, lon in enumerate(unique_lons)}
    time_to_idx = {t: i for i, t in enumerate(unique_times)}
    
    # Build 3D grid (time, lat, lon)
    n_times = len(unique_times)
    n_lats = len(unique_lats)
    n_lons = len(unique_lons)
    
    # Initialize grid with NaN (to track which cells have data)
    grid = np.full((n_times, n_lats, n_lons), np.nan)
    
    # Fill grid with labels
    for i, (t, lat, lon, label) in enumerate(zip(times, lats, lons, y_arr)):
        t_idx = time_to_idx[t]
        lat_idx = lat_to_idx[lat]
        lon_idx = lon_to_idx[lon]
        grid[t_idx, lat_idx, lon_idx] = label
    
    # Create a binary mask for valid data points
    valid_mask = ~np.isnan(grid)
    
    # Replace NaN with 0 for smoothing (will restore later)
    grid_for_smooth = np.where(valid_mask, grid, 0.0)
    
    # Apply temporal smoothing (along axis 0)
    if temporal_sigma > 0 and temporal_radius > 0:
        # Create temporal kernel
        t_kernel_size = 2 * temporal_radius + 1
        t_kernel = np.exp(-0.5 * (np.arange(-temporal_radius, temporal_radius + 1) / temporal_sigma) ** 2)
        t_kernel = t_kernel / t_kernel.sum()
        
        # Apply 1D convolution along time axis
        from scipy.ndimage import convolve1d
        grid_smoothed = convolve1d(grid_for_smooth, t_kernel, axis=0, mode='nearest')
    else:
        grid_smoothed = grid_for_smooth.copy()
    
    # Apply spatial smoothing (along lat/lon axes)
    if spatial_sigma > 0 and spatial_radius > 0:
        # Apply 2D Gaussian filter to each time slice
        for t_idx in range(n_times):
            grid_smoothed[t_idx] = gaussian_filter(
                grid_smoothed[t_idx], 
                sigma=spatial_sigma,
                mode='nearest',
                truncate=spatial_radius / spatial_sigma if spatial_sigma > 0 else 3.0
            )
    
    # Ensure original heatwave days (y=1) retain high values
    # Take maximum of smoothed value and original label
    grid_final = np.maximum(grid_smoothed, grid_for_smooth)
    
    # Clip to valid range
    grid_final = np.clip(grid_final, min_smooth_value, max_smooth_value)
    
    # Extract smoothed values back to original order
    y_smoothed = np.zeros_like(y_arr)
    for i, (t, lat, lon) in enumerate(zip(times, lats, lons)):
        t_idx = time_to_idx[t]
        lat_idx = lat_to_idx[lat]
        lon_idx = lon_to_idx[lon]
        y_smoothed[i] = grid_final[t_idx, lat_idx, lon_idx]
    
    if verbose:
        n_changed = np.sum((y_smoothed > min_smooth_value) & (y_smoothed < max_smooth_value))
        n_positive = np.sum(y_arr == 1)
        print(f"done (original positives: {n_positive:,}, intermediate values: {n_changed:,})")
    
    return pd.Series(y_smoothed, index=y.index, name=y.name)


def compute_smooth_weights(
    y: pd.Series,
    meta: pd.DataFrame,
    temporal_sigma: float = 2.0,
    temporal_radius: int = 5,
    spatial_sigma: float = 1.5,
    spatial_radius: int = 2,
    base_weight: float = 1.0,
    max_weight: float = 5.0,
    verbose: bool = True,
) -> np.ndarray:
    """Compute sample weights based on proximity to heatwave events.
    
    Samples near (but not on) heatwave days get higher weights, helping the
    model learn the transition zones better.
    
    Args:
        y: Binary labels (0 or 1)
        meta: DataFrame with 'latitude', 'longitude', 'time' columns
        temporal_sigma: Temporal smoothing sigma (days)
        temporal_radius: Temporal smoothing radius (days)  
        spatial_sigma: Spatial smoothing sigma (grid cells)
        spatial_radius: Spatial smoothing radius (grid cells)
        base_weight: Base weight for all samples
        max_weight: Maximum weight for transition zone samples
        verbose: Print progress
    
    Returns:
        Sample weights as numpy array
    """
    # Get smoothed proximity values
    proximity = apply_label_smoothing(
        y, meta,
        temporal_sigma=temporal_sigma,
        temporal_radius=temporal_radius,
        spatial_sigma=spatial_sigma,
        spatial_radius=spatial_radius,
        verbose=verbose,
    )
    
    # Samples with intermediate proximity get higher weights
    # Weight peaks at proximity ~0.5 (transition zone)
    transition_weight = 4 * proximity.values * (1 - proximity.values)  # peaks at 0.5
    weights = base_weight + (max_weight - base_weight) * transition_weight
    
    # Also upweight actual positive samples
    weights = np.where(y.values == 1, max_weight, weights)
    
    return weights


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # If the feature list is defined at the top level of the config (as in config.yaml),
    # propagate it into the data section so the loader actually trims columns.
    data_cfg = config.get("data", {})
    if "features" in config and "features" not in data_cfg:
        data_cfg = data_cfg.copy()
        data_cfg["features"] = config["features"]
        config["data"] = data_cfg

    return config


def _coord_slice(values: xr.DataArray, bounds: Tuple[float, float]) -> slice:
    """Build a slice that works for ascending or descending coordinate axes."""
    start, end = bounds
    first, last = float(values[0]), float(values[-1])
    if first > last:
        return slice(end, start)
    return slice(start, end)


def _subset_region(ds: xr.Dataset, lat_range: Tuple[float, float], lon_range: Tuple[float, float]) -> xr.Dataset:
    lat_slice = _coord_slice(ds.latitude, lat_range)
    lon_slice = slice(min(lon_range), max(lon_range))
    return ds.sel(latitude=lat_slice, longitude=lon_slice)


def _to_dataframe_fast(ds: xr.Dataset, target_var: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Convert a merged Dataset to X/y dataframes plus coordinate metadata.
    
    Optimized version that avoids slow xarray to_dataframe() by manually
    constructing arrays from flattened numpy data.
    """
    # Get coordinate arrays
    coords = {}
    for coord_name in ds.coords:
        if coord_name in ("latitude", "longitude", "time"):
            coords[coord_name] = ds.coords[coord_name].values
    
    # Build meshgrid for coordinates to match flattened data
    # Determine dimension order from the first data variable
    first_var = next(iter(ds.data_vars))
    dims = ds[first_var].dims
    
    # Create coordinate arrays matching data shape
    shape = ds[first_var].shape
    coord_arrays = {}
    
    for i, dim in enumerate(dims):
        if dim in ("latitude", "lat"):
            coord_arrays["latitude"] = np.broadcast_to(
                ds.coords[dim].values.reshape(
                    tuple(1 if j != i else -1 for j in range(len(dims)))
                ),
                shape
            ).ravel()
        elif dim in ("longitude", "lon"):
            coord_arrays["longitude"] = np.broadcast_to(
                ds.coords[dim].values.reshape(
                    tuple(1 if j != i else -1 for j in range(len(dims)))
                ),
                shape
            ).ravel()
        elif dim in ("time", "date"):
            coord_arrays["time"] = np.broadcast_to(
                ds.coords[dim].values.reshape(
                    tuple(1 if j != i else -1 for j in range(len(dims)))
                ),
                shape
            ).ravel()
    
    # Extract all variables as flattened arrays
    data_dict = {}
    
    for var_name in ds.data_vars:
        if ds[var_name].shape != shape:
            # Skip mismatched variables (should be fixed by alignment logic now, but safety first)
            continue
        data_dict[var_name] = ds[var_name].values.ravel()
    
    # Add coordinates
    data_dict.update(coord_arrays)
    
    # Create DataFrame directly from dict (much faster than to_dataframe)
    df = pd.DataFrame(data_dict)
    
    # Drop NaN rows
    mask = ~df.isna().any(axis=1)
    df = df.loc[mask].reset_index(drop=True)
    
    # Extract target
    y = df.pop(target_var).astype("int64")
    
    # Extract metadata
    meta_cols = [col for col in ("latitude", "longitude", "time") if col in df.columns]
    meta = df[meta_cols].copy()
    
    # Remove time from features if present
    if "time" in df.columns:
        df = df.drop(columns=["time"])
    
    return df, y, meta


def load_single_year(
    year: int,
    config: Dict,
    sample_fraction: float | None = None,
    random_seed: int = 42,
    return_metadata: bool = False,
    verbose: bool = True,
    apply_feature_engineering: bool = True,
    cache_dir: str | Path = "climate_data/cache_per_year",
) -> Tuple[pd.DataFrame, pd.Series] | Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load features and targets for one year, handling per-year Parquet caching."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = cache_dir / f"year_{year}.parquet"
    
    data_cfg = config["data"]
    target_var = data_cfg["target_var"]

    # 1. Try to load from Parquet
    loaded_from_cache = False
    if parquet_path.exists():
        if verbose:
            print(f"  Loading year {year} from cache...", end=" ", flush=True)
        
        try:
            df = pd.read_parquet(parquet_path)
            loaded_from_cache = True
        except Exception as e:
            if verbose:
                print(f"failed! (Corruption detected: {e})")
                print(f"  Deleting corrupted cache file: {parquet_path}")
            try:
                parquet_path.unlink()
            except FileNotFoundError:
                pass
            # Fall through to 'else' block (load from scratch) by ensuring loaded_from_cache is False

    if loaded_from_cache:
        # Check for missing features
        missing_fe_config = {}
        if apply_feature_engineering:
            missing_fe_config = get_missing_features_config(config, df.columns.tolist())
        
        if not missing_fe_config:
            # Everything we need is here!
            if verbose:
                 print(f"done (cache hit)")
        else:
            # We have missing features, need to compute them
            if verbose:
                print(f"found missing features, computing incremental...", end=" ", flush=True)
            
            # Load raw data to compute ONLY missing features
            feature_path = Path(data_cfg["features_dir"]) / data_cfg["feature_file_template"].format(year=year)
            
            # Use h5netcdf if available
            engine_kwargs = {}
            try:
                import h5netcdf
                engine_kwargs["engine"] = "h5netcdf"
            except ImportError:
                pass
                
            with xr.open_dataset(feature_path, **engine_kwargs) as ds:
                if "date" in ds.coords: ds = ds.rename({"date": "time"})
                lat_range = tuple(data_cfg["latitude_range"])
                lon_range = tuple(data_cfg["longitude_range"])
                ds = _subset_region(ds, lat_range, lon_range)
                ds = ds.load()
                
                # Compute ONLY missing features
                ds_new = engineer_features(
                    ds,
                    missing_fe_config,
                    time_dim="time",
                    lat_dim="latitude",
                    lon_dim="longitude",
                )
                
                # Extract new columns to DataFrame
                # Use target_var=None if _to_dataframe_fast supports it, or just use first variable
                # _to_dataframe_fast expects a target_var usually to pop it. 
                # Let's inspect _to_dataframe_fast - it pops target_var.
                # If we don't have target_var (we only have features), we need to be careful.
                # Let's verify _to_dataframe_fast. It takes ds and target_var.
                # If target_var is not in ds, it might fail? 
                
                # Workaround: _to_dataframe_fast expects target_var. We can add a dummy one.
                # Or better: construct DataFrame manually for features since we already have logic.
                
                # Actually, let's just make a temporary dummy target in ds_new
                ds_new["__dummy_target__"] = (ds_new[list(ds_new.data_vars)[0]] * 0).astype(int)
                new_df, _, new_meta = _to_dataframe_fast(ds_new, "__dummy_target__")
                
                # Identify actual new columns
                new_cols = [c for c in new_df.columns if c not in df.columns]
                
                # Robust Merge Approach:
                # 1. Add metadata columns to new_df to enable merging
                merge_on = []
                for c in new_meta.columns:
                    meta_col_name = f"__meta_{c}__"
                    new_df[meta_col_name] = new_meta[c]
                    if meta_col_name in df.columns:
                        merge_on.append(meta_col_name)
                
                if not merge_on:
                    # Fallback if metadata is missing (should not happen with valid cache)
                    raise ValueError("Cannot perform incremental update: Cache missing metadata columns.")
                
                if verbose:
                    print(f"(merging {len(new_cols)} new features on {len(merge_on)} keys)...", end=" ", flush=True)

                # 2. Merge new features into existing df
                # Use 'inner' join to intersect valid rows (e.g. dropping start of time series if lag introduced)
                # Select only new columns + merge keys from new_df to avoid duplication
                cols_to_merge = new_cols + merge_on
                df = pd.merge(df, new_df[cols_to_merge], on=merge_on, how="inner")
                
                if verbose:
                     print(f"done (rows: {len(df)}).", end=" ", flush=True)
            
            # Save updated cache
            df.to_parquet(parquet_path, index=False)
            if verbose:
                print("updated cache.")

    else:
        # 2. No cache, load from scratch
        if verbose:
            print(f"  Loading year {year} from NetCDF (creating cache)...", end=" ", flush=True)
            
        feature_path = Path(data_cfg["features_dir"]) / data_cfg["feature_file_template"].format(year=year)
        target_path = Path(data_cfg["target_dir"]) / data_cfg["target_file_template"].format(year=year)
        
        # Use engine="h5netcdf" for faster loading if available
        engine_kwargs = {}
        try:
            import h5netcdf
            engine_kwargs["engine"] = "h5netcdf"
        except ImportError:
            pass
        
        with xr.open_dataset(feature_path, **engine_kwargs) as features, \
             xr.open_dataset(target_path, **engine_kwargs) as target:
            
            if "date" in features.coords: features = features.rename({"date": "time"})
            
            # Handle 'valid_time' which might be used instead of 'date'/'time' for some variables (e.g. 2024 data)
            # If both exist, we need to align them.
            if "valid_time" in features.coords:
                if "time" in features.coords:
                    # Split variables by dimension usage
                    # Drop vars using 'time' to isolate 'valid_time' vars
                    try:
                        # Variables using valid_time
                        ds_vt = features.drop_dims("time", errors="ignore")
                        # Variables using time
                        ds_main = features.drop_dims("valid_time", errors="ignore")
                        
                        # Rename valid_time -> time in the subset
                        ds_vt = ds_vt.rename({"valid_time": "time"})
                        
                        # Merge back with intersection (inner join) to align timesteps
                        # This slices ds_vt (366 steps) to match ds_main (353 steps)
                        features = xr.merge([ds_main, ds_vt], join="inner")
                    except Exception as e:
                        if verbose:
                            print(f"Warning: failed to align valid_time with time: {e}")
                else:
                    # Only valid_time exists, just rename it
                    features = features.rename({"valid_time": "time"})
            # The following cache_params block was not present in the original document.
            # It is inserted here as per the user's instruction, with sample_fraction commented out.
            # This block seems to be intended for a different caching mechanism or a future feature.
            # For now, it's added as is, with the sample_fraction line commented as requested.
            # cache_params = {
            #     "years": sorted(years), # 'years' is not defined in this scope
            #     "lat_range": config["data"].get("latitude_range"),
            #     "lon_range": config["data"].get("longitude_range"),
            #     "features": selected_features, # 'selected_features' is not defined in this scope
            #     # "sample_fraction": ... (Removed: we always cache FULL data, then sample on load)
            #     "target_var": config["data"].get("target_var"),
            #     # "feature_engineering": ... (Removed to support incremental updates)
            #     # "label_smoothing": ... (Removed as smoothing applies after loading)
            # }
            
            # Subset
            lat_range = tuple(data_cfg["latitude_range"])
            lon_range = tuple(data_cfg["longitude_range"])
            features = _subset_region(features, lat_range, lon_range)
            target = _subset_region(target.reset_coords(drop=True), lat_range, lon_range)
            
            features = features.load()
            target = target.load()
            
            # Feature Engineering (Full)
            if apply_feature_engineering:
                features = engineer_features(features, config.get("feature_engineering", {}))
            
            # Merge
            merged = xr.merge([features, target[[target_var]]], join="inner")
            df, y_series, meta_df = _to_dataframe_fast(merged, target_var)
            
            # Add target/meta to df for saving
            df["__target__"] = y_series
            # Meta cols
            for c in meta_df.columns:
                df[f"__meta_{c}__"] = meta_df[c]
                
            # Save
            df.to_parquet(parquet_path, index=False)
            
            if verbose:
                print("done.")

    # 3. Process loaded DataFrame (filtering, sampling)
    
    # Extract special columns
    if "__target__" in df.columns:
        y = df.pop("__target__")
        # Ensure int64
        if pd.api.types.is_float_dtype(y):
             y = y.astype("int64")
    else:
        raise ValueError(f"Cache file {parquet_path} missing __target__")
        
    meta_cols = [c for c in df.columns if c.startswith("__meta_")]
    meta = df[meta_cols].copy()
    meta.columns = [c.replace("__meta_", "").replace("__", "") for c in meta_cols]
    df = df.drop(columns=meta_cols)
    
    # Filter columns to only what we expect/need
    expected_feats = get_expected_features(config)
    
    final_cols = [c for c in expected_feats if c in df.columns]
    
    if final_cols:
        df = df[final_cols]

    # Sampling
    frac = sample_fraction if sample_fraction is not None else data_cfg.get("sample_fraction", 1.0)
    if frac < 1.0:
        n_samples = int(len(df) * frac)
        rng = np.random.default_rng(random_seed)
        sample_idx = rng.choice(len(df), size=n_samples, replace=False)
        df = df.iloc[sample_idx].reset_index(drop=True)
        y = y.iloc[sample_idx].reset_index(drop=True)
        meta = meta.iloc[sample_idx].reset_index(drop=True)

    if return_metadata:
        return df, y, meta
    return df, y


def _load_year_worker(args: Tuple) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Worker function for parallel year loading."""
    year, config, sample_fraction, random_seed, verbose, apply_fe, cache_dir = args
    return load_single_year(
        year=year,
        config=config,
        sample_fraction=sample_fraction,
        random_seed=random_seed,
        return_metadata=True,
        verbose=verbose,
        apply_feature_engineering=apply_fe,
        cache_dir=cache_dir,
    )


def load_years(
    years: Iterable[int],
    config: Dict,
    return_metadata: bool = False,
    parallel: bool = False,
    max_workers: int | None = None,
    verbose: bool = True,
    apply_feature_engineering: bool = True,
    apply_label_smoothing_flag: bool = True,
    return_hard_labels: bool = False,
    cache_dir: str | Path = "climate_data/cache_per_year",
) -> Tuple[pd.DataFrame, pd.Series] | Tuple[pd.DataFrame, pd.Series, pd.DataFrame] | Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load and concatenate multiple years with per-year caching.
    
    Args:
        years: Years to load
        config: Configuration dictionary
        return_metadata: Whether to return coordinate metadata
        parallel: Use parallel loading (disabled by default - NetCDF/HDF5 not thread-safe)
        max_workers: Max parallel workers (defaults to number of years or CPU count)
        verbose: Print progress information
        apply_feature_engineering: Whether to apply configured feature engineering
        apply_label_smoothing_flag: Whether to apply label smoothing if configured
        return_hard_labels: Also return original hard labels (0/1) if smoothing is applied
        cache_dir: Directory for per-year cache files
    """
    years_list = list(years)
    sample_fraction = config["data"].get("sample_fraction", 1.0)
    base_seed = config["data"].get("random_seed", 42)
    
    if verbose:
        print(f"Loading {len(years_list)} year(s): {years_list}")
    
    # Prepare arguments for each year
    worker_args = [
        (year, config, sample_fraction, base_seed + idx, verbose, apply_feature_engineering, cache_dir)
        for idx, year in enumerate(years_list)
    ]
    
    if parallel and len(years_list) > 1:
        # WARNING: NetCDF/HDF5 libs may not be thread-safe - can cause segfaults
        # Only enable if your netCDF4/h5py is built with thread-safety
        n_workers = max_workers or min(len(years_list), 4)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_load_year_worker, worker_args))
    else:
        # Sequential loading (safe default)
        results = [_load_year_worker(args) for args in worker_args]
    
    # Unpack results
    frames = [r[0] for r in results]
    targets = [r[1] for r in results]
    metas = [r[2] for r in results]
    
    if verbose:
        print("Concatenating data...", end=" ", flush=True)
    
    # Concatenate with copy=False for efficiency
    X_all = pd.concat(frames, axis=0, copy=False).reset_index(drop=True)
    y_all = pd.concat(targets, axis=0, copy=False).reset_index(drop=True)
    meta_all = pd.concat(metas, axis=0, copy=False).reset_index(drop=True)
    
    # Keep copy of hard labels
    y_hard = y_all.copy()

    
    if verbose:
        n_features = len(X_all.columns)
        class_counts = y_all.value_counts().sort_index()
        print(f"done")
        print(f"Total: {len(X_all):,} samples, {n_features} features")
        print(f"Features: {list(X_all.columns)}")
        print(f"Class distribution: {dict(class_counts)}")
    
    # Apply label smoothing if configured
    ls_config = config.get("label_smoothing", {})
    if apply_label_smoothing_flag and ls_config.get("enabled", False):
        y_all = apply_label_smoothing(
            y=y_all,
            meta=meta_all,
            temporal_sigma=ls_config.get("temporal_sigma", 1.0),
            temporal_radius=ls_config.get("temporal_radius", 3),
            spatial_sigma=ls_config.get("spatial_sigma", 1.0),
            spatial_radius=ls_config.get("spatial_radius", 1),
            max_smooth_value=ls_config.get("max_smooth_value", 1.0),
            min_smooth_value=ls_config.get("min_smooth_value", 0.0),
            verbose=verbose,
        )
    
    if return_hard_labels:
        if return_metadata:
            return X_all, y_all, meta_all, y_hard
        return X_all, y_all, y_hard
        
    if return_metadata:
        return X_all, y_all, meta_all
    return X_all, y_all


# Removed old caching logic (load_years_cached, save_to_parquet, etc.) as requested.
    
    frames, targets, metas = [], [], []
    
    for idx, year in enumerate(years_list):
        path = preprocessed_dir / f"year_{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Preprocessed file not found: {path}. "
                f"Run preprocess_to_parquet() first."
            )
        
        X, y, meta = load_from_parquet(path, verbose=verbose)
        
        # Apply sampling if needed
        if sample_fraction < 1.0:
            n_samples = int(len(X) * sample_fraction)
            rng = np.random.default_rng(random_seed + idx)
            sample_idx = rng.choice(len(X), size=n_samples, replace=False)
            X = X.iloc[sample_idx].reset_index(drop=True)
            y = y.iloc[sample_idx].reset_index(drop=True)
            meta = meta.iloc[sample_idx].reset_index(drop=True)
        
        frames.append(X)
        targets.append(y)
        metas.append(meta)
    
    X_all = pd.concat(frames, axis=0, copy=False).reset_index(drop=True)
    y_all = pd.concat(targets, axis=0, copy=False).reset_index(drop=True)
    
    if verbose:
        n_features = len(X_all.columns)
        class_counts = y_all.value_counts().sort_index()
        print(f"Total: {len(X_all):,} samples, {n_features} features")
        print(f"Features: {list(X_all.columns)}")
        print(f"Class distribution: {dict(class_counts)}")
    
    if return_metadata:
        meta_all = pd.concat(metas, axis=0, copy=False).reset_index(drop=True)
        return X_all, y_all, meta_all
    return X_all, y_all
