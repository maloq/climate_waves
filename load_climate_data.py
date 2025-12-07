from __future__ import annotations

import os
import glob
import pathlib
import gc
import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict, Iterable, Tuple, List, Union

import hashlib
import json
import shutil
import time

# Import helpers from load_data
from load_data import _subset_region, apply_label_smoothing, load_config
from feature_engineering import engineer_features

CACHE_DIR = pathlib.Path("cache/climate_data")
CACHE_LIMIT_GB = 20

def get_config_hash(config: Dict, years: List[int]) -> str:
    """Generates a stable hash for the data configuration + specific years."""
    # Create a simplified config dictionary containing only data-relevant parts
    relevant_config = {
        "features": sorted(config.get("features", [])),
        "data": {k: v for k, v in config.get("data", {}).items() if k not in ["train_years", "test_years", "random_seed", "sample_fraction"]},
        "feature_engineering": config.get("feature_engineering", {}),
        "years": sorted(years)
    }
    
    # JSON dump with sort_keys for stability
    config_str = json.dumps(relevant_config, sort_keys=True)
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()

def manage_cache_size(limit_gb: float = CACHE_LIMIT_GB):
    """Enforces the cache directory size limit by removing oldest files."""
    if not CACHE_DIR.exists():
        return

    total_size = 0
    files = []
    
    for p in CACHE_DIR.rglob("*.parquet"):
        try:
            stat = p.stat()
            total_size += stat.st_size
            files.append((stat.st_mtime, p))
        except FileNotFoundError:
            pass
            
    limit_bytes = limit_gb * 1024**3
    
    if total_size > limit_bytes:
        print(f"[Cache] Size ({total_size / 1024**3:.2f} GB) exceeds limit ({limit_gb} GB). Cleaning up...")
        # Sort by mtime (oldest first)
        files.sort(key=lambda x: x[0])
        
        deleted_size = 0
        for _, p in files:
            try:
                size = p.stat().st_size
                p.unlink()
                deleted_size += size
                total_size -= size
                print(f"  Deleted {p.name} ({size / 1024**2:.1f} MB)")
                
                if total_size <= limit_bytes:
                    break
            except Exception as e:
                print(f"  Failed to delete {p}: {e}")
        print(f"[Cache] Cleanup complete. New size: {total_size / 1024**3:.2f} GB")

def load_zarr_dataset(
    climate_data_dir: str, 
    variables: List[str], 
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float]
) -> xr.Dataset:
    """
    Loads ALL available Zarr data lazily. 
    Uses a safe sequential open + concat approach to avoid malloc/threading crashes.
    """
    datasets = []
    
    for var in variables:
        # 1. Find files
        search_path = os.path.join(climate_data_dir, var, f"{var}_*.zarr")
        files = sorted(glob.glob(search_path))
        if not files:
            files = sorted(glob.glob(os.path.join(climate_data_dir, "**", f"{var}_*.zarr"), recursive=True))
            if not files:
                raise FileNotFoundError(f"No Zarr files found for variable: {var}")

        # 2. Open Lazy (Sequential to prevent Segfault)
        # We avoid open_mfdataset because it spawns threads that can crash underlying C-libs with Zarr
        var_chunks = []
        for f in files:
            try:
                # chunks="auto" defers loading (Lazy)
                ds = xr.open_dataset(f, engine="zarr", chunks="auto", decode_timedelta=False)
                var_chunks.append(ds)
            except Exception as e:
                print(f"Warning: Could not open {f}: {e}")

        if not var_chunks:
            raise IOError(f"Failed to open any files for {var}")

        # 3. Concatenate
        # Assuming files are split by time (standard for climate data)
        try:
            concat_dim = "valid_time" if "valid_time" in var_chunks[0].coords else "time"
            ds_var = xr.concat(var_chunks, dim=concat_dim)
            ds_var = ds_var.sortby(concat_dim)
        except Exception as e:
            # Fallback for unexpected structures
            print(f"Concatenation failed for {var}, trying merge. Error: {e}")
            ds_var = xr.merge(var_chunks)

        # 4. Standardize Time Name
        if "valid_time" in ds_var.coords:
            ds_var = ds_var.rename({"valid_time": "time"})
            
        # --- FIX: Ensure Unique Time Index ---
        # Dropping duplicates is essential if files overlap
        _, index = np.unique(ds_var['time'], return_index=True)
        ds_var = ds_var.isel(time=index)
        # -------------------------------------
        
        # 5. Spatial Slice (Lazy - reduces graph size)
        if lat_range:
            lat_name = 'latitude' if 'latitude' in ds_var.coords else 'lat'
            if ds_var[lat_name][0] > ds_var[lat_name][-1]:
                ds_var = ds_var.sel({lat_name: slice(lat_range[1], lat_range[0])})
            else:
                ds_var = ds_var.sel({lat_name: slice(lat_range[0], lat_range[1])})
        
        if lon_range:
            lon_name = 'longitude' if 'longitude' in ds_var.coords else 'lon'
            ds_var = ds_var.sel({lon_name: slice(lon_range[0], lon_range[1])})
            
        rename_dict = {}
        if 'lat' in ds_var.coords and 'latitude' not in ds_var.coords: rename_dict['lat'] = 'latitude'
        if 'lon' in ds_var.coords and 'longitude' not in ds_var.coords: rename_dict['lon'] = 'longitude'
        if rename_dict:
            ds_var = ds_var.rename(rename_dict)
            
        datasets.append(ds_var)

    # Merge all variables into one Dataset
    return xr.merge(datasets, join='inner')

def load_targets_lazy(
    years: Iterable[int],
    target_dir: str,
    file_template: str,
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float]
) -> xr.Dataset:
    files = []
    for yr in sorted(list(years)):
        fpath = pathlib.Path(target_dir) / file_template.format(year=yr)
        if fpath.exists():
            files.append(str(fpath))
    
    if not files:
        raise FileNotFoundError("No target files found.")

    # Disable parallel=True here too just to be safe
    ds = xr.open_mfdataset(files, combine="by_coords", chunks="auto", parallel=False)
    
    if "date" in ds.coords: ds = ds.rename({"date": "time"})
    
    # --- FIX: Ensure Unique Target Time Index ---
    _, index = np.unique(ds['time'], return_index=True)
    ds = ds.isel(time=index)
    # --------------------------------------------

    ds = _subset_region(ds, lat_range, lon_range)
    
    rename_dict = {}
    if 'lat' in ds.coords and 'latitude' not in ds.coords: rename_dict['lat'] = 'latitude'
    if 'lon' in ds.coords and 'longitude' not in ds.coords: rename_dict['lon'] = 'longitude'
    if rename_dict:
        ds = ds.rename(rename_dict)
        
    return ds

def load_years(
    years: Iterable[int],
    config: Dict,
    return_metadata: bool = False,
    verbose: bool = True,
    apply_feature_engineering: bool = True,
    return_hard_labels: bool = False,
    apply_label_smoothing_flag: bool = True,
    use_cache: bool = True,
    **kwargs
) -> Union[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series, pd.DataFrame], Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
    
    start_time_glob = pd.Timestamp.now()
    years = sorted(list(years))
    data_cfg = config["data"]
    
    sample_fraction = kwargs.get("sample_fraction", data_cfg.get("sample_fraction", 1.0))
    random_seed = kwargs.get("random_seed", data_cfg.get("random_seed", 42))
    
    # --- Caching Setup ---
    cache_subdir = None
    if use_cache:
        # Hash logic must be very robust. We hash the config relevant to data processing + list of years.
        # Actually, caching per year is better, so we can reuse years across experiments.
        # So the hash should depend on config but NOT the list of years requested (we'll look up year by year).
        config_hash = get_config_hash(config, []) # Empty list for base config hash
        cache_subdir = CACHE_DIR / config_hash
        cache_subdir.mkdir(parents=True, exist_ok=True)
        if verbose: print(f"Cache Directory: {cache_subdir}")
    # ---------------------

    # 1. Calc Buffer
    fe_config = config.get("feature_engineering", {})
    fe_enabled = fe_config.get("enabled", True) and apply_feature_engineering
    buffer_days = 0
    if fe_enabled:
        lookbacks = [0]
        if "lag" in fe_config: lookbacks.extend(fe_config["lag"].get("lags", []))
        if "ewm" in fe_config: lookbacks.append(max(fe_config["ewm"].get("spans", [0])) * 3)
        if "temporal_diff" in fe_config: lookbacks.extend(fe_config["temporal_diff"].get("periods", []))
        buffer_days = max(lookbacks) + 5

    lat_range = tuple(data_cfg["latitude_range"])
    lon_range = tuple(data_cfg["longitude_range"])
    variables = config["features"]

    # Check what needs to be computed vs loaded
    years_to_compute = []
    processed_dfs = []
    
    # 2. Try Loading from Cache
    if use_cache:
        for yr in years:
            cache_file = cache_subdir / f"{yr}.parquet"
            if cache_file.exists():
                if verbose: print(f"  [Cache] Loading {yr}...", end=" ", flush=True)
                try:
                     # Touch the file to update mtime for LRU eviction
                    cache_file.touch()
                    df_year = pd.read_parquet(cache_file)
                    
                    if sample_fraction < 1.0 and not df_year.empty:
                        df_year = df_year.sample(frac=sample_fraction, random_state=random_seed)
                    
                    processed_dfs.append(df_year)
                    if verbose: print(f"done. ({len(df_year):,} rows)")
                except Exception as e:
                    print(f"Error loading cache for {yr}: {e}. Will recompute.")
                    years_to_compute.append(yr)
            else:
                years_to_compute.append(yr)
    else:
        years_to_compute = years

    # 3. Open Global Datasets (Lazy) - ONLY if we have years to compute
    if years_to_compute:
        if verbose:
            print(f"Loading Raw Data for {years_to_compute} (Buffer: {buffer_days}d)")

        ds_features_global = load_zarr_dataset(
            data_cfg["features_dir"], variables, lat_range, lon_range
        )
        ds_targets_global = load_targets_lazy(
            years_to_compute, data_cfg["target_dir"], data_cfg["target_file_template"], lat_range, lon_range
        )

        manage_cache_size() # Pre-emptive clean

        # 4. Stream & Process Batch-by-Batch
        for yr in years_to_compute:
            if verbose: print(f"  Processing {yr}...", end=" ", flush=True)
            
            t_start = pd.Timestamp(f"{yr}-01-01")
            t_end = pd.Timestamp(f"{yr}-12-31")
            
            feat_start = t_start - pd.Timedelta(days=buffer_days)
            feat_end = t_end 
            
            # A. Slice (Lazy)
            try:
                ds_chunk_feat = ds_features_global.sel(time=slice(feat_start, feat_end))
                ds_chunk_target = ds_targets_global.sel(time=slice(t_start, t_end))
            except KeyError:
                print(f"Warning: Data missing for {yr}, skipping.")
                continue

            if ds_chunk_target.time.size == 0:
                print(f"No target data for {yr}, skipping.")
                continue

            # B. Feature Engineering (Lazy)
            if fe_enabled:
                ds_chunk_feat = engineer_features(ds_chunk_feat, fe_config)
                
            # C. Align
            # Select common times
            ds_chunk_feat = ds_chunk_feat.sel(time=ds_chunk_target.time)
            
            # Reindex features to target grid (Spatial align)
            ds_chunk_feat = ds_chunk_feat.reindex(
                latitude=ds_chunk_target.latitude, 
                longitude=ds_chunk_target.longitude, 
                method='nearest', tolerance=0.01
            )
            
            # Merge
            ds_merged = xr.merge([ds_chunk_feat, ds_chunk_target], join='inner')

            # --- OPTIMIZATION: Cast to float32 to save space/memory ---
            for var in ds_merged.data_vars:
                if ds_merged[var].dtype == 'float64':
                    ds_merged[var] = ds_merged[var].astype('float32')
            # ----------------------------------------------------------
            
            # D. MATERIALIZE
            try:
                df_year = ds_merged.to_dataframe().reset_index().dropna()
            except MemoryError:
                 print("Memory Warning! Chunking compute...")
                 df_year = ds_merged.compute().to_dataframe().reset_index().dropna()
            
            # E. Save to Cache (BEFORE subsampling)
            if use_cache and not df_year.empty:
                 cache_file = cache_subdir / f"{yr}.parquet"
                 try:
                     # Check size again before write
                     manage_cache_size() 
                     df_year.to_parquet(cache_file, index=False, compression='snappy')
                 except Exception as e:
                     print(f"Warning: Could not save to cache: {e}")

            # F. Subsample In-Memory (Fast)
            if sample_fraction < 1.0 and not df_year.empty:
                df_year = df_year.sample(frac=sample_fraction, random_state=random_seed)
                
            processed_dfs.append(df_year)
            
            del ds_merged, ds_chunk_feat, ds_chunk_target, df_year
            gc.collect()
            
            if verbose: print(f"done. ({len(processed_dfs[-1]):,} rows)")

    # 5. Final Concat
    if not processed_dfs:
        raise ValueError("No data loaded for any year!")
        
    if verbose: print("  Concatenating years...", end=" ", flush=True)
    df_final = pd.concat(processed_dfs, axis=0, ignore_index=True)
    if verbose: print(f"done. Total shape: {df_final.shape}")
    
    # 6. Extract X, y
    target_var = data_cfg["target_var"]
    y = df_final[target_var].copy()
    meta = df_final[["time", "latitude", "longitude"]].copy()
    
    # Only drop known metadata/targets, keep all other columns (including engineered features)
    drop_cols = [target_var, "time", "latitude", "longitude", "valid_time"]
    
    X = df_final.drop(columns=[c for c in drop_cols if c in df_final.columns])

    # 7. Label Smoothing
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

    if return_hard_labels and return_metadata:
        return X, y, meta, y_hard
    elif return_hard_labels:
        return X, y, y_hard
    elif return_metadata:
        return X, y, meta
    else:
        return X, y