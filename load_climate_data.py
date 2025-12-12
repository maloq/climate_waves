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

# Thread safety: Use synchronous scheduler to avoid heap corruption from concurrent Blosc/Zarr decompression
import dask
dask.config.set(scheduler='synchronous')

# Import helpers from load_data
from load_data import _subset_region, apply_label_smoothing, load_config
from feature_engineering import engineer_features

CACHE_DIR = pathlib.Path("cache/climate_data")
CACHE_LIMIT_GB = 130
CACHE_FORMAT_VERSION = 2


def _atomic_write_parquet(df: pd.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    df.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)


def _month_cache_path(cache_subdir: pathlib.Path, year: int, month: int) -> pathlib.Path:
    return cache_subdir / "monthly" / str(year) / f"{month:02d}.parquet"


def _dataset_to_dataframe_fast(
    ds: xr.Dataset,
    *,
    target_var: str,
    time_dim: str = "time",
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
) -> pd.DataFrame:
    """
    Faster alternative to `ds.to_dataframe().reset_index()` for common (time, lat, lon) grids.

    - Avoids MultiIndex creation
    - Avoids global dropna() (we only drop rows with missing target)
    """
    for dim in (time_dim, lat_dim, lon_dim):
        if dim not in ds.dims:
            raise KeyError(f"Expected dim '{dim}' in dataset dims {tuple(ds.dims)}")
    if target_var not in ds.data_vars:
        raise KeyError(f"Target var '{target_var}' not found in dataset vars {list(ds.data_vars)}")

    time_vals = ds[time_dim].values
    lat_vals = ds[lat_dim].values
    lon_vals = ds[lon_dim].values

    n_time = time_vals.size
    n_lat = lat_vals.size
    n_lon = lon_vals.size

    n_grid = n_lat * n_lon
    n_rows = n_time * n_grid

    # Coordinate columns (match C-order flattening of (time, lat, lon))
    time_col = np.repeat(time_vals, n_grid)
    lat_col = np.tile(np.repeat(lat_vals, n_lon), n_time)
    lon_col = np.tile(lon_vals, n_time * n_lat)

    # Build mask: target must be finite (labels cannot be NaN)
    target_da = ds[target_var]
    if target_da.dims != (time_dim, lat_dim, lon_dim):
        target_da = target_da.transpose(time_dim, lat_dim, lon_dim)
    y_flat = np.asarray(target_da.values).reshape(n_rows)
    keep = np.isfinite(y_flat)

    cols: dict[str, np.ndarray] = {
        "time": time_col[keep],
        "latitude": lat_col[keep],
        "longitude": lon_col[keep],
        target_var: y_flat[keep].astype(np.float32, copy=False),
    }

    for var in ds.data_vars:
        if var == target_var:
            continue
        da = ds[var]
        # Only support scalar fields on the same grid
        if da.ndim != 3 or time_dim not in da.dims or lat_dim not in da.dims or lon_dim not in da.dims:
            # Skip unexpected shapes (keeps behavior robust; downstream will warn on missing vars if needed)
            continue
        if da.dims != (time_dim, lat_dim, lon_dim):
            da = da.transpose(time_dim, lat_dim, lon_dim)
        flat = np.asarray(da.values).reshape(n_rows)
        cols[var] = flat[keep].astype(np.float32, copy=False) if flat.dtype.kind == "f" else flat[keep]

    return pd.DataFrame(cols)

def get_config_hash(config: Dict, years: List[int], runtime: Dict | None = None) -> str:
    """Generates a stable hash for the data configuration + specific years."""
    # Create a simplified config dictionary containing only data-relevant parts
    relevant_config = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "features": sorted(config.get("features", [])),
        # Include sample_fraction/random_seed/subsampling so caches are invalidated when they change.
        "data": {k: v for k, v in config.get("data", {}).items() if k not in ["train_years", "test_years"]},
        "feature_engineering": config.get("feature_engineering", {}),
        "label_smoothing": config.get("label_smoothing", {}),
        "runtime": runtime or {},
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

def load_netcdf_features_lazy(
    years: Iterable[int],
    feature_dir: str,
    file_template: str,
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    variables: List[str]
) -> xr.Dataset:
    """
    Loads features from NetCDF files lazily (Legacy support).
    """
    files = []
    # Load requested years
    unique_years = sorted(list(set(years)))
    
    for yr in unique_years:
        fpath = pathlib.Path(feature_dir) / file_template.format(year=yr)
        if fpath.exists():
            files.append(str(fpath))
    
    if not files:
        print(f"Warning: No NetCDF files found for years {unique_years} at {feature_dir}")
        return xr.Dataset()

    # Open MFDataset
    # Parallel=False to avoid threading issues inside agents or limited environments
    # Use nested combine to avoid strict coordinate equality checks across files (floating point jitter)
    ds = xr.open_mfdataset(
        files, 
        combine="nested", 
        concat_dim="date", # We know files are split by year and use 'date'
        chunks="auto", 
        parallel=False
    )
    
    if "date" in ds.coords: ds = ds.rename({"date": "time"})
    
    # Filter variables that actually exist in the dataset
    available_vars = [v for v in variables if v in ds.data_vars]
    if available_vars:
        ds = ds[available_vars]
    else:
        print(f"Warning: None of requested variables {variables} found in NetCDF files.")
        return xr.Dataset()
        
    ds = _subset_region(ds, lat_range, lon_range)
    
    rename_dict = {}
    if 'lat' in ds.coords and 'latitude' not in ds.coords: rename_dict['lat'] = 'latitude'
    if 'lon' in ds.coords and 'longitude' not in ds.coords: rename_dict['lon'] = 'longitude'
    if rename_dict:
        ds = ds.rename(rename_dict)
        
    return ds

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

    # Check if files are zarr or netcdf
    is_zarr = any(f.endswith('.zarr') for f in files)
    
    if is_zarr:
        # Open zarr files individually and concatenate
        datasets = []
        for f in files:
            ds_single = xr.open_zarr(f, chunks="auto")
            
            # Standardize time coordinate name BEFORE concatenation
            if "date" in ds_single.coords:
                ds_single = ds_single.rename({"date": "time"})
            if "valid_time" in ds_single.coords:
                ds_single = ds_single.rename({"valid_time": "time"})
            
            # Ensure 'time' is a dimension, not just a coordinate
            # This is crucial for zarr files where time might be stored differently
            if 'time' in ds_single.coords and 'time' not in ds_single.dims:
                # Time is a coordinate but not a dimension - need to expand
                ds_single = ds_single.expand_dims('time')
            
            datasets.append(ds_single)
        
        # Use combine_by_coords which handles overlapping/sequential time coordinates better
        ds = xr.combine_by_coords(datasets, combine_attrs="override")
    else:
        # Use open_mfdataset for netcdf files
        ds = xr.open_mfdataset(files, combine="by_coords", chunks="auto", parallel=False)
    
    if "date" in ds.coords: ds = ds.rename({"date": "time"})
    if "valid_time" in ds.coords: ds = ds.rename({"valid_time": "time"})
    
    # --- FIX: Ensure Unique Target Time Index ---
    if 'time' in ds.dims and ds['time'].size > 0:
        _, index = np.unique(ds['time'].values, return_index=True)
        index = np.sort(index)  # Preserve time ordering
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
    target_var = data_cfg["target_var"]
    subsampling_cfg = data_cfg.get("subsampling", {}) or {}
    subsampling_mode = str(subsampling_cfg.get("mode", "")).strip().lower()
    tail_preserving_enabled = subsampling_mode in ("tail_preserving", "tail-preserving", "tail_preserve", "tail-preserve")
    tail_positive_threshold = subsampling_cfg.get("positive_threshold", None)
    tail_near_miss_threshold = subsampling_cfg.get("near_miss_threshold", None)

    # Check if label smoothing is enabled
    ls_config = config.get("label_smoothing", {})
    ls_enabled = ls_config.get("enabled", False) and apply_label_smoothing_flag

    def _apply_chunk_smoothing(df_chunk: pd.DataFrame) -> pd.DataFrame:
        if verbose: print(f"    [Smoothing] Applying to {len(df_chunk):,} samples...", end=" ", flush=True)
        
        # Preserve hard labels
        if f"{target_var}_hard" not in df_chunk.columns:
            df_chunk[f"{target_var}_hard"] = df_chunk[target_var]
            
        y_smooth = apply_label_smoothing(
             df_chunk[target_var],
             df_chunk[["time", "latitude", "longitude"]],
             temporal_sigma=ls_config.get("temporal_sigma", 1.0),
             temporal_radius=ls_config.get("temporal_radius", 3),
             spatial_sigma=ls_config.get("spatial_sigma", 1.0),
             spatial_radius=ls_config.get("spatial_radius", 1),
             max_smooth_value=ls_config.get("max_smooth_value", 1.0),
             min_smooth_value=ls_config.get("min_smooth_value", 0.0),
             verbose=False 
        )
        df_chunk[target_var] = y_smooth
        if verbose: print("done.")
        return df_chunk

    def _tail_preserving_subsample_chunk(
        df_chunk: pd.DataFrame,
        *,
        year: int,
        month: int,
    ) -> pd.DataFrame:
        """
        Tail-preserving subsampling:
        - Keep all positives (>= positive_threshold)
        - Optionally keep all near-misses (>= near_miss_threshold)
        - Subsample remaining negatives to reach approx sample_fraction
        """
        nonlocal tail_positive_threshold, tail_near_miss_threshold

        if not tail_preserving_enabled:
            return df_chunk
        if tail_positive_threshold is None:
            raise ValueError(
                "data.subsampling.positive_threshold must be set when tail-preserving subsampling is enabled."
            )
        if target_var not in df_chunk.columns:
            raise KeyError(f"Target column '{target_var}' missing from chunk; cannot tail-preserve.")

        n_total = len(df_chunk)
        if n_total == 0:
            return df_chunk

        n_target = int(n_total * sample_fraction)
        n_target = max(n_target, 1)

        y = df_chunk[target_var]
        keep_mask = y >= float(tail_positive_threshold)
        if tail_near_miss_threshold is not None:
            keep_mask = keep_mask | (y >= float(tail_near_miss_threshold))

        df_keep = df_chunk.loc[keep_mask]
        df_rest = df_chunk.loc[~keep_mask]

        n_keep = len(df_keep)
        n_needed = max(n_target - n_keep, 0)

        # Deterministic per (year, month) seed
        chunk_seed = int(random_seed) + (int(year) * 100) + int(month)

        if n_needed <= 0 or df_rest.empty:
            out = df_keep if n_keep > 0 else df_chunk.sample(n=n_target, random_state=chunk_seed)
        else:
            n_sample = min(n_needed, len(df_rest))
            df_sampled = df_rest.sample(n=n_sample, random_state=chunk_seed)
            out = pd.concat([df_keep, df_sampled], axis=0, ignore_index=True)

        # Keep a deterministic stable order for temporal splits/inspection
        sort_cols = [c for c in ["time", "latitude", "longitude"] if c in out.columns]
        if sort_cols:
            out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
        else:
            out = out.reset_index(drop=True)
        return out
    
    # 1. Calc Buffer / FE enabled
    fe_config = config.get("feature_engineering", {})
    fe_enabled = fe_config.get("enabled", True) and apply_feature_engineering
    buffer_days = 0
    if fe_enabled:
        lookbacks = [0]
        if "lag" in fe_config: lookbacks.extend(fe_config["lag"].get("lags", []))
        if "ewm" in fe_config: lookbacks.append(max(fe_config["ewm"].get("spans", [0])) * 3)
        if "temporal_diff" in fe_config: lookbacks.extend(fe_config["temporal_diff"].get("periods", []))
        buffer_days = max(lookbacks) + 5

    # --- Caching Setup ---
    cache_subdir = None
    if use_cache:
        # Hash logic must be very robust. We hash the config relevant to data processing.
        # The hash should depend on config but NOT the list of years requested (year/month are part of cache path).
        config_hash = get_config_hash(
            config,
            [],
            runtime={
                "apply_feature_engineering": bool(fe_enabled),
                "apply_label_smoothing": bool(ls_enabled),
                "effective_sample_fraction": float(sample_fraction),
                "effective_random_seed": int(random_seed),
            },
        )
        cache_subdir = CACHE_DIR / config_hash
        cache_subdir.mkdir(parents=True, exist_ok=True)
        if verbose: print(f"Cache Directory: {cache_subdir}")
    # ---------------------

    lat_range = tuple(data_cfg["latitude_range"])
    lon_range = tuple(data_cfg["longitude_range"])
    variables = config["features"]

    # Check what needs to be computed vs loaded
    years_to_compute = []
    processed_dfs = []
    
    # 2. Try Loading from Cache
    year_dfs_map: dict[int, list[pd.DataFrame]] = {int(yr): [] for yr in years}
    months_needed_map: dict[int, set[int]] = {}

    if use_cache:
        for yr in years:
            yr = int(yr)
            loaded_months = 0
            missing_months: set[int] = set()

            for m in range(1, 13):
                cache_file = _month_cache_path(cache_subdir, yr, m)
                if cache_file.exists():
                    if verbose:
                        print(f"  [Cache] Loading {yr}-{m:02d}...", end=" ", flush=True)
                    try:
                        cache_file.touch()
                        df_month = pd.read_parquet(cache_file)
                        year_dfs_map[yr].append(df_month)
                        loaded_months += 1
                        if verbose:
                            print(f"done. ({len(df_month):,} rows)")
                    except Exception as e:
                        print(f"Error loading monthly cache for {yr}-{m:02d}: {e}. Will recompute month.")
                        missing_months.add(m)
                else:
                    missing_months.add(m)

            # If we found no monthly cache at all, fall back to legacy yearly cache (if it exists)
            if loaded_months == 0:
                legacy_cache_file = cache_subdir / f"{yr}.parquet"
                if legacy_cache_file.exists():
                    if verbose:
                        print(f"  [Cache] Loading legacy {yr}...", end=" ", flush=True)
                    try:
                        legacy_cache_file.touch()
                        df_year = pd.read_parquet(legacy_cache_file)
                        if ls_enabled and not df_year.empty:
                            try:
                                df_year = _apply_chunk_smoothing(df_year)
                            except Exception as e:
                                print(f"Warning: Label smoothing failed for cached year {yr}: {e}")
                        if sample_fraction < 1.0 and not df_year.empty:
                            if tail_preserving_enabled:
                                # Legacy cache doesn't preserve tails; force recompute instead of sampling blindly
                                raise RuntimeError("Legacy yearly cache is incompatible with tail-preserving sampling.")
                            df_year = df_year.sample(frac=sample_fraction, random_state=random_seed)
                        year_dfs_map[yr].append(df_year)
                        if verbose:
                            print(f"done. ({len(df_year):,} rows)")
                        continue
                    except Exception as e:
                        print(f"Error loading legacy cache for {yr}: {e}. Will recompute.")

                months_needed_map[yr] = set(range(1, 13))
                years_to_compute.append(yr)
                continue

            if missing_months:
                months_needed_map[yr] = missing_months
                years_to_compute.append(yr)
    else:
        years_to_compute = list(map(int, years))
        months_needed_map = {int(yr): set(range(1, 13)) for yr in years_to_compute}

    # 3. Open Global Datasets (Lazy) - ONLY if we have years to compute
    if years_to_compute:
        if verbose:
            print(f"Loading Raw Data for {years_to_compute} (Buffer: {buffer_days}d)")

        # Split variables into Zarr vs NetCDF
        zarr_vars = []
        netcdf_vars = []
        
        for var in variables:
            # Check Zarr
            search_path = os.path.join(data_cfg["features_dir"], var, f"{var}_*.zarr")
            if glob.glob(search_path) or glob.glob(os.path.join(data_cfg["features_dir"], "**", f"{var}_*.zarr"), recursive=True):
                zarr_vars.append(var)
            else:
                netcdf_vars.append(var)
        
        if verbose:
            print(f"  Zarr variables: {len(zarr_vars)}")
            print(f"  NetCDF variables: {len(netcdf_vars)}")

        ds_list = []
        
        # Load Zarr
        if zarr_vars:
            ds_zarr = load_zarr_dataset(
                data_cfg["features_dir"], zarr_vars, lat_range, lon_range
            )
            ds_list.append(ds_zarr)
            
        # Load NetCDF
        if netcdf_vars:
            unique_load_years = set(years_to_compute)
            if buffer_days > 0:
                min_yr = min(years_to_compute)
                unique_load_years.add(min_yr - 1)
            
            ds_nc = load_netcdf_features_lazy(
                list(unique_load_years), 
                "climate_data/features", 
                "features_{year}.nc",
                lat_range, lon_range, 
                netcdf_vars
            )
            if ds_nc.nbytes > 0 or len(ds_nc) > 0:
                 ds_list.append(ds_nc)

        if not ds_list:
             raise ValueError("Failed to load any features (Zarr or NetCDF).")

        # Merge global features
        if len(ds_list) > 1:
            ds_features_global = xr.merge(ds_list, join='inner')
        else:
            ds_features_global = ds_list[0]

        ds_targets_global = load_targets_lazy(
            years_to_compute, data_cfg["target_dir"], data_cfg["target_file_template"], lat_range, lon_range
        )

        # If the expected target variable name is missing, fall back to the single available one.
        # This guards against mismatches between the config `target_var` and the actual variable
        # stored in the Zarr/NetCDF files (e.g. config expects `target_wind_gust_p95` but files
        # contain `target_gust_p95`).
        if target_var not in ds_targets_global.data_vars:
            target_candidates = list(ds_targets_global.data_vars)
            if len(target_candidates) == 1:
                alt_var = target_candidates[0]
                if verbose:
                    print(f"  [Target] '{target_var}' not found, using '{alt_var}' instead.")
                ds_targets_global = ds_targets_global.rename({alt_var: target_var})
            else:
                raise KeyError(
                    f"Target variable '{target_var}' not found in loaded target data. "
                    f"Available variables: {target_candidates}"
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

            # B. Feature Engineering (Lazy -> Eager to fix Dask issues)
            if fe_enabled:
                if verbose: print("    [Compute] Materializing chunk for feature engineering...", end=" ", flush=True)
                ds_chunk_feat = ds_chunk_feat.compute()
                if verbose: print("done.")
                ds_chunk_feat = engineer_features(ds_chunk_feat, fe_config)
                
            # C. Align
            # Select common times (Nearest match to handle offsets like 09:00 vs 00:00)
            try:
                # FAST PATH: Standard selection (Works if all target times have matching features within tolerance)
                ds_chunk_feat_aligned = ds_chunk_feat.sel(time=ds_chunk_target.time, method='nearest', tolerance="12h")
                ds_chunk_feat = ds_chunk_feat_aligned
            except KeyError:
                # SLOW PATH (Fallback): Missing features for some target times (e.g. 2024 gap)
                # Use reindex to create NaNs, then drop them
                if verbose: print(f"  [Warning] Missing features for some timestamps in {yr}. Dropping incomplete samples...", end=" ")
                ds_chunk_feat = ds_chunk_feat.reindex(time=ds_chunk_target.time, method='nearest', tolerance="12h")
                
                # Drop times where ALL features/spatial points are NaN (i.e. the ones inserted by reindex)
                # Note: This assumes valid time steps have at least SOME non-NaN data.
                ds_chunk_feat = ds_chunk_feat.dropna(dim='time', how='all')
                
                # Sync target to match remaining feature times
                ds_chunk_target = ds_chunk_target.sel(time=ds_chunk_feat.time)
                
                if verbose: print(f"Retained {ds_chunk_target.time.size} samples.")
            ds_chunk_feat['time'] = ds_chunk_target.time
            
            # Reindex features to target grid (Spatial align)
            ds_chunk_feat = ds_chunk_feat.reindex(
                latitude=ds_chunk_target.latitude, 
                longitude=ds_chunk_target.longitude, 
                method='nearest', tolerance=0.6
            )
            
            # Merge
            ds_merged = xr.merge([ds_chunk_feat, ds_chunk_target], join='inner')

            # --- OPTIMIZATION: Cast to float32 to save space/memory ---
            for var in ds_merged.data_vars:
                if ds_merged[var].dtype == 'float64':
                    ds_merged[var] = ds_merged[var].astype('float32')
            # ----------------------------------------------------------
            
            # D. MATERIALIZE & PROCESS IN CHUNKS (Memory Optimization)
            # Instead of converting the whole year to DataFrame at once (which spikes RAM),
            # we iterate by month (or chunks), convert, subsample, and accumulate.
            
            year_dfs = []
            sampling_stats = {
                "total_before": 0,
                "pos_before": 0,
                "total_after": 0,
                "pos_after": 0,
                "kept_pos": 0,
                "kept_near": 0,
                "sampled_other": 0,
            }
            months_needed = months_needed_map.get(int(yr), set(range(1, 13)))
            
            # Group by month usually works well for climate data
            # Use 'time.month' for grouping or just a simple split
            # Since data is likely sorted by time, we can just split by month index if efficient, 
            # but standard selection is safer.
            
            months = np.unique(ds_merged['time'].dt.month)
            present_months = {int(x) for x in months.tolist()}
            if use_cache:
                # Write empty sentinels for months with no data so we don't attempt recomputation every run.
                for missing_m in sorted(months_needed - present_months):
                    cache_file = _month_cache_path(cache_subdir, int(yr), int(missing_m))
                    try:
                        _atomic_write_parquet(
                            pd.DataFrame(columns=["time", "latitude", "longitude", target_var]),
                            cache_file,
                        )
                    except Exception as e:
                        print(f"  [Cache] Failed to write empty monthly cache {cache_file}: {e}")
            for m in months:
                m = int(m)
                if m not in months_needed:
                    continue
                ds_month = ds_merged.sel(time=ds_merged['time'].dt.month == m)
                
                if ds_month.time.size == 0: continue
                
                try:
                    df_chunk = _dataset_to_dataframe_fast(ds_month, target_var=target_var)
                except MemoryError:
                    print(f"    [Memory] Month {m} too large, trying compute()...")
                    df_chunk = _dataset_to_dataframe_fast(ds_month.compute(), target_var=target_var)
                
                # F. Label Smoothing (Per chunk)
                if ls_enabled and not df_chunk.empty:
                     df_chunk = _apply_chunk_smoothing(df_chunk)
                     
                # G. Subsample (Per chunk)
                if sample_fraction < 1.0 and not df_chunk.empty:
                    if tail_preserving_enabled:
                        # Track before/after class balance for this chunk
                        sampling_stats["total_before"] += len(df_chunk)
                        pos_before = int((df_chunk[target_var] >= float(tail_positive_threshold)).sum())
                        sampling_stats["pos_before"] += pos_before

                        if tail_near_miss_threshold is not None:
                            keep_near = (df_chunk[target_var] >= float(tail_near_miss_threshold)) & (
                                df_chunk[target_var] < float(tail_positive_threshold)
                            )
                            sampling_stats["kept_near"] += int(keep_near.sum())

                        keep_pos = df_chunk[target_var] >= float(tail_positive_threshold)
                        sampling_stats["kept_pos"] += int(keep_pos.sum())

                        df_chunk = _tail_preserving_subsample_chunk(df_chunk, year=int(yr), month=int(m))

                        sampling_stats["total_after"] += len(df_chunk)
                        sampling_stats["pos_after"] += int((df_chunk[target_var] >= float(tail_positive_threshold)).sum())
                        # sampled_other is "everything that's not kept_pos/kept_near" in the final set
                        sampling_stats["sampled_other"] = sampling_stats["total_after"] - sampling_stats["kept_pos"] - sampling_stats["kept_near"]
                    else:
                        df_chunk = df_chunk.sample(frac=sample_fraction, random_state=random_seed)
                    
                if not df_chunk.empty:
                    year_dfs.append(df_chunk)
                    if use_cache:
                        cache_file = _month_cache_path(cache_subdir, int(yr), int(m))
                        try:
                            _atomic_write_parquet(df_chunk, cache_file)
                        except Exception as e:
                            print(f"  [Cache] Failed to write monthly cache {cache_file}: {e}")
                
                del ds_month, df_chunk
                
            if year_dfs:
                year_dfs_map[int(yr)].extend(year_dfs)

            del ds_merged, ds_chunk_feat, ds_chunk_target
            gc.collect()
            
            if verbose:
                added_rows = sum(len(d) for d in year_dfs) if year_dfs else 0
                print(f"done. (+{added_rows:,} rows)")

            if verbose and sample_fraction < 1.0 and tail_preserving_enabled:
                tb = sampling_stats["total_before"]
                ta = sampling_stats["total_after"]
                pb = sampling_stats["pos_before"]
                pa = sampling_stats["pos_after"]
                if tb > 0 and ta > 0:
                    print(
                        "  [Subsampling:tail_preserving] "
                        f"year={yr} threshold>={tail_positive_threshold} "
                        f"before: n={tb:,} pos={pb:,} ({100.0*pb/tb:.3f}%) | "
                        f"after: n={ta:,} pos={pa:,} ({100.0*pa/ta:.3f}%) | "
                        f"kept_pos={sampling_stats['kept_pos']:,} kept_near={sampling_stats['kept_near']:,} "
                        f"sampled_other={sampling_stats['sampled_other']:,}"
                    )

    # 5. Final Concat
    all_parts: list[pd.DataFrame] = []
    for yr in years:
        all_parts.extend(year_dfs_map[int(yr)])
    if not all_parts:
        raise ValueError("No data loaded for any year!")

    if verbose: print("  Concatenating years...", end=" ", flush=True)
    df_final = pd.concat(all_parts, axis=0, ignore_index=True)
    if verbose: print(f"done. Total shape: {df_final.shape}")
    
    # 6. Extract X, y
    target_var = data_cfg["target_var"]
    if target_var not in df_final.columns:
        target_candidates = [c for c in df_final.columns if "target" in c]
        raise KeyError(
            f"Target column '{target_var}' missing after processing. "
            f"Available target-like columns: {target_candidates}"
        )

    y = df_final[target_var].copy()
    meta = df_final[["time", "latitude", "longitude"]].copy()
    
    # Only drop known metadata/targets, keep all other columns (including engineered features)
    drop_cols = [target_var, "time", "latitude", "longitude", "valid_time"]
    
    X = df_final.drop(columns=[c for c in drop_cols if c in df_final.columns])

    # 7. Extract Labels (Handle Soft vs Hard)
    # y is now potentially smoothed (if ls_enabled was True)
    # y_hard should be recovered from the extra column if it exists
    
    if f"{target_var}_hard" in df_final.columns:
        y_hard = df_final[f"{target_var}_hard"].copy()
    else:
        y_hard = y.copy() # No smoothing or only hard labels available

    if return_hard_labels and return_metadata:
        return X, y, meta, y_hard
    elif return_hard_labels:
        return X, y, y_hard
    elif return_metadata:
        return X, y, meta
    else:
        return X, y
