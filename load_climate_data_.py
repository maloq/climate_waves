import pandas as pd
import os
import numpy as np
import xarray as xr
import time as time_lib
from concurrent.futures import ProcessPoolExecutor
import polars as pl
from tqdm import tqdm
import pathlib
import glob
from pandas import to_datetime
import matplotlib.pyplot as plt
import os, re


def find_climate_files(climate_data_dir: str, variable: str):
    """
    Return (file_list, variable_dir) for *variable*.
    New logic: if <climate_data_dir>/<variable> is missing, look one level
    deeper and search recursively.
    """
    import glob, os, pathlib

    # ―― primary location: <dir>/<variable>/variable_*.zarr ――――――――――――――――――
    variable_dir = os.path.join(climate_data_dir, variable)
    pattern      = os.path.join(variable_dir, f"{variable}_*.zarr")
    files        = glob.glob(pattern)

    # ―― fallback: search one level deeper (e.g. …/ECMWF/<variable>) ――――――
    if not files:
        pattern = os.path.join(climate_data_dir, "**", f"{variable}_*.zarr")
        files   = glob.glob(pattern, recursive=True)
        if files:
            # derive the “variable_dir” from the first hit (for logging only)
            variable_dir = str(pathlib.Path(files[0]).parent)

    if not files:
        raise FileNotFoundError(
            f"No NetCDFs named '{variable}_*.zarr' under {climate_data_dir}"
        )
    return files, variable_dir


def parse_filename_metadata(filename: str) -> dict:
    """
    Extract time range, lat‑range, lon‑range from filenames like
        t2m_20211002‑20211031_lat35.00_75.00_lon25.00_179.00.zarr
        tp_202004‑202005_latitude‑10.0‑10.0_longitude90.0‑100.0.zarr  ← still ok
        t2m_20211002‑20211031_lat35.00‑75.00_lon25.00‑179.00.zarr     ← new
    Recognises both prefixes: (lat|latitude) and (lon|longitude).
    """
    meta = {}
    stem = os.path.splitext(os.path.basename(filename))[0]          # no “.zarr”
    tokens = stem.split("_")

    for tok in tokens[1:]:
        m = re.fullmatch(r"(\d{8})[-‑](\d{8})", tok)               # 20210101‑20210131
        if m:
            meta["time_range"] = (m.group(1), m.group(2))
            break

    lat_re = re.compile(r"^(lat|latitude)(-?\d+\.?\d*)[-_](-?\d+\.?\d*)$")
    lon_re = re.compile(r"^(lon|longitude)(-?\d+\.?\d*)[-_](-?\d+\.?\d*)$")
    for tok in tokens[2:]:
        m = lat_re.match(tok)
        if m:
            meta["lat_range"] = (float(m.group(2)), float(m.group(3)))
            continue
        m = lon_re.match(tok)
        if m:
            meta["lon_range"] = (float(m.group(2)), float(m.group(3)))
            continue

    return meta





def load_climate_variable_mf(climate_data_dir, variable, time_range=None, lat_range=None, lon_range=None, test_mode=False, chunks_spec="auto"):
    """
    Loads data for a single climate variable using open_mfdataset for efficiency.

    Args:
        climate_data_dir: Directory containing climate data files.
        variable: Name of the climate variable directory/prefix.
        time_range: Optional tuple (start_time, end_time) pandas Timestamps/datetimes.
        lat_range: Optional tuple (min_lat, max_lat).
        lon_range: Optional tuple (min_lon, max_lon).
        test_mode: (Currently unused, consider purpose).
        chunks_spec: Chunk specification for Dask (e.g., "auto", {'latitude': 50, 'longitude': 50}).

    Returns:
        xarray Dataset (Dask-backed) containing the loaded variable data.

    Raises:
        ValueError: If data cannot be found or loaded.
        IOError: If xr.open_mfdataset fails.
    """
    load_start_time = time_lib.time()
    # 1. Find files, pre-filtering by filename convention
    try:
        files, variable_dir = find_climate_files(climate_data_dir, variable)
    except (FileNotFoundError, ValueError) as e:
        # Reraise as ValueError for the calling function to handle
        raise ValueError(f"Could not find data for variable {variable}: {e}") from e

    # 2. Open files lazily with open_mfdataset
    print(f"Opening {len(files)} files with xr.open_mfdataset …")
    time_coord_name = "valid_time"           # adjust if your data differ
    try:
        # xarray ≥ 0.23 ➜ no concat_dim when combine="by_coords"
        ds = xr.open_mfdataset(
            files,
            combine="by_coords",
            decode_timedelta=False,
            chunks=chunks_spec,
            parallel=True,
        )
        ds = ds.sortby(time_coord_name)

    except TypeError as e:
        # Older xarray (< 0.23) still expects concat_dim for 'by_coords'
        # or the user may insist on a deterministic axis: fall back to "nested"
        print(" • Falling back to combine='nested' + explicit concat_dim "
              f"because: {e}")
        ds = xr.open_mfdataset(
            files,
            concat_dim=time_coord_name,
            combine="nested",
            decode_timedelta=False,
            chunks=chunks_spec,
            parallel=True,
        ).sortby(time_coord_name)

    except Exception as e:
        print(f"Error using xr.open_mfdataset on files in {variable_dir}: {e}")
        print("Check:")
        print(f" - Time coordinate name used: '{time_coord_name}' (is it correct?)")
        print(f" - File integrity in list: {files[:3]}...")
        print(f" - Chunk specification: {chunks_spec}")
        print(f" - Available memory and Dask worker logs (if using a cluster)")
        raise IOError(f"Failed to open dataset for {variable}") from e


    if lat_range and 'latitude' in ds.coords:
        lat0, lat1 = lat_range          # 35, 75 in your example
        # True  ⇨ decreasing
        decreasing = ds.latitude[0] > ds.latitude[-1]

        if decreasing:
            ds = ds.sel(latitude=slice(lat1, lat0))   # slice(75, 35)
        else:
            ds = ds.sel(latitude=slice(lat0, lat1))
    if lon_range and 'longitude' in ds.coords:
        # Basic slice, assumes consistent longitude convention (e.g., 0-360 or -180-180)
        ds = ds.sel(longitude=slice(lon_range[0], lon_range[1]))

    # 4. Apply precise time range selection lazily
    if time_range and time_coord_name in ds.coords:
       try:
            # Convert time_range to the same dtype as the coordinate for robust slicing
            coord_dtype = ds[time_coord_name].dtype
            start_time = pd.Timestamp(time_range[0]).to_datetime64().astype(coord_dtype)
            end_time = pd.Timestamp(time_range[1]).to_datetime64().astype(coord_dtype)
            ds = ds.sel({time_coord_name: slice(start_time, end_time)})
       except Exception as e:
           print(f"Warning: Could not apply precise time slice ({time_range}) using .sel(): {e}. Relying on file filtering.")
           print(f"Dataset time coordinate type: {ds[time_coord_name].dtype}")

    # 5. Print info about the lazy dataset
    print("\n" + "="*50)
    print(f"✅ LAZY CLIMATE DATASET CREATED: {variable}")
    print("="*50)
    print(f"  Load Function Time: {time_lib.time() - load_start_time:.2f}s")
    try:
        # Use .item() safely if coordinates exist, otherwise provide 'N/A'
        def safe_min_max_str(coord, fmt='%Y-%m-%d %H:%M:%S'):
            if coord.size > 0:
                 # .compute() might be needed for Dask arrays, but min/max often work directly
                 min_val = pd.Timestamp(coord.min().compute().item())
                 max_val = pd.Timestamp(coord.max().compute().item())
                 return f"{min_val.strftime(fmt)} to {max_val.strftime(fmt)}"
            return "N/A or Empty"

        def safe_min_max_float(coord, fmt='.2f'):
             if coord.size > 0:
                  min_val = coord.min().compute().item()
                  max_val = coord.max().compute().item()
                  return f"{min_val:{fmt}} to {max_val:{fmt}}"
             return "N/A or Empty"

        time_str = safe_min_max_str(ds[time_coord_name]) if time_coord_name in ds else "N/A"
        lat_str = safe_min_max_float(ds.latitude) if 'latitude' in ds else "N/A"
        lon_str = safe_min_max_float(ds.longitude) if 'longitude' in ds else "N/A"

        print(f"📅 Approx Time Range (lazy): {time_str}")
        print(f"🌐 Approx Lat Range (lazy):  {lat_str}")
        print(f"🌐 Approx Lon Range (lazy):  {lon_str}")
        print(f"  Dimensions (lazy): {dict(ds.sizes)}")
        print(f"  Chunking reported by xarray: {ds.chunks}") # VERY IMPORTANT
    except Exception as e:
        print(f"Could not print all dataset details (might be empty or error): {e}")
        print(f"Dataset Coords: {list(ds.coords)}")
        print(f"Dataset Variables: {list(ds.data_vars)}")

    print("="*50 + "\n")

    return ds
