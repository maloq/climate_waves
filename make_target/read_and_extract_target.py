
import xarray as xr
import numpy as np
import glob
import os
import dask
import sys
import re
from datetime import timedelta

# --- Configuration ---
DATA_DIR = "/home/tehbek/code/climate_data_era5 /10m_wind_gust_since_previous_post_processing"
OUTPUT_DIR = "/home/tehbek/code/climate_data_era5 " # Save in active workspace
WINDOW_DAYS = 7
PERCENTILE = 0.95

# --- Utils ---

def get_variable_name(ds):
    """
    Identifies the wind gust variable from the dataset.
    Prioritizes 'fg10' then 'i10fg'.
    """
    if "fg10" in ds.data_vars:
        return "fg10"
    elif "i10fg" in ds.data_vars:
        return "i10fg"
    else:
        print(f"Error: Could not find wind gust variable. Available variables: {list(ds.data_vars)}")
        sys.exit(1)

def parse_year_from_filename(filename):
    """
    Extracts year from filename assuming format ..._YYYY_MM_...
    """
    basename = os.path.basename(filename)
    match = re.search(r'_(\d{4})_(\d{2})_', basename)
    if match:
        return int(match.group(1))
    return None

def process_year(year, year_files, next_year_files, gust_var_name):
    """
    Process a single year.
    Loads year_files + subset of next_year_files (for padding).
    computes target, crops to year, saves to file.
    """
    print(f"\n=== Processing Year {year} ===")
    
    # We need enough next year data to cover WINDOW_DAYS.
    # Assuming monthly files, one month is enough (covering > 7 days).
    overlap_files = [next_year_files[0]] if next_year_files else []
    
    files_to_load = year_files + overlap_files
    print(f"Loading {len(files_to_load)} files (including {len(overlap_files)} overlap files)...")
    
    try:
        # Use simple chunks, reliance on dask
        # parallel=False (default safe)
        ds = xr.open_mfdataset(
            files_to_load, 
            concat_dim="valid_time", 
            combine="nested", 
            parallel=False,
            # Let dask manage chunks naturally or enforce per-file chunks
        )
    except Exception as e:
        print(f"Error loading files for {year}: {e}")
        return

    gust_da = ds[gust_var_name]

    # Calculate window size in steps
    time_diff = (ds.valid_time[1] - ds.valid_time[0]).values
    dt_seconds = time_diff.astype('timedelta64[s]').astype(int)
    window_steps = int(WINDOW_DAYS * 24 * 3600 // dt_seconds)
    
    # Check if window steps is reasonable
    if window_steps <= 0:
        print(f"Error: Invalid window steps {window_steps}")
        return
    
    print(f"Window: {window_steps} steps ({WINDOW_DAYS} days)")

    # Rolling + Quantile
    # rolled.quantile is missing in older xarray/dask combos or requires construct
    print("Computing 95th Percentile (Forward Looking)...")
    
    # Construct window view
    # This increases memory usage by factor of window_steps effectively if materialized,
    # but dask handles it lazily.
    rolled = gust_da.rolling(valid_time=window_steps, center=False)
    target = rolled.construct("window_dim").quantile(PERCENTILE, dim="window_dim")
    
    # Shift backwards to make it forward looking
    # Result at T from rolling is based on [T-W+1, T].
    # We want result at T to be based on [T, T+W].
    # So we shift T-W to T. i.e., shift by -window_steps.
    target_shifted = target.shift(valid_time=-window_steps)
    
    # Fill NaNs at the end 
    # (Only strictly needed for the very last year if we run out of future data.
    #  For intermediate years, we have overlap data, so the year's end is valid.)
    target_filled = target_shifted.ffill(dim="valid_time")
    
    # Crop to the specific year
    # We select only the current year.
    print(f"Cropping to year {year}...")
    try:
        target_year = target_filled.sel(valid_time=str(year))
    except KeyError:
        print(f"Warning: No data found for year {year} after processing.")
        return

    # Optimization: Round, Clip, uint8
    print("Optimizing (uint8)...")
    target_rounded = target_year.clip(min=0, max=254).round().astype("uint8")
    target_rounded.name = "target_gust_p95"

    output_file = os.path.join(OUTPUT_DIR, f"target_wind_gust_p95_{year}.nc")
    
    encoding = {
        "target_gust_p95": {
            "zlib": True, 
            "complevel": 5, 
            "dtype": "uint8",
            "_FillValue": 255
        }
    }

    print(f"Saving to {output_file}...")
    # Trigger computation
    target_rounded.to_netcdf(output_file, encoding=encoding)
    print(f"Year {year} complete.")

def main():
    print("Finding NetCDF files...")
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.nc")))
    if not all_files:
        print(f"Error: No NetCDF files found in {DATA_DIR}")
        return

    # Group files by year
    files_by_year = {}
    for f in all_files:
        y = parse_year_from_filename(f)
        if y is not None:
            if y not in files_by_year:
                files_by_year[y] = []
            files_by_year[y].append(f)
    
    years = sorted(files_by_year.keys())
    print(f"Found years: {years}")

    # Check variable name once
    with xr.open_dataset(all_files[0]) as first_ds:
        gust_var_name = get_variable_name(first_ds)
    print(f"Using variable: {gust_var_name}")

    for i, year in enumerate(years):
        year_files = files_by_year[year]
        # Get next year files if available
        next_year = year + 1
        next_year_files = files_by_year.get(next_year, [])
        
        process_year(year, year_files, next_year_files, gust_var_name)
    
    print("\nAll years processed.")

if __name__ == "__main__":
    main()
