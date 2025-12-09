import xarray as xr
import numpy as np
import glob
import os
import dask
import sys
import re
from datetime import timedelta

# --- Configuration ---
DATA_DIR = "/home/tehbek/code/climate_data_era5 /reanalysis-era5-single-levels-temperature"
OUTPUT_DIR = "/home/tehbek/code/climate_data_era5 " # Save in active workspace
WINDOW_DAYS = 7
PERCENTILES = [0.05, 0.95]
VAR_NAME_GUESS = "t2m" 

# --- Utils ---

def get_variable_name(ds):
    """
    Identifies the temperature variable from the dataset.
    Prioritizes 't2m'.
    """
    if VAR_NAME_GUESS in ds.data_vars:
        return VAR_NAME_GUESS
    else:
        # Fallback or error
        # Check for other common names if needed, but for ERA5 single levels it's usually t2m
        print(f"Error: Could not find variable '{VAR_NAME_GUESS}'. Available variables: {list(ds.data_vars)}")
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

def process_year(year, year_files, next_year_files, var_name):
    """
    Process a single year.
    Loads year_files + subset of next_year_files (for padding).
    computes targets (5th and 95th), crops to year, saves to separate files.
    """
    print(f"\n=== Processing Year {year} ===")
    
    # We need enough next year data to cover WINDOW_DAYS.
    overlap_files = [next_year_files[0]] if next_year_files else []
    
    files_to_load = year_files + overlap_files
    print(f"Loading {len(files_to_load)} files (including {len(overlap_files)} overlap files)...")
    
    try:
        ds = xr.open_mfdataset(
            files_to_load, 
            concat_dim="valid_time", 
            combine="nested", 
            parallel=False,
        )
    except Exception as e:
        print(f"Error loading files for {year}: {e}")
        return

    da = ds[var_name]

    # Calculate window size in steps
    time_diff = (ds.valid_time[1] - ds.valid_time[0]).values
    dt_seconds = time_diff.astype('timedelta64[s]').astype(int)
    window_steps = int(WINDOW_DAYS * 24 * 3600 // dt_seconds)
    
    if window_steps <= 0:
        print(f"Error: Invalid window steps {window_steps}")
        return
    
    print(f"Window: {window_steps} steps ({WINDOW_DAYS} days)")

    # Construct window view
    rolled = da.rolling(valid_time=window_steps, center=False)
    rolled_construct = rolled.construct("window_dim")

    for p in PERCENTILES:
        p_str = f"p{int(p*100):02d}"
        print(f"Computing {int(p*100)}th Percentile (Forward Looking)...")
        
        target = rolled_construct.quantile(p, dim="window_dim")
        
        # Shift backwards to make it forward looking
        target_shifted = target.shift(valid_time=-window_steps)
        
        # Fill NaNs at the end 
        target_filled = target_shifted.ffill(dim="valid_time")
        
        # Crop to the specific year
        print(f"Cropping to year {year}...")
        try:
            target_year = target_filled.sel(valid_time=str(year))
        except KeyError:
            print(f"Warning: No data found for year {year} after processing.")
            continue

        target_name = f"target_temperature_{p_str}"
        target_year.name = target_name
        
        # Encoding - using float32 for precision
        target_final = target_year.astype("float32")

        output_file = os.path.join(OUTPUT_DIR, f"{target_name}_{year}.nc")
        
        encoding = {
            target_name: {
                "zlib": True, 
                "complevel": 5, 
                "dtype": "float32",
                "_FillValue": None # float32 usually handles NaN naturally, or specifies one.
            }
        }

        print(f"Saving to {output_file}...")
        target_final.to_netcdf(output_file, encoding=encoding)
        print(f"Saved {target_name} for {year}.")

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
        var_name = get_variable_name(first_ds)
    print(f"Using variable: {var_name}")

    for i, year in enumerate(years):
        year_files = files_by_year[year]
        # Get next year files if available
        next_year = year + 1
        next_year_files = files_by_year.get(next_year, [])
        
        process_year(year, year_files, next_year_files, var_name)
    
    print("\nAll years processed.")

if __name__ == "__main__":
    main()
