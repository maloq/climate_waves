import xarray as xr
import pandas as pd
import pathlib
import numpy as np
import os
import glob
import gc

def compute_climatology_sequential():
    feature_dir = pathlib.Path("climate_data/features")
    files = sorted(list(feature_dir.glob("features_*.nc")))
    
    if not files:
        print("No feature files found.")
        return

    print(f"Found {len(files)} files.")

    # Initialize accumulators
    # We don't know the exact shape yet, so we'll init on first file
    daily_sum = None
    daily_count = None
    
    # Variables to track
    temp_vars = []

    for f in files:
        print(f"Processing {f.name}...")
        try:
            with xr.open_dataset(f) as ds:
                # Standardize time
                if "date" in ds.coords:
                    ds = ds.rename({"date": "time"})
                elif "valid_time" in ds.coords:
                    ds = ds.rename({"valid_time": "time"})
                
                # Identify vars on first pass
                current_temp_vars = [v for v in ds.data_vars if "temperature" in v]
                if not temp_vars:
                    temp_vars = current_temp_vars
                    print(f"  Tracking variables: {temp_vars}")
                
                if not current_temp_vars:
                    continue

                ds_subset = ds[temp_vars].load() # Load into memory to work with numpy/pandas
                
                # Group by dayofyear
                # sum() and count()
                gb = ds_subset.groupby("time.dayofyear")
                
                # We need to manually aggregate because we want to accumulate across files
                # xarray's groupby.sum() returns a dataset with dayofyear dim
                
                local_sum = gb.sum(dim="time")
                local_count = gb.count(dim="time")
                
                if daily_sum is None:
                    daily_sum = local_sum
                    daily_count = local_count
                else:
                    # Align and add
                    # Use reindex in case some files don't have all days (e.g. leap years)
                    # or fillna(0) 
                    daily_sum = daily_sum.fillna(0) + local_sum.fillna(0)
                    daily_count = daily_count.fillna(0) + local_count.fillna(0)
                
                del ds_subset, local_sum, local_count
                gc.collect()
                
        except Exception as e:
            print(f"Error processing {f}: {e}")

    if daily_sum is None:
        print("No data processed.")
        return

    # Compute Mean
    print("Computing final mean...")
    climatology = daily_sum / daily_count
    
    # Handle possible division by zero or NaNs if any
    
    output_path = pathlib.Path("climate_data/climatology.nc")
    climatology.to_netcdf(output_path)
    print(f"Saved climatology to {output_path}")

if __name__ == "__main__":
    compute_climatology_sequential()
