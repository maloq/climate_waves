import xarray as xr
import numpy as np
import glob
import os
import sys

# --- Configuration ---
DATA_DIR = "/home/tehbek/code/climate_data_era5 "
AVG_COLD_FILE = "daily_average_cold.nc"
AVG_HEAT_FILE = "daily_average_heat.nc"
THRESHOLD = 10.0

def get_target_files(data_dir, percentile_str):
    pattern = os.path.join(data_dir, f"target_temperature_{percentile_str}_*.nc")
    files = sorted(glob.glob(pattern))
    return files

def parse_year_from_filename(filename):
    # filename format: target_temperature_pXX_YYYY.nc
    basename = os.path.basename(filename)
    parts = basename.split('_')
    # parts: ['target', 'temperature', 'pXX', 'YYYY.nc']
    if len(parts) >= 4:
        year_str = parts[-1].replace('.nc', '')
        if year_str.isdigit():
            return int(year_str)
    return None

def load_average_file(path):
    if not os.path.exists(path):
        print(f"Error: Average file not found at {path}")
        return None
    try:
        ds = xr.open_dataset(path)
        print(f"Loaded average file: {path}")
        print(ds)
        return ds
    except Exception as e:
        print(f"Error loading average file {path}: {e}")
        return None

def main():
    avg_cold_path = os.path.join(DATA_DIR, AVG_COLD_FILE)
    avg_heat_path = os.path.join(DATA_DIR, AVG_HEAT_FILE)

    # Load averages
    ds_avg_cold = load_average_file(avg_cold_path)
    ds_avg_heat = load_average_file(avg_heat_path)

    if ds_avg_cold is None or ds_avg_heat is None:
        print("Error: Could not load one or more average files. Exiting.")
        return

    # Identify variable names in average files (assuming first variable if not 't2m')
    var_cold = list(ds_avg_cold.data_vars)[0]
    var_heat = list(ds_avg_heat.data_vars)[0]
    print(f"Using variable '{var_cold}' from cold average and '{var_heat}' from heat average.")
    
    da_avg_cold = ds_avg_cold[var_cold]
    da_avg_heat = ds_avg_heat[var_heat]

    # Process p05 (Cold)
    files_p05 = get_target_files(DATA_DIR, "p05")
    if not files_p05:
        print("No target files found for p05.")
    
    for f in files_p05:
        year = parse_year_from_filename(f)
        print(f"\nProcessing p05 (Cold) for year {year}...")
        
        try:
            ds_target = xr.open_dataset(f)
            target_da = ds_target["target_temperature_p05"]
            
            # Print stats for original
            mean_val = target_da.mean().values
            min_val = target_da.min().values
            max_val = target_da.max().values
            print(f"  [Original] Mean: {mean_val:.2f}, Min: {min_val:.2f}, Max: {max_val:.2f}")

            # Align average to target
            # Need to handle time alignment. 
            # If average is dayofyear, we select using target's dayofyear.
            # If average is time-series, we select using target's time.
            
            if "dayofyear" in da_avg_cold.coords or "dayofyear" in da_avg_cold.dims:
                # Climatology
                # Group target by dayofyear is slow for large data, better to sel
                # Assuming simple broadcasting might work if encoded right, but usually xarray needs help.
                # Let's try to select based on dayofyear of the target
                avg_aligned = da_avg_cold.sel(dayofyear=target_da['valid_time'].dt.dayofyear)
            else:
                # Time series - try to sel by time (intersection)
                 avg_aligned = da_avg_cold.sel(time=target_da['valid_time'], method="nearest")

            # Condition: Target < Avg - 10
            # Cold wave: Temp is significantly lower than average
            diff = avg_aligned - THRESHOLD
            binary_target = (target_da < diff).astype("uint8")
            
            # Print stats for binary
            pos_count = (binary_target == 1).sum().values
            total_count = binary_target.size
            pos_rate = pos_count / total_count
            print(f"  [Binary] Positive Rate: {pos_rate:.4%} ({pos_count}/{total_count})")

            # Save
            out_name = f"target_binary_p05_{year}.nc"
            out_path = os.path.join(DATA_DIR, out_name)
            binary_target.name = "target_binary_p05"
            
            encoding = {"target_binary_p05": {"zlib": True, "complevel": 5, "dtype": "uint8", "_FillValue": 255}}
            binary_target.to_netcdf(out_path, encoding=encoding)
            print(f"  Saved to {out_name}")

        except Exception as e:
            print(f"  Error processing {f}: {e}")

    # Process p95 (Heat)
    files_p95 = get_target_files(DATA_DIR, "p95")
    if not files_p95:
        print("No target files found for p95.")

    for f in files_p95:
        year = parse_year_from_filename(f)
        print(f"\nProcessing p95 (Heat) for year {year}...")
        
        try:
            ds_target = xr.open_dataset(f)
            target_da = ds_target["target_temperature_p95"]
            
            # Print stats for original
            mean_val = target_da.mean().values
            min_val = target_da.min().values
            max_val = target_da.max().values
            print(f"  [Original] Mean: {mean_val:.2f}, Min: {min_val:.2f}, Max: {max_val:.2f}")

            # Align
            if "dayofyear" in da_avg_heat.coords or "dayofyear" in da_avg_heat.dims:
                avg_aligned = da_avg_heat.sel(dayofyear=target_da['valid_time'].dt.dayofyear)
            else:
                 avg_aligned = da_avg_heat.sel(time=target_da['valid_time'], method="nearest")

            # Condition: Target > Avg + 10
            diff = avg_aligned + THRESHOLD
            binary_target = (target_da > diff).astype("uint8")
            
            # Print stats
            pos_count = (binary_target == 1).sum().values
            total_count = binary_target.size
            pos_rate = pos_count / total_count
            print(f"  [Binary] Positive Rate: {pos_rate:.4%} ({pos_count}/{total_count})")

            # Save
            out_name = f"target_binary_p95_{year}.nc"
            out_path = os.path.join(DATA_DIR, out_name)
            binary_target.name = "target_binary_p95"
            
            encoding = {"target_binary_p95": {"zlib": True, "complevel": 5, "dtype": "uint8", "_FillValue": 255}}
            binary_target.to_netcdf(out_path, encoding=encoding)
            print(f"  Saved to {out_name}")

        except Exception as e:
            print(f"  Error processing {f}: {e}")

if __name__ == "__main__":
    main()
