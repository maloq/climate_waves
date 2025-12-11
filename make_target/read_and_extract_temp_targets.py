import xarray as xr
import numpy as np
import glob
import os
import sys
import re
import numba

# --- Configuration ---
DATA_DIR = "/home/tehbek/code/climate_data_era5/reanalysis-era5-single-levels-temperature"
OUTPUT_DIR = "/home/tehbek/code/climate_data_era5/results"
WINDOW_DAYS = 14
PERCENTILES = [0.05, 0.95]
VAR_NAME_GUESS = "t2m"
START_YEAR = 2010

NUM_THREADS = 28  # Use all threads for numba parallel

# Set numba threading
numba.set_num_threads(NUM_THREADS)

# --- Utils ---

def get_variable_name(ds):
    """Identifies the temperature variable from the dataset."""
    if VAR_NAME_GUESS in ds.data_vars:
        return VAR_NAME_GUESS
    else:
        print(f"Error: Could not find variable '{VAR_NAME_GUESS}'. Available variables: {list(ds.data_vars)}")
        sys.exit(1)

def parse_year_from_filename(filename):
    """Extracts year from filename assuming format ..._YYYY_MM_..."""
    basename = os.path.basename(filename)
    match = re.search(r'_(\d{4})_(\d{2})_', basename)
    if match:
        return int(match.group(1))
    return None

@numba.jit(nopython=True, cache=True)
def quantile_sorted(sorted_vals, n, q):
    """Compute quantile from pre-sorted array."""
    if n == 0:
        return np.nan
    if n == 1:
        return sorted_vals[0]
    idx = q * (n - 1)
    idx_floor = int(np.floor(idx))
    idx_ceil = min(idx_floor + 1, n - 1)
    frac = idx - idx_floor
    return sorted_vals[idx_floor] * (1 - frac) + sorted_vals[idx_ceil] * frac

@numba.jit(nopython=True, parallel=True, cache=True)
def fast_rolling_quantiles(data, window_size, q_low, q_high):
    """
    Numba-accelerated rolling window quantile computation.
    Parallelizes over spatial dimensions.
    
    data: (Time, Lat, Lon) - assumed contiguous float32
    Returns: (2, Time_out, Lat, Lon) for [q_low, q_high]
    """
    T, H, W = data.shape
    T_out = T - window_size + 1
    
    if T_out <= 0:
        return np.empty((2, 0, H, W), dtype=np.float32)
    
    result = np.empty((2, T_out, H, W), dtype=np.float32)
    
    # Flatten spatial dims for better load balancing
    HW = H * W
    
    for idx in numba.prange(HW):
        i = idx // W
        j = idx % W
        
        # Pre-allocate sort buffer for this pixel
        sort_buf = np.empty(window_size, dtype=np.float32)
        
        for t in range(T_out):
            # Copy window data
            n_valid = 0
            for k in range(window_size):
                val = data[t + k, i, j]
                if not np.isnan(val):
                    sort_buf[n_valid] = val
                    n_valid += 1
            
            if n_valid == 0:
                result[0, t, i, j] = np.nan
                result[1, t, i, j] = np.nan
            else:
                # Sort only valid values
                valid_slice = sort_buf[:n_valid]
                valid_slice.sort()
                
                result[0, t, i, j] = quantile_sorted(valid_slice, n_valid, q_low)
                result[1, t, i, j] = quantile_sorted(valid_slice, n_valid, q_high)
    
    return result

def load_files_sequential(files, var_name):
    """Load NetCDF files sequentially into memory - fast on SSD."""
    print(f"  Loading {len(files)} files into memory...")
    
    # Open without chunking for in-memory processing
    ds = xr.open_mfdataset(
        files,
        concat_dim="valid_time",
        combine="nested",
        parallel=False,  # Sequential loading - safe
        chunks=None,  # Load fully into memory
        engine="netcdf4"
    )
    
    # Load into memory as contiguous array
    data = ds[var_name].values.astype(np.float32)
    times = ds.valid_time.values
    coords = {
        'latitude': ds.latitude.values,
        'longitude': ds.longitude.values,
        'attrs': ds[var_name].attrs
    }
    ds.close()
    
    return data, times, coords

def save_zarr(data, times, coords, var_name, output_path):
    """Save to Zarr format with fast compression."""
    
    da = xr.DataArray(
        data,
        dims=['valid_time', 'latitude', 'longitude'],
        coords={
            'valid_time': times,
            'latitude': coords['latitude'],
            'longitude': coords['longitude']
        },
        name=var_name,
        attrs=coords['attrs']
    )
    
    ds = da.to_dataset()
    
    # Use zarr v3 compatible codec via zarr.codecs
    try:
        from zarr.codecs import BloscCodec
        # Zarr v3 style
        encoding = {
            var_name: {
                'compressors': [BloscCodec(cname='zstd', clevel=3, shuffle='bitshuffle')]
            }
        }
    except ImportError:
        # Zarr v2 fallback
        from numcodecs import Blosc
        encoding = {
            var_name: {
                'compressor': Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
            }
        }
    
    ds.to_zarr(output_path, mode='w', encoding=encoding, consolidated=True)

def process_year(year, year_files, prev_year_files, next_year_files, var_name):
    """
    Process a single year with optimized in-memory computation.
    """
    print(f"\n=== Processing Year {year} ===")
    
    # Get overlap files
    overlap_next = [next_year_files[0]] if next_year_files else []
    overlap_prev = [prev_year_files[-1]] if prev_year_files else []
    
    files_to_load = overlap_prev + year_files + overlap_next
    
    # Load all data into memory (fast on SSD)
    data, times, coords = load_files_sequential(files_to_load, var_name)
    print(f"  Loaded shape: {data.shape}, dtype: {data.dtype}")
    print(f"  Memory: {data.nbytes / 1e9:.2f} GB")
    
    # Calculate window size in steps
    if len(times) > 1:
        time_diff = (times[1] - times[0]).astype('timedelta64[s]').astype(int)
        window_steps = int(WINDOW_DAYS * 24 * 3600 // time_diff)
    else:
        print("  Error: Not enough time steps")
        return
    
    print(f"  Window: {window_steps} steps ({WINDOW_DAYS} days)")
    
    # Run numba parallel computation
    print(f"  Computing rolling quantiles with {NUM_THREADS} threads...")
    
    import time
    t0 = time.time()
    
    result = fast_rolling_quantiles(
        data, 
        window_steps, 
        PERCENTILES[0], 
        PERCENTILES[1]
    )
    
    t1 = time.time()
    print(f"  Computation took {t1-t0:.1f}s")
    print(f"  Result shape: {result.shape}")
    
    # Adjust times for output (first window_steps-1 times are lost)
    out_times = times[window_steps - 1:]
    
    # Find indices for this year only
    year_start = np.datetime64(f'{year}-01-01')
    year_end = np.datetime64(f'{year+1}-01-01')
    
    year_mask = (out_times >= year_start) & (out_times < year_end)
    year_indices = np.where(year_mask)[0]
    
    if len(year_indices) == 0:
        print(f"  Warning: No data found for year {year}")
        return
    
    year_times = out_times[year_mask]
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for i, p in enumerate(PERCENTILES):
        p_str = f"p{int(p*100):02d}"
        target_name = f"target_temperature_{p_str}"
        
        # Extract this year's data for this percentile
        year_data = result[i, year_indices, :, :]
        
        output_path = os.path.join(OUTPUT_DIR, f"{target_name}_{year}.zarr")
        print(f"  Saving {target_name} to {output_path}...")
        
        save_zarr(year_data, year_times, coords, target_name, output_path)
    
    print(f"  Year {year} complete.")
    
    # Clean up
    del data, result

def main():
    # Warm up numba JIT
    print("Warming up numba JIT compilation...")
    try:
        dummy = np.random.rand(100, 10, 10).astype(np.float32)
        _ = fast_rolling_quantiles(dummy, 10, 0.05, 0.95)
        print(f"Numba ready with {NUM_THREADS} threads")
    except Exception as e:
        print(f"Numba error: {e}")
        return
    
    print("\nFinding NetCDF files...")
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

    for year in years:
        if START_YEAR is not None and year < START_YEAR:
            continue

        year_files = files_by_year[year]
        prev_year_files = files_by_year.get(year - 1, [])
        next_year_files = files_by_year.get(year + 1, [])
        
        process_year(year, year_files, prev_year_files, next_year_files, var_name)
    
    print("\nAll years processed.")

if __name__ == "__main__":
    main()
