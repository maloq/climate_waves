import xarray as xr
import numpy as np
import glob
import os
import sys
import re
import numba

# --- Configuration ---
DATA_DIR = "/home/tehbek/code/climate_data_era5/10m_wind_gust_since_previous_post_processing"
OUTPUT_DIR = "/home/tehbek/code/climate_data_era5/results"
WINDOW_DAYS = 14
PERCENTILE = 0.95

NUM_THREADS = 32

# Set numba threading
numba.set_num_threads(NUM_THREADS)

# --- Utils ---

def get_variable_name(ds):
    """Identifies the wind gust variable from the dataset."""
    if "fg10" in ds.data_vars:
        return "fg10"
    elif "i10fg" in ds.data_vars:
        return "i10fg"
    else:
        print(f"Error: Could not find wind gust variable. Available variables: {list(ds.data_vars)}")
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
def ffill_numba(data):
    """Fast forward fill for NaN values along time axis."""
    T, H, W = data.shape
    result = data.copy()
    
    for idx in numba.prange(H * W):
        i = idx // W
        j = idx % W
        
        last_valid = np.nan
        for t in range(T):
            val = result[t, i, j]
            if np.isnan(val):
                if not np.isnan(last_valid):
                    result[t, i, j] = last_valid
            else:
                last_valid = val
    
    return result

@numba.jit(nopython=True, parallel=True, cache=True)
def fast_rolling_quantile_centered(data, window_size, q):
    """
    Numba-accelerated centered rolling window quantile.
    For each time T, computes quantile over a centered window.
    
    data: (Time, Lat, Lon) - assumed contiguous float32
    Returns: (Time, Lat, Lon) with NaN at edges where window doesn't fit
    """
    T, H, W = data.shape
    
    result = np.empty((T, H, W), dtype=np.float32)
    
    # Centered window: half before, half after
    half_window = window_size // 2
    
    # Flatten spatial dims for better load balancing
    HW = H * W
    
    for idx in numba.prange(HW):
        i = idx // W
        j = idx % W
        
        # Pre-allocate sort buffer
        sort_buf = np.empty(window_size, dtype=np.float32)
        
        # Centered: for time t, use data[t-half:t+half+1] or data[t-half:t+half]
        for t in range(T):
            start = t - half_window
            end = t + half_window + (window_size % 2)  # +1 if odd window size
            
            if start < 0 or end > T:
                # Not enough data on edges - will be filled later
                result[t, i, j] = np.nan
            else:
                # Copy window data
                n_valid = 0
                for k in range(start, end):
                    val = data[k, i, j]
                    if not np.isnan(val):
                        sort_buf[n_valid] = val
                        n_valid += 1
                
                if n_valid == 0:
                    result[t, i, j] = np.nan
                else:
                    # Sort and compute quantile
                    valid_slice = sort_buf[:n_valid]
                    valid_slice.sort()
                    result[t, i, j] = quantile_sorted(valid_slice, n_valid, q)
    
    return result

def load_files_sequential(files, var_name):
    """Load NetCDF files sequentially into memory - fast on SSD."""
    print(f"  Loading {len(files)} files into memory...")
    
    ds = xr.open_mfdataset(
        files,
        concat_dim="valid_time",
        combine="nested",
        parallel=False,
        chunks=None,
        engine="netcdf4"
    )
    
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
    
    # Use zarr v3 compatible codec
    try:
        from zarr.codecs import BloscCodec
        encoding = {
            var_name: {
                'compressors': [BloscCodec(cname='zstd', clevel=3, shuffle='bitshuffle')]
            }
        }
    except ImportError:
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
    Centered window: result at T is based on [T-window/2, T+window/2].
    """
    print(f"\n=== Processing Year {year} ===")
    
    # Get overlap files for centered window (need both before and after)
    overlap_next = [next_year_files[0]] if next_year_files else []
    overlap_prev = [prev_year_files[-1]] if prev_year_files else []
    
    files_to_load = overlap_prev + year_files + overlap_next
    
    # Load all data into memory
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
    
    # Run numba parallel computation (centered window)
    print(f"  Computing centered {int(PERCENTILE*100)}th percentile with {NUM_THREADS} threads...")
    
    import time
    t0 = time.time()
    
    result = fast_rolling_quantile_centered(data, window_steps, PERCENTILE)
    
    t1 = time.time()
    print(f"  Computation took {t1-t0:.1f}s")
    
    # Forward fill NaNs at the end (where window doesn't fit)
    print("  Forward filling NaN values...")
    result = ffill_numba(result)
    
    # Find indices for this year only
    year_start = np.datetime64(f'{year}-01-01')
    year_end = np.datetime64(f'{year+1}-01-01')
    
    year_mask = (times >= year_start) & (times < year_end)
    year_indices = np.where(year_mask)[0]
    
    if len(year_indices) == 0:
        print(f"  Warning: No data found for year {year}")
        return
    
    year_times = times[year_mask]
    year_data = result[year_indices, :, :]
    
    # Optimize: clip, round, convert to uint8
    print("  Optimizing (uint8)...")
    year_data = np.clip(year_data, 0, 254).round().astype(np.uint8)
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    target_name = "target_gust_p95"
    output_path = os.path.join(OUTPUT_DIR, f"target_wind_gust_p95_{year}.zarr")
    print(f"  Saving to {output_path}...")
    
    # For uint8, update coords attrs
    coords['attrs'] = {'units': 'm/s', 'long_name': f'95th percentile wind gust ({WINDOW_DAYS}-day centered)'}
    
    save_zarr(year_data, year_times, coords, target_name, output_path)
    
    print(f"  Year {year} complete.")
    
    # Clean up
    del data, result

def main():
    # Warm up numba JIT
    print("Warming up numba JIT compilation...")
    try:
        dummy = np.random.rand(100, 10, 10).astype(np.float32)
        _ = fast_rolling_quantile_centered(dummy, 10, 0.95)
        _ = ffill_numba(dummy)
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
        year_files = files_by_year[year]
        prev_year_files = files_by_year.get(year - 1, [])
        next_year_files = files_by_year.get(year + 1, [])
        
        process_year(year, year_files, prev_year_files, next_year_files, var_name)
    
    print("\nAll years processed.")

if __name__ == "__main__":
    main()
