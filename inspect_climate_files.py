import xarray as xr
import glob
import os

files = sorted(glob.glob('climate_data/features/features_20*.nc'))
print(f"Found {len(files)} files.")

for f in files:
    if '2022' not in f and '2023' not in f and '2024' not in f:
        continue
        
    print(f"\nScanning {f}...")
    try:
        ds = xr.open_dataset(f)
        print(f"  Dimensions: {ds.dims}")
        print(f"  Size: {os.path.getsize(f) / 1e6:.2f} MB")
        
        if 'time' in ds.coords:
            time = ds.time.values
            print(f"  Time range: {time[0]} to {time[-1]}")
            print(f"  Time length: {len(time)}")
            
            # Check frequency
            diffs = ds.time.diff('time').values.astype('timedelta64[h]').astype(int)
            unique_diffs = set(diffs)
            print(f"  Time step differences (hours): {unique_diffs}")
            
            if len(unique_diffs) > 1:
                 print("  WARNING: Inconsistent time steps found!")
        else:
            print("  WARNING: No time coordinate found!")
            
        print("  Variables: ", list(ds.data_vars))
        
        # Check for NaNs in first variable
        first_var = list(ds.data_vars)[0]
        nan_count = ds[first_var].isnull().sum().item()
        print(f"  NaN count in {first_var}: {nan_count}")

    except Exception as e:
        print(f"  Error reading {f}: {e}")
