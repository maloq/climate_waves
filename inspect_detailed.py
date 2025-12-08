import xarray as xr
import glob
import numpy as np

files = sorted(glob.glob('climate_data/features/features_20*.nc'))
targets = [f for f in files if '2023' in f or '2024' in f]

for f in targets:
    print(f"\n--- Checking {f} ---")
    try:
        ds = xr.open_dataset(f)
        
        # Check Coordinates
        print("  Coords:", list(ds.coords))
        if 'latitude' in ds.coords:
            lat = ds.latitude.values
            print(f"  Lat: shape={lat.shape}, min={lat.min():.2f}, max={lat.max():.2f}, diff={np.diff(lat).mean():.4f}")
        if 'longitude' in ds.coords:
            lon = ds.longitude.values
            print(f"  Lon: shape={lon.shape}, min={lon.min():.2f}, max={lon.max():.2f}, diff={np.diff(lon).mean():.4f}")
            
        # Check Time/Date
        if 'date' in ds.dims:
            print(f"  Dim 'date': {ds.dims['date']}")
        if 'valid_time' in ds.dims:
            print(f"  Dim 'valid_time': {ds.dims['valid_time']}")
            
        # Check one variable shape
        var0 = list(ds.data_vars)[0]
        print(f"  Var '{var0}' shape: {ds[var0].shape}")

    except Exception as e:
        print(f"  Error: {e}")
