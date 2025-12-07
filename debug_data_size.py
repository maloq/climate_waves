
import xarray as xr
import glob
import os
import pandas as pd

target_dir = "climate_data/target/target_heat_waves_1984_span_30_threshold_7"
features_dir = "/home/infres/vmorozov/Misc/data/climate_data_files/ECMWF"
year = 2017

# 1. Inspect Target
target_file = os.path.join(target_dir, f"extemp_7_{year}.nc")
if os.path.exists(target_file):
    print(f"Opening target: {target_file}")
    ds_t = xr.open_dataset(target_file)
    print("Target Coords:")
    print(ds_t.coords)
    print("Target Sizes:")
    print(ds_t.sizes)
    if "latitude" in ds_t.coords:
        lat = ds_t.latitude.values
        print(f"Lat range: {lat.min()} to {lat.max()}, Step: {lat[1]-lat[0] if len(lat)>1 else 'N/A'}")
    if "longitude" in ds_t.coords:
        lon = ds_t.longitude.values
        print(f"Lon range: {lon.min()} to {lon.max()}, Step: {lon[1]-lon[0] if len(lon)>1 else 'N/A'}")
    
    # Check time density
    if "time" in ds_t.coords:
        times = pd.to_datetime(ds_t.time.values)
        print(f"Time range: {times.min()} to {times.max()}")
        print(f"Num time steps: {len(times)}")
        # Check if summer only
        months = times.month.unique()
        print(f"Months present: {sorted(months)}")
else:
    print(f"Target file not found: {target_file}")

print("-" * 40)

# 2. Inspect Features (t2m)
t2m_dir = os.path.join(features_dir, "t2m")
zarr_files = glob.glob(os.path.join(t2m_dir, "t2m_*.zarr"))
if zarr_files:
    f_path = zarr_files[0]
    print(f"Opening feature file: {f_path}")
    ds_f = xr.open_dataset(f_path, engine='zarr', chunks='auto')
    print("Feature Coords:")
    print(ds_f.coords)
    print("Feature Sizes:")
    print(ds_f.sizes)
    
    # Standardize names for check
    if "latitude" not in ds_f.coords and "lat" in ds_f.coords:
        ds_f = ds_f.rename({"lat": "latitude"})
    if "longitude" not in ds_f.coords and "lon" in ds_f.coords:
        ds_f = ds_f.rename({"lon": "longitude"})

    if "latitude" in ds_f.coords:
        lat = ds_f.latitude.values
        print(f"Lat range: {lat.min()} to {lat.max()}, Step: {lat[1]-lat[0] if len(lat)>1 else 'N/A'}")
    if "longitude" in ds_f.coords:
        lon = ds_f.longitude.values
        print(f"Lon range: {lon.min()} to {lon.max()}, Step: {lon[1]-lon[0] if len(lon)>1 else 'N/A'}")

    if "time" in ds_f.coords or "valid_time" in ds_f.coords:
        t_coord = "time" if "time" in ds_f.coords else "valid_time"
        times = pd.to_datetime(ds_f[t_coord].values)
        print(f"Time range: {times.min()} to {times.max()}")
        print(f"Num time steps: {len(times)}")

else:
    print("No feature zarr files found.")
