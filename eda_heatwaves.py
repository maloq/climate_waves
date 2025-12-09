
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

# File path
FILE_PATH = "climate_data/regression_target/target_heat_waves_1984_span_30_threshold_7_no_bin/extemp_7_2024.nc"
OUTPUT_DIR = "eda_outputs_target"

def run_eda():
    print(f"Starting EDA for {FILE_PATH}...")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load dataset
    try:
        ds = xr.open_dataset(FILE_PATH)
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print(f"Error: File not found at {FILE_PATH}")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Basic Inspection
    print("\n" + "="*30)
    print("DATASET INFO")
    print("="*30)
    print(ds)
    print("\nDimensions:", dict(ds.dims))
    print("Variables:", list(ds.data_vars))
    print("Coordinates:", list(ds.coords))

    # Access the main variable (assuming 'extemp' based on previous inspection)
    if 'extemp' in ds:
        da = ds['extemp']
    else:
        # Fallback if variable name is different, pick the first data var
        var_name = list(ds.data_vars)[0]
        print(f"Variable 'extemp' not found. Using '{var_name}' instead.")
        da = ds[var_name]

    # Statistical Summary
    print("\n" + "="*30)
    print("STATISTICAL SUMMARY")
    print("="*30)
    
    # We use .load() to bring data into memory for faster numpy ops if it's not too huge.
    # If huge, we might want to keep it dask-backed, but let's assume it fits for this EDA or compute blindly.
    # given the filename, it's one year of data, likely manageable.
    print("Computing statistics (this may take a moment)...")
    
    # Basic stats
    mean_val = da.mean().item()
    std_val = da.std().item()
    min_val = da.min().item()
    max_val = da.max().item()
    
    print(f"Mean: {mean_val:.4f}")
    print(f"Std Dev: {std_val:.4f}")
    print(f"Min: {min_val:.4f}")
    print(f"Max: {max_val:.4f}")
    
    # Missing values
    null_count = np.isnan(da).sum().item()
    total_count = da.size
    print(f"Missing Values: {null_count} / {total_count} ({null_count/total_count*100:.2f}%)")

    # Quantiles
    quantiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    q_vals = da.quantile(quantiles).values
    print("\nQuantiles:")
    for q, val in zip(quantiles, q_vals):
        print(f"  {q*100:05.1f}%: {val:.4f}")

    # ==========================================
    # Visualizations
    # ==========================================
    print("\n" + "="*30)
    print(f"GENERATING VISUALIZATIONS in {OUTPUT_DIR}/")
    print("="*30)

    # 1. Histogram of values
    plt.figure(figsize=(10, 6))
    # Flatten and drop NaNs for histogram
    data_flat = da.values.flatten()
    data_flat = data_flat[~np.isnan(data_flat)]
    
    sns.histplot(data_flat, bins=50, kde=True)
    plt.title(f"Distribution of {da.name}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.yscale('log') # Log scale often helps if there are outliers or long tails
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/1_histogram.png")
    plt.close()
    print("Saved 1_histogram.png")

    # 2. Time Series of Global Spatial Mean
    # Average over lat/lon
    if 'lat' in da.dims and 'lon' in da.dims:
        spatial_dims = ['lat', 'lon']
    elif 'latitude' in da.dims and 'longitude' in da.dims:
        spatial_dims = ['latitude', 'longitude']
    else:
        spatial_dims = list(set(da.dims) - {'time'})

    ts_mean = da.mean(dim=spatial_dims)
    
    plt.figure(figsize=(12, 6))
    ts_mean.plot()
    plt.title(f"Global Mean {da.name} over Time")
    plt.ylabel("Mean Value")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/2_timeseries_mean.png")
    plt.close()
    print("Saved 2_timeseries_mean.png")

    # 3. Spatial Map: Temporal Mean
    # Average over time
    if 'time' in da.dims:
        time_mean = da.mean(dim='time')
        
        plt.figure(figsize=(12, 8))
        time_mean.plot(cmap='coolwarm') # coolwarm good for temperature/anomalies
        plt.title(f"Temporal Mean of {da.name}")
        plt.savefig(f"{OUTPUT_DIR}/3_spatial_mean.png")
        plt.close()
        print("Saved 3_spatial_mean.png")
        
        # Spatial Map: Temporal Max
        time_max = da.max(dim='time')
        plt.figure(figsize=(12, 8))
        time_max.plot(cmap='magma')
        plt.title(f"Temporal Max of {da.name}")
        plt.savefig(f"{OUTPUT_DIR}/4_spatial_max.png")
        plt.close()
        print("Saved 4_spatial_max.png")
    
    # 4. Snapshots (First, Middle, Last)
    if 'time' in da.dims and da.sizes['time'] > 0:
        times_to_plot = [0, da.sizes['time']//2, -1]
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        for i, t_idx in enumerate(times_to_plot):
            # Handle negative index for display
            if t_idx == -1:
                t_idx = da.sizes['time'] - 1
            
            # Select safely
            snapshot = da.isel(time=t_idx)
            snapshot.plot(ax=axes[i], add_colorbar=True)
            axes[i].set_title(f"Time Index: {t_idx}")
        
        plt.suptitle(f"Snapshots of {da.name}")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/5_snapshots.png")
        plt.close()
        print("Saved 5_snapshots.png")

    print("\nEDA Completed successfully.")

if __name__ == "__main__":
    run_eda()
