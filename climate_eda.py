#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) for Climate Data
NetCDF file: features_2017.nc
"""

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Configuration
DATA_FILE = "climate_data/features/features_2017.nc"
OUTPUT_DIR = Path("eda_output")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data(filepath):
    """Load NetCDF data using xarray."""
    print(f"\n{'='*60}")
    print(f"Loading data from: {filepath}")
    print(f"{'='*60}")
    ds = xr.open_dataset(filepath)
    return ds


def print_dataset_overview(ds):
    """Print comprehensive overview of the dataset."""
    print(f"\n{'='*60}")
    print("DATASET OVERVIEW")
    print(f"{'='*60}")
    print(ds)
    
    print(f"\n{'='*60}")
    print("DIMENSIONS")
    print(f"{'='*60}")
    for dim, size in ds.dims.items():
        print(f"  {dim}: {size}")
    
    print(f"\n{'='*60}")
    print("COORDINATES")
    print(f"{'='*60}")
    for coord in ds.coords:
        coord_data = ds.coords[coord]
        print(f"\n  {coord}:")
        print(f"    Shape: {coord_data.shape}")
        print(f"    Dtype: {coord_data.dtype}")
        if coord_data.size > 0:
            if np.issubdtype(coord_data.dtype, np.datetime64):
                print(f"    Range: {coord_data.values.min()} to {coord_data.values.max()}")
            elif np.issubdtype(coord_data.dtype, np.number):
                print(f"    Range: {float(coord_data.min()):.4f} to {float(coord_data.max()):.4f}")
            else:
                print(f"    First values: {coord_data.values[:5]}")
    
    print(f"\n{'='*60}")
    print("DATA VARIABLES")
    print(f"{'='*60}")
    for var in ds.data_vars:
        var_data = ds[var]
        print(f"\n  {var}:")
        print(f"    Dimensions: {var_data.dims}")
        print(f"    Shape: {var_data.shape}")
        print(f"    Dtype: {var_data.dtype}")
        if 'units' in var_data.attrs:
            print(f"    Units: {var_data.attrs['units']}")
        if 'long_name' in var_data.attrs:
            print(f"    Long name: {var_data.attrs['long_name']}")
    
    print(f"\n{'='*60}")
    print("GLOBAL ATTRIBUTES")
    print(f"{'='*60}")
    for attr, value in ds.attrs.items():
        print(f"  {attr}: {value}")


def compute_statistics(ds):
    """Compute and display statistics for each variable."""
    print(f"\n{'='*60}")
    print("STATISTICAL SUMMARY")
    print(f"{'='*60}")
    
    stats_list = []
    
    for var in ds.data_vars:
        var_data = ds[var]
        
        # Skip non-numeric variables
        if not np.issubdtype(var_data.dtype, np.number):
            continue
        
        values = var_data.values.flatten()
        valid_values = values[~np.isnan(values)]
        
        if len(valid_values) == 0:
            print(f"\n  {var}: All NaN values")
            continue
        
        stats = {
            'Variable': var,
            'Count': len(valid_values),
            'NaN Count': np.sum(np.isnan(values)),
            'NaN %': 100 * np.sum(np.isnan(values)) / len(values),
            'Min': np.min(valid_values),
            'Max': np.max(valid_values),
            'Mean': np.mean(valid_values),
            'Std': np.std(valid_values),
            'Median': np.median(valid_values),
            'Q1 (25%)': np.percentile(valid_values, 25),
            'Q3 (75%)': np.percentile(valid_values, 75),
        }
        stats_list.append(stats)
        
        print(f"\n  {var}:")
        print(f"    Valid values: {stats['Count']:,}")
        print(f"    Missing (NaN): {stats['NaN Count']:,} ({stats['NaN %']:.2f}%)")
        print(f"    Min: {stats['Min']:.4f}")
        print(f"    Max: {stats['Max']:.4f}")
        print(f"    Mean: {stats['Mean']:.4f}")
        print(f"    Std: {stats['Std']:.4f}")
        print(f"    Median: {stats['Median']:.4f}")
        print(f"    Q1 (25%): {stats['Q1 (25%)']:.4f}")
        print(f"    Q3 (75%): {stats['Q3 (75%)']:.4f}")
    
    # Save statistics to CSV
    if stats_list:
        stats_df = pd.DataFrame(stats_list)
        stats_df.to_csv(OUTPUT_DIR / "statistics_summary.csv", index=False)
        print(f"\n  Statistics saved to: {OUTPUT_DIR / 'statistics_summary.csv'}")
    
    return stats_list


def plot_distributions(ds):
    """Create distribution plots for each variable."""
    print(f"\n{'='*60}")
    print("CREATING DISTRIBUTION PLOTS")
    print(f"{'='*60}")
    
    numeric_vars = [var for var in ds.data_vars 
                    if np.issubdtype(ds[var].dtype, np.number)]
    
    if not numeric_vars:
        print("  No numeric variables found.")
        return
    
    n_vars = len(numeric_vars)
    n_cols = min(3, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_vars == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, var in enumerate(numeric_vars):
        values = ds[var].values.flatten()
        valid_values = values[~np.isnan(values)]
        
        if len(valid_values) > 0:
            # Subsample for large datasets
            if len(valid_values) > 100000:
                valid_values = np.random.choice(valid_values, 100000, replace=False)
            
            axes[i].hist(valid_values, bins=50, edgecolor='white', alpha=0.7)
            axes[i].set_title(f'{var}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
            
            # Add mean line
            mean_val = np.mean(valid_values)
            axes[i].axvline(mean_val, color='red', linestyle='--', 
                           label=f'Mean: {mean_val:.2f}')
            axes[i].legend(fontsize=8)
    
    # Hide empty subplots
    for i in range(n_vars, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "distributions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'distributions.png'}")


def plot_spatial_maps(ds):
    """Create spatial maps for variables with lat/lon dimensions."""
    print(f"\n{'='*60}")
    print("CREATING SPATIAL MAPS")
    print(f"{'='*60}")
    
    # Find lat/lon coordinates
    lat_names = ['lat', 'latitude', 'LAT', 'LATITUDE', 'y']
    lon_names = ['lon', 'longitude', 'LON', 'LONGITUDE', 'x']
    
    lat_coord = None
    lon_coord = None
    
    for name in lat_names:
        if name in ds.coords or name in ds.dims:
            lat_coord = name
            break
    
    for name in lon_names:
        if name in ds.coords or name in ds.dims:
            lon_coord = name
            break
    
    if lat_coord is None or lon_coord is None:
        print("  No lat/lon coordinates found for spatial mapping.")
        return
    
    # Find variables with spatial dimensions
    spatial_vars = []
    for var in ds.data_vars:
        if lat_coord in ds[var].dims and lon_coord in ds[var].dims:
            spatial_vars.append(var)
    
    if not spatial_vars:
        print("  No variables with spatial dimensions found.")
        return
    
    for var in spatial_vars[:6]:  # Limit to first 6 variables
        fig, ax = plt.subplots(figsize=(12, 8))
        
        var_data = ds[var]
        
        # If there are extra dimensions (like time), take the mean or first slice
        extra_dims = [d for d in var_data.dims if d not in [lat_coord, lon_coord]]
        if extra_dims:
            # Take mean over extra dimensions
            plot_data = var_data.mean(dim=extra_dims)
        else:
            plot_data = var_data
        
        # Create the plot
        im = plot_data.plot(ax=ax, cmap='viridis', add_colorbar=True)
        ax.set_title(f'{var} - Spatial Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"spatial_map_{var}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {OUTPUT_DIR / f'spatial_map_{var}.png'}")


def plot_time_series(ds):
    """Create time series plots if time dimension exists."""
    print(f"\n{'='*60}")
    print("CREATING TIME SERIES PLOTS")
    print(f"{'='*60}")
    
    # Find time coordinate
    time_names = ['time', 'TIME', 't', 'date', 'datetime']
    time_coord = None
    
    for name in time_names:
        if name in ds.coords or name in ds.dims:
            time_coord = name
            break
    
    if time_coord is None:
        print("  No time dimension found.")
        return
    
    # Find variables with time dimension
    time_vars = []
    for var in ds.data_vars:
        if time_coord in ds[var].dims:
            time_vars.append(var)
    
    if not time_vars:
        print("  No variables with time dimension found.")
        return
    
    for var in time_vars[:6]:  # Limit to first 6 variables
        fig, ax = plt.subplots(figsize=(14, 5))
        
        var_data = ds[var]
        
        # Compute spatial mean if there are spatial dimensions
        other_dims = [d for d in var_data.dims if d != time_coord]
        if other_dims:
            plot_data = var_data.mean(dim=other_dims)
        else:
            plot_data = var_data
        
        plot_data.plot(ax=ax, linewidth=0.8)
        ax.set_title(f'{var} - Time Series (Spatial Mean)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel(var)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"timeseries_{var}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {OUTPUT_DIR / f'timeseries_{var}.png'}")


def plot_correlation_matrix(ds):
    """Create correlation matrix for numeric variables."""
    print(f"\n{'='*60}")
    print("CREATING CORRELATION MATRIX")
    print(f"{'='*60}")
    
    numeric_vars = [var for var in ds.data_vars 
                    if np.issubdtype(ds[var].dtype, np.number)]
    
    if len(numeric_vars) < 2:
        print("  Need at least 2 numeric variables for correlation.")
        return
    
    # Sample data for correlation (to handle large datasets)
    sample_data = {}
    for var in numeric_vars:
        values = ds[var].values.flatten()
        # Remove NaNs
        valid_mask = ~np.isnan(values)
        if np.sum(valid_mask) > 0:
            sample_data[var] = values[valid_mask]
    
    if len(sample_data) < 2:
        print("  Not enough valid data for correlation.")
        return
    
    # Find minimum length
    min_len = min(len(v) for v in sample_data.values())
    min_len = min(min_len, 50000)  # Cap at 50k samples
    
    # Create DataFrame with same-length samples
    df_data = {}
    for var, values in sample_data.items():
        if len(values) >= min_len:
            indices = np.random.choice(len(values), min_len, replace=False)
            df_data[var] = values[indices]
    
    if len(df_data) < 2:
        print("  Not enough matched data for correlation.")
        return
    
    df = pd.DataFrame(df_data)
    corr_matrix = df.corr()
    
    # Plot correlation matrix
    fig, ax = plt.subplots(figsize=(max(10, len(df_data)), max(8, len(df_data)*0.8)))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax)
    
    ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'correlation_matrix.png'}")
    
    # Save correlation to CSV
    corr_matrix.to_csv(OUTPUT_DIR / "correlation_matrix.csv")
    print(f"  Saved: {OUTPUT_DIR / 'correlation_matrix.csv'}")


def plot_boxplots(ds):
    """Create box plots for numeric variables."""
    print(f"\n{'='*60}")
    print("CREATING BOX PLOTS")
    print(f"{'='*60}")
    
    numeric_vars = [var for var in ds.data_vars 
                    if np.issubdtype(ds[var].dtype, np.number)]
    
    if not numeric_vars:
        print("  No numeric variables found.")
        return
    
    n_vars = len(numeric_vars)
    n_cols = min(4, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 5*n_rows))
    if n_vars == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, var in enumerate(numeric_vars):
        values = ds[var].values.flatten()
        valid_values = values[~np.isnan(values)]
        
        if len(valid_values) > 0:
            # Subsample for plotting
            if len(valid_values) > 10000:
                valid_values = np.random.choice(valid_values, 10000, replace=False)
            
            axes[i].boxplot(valid_values, vert=True)
            axes[i].set_title(f'{var}', fontsize=11, fontweight='bold')
            axes[i].set_ylabel('Value')
    
    # Hide empty subplots
    for i in range(n_vars, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "boxplots.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'boxplots.png'}")


def check_missing_data(ds):
    """Analyze and visualize missing data patterns."""
    print(f"\n{'='*60}")
    print("MISSING DATA ANALYSIS")
    print(f"{'='*60}")
    
    missing_info = []
    
    for var in ds.data_vars:
        var_data = ds[var]
        total = var_data.size
        
        if np.issubdtype(var_data.dtype, np.number):
            nan_count = int(np.sum(np.isnan(var_data.values)))
        else:
            nan_count = 0
        
        missing_pct = 100 * nan_count / total if total > 0 else 0
        
        missing_info.append({
            'Variable': var,
            'Total': total,
            'Missing': nan_count,
            'Missing %': missing_pct
        })
        
        print(f"  {var}: {nan_count:,} / {total:,} missing ({missing_pct:.2f}%)")
    
    # Create missing data bar plot
    if missing_info:
        df = pd.DataFrame(missing_info)
        
        fig, ax = plt.subplots(figsize=(12, max(6, len(missing_info)*0.4)))
        
        colors = ['#ff6b6b' if x > 10 else '#4ecdc4' if x > 0 else '#95e1d3' 
                  for x in df['Missing %']]
        
        bars = ax.barh(df['Variable'], df['Missing %'], color=colors, edgecolor='white')
        ax.set_xlabel('Missing Data (%)')
        ax.set_title('Missing Data by Variable', fontsize=14, fontweight='bold')
        ax.set_xlim(0, max(100, df['Missing %'].max() * 1.1))
        
        # Add percentage labels
        for bar, pct in zip(bars, df['Missing %']):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                   f'{pct:.1f}%', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "missing_data.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved: {OUTPUT_DIR / 'missing_data.png'}")


def main():
    """Main function to run the complete EDA."""
    print("\n" + "="*60)
    print("  CLIMATE DATA - EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # Load data
    ds = load_data(DATA_FILE)
    
    # Run all analyses
    print_dataset_overview(ds)
    compute_statistics(ds)
    check_missing_data(ds)
    
    # Create visualizations
    plot_distributions(ds)
    plot_boxplots(ds)
    plot_spatial_maps(ds)
    plot_time_series(ds)
    plot_correlation_matrix(ds)
    
    # Close dataset
    ds.close()
    
    print(f"\n{'='*60}")
    print("EDA COMPLETE!")
    print(f"{'='*60}")
    print(f"All outputs saved to: {OUTPUT_DIR.absolute()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

