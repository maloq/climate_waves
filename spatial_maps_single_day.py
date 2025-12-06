#!/usr/bin/env python3
"""
Create spatial maps for a single day for all climate variables.
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
DATA_FILE = "climate_data/features/features_2024.nc"
OUTPUT_DIR = Path("eda_output/single_day_maps")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Choose which day to plot (you can change this)
# Options: 'first', 'middle', 'last', or a specific date string like '2017-07-15'
DAY_SELECTION = 'middle'  # Will select July 2nd (middle of year)


def main():
    print("Loading data...")
    ds = xr.open_dataset(DATA_FILE)
    
    # Select the day
    dates = ds['date'].values
    if DAY_SELECTION == 'first':
        selected_date = dates[0]
        day_idx = 0
    elif DAY_SELECTION == 'middle':
        day_idx = len(dates) // 2
        selected_date = dates[day_idx]
    elif DAY_SELECTION == 'last':
        selected_date = dates[-1]
        day_idx = -1
    else:
        # Specific date
        selected_date = np.datetime64(DAY_SELECTION)
        day_idx = np.where(dates == selected_date)[0][0]
    
    date_str = str(selected_date)[:10]
    print(f"\nCreating spatial maps for: {date_str}")
    print(f"(Day {day_idx + 1} of {len(dates)})")
    print("=" * 60)
    
    # Get all data variables
    variables = list(ds.data_vars)
    
    # Create individual maps for each variable
    for var in variables:
        print(f"  Processing: {var}")
        
        # Select single day
        var_data = ds[var].sel(date=selected_date)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot
        im = var_data.plot(
            ax=ax,
            cmap='viridis',
            add_colorbar=True,
            cbar_kwargs={'label': get_units(ds, var)}
        )
        
        # Title with date
        title = f"{var}\n{date_str}"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Longitude (°E)')
        ax.set_ylabel('Latitude (°N)')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{var}_{date_str}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    # Also create a combined overview figure
    print("\nCreating combined overview figure...")
    create_combined_figure(ds, selected_date, date_str)
    
    ds.close()
    
    print(f"\n{'=' * 60}")
    print(f"All maps saved to: {OUTPUT_DIR.absolute()}")
    print(f"{'=' * 60}")


def get_units(ds, var):
    """Get units for a variable if available."""
    if 'units' in ds[var].attrs:
        return ds[var].attrs['units']
    elif 'temperature' in var.lower():
        return 'K'
    elif 'humidity' in var.lower():
        return 'kg/kg'
    elif 'wind' in var.lower():
        return 'm/s'
    elif 'geopotential' in var.lower() and 'z' not in var:
        return 'm²/s²'
    return ''


def create_combined_figure(ds, selected_date, date_str):
    """Create a combined figure with all variables in a grid."""
    variables = list(ds.data_vars)
    n_vars = len(variables)
    
    # Calculate grid size
    n_cols = 4
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    axes = axes.flatten()
    
    for i, var in enumerate(variables):
        var_data = ds[var].sel(date=selected_date)
        
        im = var_data.plot(
            ax=axes[i],
            cmap='viridis',
            add_colorbar=True,
            cbar_kwargs={'shrink': 0.8}
        )
        
        # Shorter title for combined view
        short_name = var.replace('_h00_', '\n').replace('_', ' ')
        axes[i].set_title(short_name, fontsize=10, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
    
    # Hide empty subplots
    for i in range(n_vars, len(axes)):
        axes[i].set_visible(False)
    
    fig.suptitle(f'All Climate Variables - {date_str}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"all_variables_combined_{date_str}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: all_variables_combined_{date_str}.png")


if __name__ == "__main__":
    main()

