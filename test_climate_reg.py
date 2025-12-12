from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd
import numpy as np
import xarray as xr
from catboost import CatBoostRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    balanced_accuracy_score, classification_report
)

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from load_climate_data import load_config, load_years, load_targets_lazy
from prepare_land import prepare_land_data, landsea_distance

# Try to import cartopy for geographic projections, fallback to simple plots
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Warning: cartopy not installed. Using simple lat/lon plots.")


def load_external_binary_target(
    target_dir: str,
    target_file_template: str,
    target_var: str,
    meta: pd.DataFrame,
    year: int,
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load external binary target using load_targets_lazy and map to metadata samples.
    
    Args:
        target_dir: Directory containing target files
        target_file_template: Template for target file names (e.g., "extemp_minus_7_{year}.nc")
        target_var: Variable name in the NetCDF file (e.g., "extemp_minus")
        meta: DataFrame with 'time', 'latitude', 'longitude' columns
        year: Year to load
        lat_range: Latitude range tuple
        lon_range: Longitude range tuple
        
    Returns:
        Tuple of (binary_target array, valid_mask array)
    """
    # Use the existing load_targets_lazy function
    ds = load_targets_lazy(
        years=[year],
        target_dir=target_dir,
        file_template=target_file_template,
        lat_range=lat_range,
        lon_range=lon_range
    )
    
    # Extract target data
    target_data = ds[target_var].values  # Shape: (time, lat, lon)
    target_times = pd.to_datetime(ds['time'].values)
    target_lats = ds['latitude'].values
    target_lons = ds['longitude'].values
    
    # Map to metadata samples
    meta_time = pd.to_datetime(meta["time"])
    lats = meta["latitude"].values
    lons = meta["longitude"].values
    
    # Build lookup indices using nearest neighbor matching
    lat_indices = np.abs(target_lats[:, None] - lats[None, :]).argmin(axis=0)
    lon_indices = np.abs(target_lons[:, None] - lons[None, :]).argmin(axis=0)
    
    # For time, match dates
    target_dates = pd.Series(target_times).dt.date.values
    meta_dates = meta_time.dt.date.values
    
    # Create a date-to-index mapping
    date_to_idx = {d: i for i, d in enumerate(target_dates)}
    
    # Map each meta date to the corresponding target index
    time_indices = np.array([date_to_idx.get(d, -1) for d in meta_dates])
    
    # Handle missing dates
    valid_mask = time_indices >= 0
    if not valid_mask.all():
        n_missing = (~valid_mask).sum()
        print(f"  Warning: {n_missing} samples have no matching date in binary target")
    
    # Extract values (set missing to 0 for now)
    binary_target = np.zeros(len(meta), dtype=np.int32)
    valid_indices = np.where(valid_mask)[0]
    
    for idx in valid_indices:
        t_idx = time_indices[idx]
        lat_idx = lat_indices[idx]
        lon_idx = lon_indices[idx]
        binary_target[idx] = int(target_data[t_idx, lat_idx, lon_idx])
    
    ds.close()
    return binary_target, valid_mask


def load_daily_average(climatology_path: str, meta: pd.DataFrame) -> np.ndarray:
    """
    Load daily average climatology and map to metadata samples.
    
    Args:
        climatology_path: Path to the daily average NetCDF file
        meta: DataFrame with 'time', 'latitude', 'longitude' columns
        
    Returns:
        Array of daily average temperatures for each sample in meta
    """
    ds = xr.open_dataset(climatology_path)
    
    # Get the temperature variable (usually 't2m')
    temp_var = list(ds.data_vars)[0]
    
    # Extract day of year from metadata
    meta_time = pd.to_datetime(meta["time"])
    day_of_year = meta_time.dt.dayofyear
    
    # Climatology has 365 days, handle leap year day 366
    day_of_year = day_of_year.clip(upper=365)
    
    # Get unique coordinates for efficient lookup
    lats = meta["latitude"].values
    lons = meta["longitude"].values
    
    # Extract climatology data
    clim_data = ds[temp_var].values  # Shape: (365, lat, lon)
    clim_lats = ds["latitude"].values
    clim_lons = ds["longitude"].values
    
    # Build lookup indices
    lat_indices = np.abs(clim_lats[:, None] - lats[None, :]).argmin(axis=0)
    lon_indices = np.abs(clim_lons[:, None] - lons[None, :]).argmin(axis=0)
    time_indices = day_of_year.values - 1  # 0-indexed
    
    # Vectorized lookup
    daily_avg = clim_data[time_indices, lat_indices, lon_indices]
    
    ds.close()
    return daily_avg


def binarize_for_classification(
    values: np.ndarray, 
    threshold: float = 10.0, 
    mode: str = "cold"
) -> np.ndarray:
    """
    Binarize anomaly values for classification metrics.
    
    For coldwaves: extreme = anomaly < -threshold (temp much lower than average)
    For heatwaves: extreme = anomaly > threshold (temp much higher than average)
    
    Args:
        values: Temperature anomaly values
        threshold: Absolute threshold for extreme classification
        mode: "cold" for coldwaves, "heat" for heatwaves
        
    Returns:
        Binary array (1 = extreme event, 0 = normal)
    """
    if mode == "cold":
        return (values < -threshold).astype(np.int32)
    else:  # heat
        return (values > threshold).astype(np.int32)


def compute_classification_metrics(
    y_true_binary: np.ndarray, 
    y_pred_binary: np.ndarray
) -> Dict:
    """Compute classification metrics from binarized predictions."""
    metrics = {}
    
    metrics["accuracy"] = (y_true_binary == y_pred_binary).mean()
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true_binary, y_pred_binary)
    metrics["precision"] = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    metrics["recall"] = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    metrics["f1"] = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel()
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    metrics["true_positives"] = int(tp)
    
    # Specificity
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return metrics


def find_optimal_threshold(
    y_true_anomaly: np.ndarray,
    y_pred_anomaly: np.ndarray,
    base_threshold: float = 10.0,
    mode: str = "cold",
    search_range: Tuple[float, float] = (-5.0, 5.0),
    n_steps: int = 50
) -> Tuple[float, float, Dict]:
    """
    Find the optimal prediction threshold offset that maximizes F1 score.
    
    The idea: ground truth uses threshold T (e.g., -10 for coldwaves).
    But predictions might be biased. We find an offset 'delta' such that
    binarizing predictions at (T + delta) gives best F1.
    
    Args:
        y_true_anomaly: Ground truth anomaly values
        y_pred_anomaly: Predicted anomaly values
        base_threshold: Base threshold for ground truth (e.g., 10 for coldwaves)
        mode: "cold" or "heat"
        search_range: Range of offsets to search (min, max)
        n_steps: Number of steps in the search
        
    Returns:
        optimal_offset: Best offset to add to prediction threshold
        best_f1: Best F1 score achieved
        best_metrics: Full metrics at optimal threshold
    """
    # Ground truth binary (fixed threshold)
    y_true_binary = binarize_for_classification(y_true_anomaly, base_threshold, mode)
    
    offsets = np.linspace(search_range[0], search_range[1], n_steps)
    best_f1 = 0
    best_offset = 0
    best_metrics = {}
    
    for offset in offsets:
        pred_threshold = base_threshold + offset
        y_pred_binary = binarize_for_classification(y_pred_anomaly, pred_threshold, mode)
        
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_offset = offset
            best_metrics = compute_classification_metrics(y_true_binary, y_pred_binary)
            best_metrics["threshold_offset"] = offset
            best_metrics["effective_threshold"] = pred_threshold
    
    return best_offset, best_f1, best_metrics


def calibrate_predictions(
    y_pred_anomaly: np.ndarray,
    y_true_anomaly: np.ndarray,
    method: str = "mean_shift"
) -> np.ndarray:
    """
    Calibrate predictions to reduce systematic bias.
    
    Args:
        y_pred_anomaly: Predicted anomaly values
        y_true_anomaly: Ground truth anomaly values
        method: Calibration method
            - "mean_shift": Shift predictions to match mean
            - "std_scale": Scale predictions to match std
            - "both": Apply both corrections
            
    Returns:
        Calibrated predictions
    """
    pred_mean = np.mean(y_pred_anomaly)
    true_mean = np.mean(y_true_anomaly)
    pred_std = np.std(y_pred_anomaly)
    true_std = np.std(y_true_anomaly)
    
    calibrated = y_pred_anomaly.copy()
    
    if method == "mean_shift":
        # Shift to match mean
        calibrated = calibrated - pred_mean + true_mean
    elif method == "std_scale":
        # Scale to match std (centered)
        calibrated = (calibrated - pred_mean) * (true_std / pred_std) + pred_mean
    elif method == "both":
        # Full standardization then rescale
        calibrated = (calibrated - pred_mean) / pred_std * true_std + true_mean
        
    return calibrated


def create_gridded_data(
    meta: pd.DataFrame,
    values: np.ndarray,
    agg_func: str = "mean"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate point data to a regular grid for plotting.
    
    Args:
        meta: DataFrame with 'latitude', 'longitude' columns
        values: Array of values to aggregate
        agg_func: Aggregation function ('mean', 'sum', 'count')
        
    Returns:
        lats, lons, gridded_values (2D arrays)
    """
    df = pd.DataFrame({
        "lat": meta["latitude"].values,
        "lon": meta["longitude"].values,
        "val": values
    })
    
    # Round to grid resolution (0.25 degrees typical for climate data)
    df["lat_grid"] = (df["lat"] * 4).round() / 4
    df["lon_grid"] = (df["lon"] * 4).round() / 4
    
    if agg_func == "mean":
        grouped = df.groupby(["lat_grid", "lon_grid"])["val"].mean()
    elif agg_func == "sum":
        grouped = df.groupby(["lat_grid", "lon_grid"])["val"].sum()
    elif agg_func == "count":
        grouped = df.groupby(["lat_grid", "lon_grid"])["val"].count()
    else:
        grouped = df.groupby(["lat_grid", "lon_grid"])["val"].mean()
    
    # Create regular grid
    unique_lats = sorted(df["lat_grid"].unique())
    unique_lons = sorted(df["lon_grid"].unique())
    
    grid = np.full((len(unique_lats), len(unique_lons)), np.nan)
    
    lat_to_idx = {lat: i for i, lat in enumerate(unique_lats)}
    lon_to_idx = {lon: j for j, lon in enumerate(unique_lons)}
    
    for (lat, lon), val in grouped.items():
        i = lat_to_idx[lat]
        j = lon_to_idx[lon]
        grid[i, j] = val
    
    lon_grid, lat_grid = np.meshgrid(unique_lons, unique_lats)
    
    return lat_grid, lon_grid, grid


def plot_map_simple(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    data: np.ndarray,
    title: str,
    cmap: str = "RdBu_r",
    vmin: float = None,
    vmax: float = None,
    cbar_label: str = "",
    save_path: Path = None
) -> None:
    """Simple lat/lon plot without cartopy."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if vmin is None:
        vmin = np.nanpercentile(data, 2)
    if vmax is None:
        vmax = np.nanpercentile(data, 98)
    
    # Symmetric colorbar for anomalies
    if "anomaly" in title.lower() or "error" in title.lower():
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max
    
    im = ax.pcolormesh(lon_grid, lat_grid, data, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(cbar_label)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_map_cartopy(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    data: np.ndarray,
    title: str,
    cmap: str = "RdBu_r",
    vmin: float = None,
    vmax: float = None,
    cbar_label: str = "",
    save_path: Path = None
) -> None:
    """Geographic plot with cartopy projections and coastlines."""
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    if vmin is None:
        vmin = np.nanpercentile(data, 2)
    if vmax is None:
        vmax = np.nanpercentile(data, 98)
    
    # Symmetric colorbar for anomalies
    if "anomaly" in title.lower() or "error" in title.lower():
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max
    
    im = ax.pcolormesh(
        lon_grid, lat_grid, data, 
        cmap=cmap, vmin=vmin, vmax=vmax, 
        transform=ccrs.PlateCarree(),
        shading='auto'
    )
    
    # Add geographic features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS, linewidth=0.5)
    
    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    
    ax.set_title(title, fontsize=14)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.05, orientation='horizontal')
    cbar.set_label(cbar_label)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_classification_map(
    meta: pd.DataFrame,
    y_true_bin: np.ndarray,
    y_pred_bin: np.ndarray,
    title: str,
    save_path: Path = None
) -> None:
    """
    Plot classification results on a map.
    Colors: TN=gray, TP=green, FP=red, FN=blue
    """
    # Compute classification category for each point
    # 0: TN (true=0, pred=0)
    # 1: TP (true=1, pred=1)
    # 2: FP (true=0, pred=1)
    # 3: FN (true=1, pred=0)
    categories = np.zeros(len(y_true_bin), dtype=np.int32)
    categories[(y_true_bin == 1) & (y_pred_bin == 1)] = 1  # TP
    categories[(y_true_bin == 0) & (y_pred_bin == 1)] = 2  # FP
    categories[(y_true_bin == 1) & (y_pred_bin == 0)] = 3  # FN
    # TN stays 0
    
    lat_grid, lon_grid, grid = create_gridded_data(meta, categories, agg_func="mean")
    
    # Custom colormap
    colors = ['#cccccc', '#2ecc71', '#e74c3c', '#3498db']  # TN, TP, FP, FN
    cmap = mcolors.ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    if HAS_CARTOPY:
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        
        im = ax.pcolormesh(
            lon_grid, lat_grid, grid,
            cmap=cmap, norm=norm,
            transform=ccrs.PlateCarree(),
            shading='auto'
        )
        
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
        
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
    else:
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.pcolormesh(lon_grid, lat_grid, grid, cmap=cmap, norm=norm, shading='auto')
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    
    ax.set_title(title, fontsize=14)
    
    # Custom colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.05, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(['TN', 'TP', 'FP', 'FN'])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_event_frequency_map(
    meta: pd.DataFrame,
    binary_values: np.ndarray,
    title: str,
    save_path: Path = None
) -> None:
    """Plot frequency of extreme events per grid cell."""
    lat_grid, lon_grid, grid = create_gridded_data(meta, binary_values.astype(float), agg_func="mean")
    
    # Convert to percentage
    grid = grid * 100
    
    plot_func = plot_map_cartopy if HAS_CARTOPY else plot_map_simple
    plot_func(
        lat_grid, lon_grid, grid,
        title=title,
        cmap="YlOrRd",
        vmin=0,
        vmax=min(50, np.nanmax(grid)),
        cbar_label="Event Frequency (%)",
        save_path=save_path
    )


def generate_prediction_maps(
    all_meta: pd.DataFrame,
    all_y_true_anomaly: np.ndarray,
    all_y_pred_anomaly: np.ndarray,
    all_y_true_bin: np.ndarray,
    all_y_pred_bin: np.ndarray,
    output_dir: Path,
    year_label: str = "all"
) -> None:
    """Generate all prediction maps."""
    print(f"\nGenerating maps for {year_label}...")
    
    maps_dir = output_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    
    plot_func = plot_map_cartopy if HAS_CARTOPY else plot_map_simple
    
    # 1. Mean Predicted Anomaly
    lat_grid, lon_grid, pred_grid = create_gridded_data(all_meta, all_y_pred_anomaly, "mean")
    plot_func(
        lat_grid, lon_grid, pred_grid,
        title=f"Mean Predicted Anomaly ({year_label})",
        cmap="RdBu_r",
        cbar_label="Temperature Anomaly (°C)",
        save_path=maps_dir / f"predicted_anomaly_{year_label}.png"
    )
    
    # 2. Mean True Anomaly
    lat_grid, lon_grid, true_grid = create_gridded_data(all_meta, all_y_true_anomaly, "mean")
    plot_func(
        lat_grid, lon_grid, true_grid,
        title=f"Mean True Anomaly ({year_label})",
        cmap="RdBu_r",
        cbar_label="Temperature Anomaly (°C)",
        save_path=maps_dir / f"true_anomaly_{year_label}.png"
    )
    
    # 3. Mean Error (Pred - True)
    errors = all_y_pred_anomaly - all_y_true_anomaly
    lat_grid, lon_grid, error_grid = create_gridded_data(all_meta, errors, "mean")
    plot_func(
        lat_grid, lon_grid, error_grid,
        title=f"Mean Prediction Error ({year_label})",
        cmap="RdBu_r",
        cbar_label="Error (°C)",
        save_path=maps_dir / f"error_{year_label}.png"
    )
    
    # 4. RMSE per grid cell
    squared_errors = errors ** 2
    lat_grid, lon_grid, mse_grid = create_gridded_data(all_meta, squared_errors, "mean")
    rmse_grid = np.sqrt(mse_grid)
    plot_func(
        lat_grid, lon_grid, rmse_grid,
        title=f"RMSE per Grid Cell ({year_label})",
        cmap="hot_r",
        vmin=0,
        cbar_label="RMSE (°C)",
        save_path=maps_dir / f"rmse_{year_label}.png"
    )
    
    # 5. True Event Frequency
    plot_event_frequency_map(
        all_meta, all_y_true_bin,
        title=f"True Extreme Event Frequency ({year_label})",
        save_path=maps_dir / f"true_events_{year_label}.png"
    )
    
    # 6. Predicted Event Frequency
    plot_event_frequency_map(
        all_meta, all_y_pred_bin,
        title=f"Predicted Extreme Event Frequency ({year_label})",
        save_path=maps_dir / f"predicted_events_{year_label}.png"
    )
    
    # 7. Classification Map (TP/TN/FP/FN)
    plot_classification_map(
        all_meta, all_y_true_bin, all_y_pred_bin,
        title=f"Classification Results ({year_label})",
        save_path=maps_dir / f"classification_{year_label}.png"
    )
    
    # 8. Recall per grid cell (TP / (TP + FN))
    tp_mask = (all_y_true_bin == 1) & (all_y_pred_bin == 1)
    fn_mask = (all_y_true_bin == 1) & (all_y_pred_bin == 0)
    
    # Calculate recall per grid cell
    df_recall = pd.DataFrame({
        "lat": all_meta["latitude"].values,
        "lon": all_meta["longitude"].values,
        "tp": tp_mask.astype(float),
        "fn": fn_mask.astype(float),
        "positive": all_y_true_bin.astype(float)
    })
    df_recall["lat_grid"] = (df_recall["lat"] * 4).round() / 4
    df_recall["lon_grid"] = (df_recall["lon"] * 4).round() / 4
    
    grouped = df_recall.groupby(["lat_grid", "lon_grid"]).agg({
        "tp": "sum", "positive": "sum"
    })
    grouped["recall"] = grouped["tp"] / grouped["positive"].replace(0, np.nan)
    
    # Rebuild grid
    unique_lats = sorted(df_recall["lat_grid"].unique())
    unique_lons = sorted(df_recall["lon_grid"].unique())
    recall_grid = np.full((len(unique_lats), len(unique_lons)), np.nan)
    lat_to_idx = {lat: i for i, lat in enumerate(unique_lats)}
    lon_to_idx = {lon: j for j, lon in enumerate(unique_lons)}
    
    for (lat, lon), row in grouped.iterrows():
        if not np.isnan(row["recall"]):
            i = lat_to_idx[lat]
            j = lon_to_idx[lon]
            recall_grid[i, j] = row["recall"]
    
    lon_grid_r, lat_grid_r = np.meshgrid(unique_lons, unique_lats)
    
    plot_func(
        lat_grid_r, lon_grid_r, recall_grid,
        title=f"Recall per Grid Cell ({year_label})",
        cmap="RdYlGn",
        vmin=0, vmax=1,
        cbar_label="Recall",
        save_path=maps_dir / f"recall_{year_label}.png"
    )


def test_model(config: Dict) -> None:
    data_cfg = config["data"]
    output_cfg = config["output"]
    bin_cfg = config.get("binarization", {})
    bin_cmp_cfg = config.get("binary_comparison", {})

    # Always evaluate on full test data (no subsampling)
    if data_cfg.get("sample_fraction", 1.0) < 1.0:
        print(
            f"Overriding sample_fraction from {data_cfg.get('sample_fraction')} to 1.0 for evaluation."
        )
        data_cfg["sample_fraction"] = 1.0
    
    print(f"\nTest years: {data_cfg['test_years']}")
    print(f"Loading model from: {output_cfg['model_path']}")
    
    model = CatBoostRegressor()
    model.load_model(output_cfg["model_path"])
    
    # Check if we should compute anomaly or use raw target
    climatology_path = bin_cfg.get("threshold_file")
    use_anomaly = climatology_path is not None and str(climatology_path).lower() not in ("none", "")
    
    if use_anomaly:
        print(f"Loading climatology from: {climatology_path}")
    else:
        print("Using raw target values (no threshold_file specified)")
    
    # Binarization settings
    binary_threshold = bin_cfg.get("threshold_val", 10.0)
    binary_mode = "cold" if bin_cfg.get("operator", "less") == "less" else "heat"
    threshold_optimization_enabled = bin_cfg.get("threshold_optimization", True)
    print(f"Binarization: threshold={binary_threshold}, mode={binary_mode}")
    print(f"Threshold optimization: {'enabled' if threshold_optimization_enabled else 'disabled'}")
    
    # External binary comparison settings
    bin_cmp_enabled = bin_cmp_cfg.get("enabled", False)
    if bin_cmp_enabled:
        bin_cmp_target_dir = bin_cmp_cfg.get("target_dir")
        bin_cmp_file_template = bin_cmp_cfg.get("target_file_template")
        bin_cmp_target_var = bin_cmp_cfg.get("target_var")
        print(f"\nExternal binary comparison enabled:")
        print(f"  Target dir: {bin_cmp_target_dir}")
        print(f"  File template: {bin_cmp_file_template}")
        print(f"  Target var: {bin_cmp_target_var}")
            
    # Accumulate metrics and data for plotting
    all_y_true_anomaly = []
    all_y_pred_anomaly = []
    all_y_true_bin = []
    all_y_pred_bin = []
    all_meta_list = []
    
    # For external binary comparison
    all_ext_bin_target = []
    all_ext_bin_valid_mask = []
    
    table_results = []
    ext_bin_results = []  # Separate table for external binary comparison
    year_data = {}  # Store per-year data for individual plots
    
    # Process year by year
    for year in data_cfg["test_years"]:
        print(f"\nProcessing Year {year}...")
        try:
            # Load data for this year
            X, y_raw, meta = load_years(
                [year], 
                config=config, 
                return_metadata=True, 
                apply_label_smoothing_flag=False,
                return_hard_labels=False,
                verbose=False
            )
        except Exception as e:
            print(f"Skipping {year} due to load error: {e}")
            continue
            
        if X.empty:
            print(f"No data for {year}.")
            continue

        # Compute anomaly (subtract daily average) or use raw values
        if use_anomaly:
            daily_avg = load_daily_average(climatology_path, meta)
            y_anomaly = y_raw.values - daily_avg
        else:
            y_anomaly = y_raw.values

        # Feature Integration (Must match training!)
        # 1. Distance
        meta_with_dist, dist_cols = landsea_distance(meta, lat_col="latitude", lon_col="longitude")
        for col in dist_cols:
            X[col] = meta_with_dist[col].values
            
        # 2. Land Data
        land_files = [
            "climate_data/land_data/forest_data.nc",
            "climate_data/land_data/GMTED2010_15n015_00625deg.nc"
        ]
        meta_with_land, land_cols = prepare_land_data(land_files, meta, lat_col="latitude", lon_col="longitude")
        for col in land_cols: 
            if col in meta_with_land.columns:
                X[col] = meta_with_land[col].values
            
        # Predict (model outputs anomaly)
        preds_anomaly = model.predict(X)
        
        # Binarize for classification metrics
        y_true_binary = binarize_for_classification(y_anomaly, binary_threshold, binary_mode)
        y_pred_binary = binarize_for_classification(preds_anomaly, binary_threshold, binary_mode)
        
        # Store for overall analysis
        all_y_true_anomaly.extend(y_anomaly)
        all_y_pred_anomaly.extend(preds_anomaly)
        all_y_true_bin.extend(y_true_binary)
        all_y_pred_bin.extend(y_pred_binary)
        all_meta_list.append(meta)
        
        # Store per-year data for individual plots
        year_data[year] = {
            "meta": meta,
            "y_true_anomaly": y_anomaly,
            "y_pred_anomaly": preds_anomaly,
            "y_true_bin": y_true_binary,
            "y_pred_bin": y_pred_binary
        }
        
        # Per-year metrics
        yr_rmse = np.sqrt(mean_squared_error(y_anomaly, preds_anomaly))
        yr_mae = mean_absolute_error(y_anomaly, preds_anomaly)
        yr_r2 = r2_score(y_anomaly, preds_anomaly)
        
        yr_cls_metrics = compute_classification_metrics(y_true_binary, y_pred_binary)
        yr_f1 = yr_cls_metrics["f1"]
        yr_recall = yr_cls_metrics["recall"]
        yr_precision = yr_cls_metrics["precision"]
        yr_bal_acc = yr_cls_metrics["balanced_accuracy"]
        
        print(f"  Anomaly RMSE: {yr_rmse:.4f}, MAE: {yr_mae:.4f}, R2: {yr_r2:.4f}")
        print(f"  Binary - F1: {yr_f1:.4f}, Recall: {yr_recall:.4f}, Precision: {yr_precision:.4f}")
        print(f"  Events: {y_pred_binary.sum():,} predicted / {y_true_binary.sum():,} actual")
        
        table_results.append({
            "Year": str(year),
            "RMSE": yr_rmse,
            "MAE": yr_mae,
            "R2": yr_r2,
            "F1": yr_f1,
            "Recall": yr_recall,
            "Precision": yr_precision,
            "Bal_Acc": yr_bal_acc,
            "Events_True": int(y_true_binary.sum()),
            "Events_Pred": int(y_pred_binary.sum())
        })
        
        # Load external binary target if enabled
        if bin_cmp_enabled:
            try:
                lat_range = tuple(data_cfg["latitude_range"])
                lon_range = tuple(data_cfg["longitude_range"])
                
                ext_bin_target, ext_valid_mask = load_external_binary_target(
                    target_dir=bin_cmp_target_dir,
                    target_file_template=bin_cmp_file_template,
                    target_var=bin_cmp_target_var,
                    meta=meta,
                    year=year,
                    lat_range=lat_range,
                    lon_range=lon_range
                )
                
                all_ext_bin_target.extend(ext_bin_target)
                all_ext_bin_valid_mask.extend(ext_valid_mask)
                
                # Store in year_data
                year_data[year]["ext_bin_target"] = ext_bin_target
                year_data[year]["ext_valid_mask"] = ext_valid_mask
                
                # Per-year external binary comparison (only on valid samples)
                valid_idx = ext_valid_mask
                if valid_idx.sum() > 0:
                    ext_metrics = compute_classification_metrics(
                        ext_bin_target[valid_idx], 
                        y_pred_binary[valid_idx]
                    )
                    print(f"  External Binary - F1: {ext_metrics['f1']:.4f}, Recall: {ext_metrics['recall']:.4f}, Precision: {ext_metrics['precision']:.4f}")
                    print(f"  External Events: {y_pred_binary[valid_idx].sum():,} predicted / {ext_bin_target[valid_idx].sum():,} actual")
                    
                    ext_bin_results.append({
                        "Year": str(year),
                        "F1": ext_metrics["f1"],
                        "Recall": ext_metrics["recall"],
                        "Precision": ext_metrics["precision"],
                        "Bal_Acc": ext_metrics["balanced_accuracy"],
                        "Events_True": int(ext_bin_target[valid_idx].sum()),
                        "Events_Pred": int(y_pred_binary[valid_idx].sum()),
                        "Valid_Samples": int(valid_idx.sum())
                    })
            except Exception as e:
                print(f"  Warning: Could not load external binary target for {year}: {e}")

    # Final Metrics
    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    
    if all_y_true_anomaly:
        all_y_true_anomaly = np.array(all_y_true_anomaly)
        all_y_pred_anomaly = np.array(all_y_pred_anomaly)
        all_y_true_bin = np.array(all_y_true_bin)
        all_y_pred_bin = np.array(all_y_pred_bin)
        all_meta = pd.concat(all_meta_list, ignore_index=True)
        
        rmse = np.sqrt(mean_squared_error(all_y_true_anomaly, all_y_pred_anomaly))
        mae = mean_absolute_error(all_y_true_anomaly, all_y_pred_anomaly)
        r2 = r2_score(all_y_true_anomaly, all_y_pred_anomaly)
        print(f"\nREGRESSION (on anomaly):")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R2:   {r2:.4f}")
        print(f"  Pred mean: {all_y_pred_anomaly.mean():.3f}, True mean: {all_y_true_anomaly.mean():.3f}")
        print(f"  Pred std:  {all_y_pred_anomaly.std():.3f}, True std:  {all_y_true_anomaly.std():.3f}")
    
        print(f"\nCLASSIFICATION (Binarized with threshold {binary_threshold}):")
        
        overall_metrics = compute_classification_metrics(all_y_true_bin, all_y_pred_bin)
        
        print(f"  Balanced Accuracy: {overall_metrics['balanced_accuracy']:.4f}")
        print(f"  Precision:         {overall_metrics['precision']:.4f}")
        print(f"  Recall:            {overall_metrics['recall']:.4f}")
        print(f"  F1 Score:          {overall_metrics['f1']:.4f}")
        print(f"  Specificity:       {overall_metrics['specificity']:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"    TN: {overall_metrics['true_negatives']:,}  FP: {overall_metrics['false_positives']:,}")
        print(f"    FN: {overall_metrics['false_negatives']:,}  TP: {overall_metrics['true_positives']:,}")
        
        print(f"\n  Total Events: {all_y_pred_bin.sum():,} predicted / {all_y_true_bin.sum():,} actual")
        
        # Initialize variables for optional optimization
        opt_metrics = None
        opt_pred_threshold = binary_threshold
        all_y_pred_bin_opt = all_y_pred_bin.copy()
        best_offset = 0.0
        
        # --- THRESHOLD OPTIMIZATION (Optional) ---
        if threshold_optimization_enabled:
            print("\n" + "=" * 60)
            print("THRESHOLD OPTIMIZATION")
            print("=" * 60)
            
            best_offset, best_f1, opt_metrics = find_optimal_threshold(
                all_y_true_anomaly, all_y_pred_anomaly,
                base_threshold=binary_threshold,
                mode=binary_mode,
                search_range=(-5.0, 5.0),
                n_steps=100
            )
            
            print(f"\nOptimal threshold offset: {best_offset:+.2f}")
            print(f"Effective prediction threshold: {binary_threshold + best_offset:.2f}")
            print(f"(Ground truth threshold remains: {binary_threshold})")
            print(f"\nWith OPTIMIZED threshold:")
            print(f"  F1 Score:          {opt_metrics['f1']:.4f} (was {overall_metrics['f1']:.4f})")
            print(f"  Balanced Accuracy: {opt_metrics['balanced_accuracy']:.4f}")
            print(f"  Precision:         {opt_metrics['precision']:.4f}")
            print(f"  Recall:            {opt_metrics['recall']:.4f}")
            print(f"\n  Confusion Matrix (optimized):")
            print(f"    TN: {opt_metrics['true_negatives']:,}  FP: {opt_metrics['false_positives']:,}")
            print(f"    FN: {opt_metrics['false_negatives']:,}  TP: {opt_metrics['true_positives']:,}")
            
            # --- CALIBRATION ---
            print("\n" + "=" * 60)
            print("PREDICTION CALIBRATION")
            print("=" * 60)
            
            for cal_method in ["mean_shift", "std_scale", "both"]:
                calibrated_preds = calibrate_predictions(all_y_pred_anomaly, all_y_true_anomaly, method=cal_method)
                
                # Evaluate calibrated predictions at original threshold
                cal_pred_bin = binarize_for_classification(calibrated_preds, binary_threshold, binary_mode)
                cal_metrics = compute_classification_metrics(all_y_true_bin, cal_pred_bin)
                
                # Also find optimal threshold for calibrated
                cal_offset, cal_best_f1, cal_opt_metrics = find_optimal_threshold(
                    all_y_true_anomaly, calibrated_preds,
                    base_threshold=binary_threshold,
                    mode=binary_mode
                )
                
                print(f"\nCalibration method: {cal_method}")
                print(f"  At original threshold ({binary_threshold}):")
                print(f"    F1: {cal_metrics['f1']:.4f}, Recall: {cal_metrics['recall']:.4f}, Precision: {cal_metrics['precision']:.4f}")
                print(f"    Events: {cal_pred_bin.sum():,} predicted")
                print(f"  With optimal threshold ({binary_threshold + cal_offset:.2f}):")
                print(f"    F1: {cal_opt_metrics['f1']:.4f}, Recall: {cal_opt_metrics['recall']:.4f}, Precision: {cal_opt_metrics['precision']:.4f}")
            
            # Store optimized predictions for maps
            opt_pred_threshold = binary_threshold + best_offset
            all_y_pred_bin_opt = binarize_for_classification(all_y_pred_anomaly, opt_pred_threshold, binary_mode)
            
            # Update year_data with optimized binary predictions
            for year, data in year_data.items():
                data["y_pred_bin_opt"] = binarize_for_classification(
                    data["y_pred_anomaly"], opt_pred_threshold, binary_mode
                )
        
        # Full classification report
        print("\n" + "=" * 60)
        print("DETAILED CLASSIFICATION REPORT (Original Threshold)")
        print("=" * 60)
        print(classification_report(all_y_true_bin, all_y_pred_bin, zero_division=0))
        
        # --- EXTERNAL BINARY TARGET COMPARISON ---
        if bin_cmp_enabled and all_ext_bin_target:
            print("\n" + "=" * 60)
            print("EXTERNAL BINARY TARGET COMPARISON")
            print("=" * 60)
            
            all_ext_bin_target_arr = np.array(all_ext_bin_target)
            all_ext_bin_valid_mask_arr = np.array(all_ext_bin_valid_mask)
            
            # Compute metrics only on valid samples
            valid_idx = all_ext_bin_valid_mask_arr
            n_valid = valid_idx.sum()
            n_total = len(valid_idx)
            
            print(f"\nValid samples: {n_valid:,} / {n_total:,} ({100*n_valid/n_total:.1f}%)")
            
            if n_valid > 0:
                ext_true = all_ext_bin_target_arr[valid_idx]
                ext_pred = all_y_pred_bin[valid_idx]
                
                ext_overall_metrics = compute_classification_metrics(ext_true, ext_pred)
                
                print(f"\nBinarized Regression vs External Binary Target:")
                print(f"  Balanced Accuracy: {ext_overall_metrics['balanced_accuracy']:.4f}")
                print(f"  Precision:         {ext_overall_metrics['precision']:.4f}")
                print(f"  Recall:            {ext_overall_metrics['recall']:.4f}")
                print(f"  F1 Score:          {ext_overall_metrics['f1']:.4f}")
                print(f"  Specificity:       {ext_overall_metrics['specificity']:.4f}")
                print(f"\n  Confusion Matrix:")
                print(f"    TN: {ext_overall_metrics['true_negatives']:,}  FP: {ext_overall_metrics['false_positives']:,}")
                print(f"    FN: {ext_overall_metrics['false_negatives']:,}  TP: {ext_overall_metrics['true_positives']:,}")
                print(f"\n  Total Events: {ext_pred.sum():,} predicted / {ext_true.sum():,} actual (external)")
                
                # Also compare with optimized threshold predictions (if optimization was enabled)
                ext_opt_metrics = None
                if threshold_optimization_enabled:
                    ext_pred_opt = all_y_pred_bin_opt[valid_idx]
                    ext_opt_metrics = compute_classification_metrics(ext_true, ext_pred_opt)
                    
                    print(f"\nWith OPTIMIZED prediction threshold ({opt_pred_threshold:.2f}):")
                    print(f"  Balanced Accuracy: {ext_opt_metrics['balanced_accuracy']:.4f}")
                    print(f"  Precision:         {ext_opt_metrics['precision']:.4f}")
                    print(f"  Recall:            {ext_opt_metrics['recall']:.4f}")
                    print(f"  F1 Score:          {ext_opt_metrics['f1']:.4f}")
                    print(f"\n  Confusion Matrix (optimized):")
                    print(f"    TN: {ext_opt_metrics['true_negatives']:,}  FP: {ext_opt_metrics['false_positives']:,}")
                    print(f"    FN: {ext_opt_metrics['false_negatives']:,}  TP: {ext_opt_metrics['true_positives']:,}")
                
                # Compare binarized regression target with external binary target
                binarized_reg_true = all_y_true_bin[valid_idx]
                target_agreement = compute_classification_metrics(ext_true, binarized_reg_true)
                
                print(f"\nAgreement between Binarized Regression Target and External Binary Target:")
                print(f"  Accuracy:          {target_agreement['accuracy']:.4f}")
                print(f"  F1 Score:          {target_agreement['f1']:.4f}")
                print(f"  Events: {binarized_reg_true.sum():,} (binarized reg) vs {ext_true.sum():,} (external)")
                
                # Full classification report for external comparison
                print("\n" + "-" * 40)
                print("Classification Report (vs External Binary Target):")
                print(classification_report(ext_true, ext_pred, zero_division=0))
                
                # Add to external binary results table
                ext_bin_results.append({
                    "Year": "ALL",
                    "F1": ext_overall_metrics["f1"],
                    "Recall": ext_overall_metrics["recall"],
                    "Precision": ext_overall_metrics["precision"],
                    "Bal_Acc": ext_overall_metrics["balanced_accuracy"],
                    "Events_True": int(ext_true.sum()),
                    "Events_Pred": int(ext_pred.sum()),
                    "Valid_Samples": int(n_valid)
                })
                
                # Add optimized row only if optimization was enabled
                if threshold_optimization_enabled and ext_opt_metrics is not None:
                    ext_bin_results.append({
                        "Year": "OPT",
                        "F1": ext_opt_metrics["f1"],
                        "Recall": ext_opt_metrics["recall"],
                        "Precision": ext_opt_metrics["precision"],
                        "Bal_Acc": ext_opt_metrics["balanced_accuracy"],
                        "Events_True": int(ext_true.sum()),
                        "Events_Pred": int(ext_pred_opt.sum()),
                        "Valid_Samples": int(n_valid)
                    })
    
    print("=" * 60)
    
    # --- Print Aggregated Table ---
    if all_y_true_anomaly is not None and len(all_y_true_anomaly) > 0:
        # Calculate GLOBAL metrics for the "ALL" row
        rmse_global = np.sqrt(mean_squared_error(all_y_true_anomaly, all_y_pred_anomaly))
        mae_global = mean_absolute_error(all_y_true_anomaly, all_y_pred_anomaly)
        r2_global = r2_score(all_y_true_anomaly, all_y_pred_anomaly)
        
        global_cls = compute_classification_metrics(all_y_true_bin, all_y_pred_bin)
            
        table_results.append({
            "Year": "ALL",
            "RMSE": rmse_global,
            "MAE": mae_global,
            "R2": r2_global,
            "F1": global_cls["f1"],
            "Recall": global_cls["recall"],
            "Precision": global_cls["precision"],
            "Bal_Acc": global_cls["balanced_accuracy"],
            "Events_True": int(all_y_true_bin.sum()),
            "Events_Pred": int(all_y_pred_bin.sum())
        })
        
        # Add optimized results row (only if optimization was enabled)
        if threshold_optimization_enabled and opt_metrics is not None:
            table_results.append({
                "Year": "OPT",
                "RMSE": rmse_global,
                "MAE": mae_global,
                "R2": r2_global,
                "F1": opt_metrics["f1"],
                "Recall": opt_metrics["recall"],
                "Precision": opt_metrics["precision"],
                "Bal_Acc": opt_metrics["balanced_accuracy"],
                "Events_True": int(all_y_true_bin.sum()),
                "Events_Pred": int(opt_metrics["true_positives"] + opt_metrics["false_positives"])
            })
        
        print("\n" + "=" * 100)
        print("AGGREGATED RESULTS TABLE")
        print("=" * 100)
        df_results = pd.DataFrame(table_results)
        
        # Format numeric columns
        format_cols = ["RMSE", "MAE", "R2", "F1", "Recall", "Precision", "Bal_Acc"]
        for col in format_cols:
            if col in df_results.columns:
                df_results[col] = df_results[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "-")
             
        print(df_results.to_string(index=False))
        print("=" * 100 + "\n")
        
        # Print external binary comparison table if available
        if bin_cmp_enabled and ext_bin_results:
            print("\n" + "=" * 100)
            print("EXTERNAL BINARY TARGET COMPARISON TABLE")
            print("=" * 100)
            df_ext_results = pd.DataFrame(ext_bin_results)
            
            format_cols_ext = ["F1", "Recall", "Precision", "Bal_Acc"]
            for col in format_cols_ext:
                if col in df_ext_results.columns:
                    df_ext_results[col] = df_ext_results[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "-")
            
            print(df_ext_results.to_string(index=False))
            print("=" * 100 + "\n")
        
    # Save predictions and generate plots
    pred_dir = output_cfg.get("predictions_dir")
    if pred_dir and all_y_true_anomaly is not None:
        pred_dir_path = Path(pred_dir)
        pred_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_path = pred_dir_path / "test_metrics.csv"
        pd.DataFrame(table_results).to_csv(metrics_path, index=False)
        print(f"Saved test metrics to {metrics_path}")
        
        # Save external binary comparison metrics if available
        if bin_cmp_enabled and ext_bin_results:
            ext_metrics_path = pred_dir_path / "test_metrics_ext_binary.csv"
            pd.DataFrame(ext_bin_results).to_csv(ext_metrics_path, index=False)
            print(f"Saved external binary comparison metrics to {ext_metrics_path}")
        
        # Generate maps for all years combined
        print("\n" + "=" * 60)
        print("GENERATING PREDICTION MAPS")
        print("=" * 60)
        
        generate_prediction_maps(
            all_meta,
            all_y_true_anomaly,
            all_y_pred_anomaly,
            all_y_true_bin,
            all_y_pred_bin,
            pred_dir_path,
            year_label="all_years"
        )
        
        # Generate maps for each year individually
        for year, data in year_data.items():
            generate_prediction_maps(
                data["meta"],
                data["y_true_anomaly"],
                data["y_pred_anomaly"],
                data["y_true_bin"],
                data["y_pred_bin"],
                pred_dir_path,
                year_label=str(year)
            )
        
        print(f"\nAll maps saved to {pred_dir_path / 'maps'}")


def main() -> None:
    print("=" * 60)
    print("TESTING CLIMATE REGRESSION (ANOMALY-BASED)")
    print("=" * 60)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_climate_reg_cold.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    test_model(config)


if __name__ == "__main__":
    main()
