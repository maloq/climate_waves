"""
Computationally efficient feature engineering for climate data.

This module provides vectorized operations for:
- Temporal lag features
- Exponential Weighted Moving (EWM) averages
- Spatial statistics (neighborhood means, std, gradients)

All operations are designed to work on xarray Datasets before conversion
to pandas DataFrames for maximum efficiency.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import warnings


import numpy as np
import xarray as xr
import os

try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback decorator when numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

try:
    from numpy.lib.stride_tricks import sliding_window_view
    HAS_SLIDING_VIEW = True
except ImportError:
    HAS_SLIDING_VIEW = False


# =============================================================================
# TEMPORAL LAG FEATURES
# =============================================================================

def compute_lag_features(
    ds: xr.Dataset,
    variables: List[str],
    lags: List[int],
    time_dim: str = "time",
    suffix_template: str = "_lag{lag}",
) -> xr.Dataset:
    """Compute temporal lag features for specified variables.
    
    Uses xarray's shift operation which is highly optimized and vectorized.
    
    Args:
        ds: Input dataset with time dimension
        variables: List of variable names to compute lags for
        lags: List of lag values (positive = past, e.g., [1, 3, 7] for 1, 3, 7 days ago)
        time_dim: Name of the time dimension
        suffix_template: Template for naming lag features (must contain {lag})
    
    Returns:
        Dataset with original variables plus lag features
    
    Example:
        >>> lags = [1, 3, 7]  # Yesterday, 3 days ago, 1 week ago
        >>> ds_with_lags = compute_lag_features(ds, ["temperature"], lags)
    """
    result_vars = {}
    
    for var in variables:
        if var not in ds.data_vars:
            warnings.warn(f"Variable '{var}' not found in dataset, skipping lags")
            continue
            
        data = ds[var]
        
        for lag in lags:
            if lag <= 0:
                warnings.warn(f"Lag must be positive, got {lag}, skipping")
                continue
                
            # shift along time axis (positive shift = look at past values)
            lagged = data.shift({time_dim: lag})
            new_name = f"{var}{suffix_template.format(lag=lag)}"
            result_vars[new_name] = lagged
    
    # Merge new variables with original dataset efficiently
    return ds.assign(**result_vars)


# =============================================================================
# EXPONENTIAL WEIGHTED MOVING (EWM) FEATURES  
# =============================================================================

def _ewm_numpy(data: np.ndarray, alpha: float, axis: int = 0) -> np.ndarray:
    """Compute EWM along specified axis using numpy (memory efficient).
    
    Uses the standard EWM formula: y_t = alpha * x_t + (1 - alpha) * y_{t-1}
    
    This is a forward-fill EWM where each value depends on all previous values.
    """
    result = np.empty_like(data)
    decay = 1 - alpha
    
    # Move axis to first position for iteration
    data = np.moveaxis(data, axis, 0)
    result = np.moveaxis(result, axis, 0)
    
    # Initialize with first value
    result[0] = data[0]
    
    # Vectorized EWM computation along first axis
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + decay * result[i - 1]
    
    # Move axis back
    result = np.moveaxis(result, 0, axis)
    return result


if HAS_NUMBA:
    @jit(nopython=True, parallel=True, cache=True)
    def _ewm_3d_numba(data: np.ndarray, alpha: float) -> np.ndarray:
        """Numba-accelerated EWM for 3D arrays (time, lat, lon).
        
        Parallelized over spatial dimensions for maximum throughput.
        """
        n_time, n_lat, n_lon = data.shape
        result = np.empty_like(data)
        decay = 1.0 - alpha
        
        for i in prange(n_lat):
            for j in range(n_lon):
                # Initialize
                result[0, i, j] = data[0, i, j]
                
                # EWM forward pass
                for t in range(1, n_time):
                    if np.isnan(data[t, i, j]):
                        result[t, i, j] = result[t - 1, i, j]
                    elif np.isnan(result[t - 1, i, j]):
                        result[t, i, j] = data[t, i, j]
                    else:
                        result[t, i, j] = alpha * data[t, i, j] + decay * result[t - 1, i, j]
        
        return result
else:
    def _ewm_3d_numba(data: np.ndarray, alpha: float) -> np.ndarray:
        """Fallback when numba is not available."""
        return _ewm_numpy(data, alpha, axis=0)


# =============================================================================
# SPATIAL STATISTICS KERNELS (NUMBA)
# =============================================================================

if HAS_NUMBA:
    @jit(nopython=True, parallel=True, cache=True)
    def _spatial_mean_3d_numba(data: np.ndarray, window_size: int, min_periods: int) -> np.ndarray:
        n_time, n_lat, n_lon = data.shape
        result = np.full_like(data, np.nan)
        radius = window_size // 2

        for t in prange(n_time):
            for i in range(n_lat):
                for j in range(n_lon):
                    i_start = max(0, i - radius)
                    i_end = min(n_lat, i + radius + 1)
                    j_start = max(0, j - radius)
                    j_end = min(n_lon, j + radius + 1)
                    
                    count = 0
                    acc = 0.0
                    
                    for wi in range(i_start, i_end):
                        for wj in range(j_start, j_end):
                            val = data[t, wi, wj]
                            if not np.isnan(val):
                                acc += val
                                count += 1
                                
                    if count >= min_periods:
                        result[t, i, j] = acc / count
        return result

    @jit(nopython=True, parallel=True, cache=True)
    def _spatial_std_3d_numba(data: np.ndarray, window_size: int, min_periods: int) -> np.ndarray:
        n_time, n_lat, n_lon = data.shape
        result = np.full_like(data, np.nan)
        radius = window_size // 2

        for t in prange(n_time):
            for i in range(n_lat):
                for j in range(n_lon):
                    i_start = max(0, i - radius)
                    i_end = min(n_lat, i + radius + 1)
                    j_start = max(0, j - radius)
                    j_end = min(n_lon, j + radius + 1)
                    
                    count = 0
                    acc = 0.0
                    acc_sq = 0.0
                    
                    for wi in range(i_start, i_end):
                        for wj in range(j_start, j_end):
                            val = data[t, wi, wj]
                            if not np.isnan(val):
                                acc += val
                                acc_sq += val * val
                                count += 1
                                
                    if count >= min_periods:
                        mean = acc / count
                        var = (acc_sq / count) - (mean * mean)
                        if var < 0:
                            var = 0.0  # Float precision issues
                        result[t, i, j] = np.sqrt(var)
        return result

    @jit(nopython=True, parallel=True, cache=True)
    def _spatial_min_3d_numba(data: np.ndarray, window_size: int, min_periods: int) -> np.ndarray:
        n_time, n_lat, n_lon = data.shape
        result = np.full_like(data, np.nan)
        radius = window_size // 2

        for t in prange(n_time):
            for i in range(n_lat):
                for j in range(n_lon):
                    i_start = max(0, i - radius)
                    i_end = min(n_lat, i + radius + 1)
                    j_start = max(0, j - radius)
                    j_end = min(n_lon, j + radius + 1)
                    
                    count = 0
                    min_val = np.inf
                    
                    for wi in range(i_start, i_end):
                        for wj in range(j_start, j_end):
                            val = data[t, wi, wj]
                            if not np.isnan(val):
                                if val < min_val:
                                    min_val = val
                                count += 1
                                
                    if count >= min_periods:
                        result[t, i, j] = min_val
        return result

    @jit(nopython=True, parallel=True, cache=True)
    def _spatial_max_3d_numba(data: np.ndarray, window_size: int, min_periods: int) -> np.ndarray:
        n_time, n_lat, n_lon = data.shape
        result = np.full_like(data, np.nan)
        radius = window_size // 2

        for t in prange(n_time):
            for i in range(n_lat):
                for j in range(n_lon):
                    i_start = max(0, i - radius)
                    i_end = min(n_lat, i + radius + 1)
                    j_start = max(0, j - radius)
                    j_end = min(n_lon, j + radius + 1)
                    
                    count = 0
                    max_val = -np.inf
                    
                    for wi in range(i_start, i_end):
                        for wj in range(j_start, j_end):
                            val = data[t, wi, wj]
                            if not np.isnan(val):
                                if val > max_val:
                                    max_val = val
                                count += 1
                                
                    if count >= min_periods:
                        result[t, i, j] = max_val
        return result



# =============================================================================
# SPATIAL STATISTICS KERNELS (NUMPY)
# =============================================================================

def _spatial_mean_3d_numpy(data: np.ndarray, window_size: int, min_periods: int) -> np.ndarray:
    pad = window_size // 2
    # Ensure pad valid
    if pad < 0: pad = 0
    padded = np.pad(data, ((0,0), (pad, pad), (pad, pad)), mode='constant', constant_values=np.nan)
    windows = sliding_window_view(padded, window_shape=(window_size, window_size), axis=(1, 2))
    
    if min_periods > 1:
        valid_counts = np.sum(~np.isnan(windows), axis=(-2, -1))
        mask = valid_counts < min_periods
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = np.nanmean(windows, axis=(-2, -1))
        
    if min_periods > 1:
        result[mask] = np.nan
        
    return result

def _spatial_std_3d_numpy(data: np.ndarray, window_size: int, min_periods: int) -> np.ndarray:
    pad = window_size // 2
    padded = np.pad(data, ((0,0), (pad, pad), (pad, pad)), mode='constant', constant_values=np.nan)
    windows = sliding_window_view(padded, window_shape=(window_size, window_size), axis=(1, 2))
    
    if min_periods > 1:
        valid_counts = np.sum(~np.isnan(windows), axis=(-2, -1))
        mask = valid_counts < min_periods
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = np.nanstd(windows, axis=(-2, -1))
        
    if min_periods > 1:
        result[mask] = np.nan
        
    return result

def _spatial_min_3d_numpy(data: np.ndarray, window_size: int, min_periods: int) -> np.ndarray:
    pad = window_size // 2
    padded = np.pad(data, ((0,0), (pad, pad), (pad, pad)), mode='constant', constant_values=np.nan)
    windows = sliding_window_view(padded, window_shape=(window_size, window_size), axis=(1, 2))
    
    if min_periods > 1:
        valid_counts = np.sum(~np.isnan(windows), axis=(-2, -1))
        mask = valid_counts < min_periods
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = np.nanmin(windows, axis=(-2, -1))
        
    if min_periods > 1:
        result[mask] = np.nan
        
    return result

def _spatial_max_3d_numpy(data: np.ndarray, window_size: int, min_periods: int) -> np.ndarray:
    pad = window_size // 2
    padded = np.pad(data, ((0,0), (pad, pad), (pad, pad)), mode='constant', constant_values=np.nan)
    windows = sliding_window_view(padded, window_shape=(window_size, window_size), axis=(1, 2))
    
    if min_periods > 1:
        valid_counts = np.sum(~np.isnan(windows), axis=(-2, -1))
        mask = valid_counts < min_periods
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = np.nanmax(windows, axis=(-2, -1))
        
    if min_periods > 1:
        result[mask] = np.nan
        
    return result
def compute_ewm_features(
    ds: xr.Dataset,
    variables: List[str],
    spans: List[int],
    time_dim: str = "time",
    suffix_template: str = "_ewm{span}",
    use_numba: bool = True,
) -> xr.Dataset:
    """Compute Exponential Weighted Moving averages for specified variables.
    
    Uses either numba-accelerated implementation (if available) or numpy.
    
    Args:
        ds: Input dataset with time dimension
        variables: List of variable names to compute EWM for
        spans: List of span values (equivalent to pandas span parameter)
               Higher span = smoother, more weight on older values
               alpha = 2 / (span + 1)
        time_dim: Name of the time dimension
        suffix_template: Template for naming EWM features
        use_numba: Whether to use numba acceleration (if available)
    
    Returns:
        Dataset with original variables plus EWM features
    
    Example:
        >>> spans = [3, 7, 14]  # 3-day, weekly, bi-weekly smoothing
        >>> ds_with_ewm = compute_ewm_features(ds, ["temperature"], spans)
    """
    result_vars = {}
    
    for var in variables:
        if var not in ds.data_vars:
            warnings.warn(f"Variable '{var}' not found in dataset, skipping EWM")
            continue
        
        data = ds[var]
        dims = data.dims
        
        # Find time axis position
        if time_dim not in dims:
            warnings.warn(f"Time dimension '{time_dim}' not found for {var}, skipping")
            continue
        time_axis = dims.index(time_dim)
        
        values = data.values
        
        for span in spans:
            if span <= 1:
                warnings.warn(f"Span must be > 1, got {span}, skipping")
                continue
            
            alpha = 2.0 / (span + 1)
            
            # Use numba for 3D arrays with time as first dimension
            if use_numba and HAS_NUMBA and len(dims) == 3 and time_axis == 0:
                ewm_values = _ewm_3d_numba(values.astype(np.float64), alpha)
            else:
                ewm_values = _ewm_numpy(values, alpha, axis=time_axis)
            
            new_name = f"{var}{suffix_template.format(span=span)}"
            result_vars[new_name] = xr.DataArray(
                ewm_values,
                dims=dims,
                coords=data.coords,
            )
    
    return ds.assign(**result_vars)


# =============================================================================
# SPATIAL STATISTICS FEATURES
# =============================================================================

def compute_spatial_stats(
    ds: xr.Dataset,
    variables: List[str],
    window_sizes: List[int],
    stats: List[str] = ["mean", "std"],
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
    suffix_template: str = "_{stat}{window}",
    min_periods: Optional[int] = None,
) -> xr.Dataset:
    """Compute spatial neighborhood statistics using rolling windows.
    
    Efficiently uses xarray's rolling operation with construct for vectorization.
    
    Args:
        ds: Input dataset with latitude and longitude dimensions
        variables: List of variable names to compute spatial stats for
        window_sizes: List of window sizes (e.g., [3, 5, 7] for 3x3, 5x5, 7x7)
        stats: List of statistics to compute: "mean", "std", "min", "max", "median"
        lat_dim: Name of latitude dimension
        lon_dim: Name of longitude dimension
        suffix_template: Template for naming (must contain {stat} and {window})
        min_periods: Minimum number of valid values required (defaults to 1)
    
    Returns:
        Dataset with original variables plus spatial statistics
    
    Example:
        >>> window_sizes = [3, 5]  # 3x3 and 5x5 neighborhoods
        >>> stats = ["mean", "std"]
        >>> ds_spatial = compute_spatial_stats(ds, ["temperature"], window_sizes, stats)
    """
    result_vars = {}
    
    for var in variables:
        if var not in ds.data_vars:
            warnings.warn(f"Variable '{var}' not found in dataset, skipping spatial stats")
            continue
        
        data = ds[var]
        dims = data.dims
        
        # Check dimensions
        if lat_dim not in data.dims or lon_dim not in data.dims:
            warnings.warn(f"Spatial dimensions not found for {var}, skipping")
            continue
        
        for window_size in window_sizes:
            if window_size < 3 or window_size % 2 == 0:
                warnings.warn(f"Window size should be odd >= 3, got {window_size}, skipping")
                continue
            
            min_p = min_periods if min_periods is not None else 1
            
            # Check if we can use numba (needs 3D array with time, lat, lon)
            # data.dims must match logic for 3D array extraction
            can_use_numba = False
            if HAS_NUMBA:
                # Need to ensure compatible layout
                # Usually (time, lat, lon) or similar.
                # If lat/lon are not last two, or not present, we can't easily use the fixed kernels above.
                # Logic: check if data is 3D and lat/lon are last two?
                # Or just move axes.
                
                # Check for dims presence
                try:
                    lat_idx = dims.index(lat_dim)
                    lon_idx = dims.index(lon_dim)
                    time_idx = -1
                    # Find time dim (any other dim)
                    remaining_dims = [d for d in dims if d not in (lat_dim, lon_dim)]
                    if len(remaining_dims) == 1:
                        time_idx = dims.index(remaining_dims[0])
                        
                    if len(dims) == 3 and time_idx != -1:
                        # Move axes to (time, lat, lon) for Numba kernels
                        values = data.transpose(dims[time_idx], lat_dim, lon_dim).values.astype(np.float64)
                        can_use_numba = True
                except ValueError:
                    pass
            # Check if we can use numpy view (needs 3D array with time, lat, lon)
            can_use_numpy = False
            if HAS_SLIDING_VIEW and not can_use_numba:
                 # Reuse data extraction logic
                try:
                    lat_idx = dims.index(lat_dim)
                    lon_idx = dims.index(lon_dim)
                    
                    # Ensure dims are present (handled above but good to be safe)
                    remaining_dims = [d for d in dims if d not in (lat_dim, lon_dim)]
                    time_idx = -1
                    if len(remaining_dims) == 1:
                        time_idx = dims.index(remaining_dims[0])
                        
                    if len(dims) == 3 and time_idx != -1:
                        values = data.transpose(dims[time_idx], lat_dim, lon_dim).values.astype(np.float64)
                        can_use_numpy = True
                except ValueError:
                    pass

            # Create rolling window object only if needed
            rolling = None
            if not can_use_numba and not can_use_numpy:
                rolling = data.rolling(
                    {lat_dim: window_size, lon_dim: window_size},
                    center=True,
                    min_periods=min_p,
                )
            
            for stat in stats:
                result = None
                
                if stat == "mean":
                    if can_use_numba:
                        result = _spatial_mean_3d_numba(values, window_size, min_p)
                    elif can_use_numpy:
                        result = _spatial_mean_3d_numpy(values, window_size, min_p)
                    elif rolling is not None:
                        result = rolling.mean().values
                elif stat == "std":
                    if can_use_numba:
                        result = _spatial_std_3d_numba(values, window_size, min_p)
                    elif can_use_numpy:
                        result = _spatial_std_3d_numpy(values, window_size, min_p)
                    elif rolling is not None:
                        result = rolling.std().values
                elif stat == "min":
                    if can_use_numba:
                        result = _spatial_min_3d_numba(values, window_size, min_p)
                    elif can_use_numpy:
                        result = _spatial_min_3d_numpy(values, window_size, min_p)
                    elif rolling is not None:
                        result = rolling.min().values
                elif stat == "max":
                    if can_use_numba:
                        result = _spatial_max_3d_numba(values, window_size, min_p)
                    elif can_use_numpy:
                        result = _spatial_max_3d_numpy(values, window_size, min_p)
                    elif rolling is not None:
                        result = rolling.max().values
                elif stat == "median":
                    # Median is hard to optimize with simple numba without sorting
                    # Stick to xarray rolling for median
                    if rolling is None:
                         rolling = data.rolling(
                            {lat_dim: window_size, lon_dim: window_size},
                            center=True,
                            min_periods=min_p,
                        )
                    result = rolling.median().values
                else:
                    warnings.warn(f"Unknown statistic '{stat}', skipping")
                    continue
                
                if result is not None:
                    new_name = f"{var}{suffix_template.format(stat=stat, window=window_size)}"
                    result_vars[new_name] = xr.DataArray(result, dims=dims, coords=data.coords)
    
    return ds.assign(**result_vars)


def compute_spatial_gradients(
    ds: xr.Dataset,
    variables: List[str],
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
    suffix_lat: str = "_grad_lat",
    suffix_lon: str = "_grad_lon",
    suffix_mag: str = "_grad_mag",
) -> xr.Dataset:
    """Compute spatial gradients (finite differences) for specified variables.
    
    Uses numpy's gradient function which handles edge cases properly.
    
    Args:
        ds: Input dataset
        variables: Variables to compute gradients for
        lat_dim: Latitude dimension name
        lon_dim: Longitude dimension name
        suffix_lat: Suffix for latitude gradient
        suffix_lon: Suffix for longitude gradient  
        suffix_mag: Suffix for gradient magnitude
    
    Returns:
        Dataset with gradient features added
    """
    result_vars = {}
    
    for var in variables:
        if var not in ds.data_vars:
            warnings.warn(f"Variable '{var}' not found, skipping gradients")
            continue
        
        data = ds[var]
        dims = data.dims
        
        if lat_dim not in dims or lon_dim not in dims:
            warnings.warn(f"Spatial dimensions not found for {var}, skipping gradients")
            continue
        
        lat_axis = dims.index(lat_dim)
        lon_axis = dims.index(lon_dim)
        
        values = data.values
        
        # Compute gradients using numpy (handles edges with second-order accuracy)
        lat_values = ds.coords[lat_dim].values
        lon_values = ds.coords[lon_dim].values
        
        # Approximate grid spacing (assuming regular grid)
        dlat = np.abs(np.mean(np.diff(lat_values)))
        dlon = np.abs(np.mean(np.diff(lon_values)))
        
        # Compute partial derivatives
        grad_lat = np.gradient(values, dlat, axis=lat_axis)
        grad_lon = np.gradient(values, dlon, axis=lon_axis)
        
        # Gradient magnitude
        grad_mag = np.sqrt(grad_lat**2 + grad_lon**2)
        
        result_vars[f"{var}{suffix_lat}"] = xr.DataArray(grad_lat, dims=dims, coords=data.coords)
        result_vars[f"{var}{suffix_lon}"] = xr.DataArray(grad_lon, dims=dims, coords=data.coords)
        result_vars[f"{var}{suffix_mag}"] = xr.DataArray(grad_mag, dims=dims, coords=data.coords)
    
        result_vars[f"{var}{suffix_mag}"] = xr.DataArray(grad_mag, dims=dims, coords=data.coords)
    
    return ds.assign(**result_vars)


def compute_anomalies(
    ds: xr.Dataset,
    variables: List[str],
    reference_ds: Optional[xr.Dataset] = None,
    time_dim: str = "time",
    suffix: str = "_anom",
) -> xr.Dataset:
    """Compute anomalies (deviation from climatological mean).
    
    If reference_ds is provided, uses its mean; otherwise computes mean from ds.
    Useful for removing seasonal signals and focusing on extreme deviations.
    
    Args:
        ds: Input dataset
        variables: Variables to compute anomalies for
        reference_ds: Optional reference dataset for computing mean
        time_dim: Time dimension name
        suffix: Suffix for anomaly features
    
    Returns:
        Dataset with anomaly features added
    """
    result_vars = {}
    ref = reference_ds if reference_ds is not None else ds
    
    for var in variables:
        if var not in ds.data_vars or var not in ref.data_vars:
            warnings.warn(f"Variable '{var}' not found, skipping anomaly")
            continue
        
        # Compute climatological mean across time
        clim_mean = ref[var].mean(dim=time_dim, skipna=True)
        
        # Compute anomaly
        anomaly = ds[var] - clim_mean
        result_vars[f"{var}{suffix}"] = anomaly
    
    return ds.assign(**result_vars)



# =============================================================================
# EXPLICIT ANOMALIES (CLIMATOLOGY)
# =============================================================================

def compute_daily_climatology_anomalies(
    ds: xr.Dataset,
    variables: List[str],
    climatology_path: Optional[str] = None,
    time_dim: str = "time",
    suffix: str = "_anom",
) -> xr.Dataset:
    """Compute anomalies by subtracting daily climatology.
    
    Args:
        ds: Input dataset
        variables: Variables to compute anomalies for
        climatology_path: Path to the climatology NetCDF file (optional)
        time_dim: Time dimension name
        suffix: Suffix for anomaly features
    
    Returns:
        Dataset with anomaly features
    """
    result_vars = {}
    
    climatology_ds = None
    if climatology_path and os.path.exists(climatology_path):
        try:
            climatology_ds = xr.open_dataset(climatology_path)
            # Ensure dayofyear is the dimension
            if "dayofyear" not in climatology_ds.coords:
                 # Check if we can rename
                 pass
        except Exception as e:
            warnings.warn(f"Failed to load climatology from {climatology_path}: {e}")
    
    for var in variables:
        if var not in ds.data_vars:
            warnings.warn(f"Variable '{var}' not found, skipping climatology anomaly")
            continue
        
        data = ds[var]
        
        # FIX: Ensure time dimension is single chunk to avoid "chunks do not add up to shape" error
        # in dask groupby subtraction. Since we process year-by-year, this is memory-safe.
        if data.chunks is not None: 
            data = data.chunk({time_dim: -1})
        
        # Calculate dayofyear for the input data
        # We need to broadcast climatology to the input data shape based on dayofyear
        
        if climatology_ds is not None and var in climatology_ds:
            clim = climatology_ds[var]
            # Group by dayofyear is not strictly necessary if we can select
            # But xarray selection by dayofyear is tricky if dimensions don't match exactly
            # Easier approach: Use xarray's advanced indexing or groupby subtraction
            
            # This subtracts the mean corresponding to the dayofyear of each simple
            anomaly = data.groupby(f"{time_dim}.dayofyear") - clim
            # Drop the dayofyear coordinate added by groupby
            if "dayofyear" in anomaly.coords:
                anomaly = anomaly.drop_vars("dayofyear")
                
        else:
            # Fallback: Compute self-climatology (mean over loaded data)
            # Only useful if loading many years, otherwise might be biased
            warnings.warn(f"No external climatology for {var}, using self-mean (may be biased if data is short)")
            clim = data.groupby(f"{time_dim}.dayofyear").mean(dim=time_dim)
            anomaly = data.groupby(f"{time_dim}.dayofyear") - clim
            if "dayofyear" in anomaly.coords:
                anomaly = anomaly.drop_vars("dayofyear")

        result_vars[f"{var}{suffix}"] = anomaly
    
    return ds.assign(**result_vars)


# =============================================================================
# WIND CHILL INDEX
# =============================================================================

def compute_wind_chill(
    ds: xr.Dataset,
    temp_var: str,
    u_var: str,
    v_var: str,
    new_name: str = "wind_chill_index",
) -> xr.Dataset:
    """Compute Wind Chill Index.
    
    Formula: 13.12 + 0.6215*T - 11.37*V^0.16 + 0.3965*T*V^0.16
    Where T is Temperature in Celsius, V is Wind Speed in km/h.
    
    Args:
        ds: Input dataset
        temp_var: Name of temperature variable (assumed Kelvin if > 200, else Celsius)
        u_var: U component of wind (m/s)
        v_var: V component of wind (m/s)
        new_name: Name of the output feature
        
    Returns:
        Dataset with wind chill feature
    """
    if not all(v in ds.data_vars for v in [temp_var, u_var, v_var]):
        warnings.warn(f"Missing variables for wind chill ({temp_var}, {u_var}, {v_var}), skipping")
        return ds

    T = ds[temp_var]
    U = ds[u_var]
    V_vec = ds[v_var]
    
    # 1. Convert Temperature to Celsius if needed
    # Simple heuristic: if mean is > 200, it's likely Kelvin
    if T.mean() > 200:
        T_c = T - 273.15
    else:
        T_c = T
        
    # 2. Compute Wind Magnitude (Speed)
    V_ms = np.sqrt(U**2 + V_vec**2)
    
    # 3. Convert Wind Speed to km/h
    V_kmh = V_ms * 3.6
    
    # 4. Limit V for valid formula application? 
    # The formula is technically valid for V > 4.8 km/h and T < 10C, 
    # but we'll apply it everywhere as a continuous feature for ML.
    # To avoid complex numbers with negative base (unlikely for magnitude), ensure V >= 0.
    # To avoid division by zero or weird power behavior near 0, clip V? 
    # V^0.16 is safe for V >= 0.
    
    V_pow = V_kmh ** 0.16
    
    wchill = 13.12 + 0.6215 * T_c - 11.37 * V_pow + 0.3965 * T_c * V_pow
    
    return ds.assign(**{new_name: wchill})


# =============================================================================
# TEMPERATURE ADVECTION
# =============================================================================

def compute_temperature_advection(
    ds: xr.Dataset,
    temp_var: str,
    u_var: str,
    v_var: str,
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
    new_name: str = "temp_advection",
) -> xr.Dataset:
    """Compute Temperature Advection: - (u * dT/dx + v * dT/dy).
    
    Requires converting spatial gradients from degrees to meters.
    
    Args:
        ds: Input dataset
        temp_var: Temperature variable
        u_var: U component of wind (m/s)
        v_var: V component of wind (m/s)
        lat_dim: Latitude dimension name
        lon_dim: Longitude dimension name
        new_name: Output feature name
        
    Returns:
        Dataset with advection feature
    """
    if not all(v in ds.data_vars for v in [temp_var, u_var, v_var]):
        warnings.warn(f"Missing variables for advection ({temp_var}, {u_var}, {v_var}), skipping")
        return ds
        
    T = ds[temp_var]
    U = ds[u_var] # m/s (zonal, West->East)
    V = ds[v_var] # m/s (meridional, South->North)
    
    if lat_dim not in ds.coords or lon_dim not in ds.coords:
        warnings.warn("Lat/Lon coordinates needed for advection, skipping")
        return ds

    # Gradients in coordinate space (dT/dlat, dT/dlon)
    # xarray's differentiate or numpy gradient
    # differentiate returns change per unit coordinate (per degree)
    
    dT_dlat = T.differentiate(lat_dim)
    dT_dlon = T.differentiate(lon_dim)
    
    # Conversion factors (degrees -> meters)
    # R_earth approx 6371000 m
    # dy = R * dlat (in radians)
    # dx = R * cos(lat) * dlon (in radians)
    
    # 1 deg = pi / 180 radians
    deg2rad = np.pi / 180.0
    R = 6371000.0
    
    meters_per_lat = R * deg2rad # Constant ~111km
    
    # meters_per_lon varies with latitude
    lat_rad = np.deg2rad(ds.coords[lat_dim])
    meters_per_lon = R * np.cos(lat_rad) * deg2rad
    
    # Gradients in meters: dT/dy = (dT/dlat) / (meters/degree)
    dT_dx = dT_dlon / meters_per_lon
    dT_dy = dT_dlat / meters_per_lat
    
    # Advection = - (u * dT/dx + v * dT/dy)
    # u is zonal (x), v is meridional (y)
    
    adv = - (U * dT_dx + V * dT_dy)
    
    return ds.assign(**{new_name: adv})


# =============================================================================
# REGIONAL / GLOBAL AGGREGATIONS
# =============================================================================

def compute_regional_features(
    ds: xr.Dataset,
    variables: List[str],
    stats: List[str] = ["mean", "std", "min", "max"],
    suffix_template: str = "_global_{stat}",
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
) -> xr.Dataset:
    """Compute global/regional aggregated features (e.g. mean over entire domain).
    
    Captures macro-scale context (e.g. is the whole region cold?).
    
    Args:
        ds: Input dataset
        variables: Variables to aggregate
        stats: List of statistics to compute ("mean", "std", "min", "max")
        suffix_template: Template for naming features
        lat_dim: Latitude dimension name
        lon_dim: Longitude dimension name
        
    Returns:
        Dataset with global feature channels (broadcast to original shape)
    """
    result_vars = {}
    
    for var in variables:
        if var not in ds.data_vars:
            warnings.warn(f"Variable '{var}' not found, skipping regional stats")
            continue
            
        data = ds[var]
        dims = data.dims
        
        # Check spatial dims
        if lat_dim not in dims or lon_dim not in dims:
            warnings.warn(f"Spatial dimensions not found for {var}, skipping regional stats")
            continue
            
        spatial_dims = (lat_dim, lon_dim)
        
        for stat in stats:
            # Compute scalar (per time step)
            if stat == "mean":
                agg = data.mean(dim=spatial_dims, skipna=True)
            elif stat == "std":
                agg = data.std(dim=spatial_dims, skipna=True)
            elif stat == "min":
                agg = data.min(dim=spatial_dims, skipna=True)
            elif stat == "max":
                agg = data.max(dim=spatial_dims, skipna=True)
            else:
                continue
                
            # Broadcast back to original shape
            # xarray handles broadcasting automatically when assigning to dataset with aligned coords
            # But we want to ensure it has the same dimensions as the original variable
            # so it can be merged/stacked properly.
            
            # Note: agg has lost lat/lon dims. We assign it.
            # xarray's broadcast mechanism:
            
            agg_broadcast, _ = xr.broadcast(agg, data)
            
            new_name = f"{var}{suffix_template.format(stat=stat)}"
            result_vars[new_name] = agg_broadcast
            
    return ds.assign(**result_vars)



# =============================================================================
# TEMPORAL CHANGE FEATURES (DIFFERENCES)
# =============================================================================

def compute_temporal_diffs(
    ds: xr.Dataset,
    variables: List[str],
    periods: List[int],
    time_dim: str = "time",
    suffix_template: str = "_diff{period}",
) -> xr.Dataset:
    """Compute temporal differences (change from N steps ago).
    
    Useful for capturing trends and rate of change in climate variables.
    
    Args:
        ds: Input dataset
        variables: Variables to compute differences for
        periods: List of periods for differencing (e.g., [1, 7] for daily and weekly change)
        time_dim: Time dimension name
        suffix_template: Template for naming difference features
    
    Returns:
        Dataset with difference features added
    """
    result_vars = {}
    
    for var in variables:
        if var not in ds.data_vars:
            warnings.warn(f"Variable '{var}' not found, skipping temporal diffs")
            continue
        
        data = ds[var]
        
        for period in periods:
            if period <= 0:
                warnings.warn(f"Period must be positive, got {period}, skipping")
                continue
            
            # Compute difference: current - past value
            diff = data - data.shift({time_dim: period})
            new_name = f"{var}{suffix_template.format(period=period)}"
            result_vars[new_name] = diff
    
    return ds.assign(**result_vars)


# =============================================================================
# TEMPORAL CYCLICAL FEATURES (SEASONALITY)
# =============================================================================

def compute_temporal_features(
    ds: xr.Dataset,
    time_dim: str = "time",
    include_month: bool = True,
    include_day_of_year: bool = True,
    include_sin_cos_annual: bool = True,
    include_sin_cos_semiannual: bool = False,
) -> xr.Dataset:
    """Compute temporal/cyclical features from the time coordinate.
    
    Creates features that capture seasonality and temporal position:
    - month: Month of year (1-12)
    - day_of_year: Day of year (1-365/366)
    - sin_annual, cos_annual: Cyclical encoding of annual seasonality
    - sin_semiannual, cos_semiannual: Cyclical encoding of 6-month cycles
    
    The sin/cos encoding is crucial for ML models as it properly represents
    the cyclical nature of time (Dec 31 is close to Jan 1).
    
    Args:
        ds: Input dataset with time dimension
        time_dim: Name of time dimension
        include_month: Whether to add month feature (1-12)
        include_day_of_year: Whether to add day_of_year feature (1-365)
        include_sin_cos_annual: Whether to add annual sin/cos features
        include_sin_cos_semiannual: Whether to add semi-annual sin/cos features
    
    Returns:
        Dataset with temporal features added as data variables
        (broadcast to match spatial dimensions)
    
    Example:
        >>> ds_with_time = compute_temporal_features(ds)
        >>> # Adds: month, day_of_year, sin_annual, cos_annual
    """
    result_vars = {}
    
    # Get time coordinate
    if time_dim not in ds.coords:
        warnings.warn(f"Time dimension '{time_dim}' not found, skipping temporal features")
        return ds
    
    time_values = ds.coords[time_dim].values
    
    # Convert to pandas for easy datetime extraction
    import pandas as pd
    time_index = pd.DatetimeIndex(time_values)
    
    # Get the shape we need to broadcast to (time, lat, lon or similar)
    # Find a data variable to get the target shape
    sample_var = next(iter(ds.data_vars), None)
    if sample_var is None:
        warnings.warn("No data variables found, skipping temporal features")
        return ds
    
    target_dims = ds[sample_var].dims
    target_shape = ds[sample_var].shape
    
    # Find time axis position
    if time_dim not in target_dims:
        warnings.warn(f"Time dimension not in data variables, skipping temporal features")
        return ds
    time_axis = target_dims.index(time_dim)
    
    # Create shape for broadcasting: 1 everywhere except time dimension
    broadcast_shape = [1] * len(target_dims)
    broadcast_shape[time_axis] = len(time_index)
    
    if include_month:
        # Month: 1-12
        month_values = time_index.month.values.reshape(broadcast_shape)
        month_broadcast = np.broadcast_to(month_values, target_shape)
        result_vars["month"] = xr.DataArray(
            month_broadcast.astype(np.int8),
            dims=target_dims,
            coords=ds[sample_var].coords,
        )
    
    if include_day_of_year:
        # Day of year: 1-365 (or 366)
        doy_values = time_index.dayofyear.values.reshape(broadcast_shape)
        doy_broadcast = np.broadcast_to(doy_values, target_shape)
        result_vars["day_of_year"] = xr.DataArray(
            doy_broadcast.astype(np.int16),
            dims=target_dims,
            coords=ds[sample_var].coords,
        )
    
    if include_sin_cos_annual:
        # Annual cycle: period = 365.25 days
        # sin and cos encoding ensures Dec 31 is close to Jan 1
        doy = time_index.dayofyear.values
        annual_phase = 2 * np.pi * doy / 365.25
        
        sin_annual = np.sin(annual_phase).reshape(broadcast_shape)
        cos_annual = np.cos(annual_phase).reshape(broadcast_shape)
        
        result_vars["sin_annual"] = xr.DataArray(
            np.broadcast_to(sin_annual, target_shape).astype(np.float32),
            dims=target_dims,
            coords=ds[sample_var].coords,
        )
        result_vars["cos_annual"] = xr.DataArray(
            np.broadcast_to(cos_annual, target_shape).astype(np.float32),
            dims=target_dims,
            coords=ds[sample_var].coords,
        )
    
    if include_sin_cos_semiannual:
        # Semi-annual cycle: period = 182.625 days (6 months)
        # Captures phenomena with 2 peaks per year
        doy = time_index.dayofyear.values
        semiannual_phase = 2 * np.pi * doy / 182.625
        
        sin_semiannual = np.sin(semiannual_phase).reshape(broadcast_shape)
        cos_semiannual = np.cos(semiannual_phase).reshape(broadcast_shape)
        
        result_vars["sin_semiannual"] = xr.DataArray(
            np.broadcast_to(sin_semiannual, target_shape).astype(np.float32),
            dims=target_dims,
            coords=ds[sample_var].coords,
        )
        result_vars["cos_semiannual"] = xr.DataArray(
            np.broadcast_to(cos_semiannual, target_shape).astype(np.float32),
            dims=target_dims,
            coords=ds[sample_var].coords,
        )
    
    return ds.assign(**result_vars)


# =============================================================================
# ALL-IN-ONE FEATURE ENGINEERING
# =============================================================================

def engineer_features(
    ds: xr.Dataset,
    config: Dict,
    time_dim: str = "time",
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
) -> xr.Dataset:
    """Apply all configured feature engineering steps to a dataset.
    
    Args:
        ds: Input xarray Dataset
        config: Feature engineering configuration dict with keys:
            - lag: {"variables": [...], "lags": [1, 3, 7]}
            - ewm: {"variables": [...], "spans": [3, 7, 14]}
            - spatial: {"variables": [...], "window_sizes": [3, 5], "stats": ["mean", "std"]}
            - gradients: {"variables": [...]}
            - temporal_diff: {"variables": [...], "periods": [1, 7]}
            - anomaly: {"variables": [...]}
            - temporal: {"month": true, "day_of_year": true, "sin_cos_annual": true, "sin_cos_semiannual": false}
        time_dim: Name of time dimension
        lat_dim: Name of latitude dimension
        lon_dim: Name of longitude dimension
    
    Returns:
        Dataset with all engineered features
    
    Example config:
        fe_config = {
            "lag": {
                "variables": ["temperature_h00_lvl850"],
                "lags": [1, 3, 7],
            },
            "ewm": {
                "variables": ["temperature_h00_lvl850"],
                "spans": [3, 7, 14],
            },
            "spatial": {
                "variables": ["temperature_h00_lvl850"],
                "window_sizes": [3, 5],
                "stats": ["mean", "std"],
            },
            "gradients": {
                "variables": ["geopotential_h00_lvl850"],
            },
            "temporal_diff": {
                "variables": ["temperature_h00_lvl850"],
                "periods": [1, 3],
            },
            "temporal": {
                "month": True,
                "day_of_year": True,
                "sin_cos_annual": True,
            },
        }
    """
    result = ds
    
    # 0. Temporal/seasonality features (add first as they don't depend on other features)
    if "temporal" in config:
        temporal_cfg = config["temporal"]
        result = compute_temporal_features(
            result,
            time_dim=time_dim,
            include_month=temporal_cfg.get("month", True),
            include_day_of_year=temporal_cfg.get("day_of_year", True),
            include_sin_cos_annual=temporal_cfg.get("sin_cos_annual", True),
            include_sin_cos_semiannual=temporal_cfg.get("sin_cos_semiannual", False),
        )
    
    # 1. Lag features
    if "lag" in config:
        lag_cfg = config["lag"]
        result = compute_lag_features(
            result,
            variables=lag_cfg.get("variables", []),
            lags=lag_cfg.get("lags", []),
            time_dim=time_dim,
        )
    
    # 2. EWM features
    if "ewm" in config:
        ewm_cfg = config["ewm"]
        result = compute_ewm_features(
            result,
            variables=ewm_cfg.get("variables", []),
            spans=ewm_cfg.get("spans", []),
            time_dim=time_dim,
        )
    
    # 3. Spatial statistics
    if "spatial" in config:
        spatial_cfg = config["spatial"]
        result = compute_spatial_stats(
            result,
            variables=spatial_cfg.get("variables", []),
            window_sizes=spatial_cfg.get("window_sizes", []),
            stats=spatial_cfg.get("stats", ["mean", "std"]),
            lat_dim=lat_dim,
            lon_dim=lon_dim,
        )
    
    # 4. Spatial gradients
    if "gradients" in config:
        grad_cfg = config["gradients"]
        result = compute_spatial_gradients(
            result,
            variables=grad_cfg.get("variables", []),
            lat_dim=lat_dim,
            lon_dim=lon_dim,
        )
    
    # 5. Temporal differences
    if "temporal_diff" in config:
        diff_cfg = config["temporal_diff"]
        result = compute_temporal_diffs(
            result,
            variables=diff_cfg.get("variables", []),
            periods=diff_cfg.get("periods", []),
            time_dim=time_dim,
        )
    
    # 6. Climatology Anomalies
    if "climatology" in config:
        clim_cfg = config["climatology"]
        result = compute_daily_climatology_anomalies(
            result,
            variables=clim_cfg.get("variables", []),
            climatology_path=clim_cfg.get("path", None),
            time_dim=time_dim,
        )

    # 7. Wind Chill
    if "wind_chill" in config:
        wc_cfg = config["wind_chill"]
        if wc_cfg.get("enabled", False):
            result = compute_wind_chill(
                result,
                temp_var=wc_cfg.get("temp_var", "temperature_h00_lvl1000"),
                u_var=wc_cfg.get("u_var", "u_component_of_wind_h00_lvl1000"),
                v_var=wc_cfg.get("v_var", "v_component_of_wind_h00_lvl1000"),
            )

    # 8. Advection
    if "advection" in config:
        adv_cfg = config["advection"]
        if adv_cfg.get("enabled", False):
            result = compute_temperature_advection(
                result,
                temp_var=adv_cfg.get("temp_var", "temperature_h00_lvl1000"),
                u_var=adv_cfg.get("u_var", "u_component_of_wind_h00_lvl1000"),
                v_var=adv_cfg.get("v_var", "v_component_of_wind_h00_lvl1000"),
                lat_dim=lat_dim,
                lon_dim=lon_dim,
            )

    # 9. Regional/Global Features [NEW]
    if "regional" in config:
        reg_cfg = config["regional"]
        result = compute_regional_features(
            result,
            variables=reg_cfg.get("variables", []),
            stats=reg_cfg.get("stats", ["mean", "std", "min", "max"]),
            lat_dim=lat_dim,
            lon_dim=lon_dim,
        )

    # 9. Simple Mean Anomalies (Legacy)
    if "anomaly" in config:
        anom_cfg = config["anomaly"]
        result = compute_anomalies(
            result,
            variables=anom_cfg.get("variables", []),
            time_dim=time_dim,
        )
    
    return result


# =============================================================================
# MEMORY-EFFICIENT CHUNK PROCESSING
# =============================================================================

def engineer_features_chunked(
    ds: xr.Dataset,
    config: Dict,
    chunk_size: int = 30,
    time_dim: str = "time",
    lat_dim: str = "latitude", 
    lon_dim: str = "longitude",
    overlap: int = 14,
) -> xr.Dataset:
    """Process feature engineering in temporal chunks for memory efficiency.
    
    Useful for very large datasets that don't fit in memory.
    
    Args:
        ds: Input dataset
        config: Feature engineering configuration
        chunk_size: Number of time steps per chunk
        time_dim: Time dimension name
        lat_dim: Latitude dimension name
        lon_dim: Longitude dimension name
        overlap: Number of overlapping time steps between chunks
                 (needed for lag/ewm features to be accurate)
    
    Returns:
        Dataset with engineered features
    """
    n_time = ds.dims[time_dim]
    
    if n_time <= chunk_size:
        # Small enough to process at once
        return engineer_features(ds, config, time_dim, lat_dim, lon_dim)
    
    # Determine maximum lookback needed
    max_lookback = 0
    if "lag" in config:
        max_lookback = max(max_lookback, max(config["lag"].get("lags", [0])))
    if "ewm" in config:
        # EWM needs ~3x span for reasonable accuracy
        max_span = max(config["ewm"].get("spans", [0]))
        max_lookback = max(max_lookback, max_span * 3)
    if "temporal_diff" in config:
        max_lookback = max(max_lookback, max(config["temporal_diff"].get("periods", [0])))
    
    # Ensure overlap covers lookback
    overlap = max(overlap, max_lookback)
    
    chunks = []
    start = 0
    
    while start < n_time:
        # Include overlap from previous chunk
        chunk_start = max(0, start - overlap)
        chunk_end = min(n_time, start + chunk_size)
        
        # Extract chunk
        chunk_ds = ds.isel({time_dim: slice(chunk_start, chunk_end)})
        
        # Process chunk
        chunk_result = engineer_features(chunk_ds, config, time_dim, lat_dim, lon_dim)
        
        # Trim overlap from beginning (except first chunk)
        if start > 0:
            trim_start = overlap
            chunk_result = chunk_result.isel({time_dim: slice(trim_start, None)})
        
        chunks.append(chunk_result)
        start += chunk_size
    
    # Concatenate chunks
    return xr.concat(chunks, dim=time_dim)


# =============================================================================
# UTILITY: FEATURE NAME GENERATION
# =============================================================================

def get_feature_names(
    base_features: List[str],
    config: Dict,
) -> List[str]:
    """Generate list of all feature names that will be created.
    
    Useful for understanding what features will be generated without
    actually running the computation.
    
    Args:
        base_features: List of original feature names
        config: Feature engineering configuration
    
    Returns:
        List of all feature names (original + engineered)
    """
    names = list(base_features)
    
    # Temporal/seasonality features
    if "temporal" in config:
        temporal_cfg = config["temporal"]
        if temporal_cfg.get("month", True):
            names.append("month")
        if temporal_cfg.get("day_of_year", True):
            names.append("day_of_year")
        if temporal_cfg.get("sin_cos_annual", True):
            names.append("sin_annual")
            names.append("cos_annual")
        if temporal_cfg.get("sin_cos_semiannual", False):
            names.append("sin_semiannual")
            names.append("cos_semiannual")
    
    if "lag" in config:
        for var in config["lag"].get("variables", []):
            if var in base_features:
                for lag in config["lag"].get("lags", []):
                    names.append(f"{var}_lag{lag}")
    
    if "ewm" in config:
        for var in config["ewm"].get("variables", []):
            if var in base_features:
                for span in config["ewm"].get("spans", []):
                    names.append(f"{var}_ewm{span}")
    
    if "spatial" in config:
        for var in config["spatial"].get("variables", []):
            if var in base_features:
                for window in config["spatial"].get("window_sizes", []):
                    for stat in config["spatial"].get("stats", ["mean", "std"]):
                        names.append(f"{var}_{stat}{window}")
    
    if "gradients" in config:
        for var in config["gradients"].get("variables", []):
            if var in base_features:
                names.append(f"{var}_grad_lat")
                names.append(f"{var}_grad_lon")
                names.append(f"{var}_grad_mag")
    
    if "temporal_diff" in config:
        for var in config["temporal_diff"].get("variables", []):
            if var in base_features:
                for period in config["temporal_diff"].get("periods", []):
                    names.append(f"{var}_diff{period}")
    
    if "anomaly" in config:
        for var in config["anomaly"].get("variables", []):
            if var in base_features:
                names.append(f"{var}_anom")
    
    if "climatology" in config:
        for var in config["climatology"].get("variables", []):
            if var in base_features:
                names.append(f"{var}_anom")

    if "wind_chill" in config:
        if config["wind_chill"].get("enabled", False):
            names.append("wind_chill_index")

    if "advection" in config:
        if config["advection"].get("enabled", False):
            names.append("temp_advection")

    if "regional" in config:
        for var in config["regional"].get("variables", []):
            if var in base_features:
                for stat in config["regional"].get("stats", ["mean", "std", "min", "max"]):
                    names.append(f"{var}_global_{stat}")

    return names

