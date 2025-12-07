import netCDF4 as nc
import pandas as pd
import numpy as np
import time as time_lib
import xarray as xr
import polars as pl
from typing import Union, List, Optional
import geopandas as gpd
from shapely.geometry import Point
import os
from scipy.ndimage import distance_transform_edt, binary_opening, binary_closing, binary_dilation, binary_erosion


def get_elevation_stats(nc_path: str, target_df: pd.DataFrame, window_sizes=[0.01, 0.05, 0.1, 0.25],
                        lat_col: str = "lat_rounded", lon_col: str = "lon_rounded",
                        elevation_var: str = "elevation") -> tuple:
    """
    Get elevation statistics for regions centered at specified coordinates,
    and computes gradient statistics (min, max, mean, std) for the elevation map
    for each window size.

    Args:
        nc_path (str): Path to the NetCDF elevation file.
        target_df (pd.DataFrame): DataFrame containing at least the coordinate columns.
        window_sizes (Union[float, List[float]]): Size(s) of the region in degrees (default 0.25).
        lat_col (str): Name of the latitude column in target_df (default "lat_rounded").
        lon_col (str): Name of the longitude column in target_df (default "lon_rounded").
        elevation_var (str): Name of the elevation variable in the NetCDF file. Defaults to "elevation".

    Returns:
        tuple:
            pd.DataFrame: DataFrame containing elevation and gradient statistics for each location
                          and window size.
            list: List of feature names (i.e. the DataFrame columns).
    """
    # Convert single window size to list if needed.
    if not isinstance(window_sizes, list):
        window_sizes = [window_sizes]

    # Extract latitude and longitude from the target DataFrame.
    lats = target_df[lat_col]
    lons = target_df[lon_col]

    with nc.Dataset(nc_path) as src:
        # Determine coordinate variable names dynamically
        lat_var_name = 'latitude' if 'latitude' in src.variables else 'lat'
        lon_var_name = 'longitude' if 'longitude' in src.variables else 'lon'
        
        if lat_var_name not in src.variables or lon_var_name not in src.variables:
            raise KeyError(f"Could not find latitude/longitude coordinates in {nc_path}")
        if elevation_var not in src.variables:
            raise KeyError(f"Variable '{elevation_var}' not found in {nc_path}")

        latitudes = src.variables[lat_var_name][:]
        longitudes = src.variables[lon_var_name][:]
        elevation_data = src.variables[elevation_var][:]
        
        # Squeeze data to remove singleton dimensions (e.g., time=[1])
        if elevation_data.ndim > 2:
            elevation_data = np.squeeze(elevation_data)
        
        if elevation_data.ndim != 2:
            raise ValueError(
                f"Variable '{elevation_var}' in {nc_path} is not a 2D array after squeezing. "
                f"Shape is {elevation_data.shape}."
            )

        # Ensure coordinates are ascending for np.searchsorted
        if latitudes[0] > latitudes[-1]:
            print("Info: Latitude coordinates are descending. Reversing data and coords.")
            latitudes = latitudes[::-1]
            elevation_data = np.flip(elevation_data, axis=0)

        if longitudes[0] > longitudes[-1]:
            print("Info: Longitude coordinates are descending. Reversing data and coords.")
            longitudes = longitudes[::-1]
            elevation_data = np.flip(elevation_data, axis=1)

        # Convert lat/lon to row/col indices.
        lat_indices = np.searchsorted(latitudes, lats.values)
        lon_indices = np.searchsorted(longitudes, lons.values)
        lat_indices_original = np.clip(lat_indices, 0, len(latitudes) - 1)
        lon_indices_original = np.clip(lon_indices, 0, len(longitudes) - 1)

        # Initialize results dictionary.
        stats_dict = {
            'latitude': lats.values,
            'longitude': lons.values,
        }
        
        # Safely add point-wise stats if they exist
        point_stats_mapping = {
            'elevation_point_stddev': 'elevation_stddev',
            'elevation_point_max': 'elevation_max',
            'elevation_point_min': 'elevation_min'
        }
        for feat_name, var_name in point_stats_mapping.items():
            if var_name in src.variables:
                stats_dict[feat_name] = src.variables[var_name][:][lat_indices_original, lon_indices_original]

        # Compute resolution.
        lat_res = np.abs(latitudes[1] - latitudes[0])
        lon_res = np.abs(longitudes[1] - longitudes[0])

        # Process each window size.
        for window_size in window_sizes:
            # Compute half window size, ensuring at least one pixel.
            half_window_rows = max(1, int(np.ceil((window_size / 2) / lat_res)))
            half_window_cols = max(1, int(np.ceil((window_size / 2) / lon_res)))

            # Pad the elevation data to handle edge cases.
            padded_data = np.pad(
                elevation_data,
                ((half_window_rows, half_window_rows),
                 (half_window_cols, half_window_cols)),
                mode='edge'
            )

            # Adjust indices for padding.
            lat_indices_padded = lat_indices_original + half_window_rows
            lon_indices_padded = lon_indices_original + half_window_cols

            # Create offsets for window extraction, ensuring inclusion of the center pixel.
            row_offsets = np.arange(-half_window_rows, half_window_rows + 1)
            col_offsets = np.arange(-half_window_cols, half_window_cols + 1)

            # Generate row and column indices.
            rows = lat_indices_padded[:, None] + row_offsets  # shape: (n, R)
            cols = lon_indices_padded[:, None] + col_offsets   # shape: (n, C)

            # Extract windows with proper broadcasting to get array of shape (n, R, C).
            data_windows = padded_data[rows[:, :, None], cols[:, None, :]]

            # Define suffix for the current window size in feature names.
            suffix = f'_{window_size}deg'
            # Compute basic elevation statistics.
            stats_dict.update({
                f'{elevation_var}_min{suffix}': np.min(data_windows, axis=(1, 2)),
                f'{elevation_var}_max{suffix}': np.max(data_windows, axis=(1, 2)),
                f'{elevation_var}_mean{suffix}': np.mean(data_windows, axis=(1, 2)),
                f'{elevation_var}_std{suffix}': np.std(data_windows, axis=(1, 2))
            })

            # Compute the gradient for each window.
            grad_lat, grad_lon = np.gradient(data_windows, lat_res, lon_res, axis=(1, 2))
            grad_magnitude = np.sqrt(grad_lat**2 + grad_lon**2)
            stats_dict.update({
                f'{elevation_var}_gradient_min{suffix}': np.min(grad_magnitude, axis=(1, 2)),
                f'{elevation_var}_gradient_max{suffix}': np.max(grad_magnitude, axis=(1, 2)),
                f'{elevation_var}_gradient_mean{suffix}': np.mean(grad_magnitude, axis=(1, 2)),
                f'{elevation_var}_gradient_std{suffix}': np.std(grad_magnitude, axis=(1, 2))
            })

        # Compile results into a DataFrame and extract feature names.
        stats = pd.DataFrame(stats_dict)
        feature_names = list(stats.columns)

    return stats, feature_names


def _add_features_with_renaming(
    target_df: pl.DataFrame, 
    new_features_df: pl.DataFrame, 
    source_file_path: str
) -> tuple[pl.DataFrame, list[str]]:
    """
    Adds columns from new_features_df to target_df, renaming to avoid conflicts.
    Returns the updated target_df and a list of the final names of added features.
    """
    added_feature_final_names = []
    
    # Create a dictionary for columns to be added to avoid direct modification issues
    # and to correctly handle renaming if new_features_df itself has conflicting names (though unlikely).
    current_batch_cols_to_add = {}

    for col_name in new_features_df.columns:
        final_col_name = col_name
        
        # Check if column name (original or after previous renaming in this batch) already exists
        if final_col_name in target_df.columns or final_col_name in current_batch_cols_to_add:
            base_name_from_source = os.path.basename(source_file_path).split('.')[0]
            sanitized_source_name = "".join(c if c.isalnum() else "_" for c in base_name_from_source)
            
            # Attempt to create a unique name
            original_name_to_prefix = col_name # Use the original col_name for generating variants
            suffix_idx = 1
            # First attempt: feature_source
            potential_name = f"{original_name_to_prefix}_{sanitized_source_name}"
            while potential_name in target_df.columns or potential_name in current_batch_cols_to_add:
                # Subsequent attempts: feature_source_idx
                potential_name = f"{original_name_to_prefix}_{sanitized_source_name}_{suffix_idx}"
                suffix_idx += 1
            final_col_name = potential_name
            print(f"Info: Renaming feature '{col_name}' from source '{source_file_path}' to '{final_col_name}' to avoid conflict.")
            
        current_batch_cols_to_add[final_col_name] = new_features_df[col_name] # This is a Polars Series
        added_feature_final_names.append(final_col_name)
    
    if current_batch_cols_to_add:
        # Create a new DataFrame from the potentially renamed columns
        renamed_new_features_pl_df = pl.DataFrame(current_batch_cols_to_add)
        target_df = target_df.hstack(renamed_new_features_pl_df)
        
    return target_df, added_feature_final_names


def _prefer_nearest_indices(
    coordinate_values: np.ndarray,
    requested: np.ndarray,
) -> np.ndarray:
    """Return the index of the nearest coordinate for each requested value."""

    idx = np.searchsorted(coordinate_values, requested)
    idx = np.clip(idx, 0, coordinate_values.size - 1)

    prev_idx = np.clip(idx - 1, 0, coordinate_values.size - 1)
    prev_dist = np.abs(coordinate_values[prev_idx] - requested)
    curr_dist = np.abs(coordinate_values[idx] - requested)
    use_prev = (prev_dist < curr_dist) & (idx > 0)
    return np.where(use_prev, prev_idx, idx)


def _process_netcdf_file(
    ds: xr.Dataset,
    lats_requested: np.ndarray,
    lons_requested: np.ndarray,
    exclude_keys: list,
    file_path: str,
) -> dict:
    """Helper function to process a single NetCDF file using vectorised NumPy lookups."""

    time_dim_name = next((dim for dim in ["time", "valid_time", "t"] if dim in ds.dims), None)
    if time_dim_name:
        print(f"Selecting last time step from dimension '{time_dim_name}'")
        ds = ds.isel({time_dim_name: -1})

    try:
        lat_coord_name = "latitude" if "latitude" in ds.coords else "lat"
        lon_coord_name = "longitude" if "longitude" in ds.coords else "lon"
    except KeyError:
        print(f"Warning: Could not find standard latitude/longitude coordinates in {file_path}. Skipping file.")
        return {}

    # Ensure coordinates are monotonically increasing so searchsorted works as expected
    try:
        if not np.all(np.diff(ds[lat_coord_name].values) >= 0):
            ds = ds.sortby(lat_coord_name)
        if not np.all(np.diff(ds[lon_coord_name].values) >= 0):
            ds = ds.sortby(lon_coord_name)
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(f"Warning: Failed to sort coordinates for {file_path}: {exc}")
        ds = ds.sortby(lat_coord_name)
        ds = ds.sortby(lon_coord_name)

    latitudes = ds[lat_coord_name].values
    longitudes = ds[lon_coord_name].values

    if latitudes.ndim != 1 or longitudes.ndim != 1:
        print(f"Warning: Unexpected coordinate dimensionality in {file_path}. Skipping file.")
        return {}

    lat_indices = _prefer_nearest_indices(latitudes, lats_requested)
    lon_indices = _prefer_nearest_indices(longitudes, lons_requested)

    data_vars = [var for var in ds.data_vars if var not in ds.coords and var not in exclude_keys]
    print(f"Found {len(data_vars)} data variables to extract from NetCDF: {file_path}")

    land_data_for_file: dict[str, np.ndarray] = {}
    for key in data_vars:
        data_array = ds[key]

        if lat_coord_name not in data_array.dims or lon_coord_name not in data_array.dims:
            print(f"Info: Skipping variable '{key}' in {file_path} because it lacks lat/lon dimensions.")
            continue

        # Remove trivial singleton dimensions and enforce lat/lon order for fast indexing.
        squeezed = data_array.squeeze(drop=True)
        try:
            ordered = squeezed.transpose(lat_coord_name, lon_coord_name)
        except ValueError as exc:
            print(f"Warning: Could not align dimensions for '{key}' in {file_path}: {exc}")
            continue

        values_2d = np.asarray(ordered)
        if values_2d.ndim != 2:
            print(f"Info: Skipping '{key}' in {file_path} due to unsupported dimensionality ({values_2d.shape}).")
            continue

        try:
            extracted = values_2d[lat_indices, lon_indices]
        except Exception as exc:
            print(f"Warning: Failed vector extraction for '{key}' in {file_path}: {exc}")
            continue

        land_data_for_file[key] = extracted

    return land_data_for_file


def _process_geopackage_file(
    gpkg_gdf,
    lats_requested,
    lons_requested,
    exclude_keys,
    file_path,
    lat_col,
    lon_col,
    col_names=["population"],
    radius_meters: Optional[float] = None,
):
    """Helper function to process a single GeoPackage file."""
    # Create points GeoDataFrame
    points_gdf = gpd.GeoDataFrame(
        {'original_index': np.arange(len(lats_requested)), lon_col: lons_requested, lat_col: lats_requested},
        geometry=gpd.points_from_xy(lons_requested, lats_requested),
        crs="EPSG:4326"
    )

    # Handle CRS
    if gpkg_gdf.crs is None:
        gpkg_gdf = gpkg_gdf.set_crs("EPSG:4326", allow_override=True)
    gpkg_gdf = gpkg_gdf.to_crs(points_gdf.crs) if gpkg_gdf.crs != points_gdf.crs else gpkg_gdf

    # Check which requested columns exist in the GeoPackage
    existing_cols = [col for col in col_names if col in gpkg_gdf.columns]
    missing_cols = [col for col in col_names if col not in gpkg_gdf.columns]
    
    if missing_cols:
        print(f"Warning: Column(s) {missing_cols} not found in {file_path}")
    
    if not existing_cols:
        print(f"Warning: None of the requested columns {col_names} found in {file_path}")
        return {}

    # Decide on spatial extraction strategy based on radius_meters
    cols_to_select = existing_cols + ["geometry"]

    if radius_meters is None or radius_meters <= 0:
        # Original behaviour: take the attribute of the polygon that contains the point
        joined_gdf = gpd.sjoin(points_gdf, gpkg_gdf[cols_to_select], how="left", predicate="within")
        joined_gdf = (
            joined_gdf.drop_duplicates(subset=["original_index"], keep="first").set_index("original_index")
        )

        result_dict = {}
        for col in existing_cols:
            aligned_series = joined_gdf[col].reindex(np.arange(len(lats_requested)))
            result_dict[col] = pd.to_numeric(aligned_series, errors="coerce").values
    else:
        # Buffer the points and compute average of intersecting polygons
        # Transform both layers to a metric CRS for buffering in meters.
        metric_crs = 3857  # EPSG:3857 – units in meters, adequate for buffering.

        points_metric = points_gdf.to_crs(epsg=metric_crs)
        gpkg_metric = gpkg_gdf.to_crs(epsg=metric_crs)

        # Create buffer polygons around each point
        points_metric["geometry"] = points_metric.geometry.buffer(radius_meters)

        # Spatial join using intersects to capture polygons intersecting the buffer
        joined_gdf = gpd.sjoin(points_metric, gpkg_metric[cols_to_select], how="left", predicate="intersects")

        # Aggregate by original_index (each original point) taking the mean of each attribute
        grouped = joined_gdf.groupby("original_index")[existing_cols].mean()

        # Reindex to ensure alignment with requested points
        result_dict = {}
        for col in existing_cols:
            aligned_series = grouped[col].reindex(np.arange(len(lats_requested)))
            result_dict[col] = pd.to_numeric(aligned_series, errors="coerce").values

    return result_dict


def prepare_land_data(
    land_data_files: List[str],
    target_df: pd.DataFrame,
    lat_col: str = "lat_rounded",
    lon_col: str = "lon_rounded",
    exclude_keys: List[str] = None,
    radius_meters: Optional[float] = None,
):
    """
    Loads and merges land data from multiple NetCDF or GeoPackage files for each row in the target DataFrame.
    For NetCDF: Extracts all available data variables (not coordinate variables).
                 Handles potentially different resolutions by finding the nearest grid point.
    For GeoPackage: Extracts attributes from vector layers by spatially joining with target points.
                    Assumes target points and GeoPackage geometries are in WGS-84 (EPSG:4326) or can be transformed.

    Parameters:
      land_data_files (List[str]): List of paths to the land data files (e.g., .nc, .gpkg files).
      target_df (pandas.DataFrame or similar): DataFrame containing target rows.
          Must have coordinate columns specified by lat_col and lon_col.
      lat_col (str): Column name for latitude in target_df (default "lat_rounded").
      lon_col (str): Column name for longitude in target_df (default "lon_rounded").
      exclude_keys (List[str]): Optional list of variable/attribute keys to exclude (default None).
      radius_meters (Optional[float]): Optional buffer radius in meters for GeoPackage processing (default None).

    Returns:
      Tuple containing:
        - pandas.DataFrame: The input DataFrame with additional columns for each data variable/attribute.
        - List[str]: Names of all new features/columns added to the DataFrame (after potential renaming).
    """
    start_time = time_lib.time()
    target_pl_df = pl.DataFrame(target_df)

    if exclude_keys is None:
        exclude_keys = []

    added_features = []

    lats_requested = target_pl_df[lat_col].to_numpy()
    lons_requested = target_pl_df[lon_col].to_numpy()

    for file_path in land_data_files:
        print(f"Processing data file: {file_path}")
        try:
            if file_path.lower().endswith(".nc"):
                with xr.open_dataset(file_path) as ds:
                    land_data_for_file = _process_netcdf_file(ds, lats_requested, lons_requested, exclude_keys, file_path)
                    if land_data_for_file:
                        new_features_pl_df = pl.DataFrame(land_data_for_file)
                        target_pl_df, new_names = _add_features_with_renaming(target_pl_df, new_features_pl_df, file_path)
                        added_features.extend(new_names)
                        print(f"Added {len(new_names)} features from {file_path}")
                    else:
                        print(f"Info: No data variables extracted from {file_path}")

            elif file_path.lower().endswith(".gpkg"):
                gpkg_gdf = gpd.read_file(file_path)
                gpkg_features_dict = _process_geopackage_file(
                    gpkg_gdf,
                    lats_requested,
                    lons_requested,
                    exclude_keys,
                    file_path,
                    lat_col,
                    lon_col,
                    radius_meters=radius_meters,
                )
                
                if gpkg_features_dict:
                    new_features_pl_df = pl.DataFrame(gpkg_features_dict)
                    if target_pl_df.height == new_features_pl_df.height:
                        target_pl_df, new_names = _add_features_with_renaming(target_pl_df, new_features_pl_df, file_path)
                        added_features.extend(new_names)
                        print(f"Added {len(new_names)} features from {file_path} (GeoPackage)")
                    else: 
                        print(f"Error: Row count mismatch when adding features from {file_path}. Expected {target_pl_df.height}, got {new_features_pl_df.height}. Skipping this file.")
                else:
                    print(f"Info: No data attributes extracted from {file_path} (GeoPackage)")
            
            else:
                print(f"Warning: Unsupported file type for {file_path} (expected .nc or .gpkg). Skipping.")

        except FileNotFoundError:
            print(f"Error: Data file not found: {file_path}. Skipping.")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}. Skipping.")

    total_time = time_lib.time() - start_time
    print(f"prepare_land_data function completed processing {len(land_data_files)} file(s) in {total_time:.2f} seconds")
    print(f"Total new features added (actual count after potential renaming): {len(added_features)}")

    result_df = target_pl_df.to_pandas()
    
    final_added_features = [feat for feat in added_features if feat in result_df.columns]
    if len(final_added_features) != len(added_features):
        print(f"Warning: Mismatch between tracked added features and columns in final DataFrame. Tracked: {len(added_features)}, Found: {len(final_added_features)}")

    if "slt" in result_df.columns:
        result_df["slt"] = result_df["slt"].fillna(99)
        result_df["slt"] = result_df["slt"].astype(int)
    


    print(f"Final list of added features: {final_added_features}")
    return result_df, final_added_features


def landsea_distance(
        df: pd.DataFrame,
        lat_col: str = "lat_rounded",
        lon_col: str = "lon_rounded"
) -> tuple[pd.DataFrame, list[str]]:
    """
    Calculates the distance from each point to the nearest coastline for both a
    standard and a dilated land-sea mask.

    It loads a land-sea mask, generates two distance maps (one standard, one dilated),
    caches them for future use, and then finds the distance for each point in the input DataFrame.
    The distance is positive for inland points and negative for points at sea.

    Args:
        df (pd.DataFrame): DataFrame with latitude and longitude columns.
        lat_col (str): Name of the latitude column.
        lon_col (str): Name of the longitude column.

    Returns:
        tuple[pd.DataFrame, list[str]]:
            - A DataFrame containing two new features: 'distance_to_coast_km' and 'distance_to_coast_dilated_km'.
            - A list with the names of the new features.
    """
    from scipy.ndimage import distance_transform_edt, binary_opening, binary_closing, binary_dilation
    import numpy as np

    # Path to the land-sea mask and the pre-calculated distance maps
    mask_path = "climate_data/land_data/IMERG_land_sea_mask.nc"
    dist_path = "climate_data/land_data/IMERG_land_sea_mask_distances.nc"
    feature_names = ["distance_to_coast_km", "distance_to_coast_dilated_km"]

    if os.path.exists(dist_path):
        print(f"Loading existing distance-to-coast maps from: {dist_path}")
        distance_ds = xr.open_dataset(dist_path)
    else:
        print("Distance-to-coast maps not found. Generating and saving them...")

        with xr.open_dataset(mask_path) as mask_ds:
            try:
                lat_coord, lon_coord = mask_ds['lat'], mask_ds['lon']
            except KeyError:
                lat_coord, lon_coord = mask_ds['latitude'], mask_ds['longitude']

            land_sea_mask = mask_ds['landseamask'].values.astype(np.int8)
            landseamask_da = mask_ds['landseamask']

            # 1. Cleaned Mask (non-dilated)
            # cleaned_mask = binary_opening(land_sea_mask)
            # cleaned_mask = binary_closing(cleaned_mask).astype(cleaned_mask.dtype)
            cleaned_mask = land_sea_mask
            # 2. Dilated Mask
            dilated_mask = binary_dilation(cleaned_mask.copy() )
            dilated_mask = binary_erosion(dilated_mask.copy())

            lat_res = abs(lat_coord[1] - lat_coord[0]).item()
            lon_res = abs(lon_coord[1] - lon_coord[0]).item()

            # --- Calculate distances for CLEANED mask ---
            dist_to_sea_clean = distance_transform_edt(cleaned_mask, sampling=[lat_res, lon_res])
            dist_to_land_clean = distance_transform_edt(1 - cleaned_mask, sampling=[lat_res, lon_res])
            distance_map_km_clean = (dist_to_sea_clean - dist_to_land_clean) * 111.1

            # --- Calculate distances for DILATED mask ---
            dist_to_sea_dilated = distance_transform_edt(dilated_mask, sampling=[lat_res, lon_res])
            dist_to_land_dilated = distance_transform_edt(1 - dilated_mask, sampling=[lat_res, lon_res])
            distance_map_km_dilated = (dist_to_sea_dilated - dist_to_land_dilated) * 111.1

            # Create DataArrays for both maps
            distance_da_clean = xr.DataArray(
                distance_map_km_clean,
                coords=landseamask_da.coords,
                dims=landseamask_da.dims,
                name=feature_names[0]
            )
            distance_da_dilated = xr.DataArray(
                distance_map_km_dilated,
                coords=landseamask_da.coords,
                dims=landseamask_da.dims,
                name=feature_names[1]
            )

            # Combine into a single dataset and save
            distance_ds = xr.Dataset({
                feature_names[0]: distance_da_clean,
                feature_names[1]: distance_da_dilated
            })
            distance_ds.to_netcdf(dist_path)
            print(f"Saved new distance-to-coast maps to: {dist_path}")


    lats_requested = df[lat_col].values
    lons_requested = df[lon_col].values

    try:
        ds_lats, ds_lons = distance_ds.lat.values, distance_ds.lon.values
    except AttributeError:
        ds_lats, ds_lons = distance_ds.latitude.values, distance_ds.longitude.values
    
    # --- Lookup for both features ---
    result_dict = {}
    for feat_name in feature_names:
        distance_data = distance_ds[feat_name].values

        lat_indices = np.searchsorted(ds_lats, lats_requested)
        lat_indices = np.clip(lat_indices, 0, len(ds_lats) - 1)
        lat_indices[lats_requested >= ds_lats[-1]] = len(ds_lats) - 1
        lat_indices_prev = np.clip(lat_indices - 1, 0, None)
        lat_dist_prev = np.abs(ds_lats[lat_indices_prev] - lats_requested)
        lat_dist_curr = np.abs(ds_lats[lat_indices] - lats_requested)
        lat_indices = np.where((lat_dist_prev < lat_dist_curr) & (lat_indices > 0), lat_indices_prev, lat_indices)

        lon_indices = np.searchsorted(ds_lons, lons_requested)
        lon_indices = np.clip(lon_indices, 0, len(ds_lons) - 1)
        lon_indices[lons_requested >= ds_lons[-1]] = len(ds_lons) - 1
        lon_indices_prev = np.clip(lon_indices - 1, 0, None)
        lon_dist_prev = np.abs(ds_lons[lon_indices_prev] - lons_requested)
        lon_dist_curr = np.abs(ds_lons[lon_indices] - lons_requested)
        lon_indices = np.where((lon_dist_prev < lon_dist_curr) & (lon_indices > 0), lon_indices_prev, lon_indices)

        distances = distance_data[lat_indices, lon_indices]
        result_dict[feat_name] = distances

    result_df = pd.DataFrame(result_dict, index=df.index)

    return result_df, feature_names


def assign_ecoregion(
        df: pd.DataFrame,
        lat_col: str = "lat_rounded",
        lon_col: str = "lon_rounded",
        wwf_shp: str = r"data/wwf_terr_ecos",
        out_cols: tuple[str, str] = ("ECO_NAME", "REALM"),
        how: str = "left"
    ) -> tuple[pd.DataFrame, list[str]]:
    """
    Spatially joins WWF terrestrial ecoregions to a point table.

    Parameters
    ----------
    df         : pandas.DataFrame
        Must contain latitude & longitude columns in decimal degrees (WGS-84).
    lat_col    : str, default 'lat'
        Column name for latitude.
    lon_col    : str, default 'lon'
        Column name for longitude.
    wwf_shp    : str
        Path to `wwf_terr_ecos.shp` (other component files must sit alongside).
    out_cols   : tuple[str, str]
        Two attribute fields from the WWF layer to copy into the result.
        Defaults to ('ECO_NAME', 'REALM').
    how        : {'left', 'inner'}, default 'left'
        Whether to keep all input rows (`left`) or only matches (`inner`).

    Returns
    -------
    tuple
        pandas.DataFrame: Only the new feature columns, with same index as input.
                         Rows falling outside any ecoregion get NaN in the new columns.
        list[str]: Names of the new feature columns added to the DataFrame.
    """
    # -------------------- 1. Load WWF ecoregions ---------------------------
    wwf = gpd.read_file(wwf_shp)[list(out_cols) + ["geometry"]]

    # Ensure the layer is in geographic WGS-84 (EPSG:4326)
    if wwf.crs is None:
        wwf.set_crs("EPSG:4326", inplace=True)
    elif wwf.crs.to_epsg() != 4326:
        wwf = wwf.to_crs(4326)

    # -------------------- 2. Build a GeoDataFrame of points ---------------
    gdf_points = gpd.GeoDataFrame(
        df.copy(),  # preserve >154 other columns
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326"
    )

    # -------------------- 3. Spatial join (vector overlay) ----------------
    joined = gpd.sjoin(gdf_points, wwf, how=how, predicate="within")

    # -------------------- 4. Extract only new features & return -----------
    # Extract only the new feature columns, preserving original index
    new_features = joined[list(out_cols)].copy()
    
    # Return the feature names that were added
    feature_names = list(out_cols)
    # Fill NaN with "Unknown"
    new_features = new_features.fillna("Unknown")
    # Change feature names: ECO_NAME -> ecoregion_name, REALM -> ecoregion_realm
    new_features.columns = new_features.columns.str.replace("ECO_NAME", "ecoregion_name").str.replace("REALM", "ecoregion_realm")
    feature_names = [feat.replace("ECO_NAME", "ecoregion_name").replace("REALM", "ecoregion_realm") for feat in feature_names]

    return new_features, feature_names
