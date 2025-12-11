from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import xarray as xr
from catboost import CatBoostRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    precision_score, recall_score, f1_score, confusion_matrix, 
    balanced_accuracy_score, roc_auc_score
)

from load_climate_data import load_config, load_years
from prepare_land import prepare_land_data, landsea_distance


def load_daily_average(climatology_path: str, meta: pd.DataFrame) -> np.ndarray:
    """
    Load daily average climatology and map to metadata samples.
    
    Args:
        climatology_path: Path to the daily average NetCDF file
        meta: DataFrame with 'time', 'latitude', 'longitude' columns
        
    Returns:
        Array of daily average temperatures for each sample in meta
    """
    print(f"  Loading daily average from: {climatology_path}")
    ds = xr.open_dataset(climatology_path)
    
    # Get the temperature variable (usually 't2m')
    temp_var = list(ds.data_vars)[0]
    print(f"  Using variable: {temp_var}")
    
    # Extract day of year from metadata
    meta_time = pd.to_datetime(meta["time"])
    day_of_year = meta_time.dt.dayofyear
    
    # Climatology has 365 days, handle leap year day 366
    day_of_year = day_of_year.clip(upper=365)
    
    # Get unique coordinates for efficient lookup
    lats = meta["latitude"].values
    lons = meta["longitude"].values
    
    # Create output array
    daily_avg = np.zeros(len(meta), dtype=np.float32)
    
    # Extract climatology data
    clim_data = ds[temp_var].values  # Shape: (365, lat, lon)
    clim_lats = ds["latitude"].values
    clim_lons = ds["longitude"].values
    
    # Build lookup indices
    # Find nearest lat/lon indices
    lat_indices = np.abs(clim_lats[:, None] - lats[None, :]).argmin(axis=0)
    lon_indices = np.abs(clim_lons[:, None] - lons[None, :]).argmin(axis=0)
    time_indices = day_of_year.values - 1  # 0-indexed
    
    # Vectorized lookup
    daily_avg = clim_data[time_indices, lat_indices, lon_indices]
    
    ds.close()
    return daily_avg


def compute_sample_weights(
    anomaly: np.ndarray,
    weight_normal: float = 1.0,
    weight_moderate: float = 2.0,
    weight_extreme: float = 5.0,
    weight_very_extreme: float = 10.0,
    threshold_moderate: float = 5.0,
    threshold_extreme: float = 10.0,
    threshold_very_extreme: float = 15.0
) -> np.ndarray:
    """
    Compute sample weights based on temperature anomaly extremity.
    More extreme temperatures get higher weights.
    
    Uses absolute anomaly so it works for both coldwaves (negative) 
    and heatwaves (positive).
    
    Args:
        anomaly: Temperature anomaly array (temp - daily_avg)
        weight_*: Weights for different extremity levels
        threshold_*: Absolute anomaly thresholds for each level
        
    Returns:
        Array of sample weights
    """
    abs_anomaly = np.abs(anomaly)
    weights = np.ones_like(anomaly, dtype=np.float32) * weight_normal
    
    # Moderate: 5 <= |anomaly| < 10
    mask_moderate = (abs_anomaly >= threshold_moderate) & (abs_anomaly < threshold_extreme)
    weights[mask_moderate] = weight_moderate
    
    # Extreme: 10 <= |anomaly| < 15 (cold/heat wave territory)
    mask_extreme = (abs_anomaly >= threshold_extreme) & (abs_anomaly < threshold_very_extreme)
    weights[mask_extreme] = weight_extreme
    
    # Very extreme: |anomaly| >= 15
    mask_very_extreme = abs_anomaly >= threshold_very_extreme
    weights[mask_very_extreme] = weight_very_extreme
    
    return weights


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


def train_model(config: Dict) -> None:
    data_cfg = config["data"]
    model_cfg = config["model"].copy()
    train_cfg = config["training"]
    output_cfg = config["output"]
    binarization_cfg = config.get("binarization", {})

    verbose = model_cfg.pop("verbose", 10)
    
    # Regression specific cleanups
    model_cfg.pop("use_class_weights", None)
    model_cfg.pop("scale_pos_weight", None)
    model_cfg.pop("class_weights", None)
    model_cfg.pop("decision_threshold", None)

    print(f"\nTrain years: {data_cfg['train_years']}")
    print(f"Output model path: {output_cfg['model_path']}")
    print("\nLoading training data...")
    
    # load_years returns X, y, (meta), (y_hard)
    # We don't need y_hard or label smoothing for regression training
    X, y, meta = load_years(
        data_cfg["train_years"], 
        config=config, 
        return_metadata=True,
        apply_label_smoothing_flag=False,  # No smoothing for regression
        return_hard_labels=False
    )
    
    print(f"Data loaded: {len(X):,} samples, {len(X.columns)} features")
    print(f"Raw target stats: min={y.min():.3f}, max={y.max():.3f}, mean={y.mean():.3f}, std={y.std():.3f}")

    # --- Subtract Daily Average to Create Anomaly ---
    print("\n" + "=" * 60)
    print("COMPUTING TEMPERATURE ANOMALY")
    print("=" * 60)
    
    climatology_path = binarization_cfg.get("threshold_file", "climate_data/target_reg/daily_average_cold.nc")
    daily_avg = load_daily_average(climatology_path, meta)
    
    # Store original target for reference
    y_raw = y.copy()
    
    # Compute anomaly: temperature - daily_average
    y_anomaly = y.values - daily_avg
    y = pd.Series(y_anomaly, index=y.index)
    
    print(f"  Daily average stats: min={daily_avg.min():.3f}, max={daily_avg.max():.3f}, mean={daily_avg.mean():.3f}")
    print(f"  Anomaly stats: min={y.min():.3f}, max={y.max():.3f}, mean={y.mean():.3f}, std={y.std():.3f}")
    
    # --- Compute Sample Weights ---
    print("\n" + "=" * 60)
    print("COMPUTING SAMPLE WEIGHTS")
    print("=" * 60)
    
    weight_cfg = config.get("sample_weights", {})
    sample_weights = compute_sample_weights(
        y.values,
        weight_normal=weight_cfg.get("weight_normal", 1.0),
        weight_moderate=weight_cfg.get("weight_moderate", 2.0),
        weight_extreme=weight_cfg.get("weight_extreme", 5.0),
        weight_very_extreme=weight_cfg.get("weight_very_extreme", 10.0),
        threshold_moderate=weight_cfg.get("threshold_moderate", 5.0),
        threshold_extreme=weight_cfg.get("threshold_extreme", 10.0),
        threshold_very_extreme=weight_cfg.get("threshold_very_extreme", 15.0)
    )
    
    # Print weight distribution
    unique_weights, counts = np.unique(sample_weights, return_counts=True)
    print("  Weight distribution:")
    for w, c in zip(unique_weights, counts):
        pct = 100.0 * c / len(sample_weights)
        print(f"    Weight {w:.1f}: {c:,} samples ({pct:.2f}%)")

    # --- Integrate Static Land Features ---
    print("\nIntegrating static land features...")
    
    # 1. Distance to Coast
    print("  Calculating distance to coast...")
    meta_with_dist, dist_cols = landsea_distance(meta, lat_col="latitude", lon_col="longitude")
    
    for col in dist_cols:
        X[col] = meta_with_dist[col].values
    print(f"  Added {len(dist_cols)} distance features: {dist_cols}")
    
    # 2. Other Land Data
    land_files = [
        "climate_data/land_data/forest_data.nc",
        "climate_data/land_data/GMTED2010_15n015_00625deg.nc"
    ]
    print(f"  Processing land data files: {land_files}")
    
    meta_with_land, land_cols = prepare_land_data(
        land_files, 
        meta, 
        lat_col="latitude", 
        lon_col="longitude"
    )
    
    for col in land_cols:
        if col in meta_with_land.columns:
            X[col] = meta_with_land[col].values
            
    print(f"  Added {len(land_cols)} features from land data files.")
    print(f"  Total features after integration: {len(X.columns)}")
    # --------------------------------------
    
    print("Splitting data into train/validation sets (Temporal Split)...")
    
    if "time" not in meta.columns:
        raise ValueError("Metadata 'time' column required for temporal splitting")
        
    if not np.issubdtype(meta["time"].dtype, np.datetime64):
        meta["time"] = pd.to_datetime(meta["time"])

    meta_time = meta["time"]
    all_years = sorted(meta_time.dt.year.unique())

    if len(all_years) < 2:
        raise ValueError(f"Need at least 2 distinct years for temporal split, found {len(all_years)}: {all_years}")

    val_size_cfg = train_cfg.get("validation_size", 0.1)
    if isinstance(val_size_cfg, float) and val_size_cfg < 1.0:
        val_year_count = max(1, int(np.ceil(len(all_years) * val_size_cfg)))
    else:
        val_year_count = int(val_size_cfg) if val_size_cfg is not None else 1

    if val_year_count >= len(all_years):
        val_year_count = len(all_years) - 1
        
    val_years = all_years[-val_year_count:]
    train_years_list = all_years[:-val_year_count]
    
    print(f"  Training years: {train_years_list}")
    print(f"  Validation years: {val_years}")
    
    val_mask = meta_time.dt.year.isin(val_years)
    train_mask = meta_time.dt.year.isin(train_years_list)
    
    X_train = X[train_mask].copy()
    X_val = X[val_mask].copy()
    
    y_train = y[train_mask].copy()
    y_val = y[val_mask].copy()
    
    weights_train = sample_weights[train_mask]
    
    meta_val = meta[val_mask].copy()

    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Features ({len(X_train.columns)}):")
    for i, feat in enumerate(X_train.columns, 1):
        print(f"  {i:2d}. {feat}")
    print(f"\nTraining samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Training with sample weights: Yes (effective weight sum: {weights_train.sum():.0f})")
    print("=" * 60 + "\n")

    print("Initializing CatBoost Regressor...")
    print(f"Training on device: {model_cfg.get('task_type', 'CPU')}")
    model = CatBoostRegressor(**model_cfg)
    print("Starting training...\n")
    
    # Train with sample weights
    model.fit(
        X_train, y_train, 
        sample_weight=weights_train,
        eval_set=(X_val, y_val), 
        verbose=verbose
    )

    # --- Evaluation ---
    print("\nEvaluating on validation set...")
    val_pred = model.predict(X_val)
    train_pred = model.predict(X_train)
    
    # Regression metrics
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_mae = mean_absolute_error(y_val, val_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    
    report_str = (
        "Regression Metrics (on anomaly):\n"
        f"Train RMSE: {train_rmse:.4f}\n"
        f"Val RMSE:   {val_rmse:.4f}\n"
        f"Val MAE:    {val_mae:.4f}\n"
        f"Val R2:     {val_r2:.4f}\n"
    )
    
    print("\n" + report_str)

    # --- Classification Metrics via Binarization ---
    print("\n" + "=" * 60)
    print("CLASSIFICATION METRICS (Binarized)")
    print("=" * 60)
    
    binary_threshold = binarization_cfg.get("threshold_val", 10.0)
    binary_mode = "cold" if binarization_cfg.get("operator", "less") == "less" else "heat"
    
    print(f"Binarization threshold: {binary_threshold}")
    print(f"Mode: {binary_mode} (extreme = anomaly {'< -' if binary_mode == 'cold' else '> '}{binary_threshold})")
    
    # Binarize ground truth and predictions
    y_val_binary = binarize_for_classification(y_val.values, binary_threshold, binary_mode)
    val_pred_binary = binarize_for_classification(val_pred, binary_threshold, binary_mode)
    
    y_train_binary = binarize_for_classification(y_train.values, binary_threshold, binary_mode)
    train_pred_binary = binarize_for_classification(train_pred, binary_threshold, binary_mode)
    
    print(f"\nValidation ground truth: {y_val_binary.sum():,} extreme events ({100*y_val_binary.mean():.2f}%)")
    print(f"Validation predictions:  {val_pred_binary.sum():,} extreme events ({100*val_pred_binary.mean():.2f}%)")
    
    val_cls_metrics = compute_classification_metrics(y_val_binary, val_pred_binary)
    train_cls_metrics = compute_classification_metrics(y_train_binary, train_pred_binary)
    
    cls_report_str = (
        f"\nValidation Classification Metrics:\n"
        f"  Balanced Accuracy: {val_cls_metrics['balanced_accuracy']:.4f}\n"
        f"  Precision:         {val_cls_metrics['precision']:.4f}\n"
        f"  Recall:            {val_cls_metrics['recall']:.4f}\n"
        f"  F1 Score:          {val_cls_metrics['f1']:.4f}\n"
        f"  Specificity:       {val_cls_metrics['specificity']:.4f}\n"
        f"\n  Confusion Matrix:\n"
        f"    TN: {val_cls_metrics['true_negatives']:,}  FP: {val_cls_metrics['false_positives']:,}\n"
        f"    FN: {val_cls_metrics['false_negatives']:,}  TP: {val_cls_metrics['true_positives']:,}\n"
        f"\nTraining Classification Metrics:\n"
        f"  Balanced Accuracy: {train_cls_metrics['balanced_accuracy']:.4f}\n"
        f"  F1 Score:          {train_cls_metrics['f1']:.4f}\n"
    )
    
    print(cls_report_str)
    report_str += "\n" + "=" * 60 + "\n" + cls_report_str

    # --- Save Model & Report ---
    model_path = Path(output_cfg["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)

    report_path = model_path.with_suffix(".report.txt")
    with report_path.open("w") as f:
        f.write("Training Regression Report (Anomaly-based)\n")
        f.write(f"Climatology file: {climatology_path}\n")
        f.write(f"Binarization threshold: {binary_threshold}\n")
        f.write(f"Mode: {binary_mode}\n")
        f.write("\n")
        f.write(report_str)
        f.write("\n" + "=" * 60 + "\n\n")

    print(f"Saved model to {model_path}")

    # Save validation predictions
    pred_dir = output_cfg.get("predictions_dir")
    if pred_dir:
        pred_dir_path = Path(pred_dir)
        pred_dir_path.mkdir(parents=True, exist_ok=True)
        
        val_out = meta_val.copy()
        val_out["prediction_anomaly"] = val_pred
        val_out["target_anomaly"] = y_val.values
        val_out["prediction_binary"] = val_pred_binary
        val_out["target_binary"] = y_val_binary
            
        val_pred_path = pred_dir_path / "predictions_validation.parquet"
        val_out.to_parquet(val_pred_path, index=False)
        print(f"Saved validation predictions to {val_pred_path}")

    # Feature Importance
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)
    
    feature_importance = model.get_feature_importance(type="FeatureImportance")
    feature_names = X_train.columns
    
    fi_df = pd.DataFrame({"feature": feature_names, "importance": feature_importance})
    fi_df = fi_df.sort_values(by="importance", ascending=False)
    
    print(fi_df.head(20).to_string(index=False))

    fi_path = model_path.with_suffix(".importance.csv")
    fi_df.to_csv(fi_path, index=False)
    print(f"\nSaved feature importance to {fi_path}")


def main() -> None:
    print("=" * 60)
    print("MODEL TRAINING (REGRESSION ON ANOMALY)")
    print("=" * 60)

    parser = argparse.ArgumentParser(description="Train CatBoost regression model on temperature anomaly")
    parser.add_argument("--config", default="configs/config_climate_reg_cold.yaml", help="Path to config YAML")
    args = parser.parse_args()

    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    print("Config loaded successfully.")
    
    train_model(config)


if __name__ == "__main__":
    main()
