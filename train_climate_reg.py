from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from load_climate_data import load_config, load_years
from prepare_land import prepare_land_data, landsea_distance


def train_model(config: Dict) -> None:
    data_cfg = config["data"]
    model_cfg = config["model"].copy()
    train_cfg = config["training"]
    output_cfg = config["output"]

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
        apply_label_smoothing_flag=False, # No smoothing for regression
        return_hard_labels=False
    )
    
    print(f"Data loaded: {len(X):,} samples, {len(X.columns)} features")

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
    
    print(f"Target stats: min={y.min():.3f}, max={y.max():.3f}, mean={y.mean():.3f}, std={y.std():.3f}")
    
    print("Splitting data into train/validation sets (Temporal Split)...")
    
    if "time" not in meta.columns:
        raise ValueError("Metadata 'time' column required for temporal splitting")
        
    if not np.issubdtype(meta["time"].dtype, np.datetime64):
        meta["time"] = pd.to_datetime(meta["time"])

    meta_time = meta["time"]
    all_years = sorted(meta_time.dt.year.unique())

    if len(all_years) < 2:
        raise ValueError(f"Need at least 2 distinct years for temporal split, found {len(all_years)}: {all_years}")

    val_size_cfg = train_cfg.get("validation_size", 0.2)
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
    
    meta_val = meta[val_mask].copy()
    
    # ------------------------------------------------------------
    # NO UNDERSAMPLING FOR REGRESSION
    # ------------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Features ({len(X_train.columns)}):")
    for i, feat in enumerate(X_train.columns, 1):
        print(f"  {i:2d}. {feat}")
    print(f"\nTraining samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print("=" * 60 + "\n")

    print("Initializing CatBoost Regressor...")
    print(f"Training on device: {model_cfg.get('task_type', 'CPU')}")
    model = CatBoostRegressor(**model_cfg)
    print("Starting training...\n")
    
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=verbose)

    # --- Evaluation ---
    print("\nEvaluating on validation set...")
    val_pred = model.predict(X_val)
    train_pred = model.predict(X_train)
    
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_mae = mean_absolute_error(y_val, val_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    
    report_str = (
        "Regression Metrics:\n"
        f"Train RMSE: {train_rmse:.4f}\n"
        f"Val RMSE:   {val_rmse:.4f}\n"
        f"Val MAE:    {val_mae:.4f}\n"
        f"Val R2:     {val_r2:.4f}\n"
    )
    
    print("\n" + report_str)

    # --- Save Model & Report ---
    model_path = Path(output_cfg["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)

    report_path = model_path.with_suffix(".report.txt")
    with report_path.open("w") as f:
        f.write("Training Regression Report\n")
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
        val_out["prediction"] = val_pred
        val_out["target"] = y_val.values
            
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
    print("MODEL TRAINING (REGRESSION)")
    print("=" * 60)

    parser = argparse.ArgumentParser(description="Train CatBoost regression model")
    parser.add_argument("--config", default="configs/config_climate_reg_cold.yaml", help="Path to config YAML")
    args = parser.parse_args()

    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    print("Config loaded successfully.")
    
    train_model(config)

if __name__ == "__main__":
    main()
