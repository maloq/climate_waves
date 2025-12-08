from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, average_precision_score
from sklearn.model_selection import train_test_split

# Use the new data loader
from load_climate_data import load_config, load_years
from prepare_land import prepare_land_data, landsea_distance

def split_data(
    X: pd.DataFrame, y: pd.Series, validation_size: float, random_seed: int,
    y_hard: pd.Series | None = None,
    use_soft_labels: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    # For soft labels, stratify on binarized labels (>=0.5 threshold)
    # For hard labels, stratify directly on y

    if use_soft_labels and y_hard is not None:
        stratify_labels = y_hard 
    elif use_soft_labels:
        stratify_labels = (y >= 0.5).astype(int)
    else:
        stratify_labels = y
    
    if y_hard is not None:
        X_train, X_val, y_train, y_val, y_train_hard, y_val_hard = train_test_split(
            X,
            y,
            y_hard,
            test_size=validation_size,
            random_state=random_seed,
            shuffle=False,
        )
        return X_train, X_val, y_train, y_val, y_train_hard, y_val_hard
    
    # Fallback to standard split if no hard labels provided (though we expect them)
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=validation_size,
        random_state=random_seed,
        shuffle=False,
    )
    return X_train, X_val, y_train, y_val, y_train, y_val # Return y as hard substitute


def train_model(config: Dict) -> None:
    data_cfg = config["data"]
    model_cfg = config["model"].copy()
    train_cfg = config["training"]
    output_cfg = config["output"]
    ls_config = config.get("label_smoothing", {})

    decision_threshold = model_cfg.pop("decision_threshold", 0.5)
    verbose = model_cfg.pop("verbose", 10)
    use_class_weights = model_cfg.pop("use_class_weights", False)
    class_weight_ratio = model_cfg.pop("class_weight_ratio", 1.0)
    
    # Check if label smoothing is enabled
    # Check if label smoothing is enabled
    use_label_smoothing = ls_config.get("enabled", False)
    if use_label_smoothing:
        print("\n[Label Smoothing] Enabled - using CrossEntropy loss for soft targets")
        model_cfg["loss_function"] = "CrossEntropy"
    elif model_cfg.get("loss_function") == "Focal":
        # Construct Focal loss string if specific params are given
        alpha = model_cfg.pop("focal_alpha", None)
        gamma = model_cfg.pop("focal_gamma", None)
        
        if alpha is not None or gamma is not None:
            focal_str = "Focal:"
            params = []
            if alpha is not None: params.append(f"focal_alpha={alpha}")
            if gamma is not None: params.append(f"focal_gamma={gamma}")
            focal_str += ";".join(params)
            model_cfg["loss_function"] = focal_str
            print(f"\n[Focal Loss] Enabled - using {focal_str}")
        else:
            print("\n[Focal Loss] Enabled - using Default Focal Loss")

    print(f"\nTrain years: {data_cfg['train_years']}")
    print(f"Output model path: {output_cfg['model_path']}")
    print("\nLoading training data...")
    
    X, y, meta, y_hard = load_years(
        data_cfg["train_years"], 
        config=config, 
        return_hard_labels=True,
        return_metadata=True,
        apply_label_smoothing_flag=True,
        align_to_targets=True
    )
    
    # [Data Leakage Fix] Drop hard target column from features if it leaked there
    target_var = data_cfg["target_var"]
    leak_col = f"{target_var}_hard"
    if leak_col in X.columns:
        print(f"[Data Leakage] Dropping leaked target column '{leak_col}' from features.")
        X = X.drop(columns=[leak_col])
        
    print(f"Data loaded: {len(X):,} samples, {len(X.columns)} features")



    # --- Integrate Static Land Features ---
    print("\nIntegrating static land features...")
    
    # 1. Distance to Coast
    # Note: landsea_distance uses default paths we updated in prepare_land.py
    print("  Calculating distance to coast...")
    meta_with_dist, dist_cols = landsea_distance(meta, lat_col="latitude", lon_col="longitude")
    
    # Add to X (ensuring alignment)
    for col in dist_cols:
        X[col] = meta_with_dist[col].values
    print(f"  Added {len(dist_cols)} distance features: {dist_cols}")
    
    # 2. Other Land Data (Forest, Elevation)
    land_files = [
        "climate_data/land_data/forest_data.nc",
        "climate_data/land_data/GMTED2010_15n015_00625deg.nc"
    ]
    print(f"  Processing land data files: {land_files}")
    
    # We pass 'meta' as the target dataframe. 
    # prepare_land_data returns the dataframe with new columns + list of new column names.
    meta_with_land, land_cols = prepare_land_data(
        land_files, 
        meta, 
        lat_col="latitude", 
        lon_col="longitude"
    )
    
    # Add to X
    # Note: process uses Polars internally and returns Pandas with potentially reset index, 
    # but row order is preserved. X has RangeIndex from load_years merge.
    if len(meta_with_land) != len(X):
        print(f"Warning: Land feature integration resulted in {len(meta_with_land)} rows, expected {len(X)}")
    
    # Assign new columns to X
    # Using values to ignore index mismatches if any (though they should align by position)
    for col in land_cols:
        if col in meta_with_land.columns:
            X[col] = meta_with_land[col].values
            
    print(f"  Added {len(land_cols)} features from land data files.")
    print(f"  Total features after integration: {len(X.columns)}")
    # --------------------------------------
    
    if use_label_smoothing:
        print(f"[Label Smoothing] Label stats: min={y.min():.3f}, max={y.max():.3f}, mean={y.mean():.3f}")
    
    print("Splitting data into train/validation sets (Temporal Split)...")
    
    # Extract years from metadata to perform temporal split
    # We assume 'time' is available in meta and is a datetime object
    if "time" not in meta.columns:
        raise ValueError("Metadata 'time' column required for temporal splitting")
        
    # Standardize time column to datetime if needed
    if not np.issubdtype(meta["time"].dtype, np.datetime64):
         meta["time"] = pd.to_datetime(meta["time"])

    all_years = sorted(meta["time"].dt.year.unique())
    if len(all_years) < 2:
        print(f"Warning: Only {len(all_years)} year(s) found {all_years}. Cannot perform temporal split. Using random split.")
        # Fallback to random split if only 1 year
        X_train, X_val, y_train, y_val, y_train_hard, y_val_hard, meta_train, meta_val = train_test_split(
            X, y, y_hard, meta,
            test_size=train_cfg.get("validation_size", 0.4),
            random_state=data_cfg.get("random_seed", 42),
            shuffle=False,
            stratify=None
        )
    else:
        # Use the last year for validation
        val_year = all_years[-1]
        train_years_list = all_years[:-1]
        
        print(f"  Training years: {train_years_list}")
        print(f"  Validation year: {val_year}")
        
        val_mask = meta["time"].dt.year == val_year
        train_mask = ~val_mask
        
        X_train = X[train_mask].copy()
        X_val = X[val_mask].copy()
        
        y_train = y[train_mask].copy()
        y_val = y[val_mask].copy()
        
        if y_hard is not None:
            y_train_hard = y_hard[train_mask].copy()
            y_val_hard = y_hard[val_mask].copy()
        else:
            y_train_hard = None
            y_val_hard = None
            
        meta_train = meta[train_mask].copy()
        meta_val = meta[val_mask].copy()
        
        # --- Undersampling Training Data (10:1 Negative:Positive) ---
        print("\n[Undersampling] Balancing training data...")
        # Identify positive indices
        # Use y_train_hard if available, else threshold y_train
        if y_train_hard is not None:
            pos_mask_train = y_train_hard == 1
        else:
            pos_mask_train = y_train >= 0.5
            
        pos_indices = np.where(pos_mask_train)[0]
        neg_indices = np.where(~pos_mask_train)[0]
        
        n_pos = len(pos_indices)
        n_neg = len(neg_indices)
        target_neg = n_pos * 50
        
        print(f"  Original Train Counts: Pos={n_pos:,}, Neg={n_neg:,}")
        
        if n_neg > target_neg:
            print(f"  Undersampling Negatives to {target_neg:,} (50:1 ratio)")
            # Randomly sample negative indices
            # Set seed for reproducibility
            np.random.seed(data_cfg.get("random_seed", 42))
            undersampled_neg_indices = np.random.choice(neg_indices, size=target_neg, replace=False)
            
            # Combine and shuffle
            keep_indices = np.concatenate([pos_indices, undersampled_neg_indices])
            np.random.shuffle(keep_indices)
            
            # Apply to all training arrays
            # Note: X_train is a DataFrame, y_train Series. iloc is safest.
            X_train = X_train.iloc[keep_indices]
            y_train = y_train.iloc[keep_indices]
            
            if y_train_hard is not None:
                y_train_hard = y_train_hard.iloc[keep_indices]
                
            meta_train = meta_train.iloc[keep_indices]
            print(f"  New Train Size: {len(X_train):,}")
        else:
            print("  Negatives count below target ratio, skipping undersampling.")
        # ------------------------------------------------------------

    class_weights = train_cfg.get("class_weights")
    if class_weights is not None:
        model_cfg["class_weights"] = class_weights
    elif use_class_weights:
        if model_cfg.get("loss_function") == "CrossEntropy":
            print(f"[Class Weights] CrossEntropy active: scale_pos_weight is not supported. Will use sample_weight during fit instead.")
        else:
            print(f"[Class Weights] Enabled - using scale_pos_weight={class_weight_ratio}")
            model_cfg["scale_pos_weight"] = class_weight_ratio

    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Features ({len(X_train.columns)}):")
    for i, feat in enumerate(X_train.columns, 1):
        print(f"  {i:2d}. {feat}")
    print(f"\nTraining samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print("=" * 60 + "\n")

    print("Initializing CatBoost model...")
    print(f"Training on device: {model_cfg.get('task_type', 'CPU')}")
    model = CatBoostClassifier(**model_cfg)
    print("Starting training...\n")
    
    # Prepare sample weights if needed (for CrossEntropy with class weights)
    train_sample_weight = None
    if use_class_weights and model_cfg.get("loss_function") == "CrossEntropy":
        print(f"Creating sample weights with positive weight {class_weight_ratio}...")
        # Use hard labels for weighting to simulate scale_pos_weight
        if y_train_hard is not None:
             ref_labels = y_train_hard
        else:
             ref_labels = (y_train >= 0.5).astype(int)
        
        train_sample_weight = np.ones(len(y_train), dtype=np.float32)
        train_sample_weight[ref_labels == 1] = class_weight_ratio

    model.fit(X_train, y_train, sample_weight=train_sample_weight, eval_set=(X_val, y_val_hard), verbose=verbose)

    # --- Threshold Tuning ---
    print("\nFinding optimal decision threshold on validation set...")
    val_proba = model.predict_proba(X_val)[:, 1]

    # Calculate PR AUC (Average Precision)
    # We use y_true_tuning which is either hard labels or binarized soft labels
    y_true_tuning = (y_val_hard if y_val_hard is not None else (y_val >= 0.5)).astype(int)
    pr_auc = average_precision_score(y_true_tuning, val_proba)
    print(f"PR AUC (Average Precision): {pr_auc:.4f}")


    
    
    # thresholds = np.arange(0.05, 0.96, 0.05)
    # best_threshold = 0.5
    # best_f1 = -1.0
    # best_report = ""
    
    # # We need true labels for tuning
    # # If y_val_hard is None (soft labels only, rare), we must threshold y_val
    # # y_true_tuning defined above

    thresholds = np.arange(0.05, 0.96, 0.05)
    best_threshold = 0.5
    best_f1 = -1.0
    best_report = ""

    for thresh in thresholds:
        preds = (val_proba >= thresh).astype(int)
        report_dict = classification_report(y_true_tuning, preds, output_dict=True, zero_division=0)
        # Use macro avg F1 or positive class F1? Usually positive class F1 is what we care about in imbalance
        # "1" is the positive class
        f1 = report_dict.get("1", {}).get("f1-score", 0.0)
        

        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            best_report = classification_report(y_true_tuning, preds, zero_division=0)

    print(f"Optimal Threshold: {best_threshold:.2f} (F1={best_f1:.4f})")
    print(f"Default Threshold (0.5) F1: {classification_report(y_true_tuning, (val_proba >= 0.5).astype(int), output_dict=True, zero_division=0).get('1', {}).get('f1-score', 0.0):.4f}")
    
    # Use optimal threshold for final outputs
    decision_threshold = best_threshold
    val_pred = (val_proba >= decision_threshold).astype(int)
    val_report = best_report
    
    # Re-calc train report with new threshold for consistency
    train_proba = model.predict_proba(X_train)[:, 1]
    train_pred = (train_proba >= decision_threshold).astype(int)
    train_report = classification_report(y_train_hard, train_pred, zero_division=0)
    
    # --- Save Model & Report ---
    model_path = Path(output_cfg["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)

    report_path = model_path.with_suffix(".report.txt")
    with report_path.open("w") as f:
        f.write("Training classification report\n")
        f.write(f"Decision threshold: {decision_threshold}\n")
        f.write(f"PR AUC: {pr_auc:.4f}\n")
        f.write("\n")
        f.write(train_report)
        f.write("\n" + "=" * 60 + "\n\n")
        f.write("Validation classification report\n")
        f.write(f"Decision threshold: {decision_threshold}\n")
        f.write(f"PR AUC: {pr_auc:.4f}\n\n")
        f.write(val_report)
    # ---------------------------

    print("\nTraining metrics (at optimal threshold):")
    print(train_report)
    print("\nValidation metrics (at optimal threshold):")
    print(val_report)
    print(f"\nSaved model to {model_path}")

    # Save validation predictions
    pred_dir = output_cfg.get("predictions_dir")
    if pred_dir:
        pred_dir_path = Path(pred_dir)
        pred_dir_path.mkdir(parents=True, exist_ok=True)
        
        val_out = meta_val.copy()
        val_out["probability"] = val_proba
        val_out["prediction"] = val_pred
        val_out["target"] = y_val_hard.values
        if use_label_smoothing:
            val_out["target_smooth"] = y_val.values
            
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
    print("MODEL TRAINING (NEW CLIMATE SOURCE)")
    print("=" * 60)

    parser = argparse.ArgumentParser(description="Train CatBoost event model with new data")
    parser.add_argument("--config", default="configs/config_climate_coldwave.yaml", help="Path to config YAML")
    args = parser.parse_args()

    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    print("Config loaded successfully.")
    
    train_model(config)

if __name__ == "__main__":
    main()
