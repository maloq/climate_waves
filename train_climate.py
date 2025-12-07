from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Use the new data loader
from load_climate_data import load_config, load_years

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
    use_label_smoothing = ls_config.get("enabled", False)
    if use_label_smoothing:
        print("\n[Label Smoothing] Enabled - using CrossEntropy loss for soft targets")
        model_cfg["loss_function"] = "CrossEntropy"

    print(f"\nTrain years: {data_cfg['train_years']}")
    print(f"Output model path: {output_cfg['model_path']}")
    print("\nLoading training data...")
    
    # Load data using the new module
    X, y, meta, y_hard = load_years(
        data_cfg["train_years"], 
        config=config, 
        return_hard_labels=True,
        return_metadata=True,
        apply_label_smoothing_flag=True
    )
    print(f"Data loaded: {len(X):,} samples, {len(X.columns)} features")
    
    if use_label_smoothing:
        print(f"[Label Smoothing] Label stats: min={y.min():.3f}, max={y.max():.3f}, mean={y.mean():.3f}")
    
    print("Splitting data into train/validation sets...")
    
    # Stratified split using metadata
    # We need to split X, y, y_hard, and meta consistently
    if use_label_smoothing and y_hard is not None:
        stratify_labels = y_hard 
    elif use_label_smoothing:
        stratify_labels = (y >= 0.5).astype(int)
    else:
        stratify_labels = y
        
    # Using train_test_split directly here to handle the 4th array (meta) easier than modifying split_data helper
    # which is getting complicated with all optional args
    X_train, X_val, y_train, y_val, y_train_hard, y_val_hard, meta_train, meta_val = train_test_split(
        X, y, y_hard, meta,
        test_size=train_cfg.get("validation_size", 0.2),
        random_state=data_cfg.get("random_seed", 42),
        shuffle=train_cfg.get("shuffle", True),
        stratify=stratify_labels
    )

    class_weights = train_cfg.get("class_weights")
    if class_weights is not None:
        model_cfg["class_weights"] = class_weights
    elif use_class_weights:
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
    
    model.fit(X_train, y_train, eval_set=(X_val, y_val_hard), verbose=verbose)

    # Training metrics
    train_proba = model.predict_proba(X_train)[:, 1]
    train_pred = (train_proba >= decision_threshold).astype(int)
    train_report = classification_report(y_train_hard, train_pred)

    # Validation metrics
    val_proba = model.predict_proba(X_val)[:, 1]
    val_pred = (val_proba >= decision_threshold).astype(int)
    val_report = classification_report(y_val_hard, val_pred)

    model_path = Path(output_cfg["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)

    report_path = model_path.with_suffix(".report.txt")
    with report_path.open("w") as f:
        f.write("Training classification report\n")
        f.write(f"Decision threshold: {decision_threshold}\n")
        f.write("\n")
        f.write(train_report)
        f.write("\n" + "=" * 60 + "\n\n")
        f.write("Validation classification report\n")
        f.write(f"Decision threshold: {decision_threshold}\n\n")
        f.write(val_report)

    print("\nTraining metrics:")
    print(train_report)
    print("\nValidation metrics:")
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
    parser.add_argument("--config", default="configs/config_climate_coldwaves.yaml", help="Path to config YAML")
    args = parser.parse_args()

    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    print("Config loaded successfully.")
    
    train_model(config)

if __name__ == "__main__":
    main()
