from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report

from load_climate_data import load_config, load_years
from prepare_land import prepare_land_data, landsea_distance
from visualization import plot_monthly_maps


def evaluate_year(
    year: int, model: CatBoostClassifier, config: Dict, threshold: float, pred_dir: Path
) -> None:
    # Use new loader
    X, y, meta = load_years(
        years=[year],
        config=config,
        return_metadata=True,
        verbose=True,
        apply_label_smoothing_flag=False,
        align_to_targets=True,
    )

    # --- Integrate Static Land Features (Match Training Logic) ---
    print(f"Integrating static land features for year {year}...")
    
    # 1. Distance to Coast
    print("  Calculating distance to coast...")
    meta_with_dist, dist_cols = landsea_distance(meta, lat_col="latitude", lon_col="longitude")
    
    # Add to X (ensuring alignment)
    for col in dist_cols:
        X[col] = meta_with_dist[col].values
    
    # 2. Other Land Data (Forest, Elevation)
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
    
    # Add to X
    for col in land_cols:
        if col in meta_with_land.columns:
            X[col] = meta_with_land[col].values
            
    print(f"  Added static features. Total features: {len(X.columns)}")
    # --------------------------------------



    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)

    report = classification_report(y, preds)
    print(f"Year {year} metrics:")
    print(report)
    
    # Calculate and print relaxed metrics
    calculate_relaxed_metrics(y, preds, proba, meta, year)

    out = meta.copy()
    out["probability"] = proba
    out["prediction"] = preds
    out["target"] = y.values

    pred_dir.mkdir(parents=True, exist_ok=True)
    out_path = pred_dir / f"predictions_{year}.parquet"
    out.to_parquet(out_path, index=False)
    print(f"Saved predictions to {out_path}")

    # Generate Monthly Maps
    print(f"Generating monthly maps for {year}...")
    plot_monthly_maps(meta, y, proba, preds, year, pred_dir)


def calculate_relaxed_metrics(y_true, y_pred, y_proba, meta, year):
    """
    Calculates and prints relaxed precision, recall, and F1 score.
    A prediction is considered correct if it is within +/- 1 day of a ground truth event.
    """
    print(f"\nYear {year} Relaxed Metrics (+/- 1 day):")
    
    # 1. Prepare DataFrame for window operations
    df_eval = meta.copy()
    df_eval["target"] = y_true.values
    df_eval["prediction"] = y_pred
    
    # Ensure sorted by space then time for correct shifting
    if not pd.api.types.is_datetime64_any_dtype(df_eval["time"]):
         df_eval["time"] = pd.to_datetime(df_eval["time"])
         
    df_eval = df_eval.sort_values(["latitude", "longitude", "time"])
    
    # 2. Define Window Helpers
    indexer = df_eval.groupby(["latitude", "longitude"])
    
    # Relaxed Recall: Did we predict 1 loosely around the actual event?
    # Expand Predictions: if Pred(t)=1, then Pred_Relaxed(t-1)=1, Pred_Relaxed(t)=1, Pred_Relaxed(t+1)=1.
    p = df_eval["prediction"]
    p_prev = indexer["prediction"].shift(1).fillna(0)
    p_next = indexer["prediction"].shift(-1).fillna(0)
    pred_relaxed = ((p == 1) | (p_prev == 1) | (p_next == 1)).astype(int)
    
    # Relaxed Precision: Was the prediction close to an actual event?
    # Expand Targets: if Target(t)=1, then Target_Relaxed(t-1)=1, Target_Relaxed(t)=1, Target_Relaxed(t+1)=1.
    t = df_eval["target"]
    t_prev = indexer["target"].shift(1).fillna(0)
    t_next = indexer["target"].shift(-1).fillna(0)
    target_relaxed = ((t == 1) | (t_prev == 1) | (t_next == 1)).astype(int)
    
    # Calculate Metrics
    # Precision: TP / (TP + FP) where "TP" means Pred=1 & RelaxedTarget=1
    true_pos_relaxed_prec = ((df_eval["prediction"] == 1) & (target_relaxed == 1)).sum()
    total_pred_pos = (df_eval["prediction"] == 1).sum()
    
    relaxed_precision = true_pos_relaxed_prec / total_pred_pos if total_pred_pos > 0 else 0.0
    
    # Recall: TP / (TP + FN) where "TP" means Target=1 & RelaxedPred=1
    true_pos_relaxed_rec = ((df_eval["target"] == 1) & (pred_relaxed == 1)).sum()
    total_target_pos = (df_eval["target"] == 1).sum()
    
    relaxed_recall = true_pos_relaxed_rec / total_target_pos if total_target_pos > 0 else 0.0
    
    # F1
    if (relaxed_precision + relaxed_recall) > 0:
        relaxed_f1 = 2 * (relaxed_precision * relaxed_recall) / (relaxed_precision + relaxed_recall)
    else:
        relaxed_f1 = 0.0
        
    print(f"  Precision (Relaxed): {relaxed_precision:.4f}")
    print(f"  Recall (Relaxed)   : {relaxed_recall:.4f}")
    print(f"  F1 Score (Relaxed) : {relaxed_f1:.4f}")
    print("-" * 30)




def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained model on holdout years (New Climate Source)")
    parser.add_argument("--config", default="configs/config_climate_coldwave_local.yaml", help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Handle potentially missing model path in config
    if "output" in config and "model_path" in config["output"]:
        model_path = Path(config["output"]["model_path"])
    else:
        model_path = Path("artifacts/model.cbm")
        
    if "model" in config:
        threshold = config["model"].get("decision_threshold", 0.5)
    else:
        threshold = 0.5
        
    if "output" in config and "predictions_dir" in config["output"]:
        pred_dir = Path(config["output"]["predictions_dir"])
    else:
        pred_dir = Path("artifacts/predictions")

    if not model_path.exists():
        print(f"Model not found at {model_path}. Please train usage train_climate.py first.")
        return
        
    # Force full sampling for evaluation to avoid map holes
    if config["data"].get("sample_fraction", 1.0) < 1.0:
        print(f"Overriding sample_fraction from {config['data']['sample_fraction']} to 1.0 for evaluation.")
        config["data"]["sample_fraction"] = 1.0

    model = CatBoostClassifier()
    model.load_model(model_path)

    for year in config["data"]["test_years"]:
        evaluate_year(
            year=year,
            model=model,
            config=config,
            threshold=threshold,
            pred_dir=pred_dir,
        )


if __name__ == "__main__":
    main()
