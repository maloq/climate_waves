from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report

from load_climate_data import load_config, load_years
from prepare_land import prepare_land_data, landsea_distance
from visualization import plot_monthly_maps


def load_regression_predictions(
    pred_dir: str,
    meta: pd.DataFrame,
    verbose: bool = True
) -> Optional[np.ndarray]:
    """
    Load regression predictions and map them to the given metadata.
    
    Looks for predictions in pred_dir/reg_predictions_<year>.parquet files.
    Maps predictions to meta by matching time, latitude, longitude.
    
    Args:
        pred_dir: Directory containing regression prediction files
        meta: DataFrame with 'time', 'latitude', 'longitude' columns
        verbose: Print progress
        
    Returns:
        Array of regression predictions aligned to meta, or None if not found
    """
    pred_path = Path(pred_dir)
    
    if not pred_path.exists():
        if verbose:
            print(f"  [Reg Predictions] Directory not found: {pred_dir}")
        return None
    
    # Get unique years from meta
    meta_time = pd.to_datetime(meta["time"])
    years = meta_time.dt.year.unique()
    
    if verbose:
        print(f"  [Reg Predictions] Loading predictions for years: {sorted(years)}")
    
    # Load predictions for each year
    pred_dfs = []
    for year in sorted(years):
        year_file = pred_path / f"reg_predictions_{year}.parquet"
        if year_file.exists():
            df = pd.read_parquet(year_file)
            pred_dfs.append(df)
            if verbose:
                print(f"    Loaded {len(df):,} predictions from {year_file.name}")
        else:
            if verbose:
                print(f"    [Warning] Missing predictions file: {year_file}")
    
    if not pred_dfs:
        if verbose:
            print(f"  [Reg Predictions] No prediction files found")
        return None
    
    # Combine all predictions
    preds_df = pd.concat(pred_dfs, ignore_index=True)
    preds_df["time"] = pd.to_datetime(preds_df["time"])
    
    # Create lookup key for efficient joining
    # Round lat/lon to handle floating point comparison
    preds_df["lat_key"] = (preds_df["latitude"] * 100).round().astype(int)
    preds_df["lon_key"] = (preds_df["longitude"] * 100).round().astype(int)
    preds_df["date_key"] = preds_df["time"].dt.date
    
    # Drop duplicates to ensure unique keys (take first occurrence)
    preds_df = preds_df.drop_duplicates(subset=["lat_key", "lon_key", "date_key"], keep="first")
    if verbose:
        print(f"  [Reg Predictions] Unique prediction entries: {len(preds_df):,}")
    
    # Create same keys for meta
    meta_lookup = meta.copy()
    meta_lookup["time"] = pd.to_datetime(meta_lookup["time"])
    meta_lookup["lat_key"] = (meta_lookup["latitude"] * 100).round().astype(int)
    meta_lookup["lon_key"] = (meta_lookup["longitude"] * 100).round().astype(int)
    meta_lookup["date_key"] = meta_lookup["time"].dt.date
    meta_lookup["_idx"] = np.arange(len(meta_lookup))
    
    n_meta = len(meta_lookup)
    
    # Merge
    merged = meta_lookup.merge(
        preds_df[["lat_key", "lon_key", "date_key", "reg_pred_anomaly"]],
        on=["lat_key", "lon_key", "date_key"],
        how="left"
    )
    
    # Handle potential duplicates from merge (shouldn't happen after drop_duplicates, but safety check)
    if len(merged) > n_meta:
        if verbose:
            print(f"  [Reg Predictions] Warning: Merge created duplicates ({len(merged)} vs {n_meta}), taking first match")
        merged = merged.drop_duplicates(subset=["_idx"], keep="first")
    
    # Sort back to original order
    merged = merged.sort_values("_idx")
    
    # Check for missing matches
    n_missing = merged["reg_pred_anomaly"].isna().sum()
    n_total = len(merged)
    
    if verbose:
        print(f"  [Reg Predictions] Matched {n_total - n_missing:,}/{n_total:,} samples")
        if n_missing > 0:
            print(f"    [Warning] {n_missing:,} samples have no matching regression prediction")
    
    # Fill missing with 0 (neutral anomaly)
    result = merged["reg_pred_anomaly"].fillna(0.0).values
    
    return result


def evaluate_year(
    year: int, model: CatBoostClassifier, config: Dict, threshold: float, pred_dir: Path
) -> Tuple[pd.Series, np.ndarray, np.ndarray, pd.DataFrame, Dict, Dict]:
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
    
    # 3. Regression Predictions as Feature
    reg_pred_cfg = config.get("regression_predictions", {})
    reg_pred_dir = reg_pred_cfg.get("predictions_dir", "artifacts/coldwave_reg/predictions")
    use_reg_pred = reg_pred_cfg.get("enabled", True)  # Default to True if predictions exist
    
    if use_reg_pred:
        print(f"  Loading regression predictions as feature...")
        reg_preds = load_regression_predictions(reg_pred_dir, meta, verbose=True)
        
        if reg_preds is not None:
            X["reg_pred_anomaly"] = reg_preds
            print(f"  Added regression prediction feature: reg_pred_anomaly")
        else:
            print("  [Warning] Regression predictions not available, skipping this feature")
            
    print(f"  Total features after integration: {len(X.columns)}")
    # --------------------------------------

    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)

    report = classification_report(y, preds)
    print(f"Year {year} metrics:")
    print(report)
    
    # Calculate and print relaxed metrics
    # Calculate and print relaxed metrics
    # calculate_relaxed_metrics(y, preds, proba, meta, year) # Called later now to capture return

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

    # Return data for aggregation
    # Standard metrics
    std_rep = classification_report(y, preds, output_dict=True, zero_division=0)
    std_metrics = {
        "precision": std_rep["1"]["precision"],
        "recall": std_rep["1"]["recall"],
        "f1": std_rep["1"]["f1-score"]
    }
    
    # Relaxed metrics
    rel_prec, rel_rec, rel_f1 = calculate_relaxed_metrics(y, preds, proba, meta, year)
    rel_metrics = {
        "rel_precision": rel_prec,
        "rel_recall": rel_rec,
        "rel_f1": rel_f1
    }
    
    return y, preds, proba, meta, std_metrics, rel_metrics


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
    
    return relaxed_precision, relaxed_recall, relaxed_f1




def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained model on holdout years (New Climate Source)")
    parser.add_argument("--config", default="configs/config_climate_coldwave.yaml", help="Path to config YAML")
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

    all_results = []
    
    # Accumulate for global relaxed
    all_meta_list = []
    all_y_list = []
    all_preds_list = []
    all_proba_list = []

    for year in config["data"]["test_years"]:
        y, preds, proba, meta, std_m, rel_m = evaluate_year(
            year=year,
            model=model,
            config=config,
            threshold=threshold,
            pred_dir=pred_dir,
        )
        
        # Store per-year row
        row = {
            "Year": str(year),
            "Precision": std_m["precision"],
            "Recall": std_m["recall"],
            "F1": std_m["f1"],
            "Rel_Precision": rel_m["rel_precision"],
            "Rel_Recall": rel_m["rel_recall"],
            "Rel_F1": rel_m["rel_f1"]
        }
        all_results.append(row)
        
        all_meta_list.append(meta)
        all_y_list.append(y)
        all_preds_list.append(preds)
        all_proba_list.append(proba)

    # --- Calculate Aggregated Metrics ---
    if all_y_list:
        # 1. Standard Aggregated
        full_y = pd.concat(all_y_list)
        full_preds = np.concatenate(all_preds_list)
        
        full_rep = classification_report(full_y, full_preds, output_dict=True, zero_division=0)
        
        # 2. Relaxed Aggregated
        # We need to stitch meta carefully or just reuse the logic.
        # Relaxed metrics depend on temporal adjacency. Concatenating years might break continuity 
        # at boundaries, but since they are separate years, that's expected.
        # We can pass the concatenated dfs to calculate_relaxed_metrics, 
        # assuming it handles the gaps via the groupby(['latitude', 'longitude']) 
        # and time sorting. It should just see a jump in time and not match +/- 1 day across year gap.
        
        full_meta = pd.concat(all_meta_list)
        full_proba = np.concatenate(all_proba_list)
        
        # Suppress prints for the aggregated calc
        print("\nCalculating Aggregated Relaxed Metrics...")
        rel_prec, rel_rec, rel_f1 = calculate_relaxed_metrics(full_y, full_preds, full_proba, full_meta, "ALL_YEARS")
        
        agg_row = {
            "Year": "ALL",
            "Precision": full_rep["1"]["precision"],
            "Recall": full_rep["1"]["recall"],
            "F1": full_rep["1"]["f1-score"],
            "Rel_Precision": rel_prec,
            "Rel_Recall": rel_rec,
            "Rel_F1": rel_f1
        }
        all_results.append(agg_row)

    # --- Print Table ---
    print("\n" + "="*80)
    print("AGGREGATED RESULTS TABLE")
    print("="*80)
    df_results = pd.DataFrame(all_results)
    # Format floats
    format_cols = ["Precision", "Recall", "F1", "Rel_Precision", "Rel_Recall", "Rel_F1"]
    for col in format_cols:
        df_results[col] = df_results[col].apply(lambda x: f"{x:.4f}")
        
    print(df_results.to_string(index=False))
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
