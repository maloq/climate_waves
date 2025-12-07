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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained model on holdout years (New Climate Source)")
    parser.add_argument("--config", default="configs/config_climate_wind.yaml", help="Path to config YAML")
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
