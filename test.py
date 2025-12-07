from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report

from load_data import load_config, load_years
from visualization import plot_monthly_maps


def evaluate_year(
    year: int, model: CatBoostClassifier, config: Dict, threshold: float, pred_dir: Path
) -> None:
    # Use cached loader so repeated evaluations avoid slow NetCDF reads.
    X, y, meta = load_years(
        years=[year],
        config=config,
        return_metadata=True,
        verbose=True,
        apply_label_smoothing_flag=False,
    )

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
    parser = argparse.ArgumentParser(description="Evaluate trained model on holdout years")
    parser.add_argument("--config", default="configs/config_wind.yaml", help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Handle potentially missing model path in config (backward compatibility)
    if "output" in config and "model_path" in config["output"]:
        model_path = Path(config["output"]["model_path"])
    else:
        # Fallback if specific config structure differs
        model_path = Path("artifacts/model.cbm")
        
    if "model" in config:
        threshold = config["model"].get("decision_threshold", 0.5)
    else:
        threshold = 0.5
        
    if "output" in config and "predictions_dir" in config["output"]:
        pred_dir = Path(config["output"]["predictions_dir"])
    else:
        pred_dir = Path("artifacts/predictions")

    model = CatBoostClassifier()
    if model_path.exists():
        model.load_model(model_path)
    else:
        print(f"Error: Model not found at {model_path}")
        return
    
    # Force full sampling for evaluation to avoid map holes
    if config["data"].get("sample_fraction", 1.0) < 1.0:
        print(f"Overriding sample_fraction from {config['data']['sample_fraction']} to 1.0 for evaluation.")
        config["data"]["sample_fraction"] = 1.0

    test_years = config["data"].get("test_years", [])
    if not test_years:
         print("No test years defined in config.")
         return

    for year in test_years:
        evaluate_year(
            year=year,
            model=model,
            config=config,
            threshold=threshold,
            pred_dir=pred_dir,
        )


if __name__ == "__main__":
    main()
