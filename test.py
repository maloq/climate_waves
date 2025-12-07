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


def plot_monthly_maps(
    meta: pd.DataFrame,
    targets: pd.Series,
    probs: np.ndarray,
    preds: np.ndarray,
    year: int,
    output_dir: Path,
) -> None:
    """aggregates predictions and targets by month and plots spatial means."""
    output_dir = output_dir / "maps"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a dataframe with all necessary data
    df = meta.copy()
    df["target"] = targets.values
    df["prob"] = probs
    df["pred"] = preds
    df["month"] = pd.to_datetime(df["time"]).dt.month

    # Get unique months present in the data
    months = sorted(df["month"].unique())

    for month in months:
        monthly_df = df[df["month"] == month]
        n_days = monthly_df["time"].nunique()
        n_records = len(monthly_df)
        target_mean = monthly_df["target"].mean()
        print(f"[DEBUG] Month {month}: {n_days} unique days, {n_records} records, target mean: {target_mean:.4f}")
        
        # Group by lat/lon to get mean values for the month
        grid_df = monthly_df.groupby(["latitude", "longitude"])[["target", "prob", "pred"]].mean().reset_index()
        
        # Pivot to create 2D arrays for plotting
        target_grid = grid_df.pivot(index="latitude", columns="longitude", values="target")
        prob_grid = grid_df.pivot(index="latitude", columns="longitude", values="prob")
        pred_grid = grid_df.pivot(index="latitude", columns="longitude", values="pred")
        
        # Sort index/columns
        target_grid = target_grid.sort_index(ascending=True).sort_index(axis=1)
        prob_grid = prob_grid.sort_index(ascending=True).sort_index(axis=1)
        pred_grid = pred_grid.sort_index(ascending=True).sort_index(axis=1)

        lons = prob_grid.columns
        lats = prob_grid.index
        
        # Plotting - 3 panels now
        fig, axes = plt.subplots(1, 3, figsize=(24, 6))
        
        # 1. Ground Truth (Target)
        im0 = axes[0].pcolormesh(lons, lats, target_grid.values, shading='auto', cmap='viridis', vmin=0, vmax=1)
        axes[0].set_title(f"Ground Truth (Mean Target) - {year}-{month:02d}")
        axes[0].set_xlabel("Longitude")
        axes[0].set_ylabel("Latitude")
        plt.colorbar(im0, ax=axes[0], label="Proportion")
        
        # 2. Predicted Probability
        im1 = axes[1].pcolormesh(lons, lats, prob_grid.values, shading='auto', cmap='viridis', vmin=0, vmax=1)
        axes[1].set_title(f"Predicted Probability - {year}-{month:02d}")
        axes[1].set_xlabel("Longitude")
        axes[1].set_ylabel("Latitude")
        plt.colorbar(im1, ax=axes[1], label="Probability")

        # 3. Binary Prediction
        im2 = axes[2].pcolormesh(lons, lats, pred_grid.values, shading='auto', cmap='viridis', vmin=0, vmax=1)
        axes[2].set_title(f"Binary Prediction - {year}-{month:02d}")
        axes[2].set_xlabel("Longitude")
        axes[2].set_ylabel("Latitude")
        plt.colorbar(im2, ax=axes[2], label="Proportion Predicted 1")

        plt.tight_layout()
        save_path = output_dir / f"{year}_{month:02d}.png"
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved map to {save_path}")


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
    parser.add_argument("--config", default="configs/config_coldwave.yaml", help="Path to config YAML")
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
