
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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
        # Auto-scale vmax to make rare events visible, but keep vmin=0
        max_target = np.nanmax(target_grid.values)
        if np.isnan(max_target) or max_target == 0: max_target = 0.01 # Avoid 0 range

        im0 = axes[0].pcolormesh(lons, lats, target_grid.values, shading='auto', cmap='viridis', vmin=0)
        axes[0].set_title(f"Ground Truth (Mean Target) - {year}-{month:02d} (Max: {max_target:.3f})")
        axes[0].set_xlabel("Longitude")
        axes[0].set_ylabel("Latitude")
        plt.colorbar(im0, ax=axes[0], label="Proportion")
        
        # 2. Predicted Probability
        max_prob = np.nanmax(prob_grid.values)
        im1 = axes[1].pcolormesh(lons, lats, prob_grid.values, shading='auto', cmap='viridis', vmin=0)
        axes[1].set_title(f"Predicted Probability - {year}-{month:02d} (Max: {max_prob:.3f})")
        axes[1].set_xlabel("Longitude")
        axes[1].set_ylabel("Latitude")
        plt.colorbar(im1, ax=axes[1], label="Probability")

        # 3. Binary Prediction
        max_pred = np.nanmax(pred_grid.values)
        im2 = axes[2].pcolormesh(lons, lats, pred_grid.values, shading='auto', cmap='viridis', vmin=0)
        axes[2].set_title(f"Binary Prediction - {year}-{month:02d} (Max: {max_pred:.3f})")
        axes[2].set_xlabel("Longitude")
        axes[2].set_ylabel("Latitude")
        plt.colorbar(im2, ax=axes[2], label="Proportion Predicted 1")

        plt.tight_layout()
        save_path = output_dir / f"{year}_{month:02d}.png"
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved map to {save_path}")
