from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report

from load_data import load_config, load_years


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained model on holdout years")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    model_path = Path(config["output"]["model_path"])
    threshold = config["model"].get("decision_threshold", 0.5)
    pred_dir = Path(config["output"]["predictions_dir"])

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
