"""
Optuna hyperparameter optimization for heatwave prediction model (Climate Source).

Optimizes:
1. Feature selection (which base columns to include)
2. Feature engineering toggles
3. CatBoost model hyperparameters

Uses 2022 for validation and 2023 for final test evaluation (or as configured).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
import xarray as xr
import yaml
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# Use new loader
# apply_label_smoothing and load_config reused from load_data via load_climate_data or directly
# But load_years MUST come from load_climate_data
from load_climate_data import load_config, load_years, apply_label_smoothing
from feature_engineering import get_feature_names

warnings.filterwarnings("ignore", category=UserWarning)


def create_trial_config(
    trial: optuna.Trial, 
    base_config: Dict, 
    mode: str = "full", 
    fixed_model_params: Dict = None
) -> Dict:
    """Create a configuration dictionary based on Optuna trial suggestions."""
    config = deepcopy(base_config)
    
    all_features = config.get("features", [])
    
    # Define core features for this dataset
    # e.g. t2m (temperature) is critical
    core_features = [f for f in all_features if f in ["t2m", "d2m"]]
    if not core_features and all_features:
        core_features = [all_features[0]]

    # =========================================================================
    # 0. LABEL SMOOTHING CONFIGURATION
    # =========================================================================
    # Label smoothing is removed as per request
    config["label_smoothing"] = {"enabled": False}

    
    # =========================================================================
    # 1. FEATURE SELECTION
    # =========================================================================
    if mode in ["features", "full"]:
        selected_features = list(core_features)
        optional_features = [f for f in all_features if f not in core_features]
        
        for feat in optional_features:
            if trial.suggest_categorical(f"use_{feat}", [True, False]):
                selected_features.append(feat)
        
        config["features"] = selected_features
        config["data"]["features"] = selected_features # Ensure sync if needed
    else:
        config["features"] = all_features
        config["data"]["features"] = all_features
    
    # =========================================================================
    # 2. FEATURE ENGINEERING CONFIGURATION
    # =========================================================================
    fe_config = config.get("feature_engineering", {}).copy()
    if not fe_config:
        fe_config = {"enabled": True, "temporal": {}, "lag": {}, "ewm": {}, "spatial": {}, "gradients": {}, "temporal_diff": {}}

    if mode in ["features", "full"]:
        # Temporal
        if "temporal" not in fe_config: fe_config["temporal"] = {}
        fe_config["temporal"]["month"] = trial.suggest_categorical("fe_month", [True, False])
        fe_config["temporal"]["day_of_year"] = trial.suggest_categorical("fe_day_of_year", [True, False])
        fe_config["temporal"]["sin_cos_annual"] = trial.suggest_categorical("fe_sin_cos_annual", [True, False])
        
        # Candidates for complex FE: t2m, d2m, stl1, tp
        candidates = [f for f in config["features"] if f in ["t2m", "d2m", "stl1", "tp"]]
        if not candidates: candidates = config["features"][:3]

        # Use settings from base_config as the "Universe" of options
        # If user configured [5, 11] windows, we optimize usage of 5 and 11.
        
        base_fe = base_config.get("feature_engineering", {})

        # Lag
        use_lag = trial.suggest_categorical("use_lag_features", [True, False])
        if use_lag:
            if "lag" not in fe_config: fe_config["lag"] = {}
            lag_vars = []
            for var in candidates:
                if trial.suggest_categorical(f"lag_{var}", [True, False]):
                    lag_vars.append(var)
            
            if lag_vars:
                fe_config["lag"]["variables"] = lag_vars
                base_lags = base_fe.get("lag", {}).get("lags", [1, 3, 7])
                lags = []
                for lg in base_lags:
                     if trial.suggest_categorical(f"lag_{lg}", [True, False]):
                         lags.append(lg)
                if not lags and base_lags: lags = [base_lags[0]] # fallback
                fe_config["lag"]["lags"] = lags
        else:
            fe_config["lag"] = {}
        
        # EWM
        use_ewm = trial.suggest_categorical("use_ewm_features", [True, False])
        if use_ewm:
            if "ewm" not in fe_config: fe_config["ewm"] = {}
            ewm_vars = []
            for var in candidates:
                if trial.suggest_categorical(f"ewm_{var}", [True, False]):
                    ewm_vars.append(var)
            
            if ewm_vars:
                fe_config["ewm"]["variables"] = ewm_vars
                base_spans = base_fe.get("ewm", {}).get("spans", [3, 7, 14])
                spans = []
                for sp in base_spans:
                     if trial.suggest_categorical(f"ewm_span_{sp}", [True, False]):
                         spans.append(sp)
                if not spans and base_spans: spans = [base_spans[0]]
                fe_config["ewm"]["spans"] = spans
        else:
            fe_config["ewm"] = {}

        # Spatial
        use_spatial = trial.suggest_categorical("use_spatial_features", [True, False])
        if use_spatial:
            if "spatial" not in fe_config: fe_config["spatial"] = {}
            spatial_vars = []
            for var in candidates:
                if trial.suggest_categorical(f"spatial_{var}", [True, False]):
                    spatial_vars.append(var)
            
            if spatial_vars:
                fe_config["spatial"]["variables"] = spatial_vars
                base_windows = base_fe.get("spatial", {}).get("window_sizes", [5])
                window_sizes = []
                for win in base_windows:
                     if trial.suggest_categorical(f"spatial_window_{win}", [True, False]):
                         window_sizes.append(win)
                if not window_sizes and base_windows: window_sizes = [base_windows[0]]
                fe_config["spatial"]["window_sizes"] = window_sizes
                fe_config["spatial"]["stats"] = ["mean", "std"]
        else:
            fe_config["spatial"] = {}

        # Gradients
        use_gradients = trial.suggest_categorical("use_gradient_features", [True, False])
        if use_gradients:
            if "gradients" not in fe_config: fe_config["gradients"] = {}
            grad_vars = []
            for var in candidates:
                if trial.suggest_categorical(f"gradient_{var}", [True, False]):
                    grad_vars.append(var)
            
            if grad_vars:
                fe_config["gradients"]["variables"] = grad_vars
        else:
            fe_config["gradients"] = {}

        # Temporal diff
        use_temporal_diff = trial.suggest_categorical("use_temporal_diff_features", [True, False])
        if use_temporal_diff:
            if "temporal_diff" not in fe_config: fe_config["temporal_diff"] = {}
            diff_vars = []
            for var in candidates:
                if trial.suggest_categorical(f"diff_{var}", [True, False]):
                    diff_vars.append(var)
            
            if diff_vars:
                fe_config["temporal_diff"]["variables"] = diff_vars
                base_periods = base_fe.get("temporal_diff", {}).get("periods", [1, 3])
                periods = []
                for p in base_periods:
                    if trial.suggest_categorical(f"diff_period_{p}", [True, False]):
                        periods.append(p)
                if not periods and base_periods: periods = [base_periods[0]]
                fe_config["temporal_diff"]["periods"] = periods
        else:
            fe_config["temporal_diff"] = {}
            
        config["feature_engineering"] = fe_config
    else:
        pass
    
    # =========================================================================
    # 3. MODEL HYPERPARAMETERS
    # =========================================================================
    loss_function = "Logloss"
    
    model_config = config.get("model", {}).copy()
    model_config["loss_function"] = loss_function
    if "eval_metric" not in model_config:
        model_config["eval_metric"] = "F1"

    if mode in ["model", "full"]:
        model_config.update({
            "iterations": trial.suggest_int("iterations", 50, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
            "random_seed": 42,
            "verbose": True,
        })
        
        grow_policy = trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"])
        model_config["grow_policy"] = grow_policy
        
        if grow_policy == "Lossguide":
            model_config["max_leaves"] = trial.suggest_int("max_leaves", 16, 128)
        else:
            model_config.pop("max_leaves", None)
            
        bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "No"])
        model_config["bootstrap_type"] = bootstrap_type
        
        if bootstrap_type == "Bayesian":
            model_config["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 2.0)
        else:
            model_config.pop("bagging_temperature", None)
            
        model_config["decision_threshold"] = trial.suggest_float("decision_threshold", 0.3, 0.7)
        
        use_class_weights = trial.suggest_categorical("use_class_weights", [True, False])
        if use_class_weights:
            weight_ratio = trial.suggest_float("class_weight_ratio", 1.0, 10.0)
            config["training"]["class_weights"] = [1.0, weight_ratio]
        
    elif mode == "features":
        if fixed_model_params:
            for k, v in fixed_model_params.items():
                if k not in ["task_type", "devices", "thread_count"]:
                    model_config[k] = v
            if "decision_threshold" in fixed_model_params:
                model_config["decision_threshold"] = fixed_model_params["decision_threshold"]

    # Enforce conditional parameter logic (cleanup invalid params)
    if model_config.get("bootstrap_type") != "Bayesian":
        model_config.pop("bagging_temperature", None)
        
    if model_config.get("grow_policy") != "Lossguide":
        model_config.pop("max_leaves", None)

    config["model"] = model_config
    
    if mode == "features" and fixed_model_params:
        if "class_weights_ratio" in fixed_model_params:
             config["training"]["class_weights"] = [1.0, fixed_model_params["class_weights_ratio"]]

    return config


def evaluate_model(
    model: CatBoostClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
    use_soft_labels: bool = False,
) -> Dict[str, float]:
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)
    y_binary = (y >= 0.5).astype(int) if use_soft_labels else y
    
    return {
        "f1": f1_score(y_binary, preds),
        "precision": precision_score(y_binary, preds, zero_division=0),
        "recall": recall_score(y_binary, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_binary, proba) if len(np.unique(y_binary)) > 1 else 0.0,
    }



def generate_super_config(base_config: Dict) -> Dict:
    """Generate a configuration with ALL possible features enabled."""
    super_config = deepcopy(base_config)
    
    fe_config = super_config.get("feature_engineering", {})
    if not fe_config:
        fe_config = {}
    
    # Enable feature engineering globally
    fe_config["enabled"] = True
    super_config["feature_engineering"] = fe_config
    
    # Disable label smoothing here
    if "label_smoothing" not in super_config:
        super_config["label_smoothing"] = {}
    super_config["label_smoothing"]["enabled"] = False
    
    return super_config



class OptunaObjective:
    def __init__(
        self,
        base_config: Dict,
        train_years: List[int],
        val_years: List[int],
        metric: str = "f1",
        early_stopping_rounds: int = 20,
        verbose: bool = False,
        mode: str = "full",
        fixed_model_params: Dict = None,
    ):
        self.base_config = base_config
        self.train_years = train_years
        self.val_years = val_years
        self.metric = metric
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.mode = mode
        self.fixed_model_params = fixed_model_params
        
        # Pre-load ALL data once
        if self.verbose:
            print("Pre-loading all data for optimization (Super Config)...")
            
        super_config = generate_super_config(base_config)
        
        # Load Training Data
        self.X_train_full, self.y_train_raw, self.meta_train = load_years(
            self.train_years,
            super_config,
            verbose=self.verbose,
            apply_feature_engineering=True,
            return_hard_labels=False, # Raw y is hard because smoothing is disabled
            return_metadata=True,
            apply_label_smoothing_flag=False,
            align_to_targets=True
        )
        
        # Load Validation Data
        self.X_val_full, self.y_val_raw, self.meta_val = load_years(
            self.val_years,
            super_config,
            verbose=self.verbose,
            apply_feature_engineering=True,
            return_hard_labels=False,
            return_metadata=True,
            apply_label_smoothing_flag=False,
            align_to_targets=True
        )
        
        if self.verbose:
            print(f"Data loaded. Train shape: {self.X_train_full.shape}, Val shape: {self.X_val_full.shape}")
    
    def __call__(self, trial: optuna.Trial) -> float:
        try:
            config = create_trial_config(trial, self.base_config, mode=self.mode, fixed_model_params=self.fixed_model_params)
            
            ls_config = config.get("label_smoothing", {})
            use_label_smoothing = ls_config.get("enabled", False)
            
            # Determine features for this trial
            required_features = get_feature_names(config["features"], config["feature_engineering"])
            
            # Intersect with available features (just in case)
            valid_features = [f for f in required_features if f in self.X_train_full.columns]
            
            if self.verbose and trial.number == 0:
                 print(f"Trial 0: Selected {len(valid_features)} features from {len(self.X_train_full.columns)} available.")
            
            # Select features (Create views/copies)
            X_train = self.X_train_full[valid_features]
            X_val = self.X_val_full[valid_features]
            
            # Prepare Targets (No Label Smoothing)
            y_train = self.y_train_raw
            y_val = self.y_val_raw
            
            model_cfg = config["model"].copy()
            decision_threshold = model_cfg.pop("decision_threshold", 0.5)
            model_cfg.pop("verbose", None)
            
            # Handle class weights params
            model_cfg.pop("use_class_weights", None)
            weight_ratio = model_cfg.pop("class_weight_ratio", None)
            
            # Construct class_weights parameter for CatBoost if enabled
            if config["model"].get("use_class_weights", False) and weight_ratio is not None:
                model_cfg["class_weights"] = [1.0, weight_ratio]
            
            # Clean up other non-catboost params
            model_cfg.pop("task_type", None)
            model_cfg.pop("devices", None)
            model_cfg.pop("thread_count", None)
            # Re-add these if they were in fixed_model_params 
            if self.fixed_model_params:
                if "task_type" in self.fixed_model_params: model_cfg["task_type"] = self.fixed_model_params["task_type"]
                if "devices" in self.fixed_model_params: model_cfg["devices"] = self.fixed_model_params["devices"]
                if "thread_count" in self.fixed_model_params: model_cfg["thread_count"] = self.fixed_model_params["thread_count"]

            # If class_weights are defined in config["training"] (legacy), respect that if not overridden
            class_weights = config["training"].get("class_weights")
            if "class_weights" not in model_cfg and class_weights is not None:
                model_cfg["class_weights"] = class_weights
            
            model_cfg["early_stopping_rounds"] = self.early_stopping_rounds
            
            model = CatBoostClassifier(**model_cfg)
            model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                verbose=False,
            )
            
            metrics = evaluate_model(
                model, X_val, y_val, 
                threshold=decision_threshold,
                use_soft_labels=False, 
            )
            
            trial.set_user_attr("val_f1", metrics["f1"])
            trial.set_user_attr("val_roc_auc", metrics["roc_auc"])
            trial.set_user_attr("n_features", len(X_train.columns))
            
            return metrics[self.metric]
            
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned()

def run_optimization(
    config_path: str = "config_climate.yaml",
    n_trials: int = 20,
    study_name: str = "heatwave_optimization_climate",
    storage: str | None = None,
    output_path: str = "optimized_params_climate.yaml",
    metric: str = "f1",
) -> None:
    base_config = load_config(config_path)
    train_years = base_config["data"]["train_years"]
    val_years = base_config["data"]["test_years"] # Use test years as val for optimization loop or split train?
    # Actually better to take last year of train as val, and keep test separate?
    # Original config had train [2017-2022] and test [2023, 2024].
    # optuna_optimize.py used 2022 for validation.
    
    # Let's split train_years for optuna:
    # Train: 2017-2021
    # Val: 2022
    opt_train_years = [y for y in train_years if y < 2022]
    opt_val_years = [2022, 2023, 2024]
    
    print(f"Optimization split: Train={opt_train_years}, Val={opt_val_years}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    objective = OptunaObjective(
        base_config=base_config,
        train_years=opt_train_years,
        val_years=opt_val_years,
        metric=metric,
        verbose=True,
        mode="model"
    )
    
    # Stage 1: Model
    print("STAGE 1: MODEL")
    study_model = optuna.create_study(
        study_name=f"{study_name}_model",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study_model.optimize(objective, n_trials=n_trials)
    
    best_trial_model = study_model.best_trial
    print(f"Stage 1 Best {metric}: {best_trial_model.value:.4f}")
    
    best_config_stage1 = create_trial_config(best_trial_model, base_config, mode="model")
    best_model_params = best_config_stage1["model"]
    if "class_weights" in best_config_stage1["training"] and best_config_stage1["training"]["class_weights"] is not None:
        best_training_params = best_config_stage1["training"]
        best_model_params["class_weights_ratio"] = best_training_params["class_weights"][1]

    # Stage 2: Features
    print("STAGE 2: FEATURES")
    objective.mode = "features"
    objective.fixed_model_params = best_model_params
    
    study_features = optuna.create_study(
        study_name=f"{study_name}_features",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study_features.optimize(objective, n_trials=n_trials)
    
    best_trial_features = study_features.best_trial
    print(f"Stage 2 Best {metric}: {best_trial_features.value:.4f}")
    
    # Save results
    best_config = create_trial_config(
        best_trial_features, 
        base_config, 
        mode="features", 
        fixed_model_params=best_model_params 
    )
    
    with open(output_path, "w") as f:
        yaml.dump(best_config, f)
    print(f"Saved optimized config to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_climate.yaml")
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()
    
    run_optimization(config_path=args.config, n_trials=args.trials)

if __name__ == "__main__":
    main()
