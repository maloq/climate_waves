"""
Optuna hyperparameter optimization for event prediction model.

Optimizes:
1. Feature selection (which base columns to include)
2. Feature engineering toggles
3. CatBoost model hyperparameters

Uses 2022 for validation and 2023 for final test evaluation.
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
from sklearn.model_selection import train_test_split

from feature_engineering import engineer_features
from load_data import apply_label_smoothing, load_config, load_years

warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# BASE FEATURES AVAILABLE FOR SELECTION
# =============================================================================

# Features are now derived from config.yaml


def create_trial_config(
    trial: optuna.Trial, 
    base_config: Dict, 
    mode: str = "full", 
    fixed_model_params: Dict = None
) -> Dict:
    """Create a configuration dictionary based on Optuna trial suggestions.
    
    Args:
        trial: Optuna trial object
        base_config: Base configuration dictionary
        mode: Optimization mode - "model", "features", or "full"
        fixed_model_params: Dictionary of fixed model parameters (used in "features" mode)
    """
    config = deepcopy(base_config)
    
    # Get all available features from config
    all_features = config.get("features", [])
    
    # Define core features (heuristic: temperature variables are critical)
    core_features = [f for f in all_features if "temperature" in f]
    if not core_features and all_features:
        # If no temperature features, pick the first few as core to avoid empty set
        core_features = all_features[:3]
    
    # =========================================================================
    # 0. LABEL SMOOTHING CONFIGURATION
    # =========================================================================
    # Only optimize label smoothing in "model" or "full" mode
    if mode in ["model", "full"]:
        use_label_smoothing = trial.suggest_categorical("use_label_smoothing", [True, False])
        
        if use_label_smoothing:
            ls_config = {
                "enabled": True,
                "temporal_sigma": trial.suggest_float("ls_temporal_sigma", 0.5, 3.0),
                "temporal_radius": trial.suggest_int("ls_temporal_radius", 1, 5),
                "spatial_sigma": trial.suggest_float("ls_spatial_sigma", 0.5, 2.0),
                "spatial_radius": trial.suggest_int("ls_spatial_radius", 1, 3),
                "max_smooth_value": 1.0,
                "min_smooth_value": 0.0,
            }
            config["label_smoothing"] = ls_config
        else:
            config["label_smoothing"] = {"enabled": False}
    else:
        # In "features" mode, assume label smoothing config is fixed/passed in base_config
        pass
    
    # =========================================================================
    # 1. FEATURE SELECTION
    # =========================================================================
    if mode in ["features", "full"]:
        selected_features = list(core_features)
        
        # Optional features to include/exclude
        optional_features = [f for f in all_features if f not in core_features]
        
        for feat in optional_features:
            if trial.suggest_categorical(f"use_{feat}", [True, False]):
                selected_features.append(feat)
        
        config["features"] = selected_features
        config["data"]["features"] = selected_features
    else:
        # Mode "model": Use ALL features
        config["features"] = all_features
        config["data"]["features"] = all_features
    
    # =========================================================================
    # 2. FEATURE ENGINEERING CONFIGURATION
    # =========================================================================
    fe_config = config.get("feature_engineering", {}).copy()
    if not fe_config:
        fe_config = {"enabled": True, "temporal": {}, "lag": {}, "ewm": {}, "spatial": {}, "gradients": {}, "temporal_diff": {}}

    if mode in ["features", "full"]:
        # Optimize FE toggles
        
        # Temporal features
        if "temporal" not in fe_config: fe_config["temporal"] = {}
        fe_config["temporal"]["month"] = trial.suggest_categorical("fe_month", [True, False])
        fe_config["temporal"]["day_of_year"] = trial.suggest_categorical("fe_day_of_year", [True, False])
        fe_config["temporal"]["sin_cos_annual"] = trial.suggest_categorical("fe_sin_cos_annual", [True, False])
        fe_config["temporal"]["sin_cos_semiannual"] = trial.suggest_categorical("fe_sin_cos_semiannual", [True, False])
        
        # Lag features
        use_lag = trial.suggest_categorical("use_lag_features", [True, False])
        if use_lag:
            if "lag" not in fe_config: fe_config["lag"] = {}
            lag_vars = []
            # Propose lags for temperature features (commonly useful)
            candidates = [f for f in config["features"] if "temperature" in f or "geopotential" in f]
            
            for var in candidates:
                if trial.suggest_categorical(f"lag_{var}", [True, False]):
                    lag_vars.append(var)
            
            if lag_vars:
                lags = []
                if trial.suggest_categorical("lag_1", [True, False]): lags.append(1)
                if trial.suggest_categorical("lag_3", [True, False]): lags.append(3)
                if trial.suggest_categorical("lag_7", [True, False]): lags.append(7)
                
                if lags:
                    fe_config["lag"]["variables"] = lag_vars
                    fe_config["lag"]["lags"] = lags
        else:
            fe_config["lag"] = {} # Disable if not used
        
        # EWM features
        use_ewm = trial.suggest_categorical("use_ewm_features", [True, False])
        if use_ewm:
            if "ewm" not in fe_config: fe_config["ewm"] = {}
            ewm_vars = []
            candidates = [f for f in config["features"] if "temperature" in f]
            
            for var in candidates:
                if trial.suggest_categorical(f"ewm_{var}", [True, False]):
                    ewm_vars.append(var)
            
            if ewm_vars:
                spans = []
                if trial.suggest_categorical("ewm_span_3", [True, False]): spans.append(3)
                if trial.suggest_categorical("ewm_span_7", [True, False]): spans.append(7)
                if trial.suggest_categorical("ewm_span_14", [True, False]): spans.append(14)
                
                if spans:
                    fe_config["ewm"]["variables"] = ewm_vars
                    fe_config["ewm"]["spans"] = spans
        else:
            fe_config["ewm"] = {}

        # Spatial features
        use_spatial = trial.suggest_categorical("use_spatial_features", [True, False])
        if use_spatial:
            if "spatial" not in fe_config: fe_config["spatial"] = {}
            spatial_vars = []
            candidates = [f for f in config["features"] if "temperature" in f or "geopotential" in f]
            
            for var in candidates:
                if trial.suggest_categorical(f"spatial_{var}", [True, False]):
                    spatial_vars.append(var)
            
            if spatial_vars:
                window_sizes = []
                if trial.suggest_categorical("spatial_window_3", [True, False]): window_sizes.append(3)
                if trial.suggest_categorical("spatial_window_5", [True, False]): window_sizes.append(5)
                
                if window_sizes:
                    fe_config["spatial"]["variables"] = spatial_vars
                    fe_config["spatial"]["window_sizes"] = window_sizes
                    fe_config["spatial"]["stats"] = ["mean", "std"]
        else:
            fe_config["spatial"] = {}

        # Gradient features
        use_gradients = trial.suggest_categorical("use_gradient_features", [True, False])
        if use_gradients:
            if "gradients" not in fe_config: fe_config["gradients"] = {}
            grad_vars = []
            candidates = [f for f in config["features"] if "temperature" in f or "geopotential" in f]
            
            for var in candidates:
                if trial.suggest_categorical(f"gradient_{var}", [True, False]):
                    grad_vars.append(var)
            
            if grad_vars:
                fe_config["gradients"]["variables"] = grad_vars
        else:
            fe_config["gradients"] = {}

        # Temporal diff features
        use_temporal_diff = trial.suggest_categorical("use_temporal_diff_features", [True, False])
        if use_temporal_diff:
            if "temporal_diff" not in fe_config: fe_config["temporal_diff"] = {}
            diff_vars = []
            candidates = [f for f in config["features"] if "temperature" in f]
            
            for var in candidates:
                if trial.suggest_categorical(f"diff_{var}", [True, False]):
                    diff_vars.append(var)
            
            if diff_vars:
                periods = []
                if trial.suggest_categorical("diff_period_1", [True, False]): periods.append(1)
                if trial.suggest_categorical("diff_period_3", [True, False]): periods.append(3)
                
                if periods:
                    fe_config["temporal_diff"]["variables"] = diff_vars
                    fe_config["temporal_diff"]["periods"] = periods
        else:
            fe_config["temporal_diff"] = {}
            
        config["feature_engineering"] = fe_config
    else:
        # Mode "model": Use default FE config from base_config
        pass
    
    # =========================================================================
    # 3. MODEL HYPERPARAMETERS
    # =========================================================================
    
    ls_enabled = config.get("label_smoothing", {}).get("enabled", False)
    loss_function = "CrossEntropy" if ls_enabled else "Logloss"
    
    # Start with base model config
    model_config = config.get("model", {}).copy()
    
    # Ensure correct loss for usage
    model_config["loss_function"] = loss_function
    # Use F1 (or configured metric) on hard labels even if label smoothing is on
    if "eval_metric" not in model_config:
        model_config["eval_metric"] = "F1"

    if mode in ["model", "full"]:
        # Update with trial suggestions
        model_config.update({
            "iterations": trial.suggest_int("iterations", 50, 800, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
            # "border_count": trial.suggest_int("border_count", 32, 254),
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
            
        # Decision threshold
        model_config["decision_threshold"] = trial.suggest_float("decision_threshold", 0.3, 0.7)
        
        # Class weights for imbalanced data
        use_class_weights = trial.suggest_categorical("use_class_weights", [True, False])
        if use_class_weights:
            weight_ratio = trial.suggest_float("class_weight_ratio", 1.0, 10.0)
            config["training"]["class_weights"] = [1.0, weight_ratio]
        
    elif mode == "features":
        # Use fixed model params
        if fixed_model_params:
            # We only update keys that are actually hyperparameters, preserving device settings
            for k, v in fixed_model_params.items():
                if k not in ["task_type", "devices", "thread_count"]:
                    model_config[k] = v
        
            # Restore decision_threshold if it was optimized
            if "decision_threshold" in fixed_model_params:
                model_config["decision_threshold"] = fixed_model_params["decision_threshold"]
            
            # Restore class weights if they were optimized
            if "class_weight_ratio" in fixed_model_params:
                # Reconstruct class weights list
                 # This part is tricky because we don't save the list structure in simple params
                 # We need to pass the full 'training' config or reconstruct it.
                 # Let's assume fixed_model_params dictates model params, 
                 # and we should also fix training params.
                 pass

    config["model"] = model_config
    
    # For training config (class_weights), if mode == features, we might need to carry it over.
    if mode == "features" and fixed_model_params:
        # Check if we have class weights info
        if "class_weights_ratio" in fixed_model_params: # Assuming we saved it with this key
             config["training"]["class_weights"] = [1.0, fixed_model_params["class_weights_ratio"]]
        elif "use_class_weights" in fixed_model_params and fixed_model_params["use_class_weights"]:
             # If we just have the params, we might need to be consistent.
             pass

    return config


def load_data_for_optimization(
    config: Dict,
    train_years: List[int],
    val_years: List[int],
    test_years: List[int],
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load data for training, validation, and testing."""
    if verbose:
        print(f"Loading training data (years: {train_years})...")
    
    # Modify config for each data load
    train_config = deepcopy(config)
    train_config["data"]["train_years"] = train_years
    X_train, y_train = load_years(train_years, config=train_config, verbose=verbose)
    
    if verbose:
        print(f"Loading validation data (years: {val_years})...")
    val_config = deepcopy(config)
    val_config["data"]["train_years"] = val_years
    X_val, y_val = load_years(val_years, config=val_config, verbose=verbose)
    
    if verbose:
        print(f"Loading test data (years: {test_years})...")
    test_config = deepcopy(config)
    test_config["data"]["train_years"] = test_years
    X_test, y_test = load_years(test_years, config=test_config, verbose=verbose)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def evaluate_model(
    model: CatBoostClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
    use_soft_labels: bool = False,
) -> Dict[str, float]:
    """Evaluate model and return metrics.
    
    Args:
        model: Trained CatBoost model
        X: Features
        y: Labels (can be soft labels if use_soft_labels=True)
        threshold: Decision threshold for predictions
        use_soft_labels: If True, binarize y for metrics that require binary labels
    """
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)
    
    # For soft labels, binarize ground truth for classification metrics
    y_binary = (y >= 0.5).astype(int) if use_soft_labels else y
    
    return {
        "f1": f1_score(y_binary, preds),
        "precision": precision_score(y_binary, preds, zero_division=0),
        "recall": recall_score(y_binary, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_binary, proba) if len(np.unique(y_binary)) > 1 else 0.0,
    }





class OptunaObjective:
    """Optuna objective function for hyperparameter optimization.
    
    Loads all raw data once at initialization and applies feature selection
    and engineering in memory for each trial.
    """
    
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
        
        # Cache for processed data (features+FE config -> DataFrames)
        # Value is (X_train, y_train, y_train_hard, X_val, y_val, y_val_hard)
        self._data_cache: Dict[str, Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.Series]] = {}
    
    def _get_cache_key(self, config: Dict) -> str:
        """Generate cache key based on features, feature engineering, and label smoothing config."""
        cache_params = {
            "features": sorted(config["features"]),
            "feature_engineering": config.get("feature_engineering", {}),
            "label_smoothing": config.get("label_smoothing", {}),
            "train_years": sorted(self.train_years),
            "val_years": sorted(self.val_years),
        }
        param_str = json.dumps(cache_params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()[:16]

    def __call__(self, trial: optuna.Trial) -> float:
        try:
            # Create trial-specific config
            config = create_trial_config(trial, self.base_config, mode=self.mode, fixed_model_params=self.fixed_model_params)
            
            # Check if using label smoothing
            ls_config = config.get("label_smoothing", {})
            use_label_smoothing = ls_config.get("enabled", False)
            
            # Check cache
            cache_key = self._get_cache_key(config)
            
            if cache_key not in self._data_cache:
                if self.verbose:
                    print(f"  Trial {trial.number}: Loading/Computing features...", end=" ", flush=True)
                
                # Load training data
                X_train, y_train, y_train_hard = load_years(
                    self.train_years,
                    config,
                    verbose=False, # Reduce verbosity of inner loader
                    apply_feature_engineering=True,
                    return_hard_labels=True,
                    apply_label_smoothing_flag=True
                )

                # Load validation data
                X_val, y_val, y_val_hard = load_years(
                    self.val_years,
                    config,
                    verbose=False,
                    apply_feature_engineering=True,
                    return_hard_labels=True,
                    apply_label_smoothing_flag=True
                )
                
                self._data_cache[cache_key] = (X_train, y_train, y_train_hard, X_val, y_val, y_val_hard)

                
                if self.verbose:
                    print("done.")
                    ls_info = " (label smoothing)" if use_label_smoothing else ""
                    print(f"  Trial {trial.number}: {len(X_train.columns)} features, "
                          f"{len(X_train):,} train samples{ls_info}")
            else:
                X_train, y_train, y_train_hard, X_val, y_val, y_val_hard = self._data_cache[cache_key]
                if self.verbose:
                     print(f"  Trial {trial.number}: Cache hit (features/config).")
            
            # Get model config
            model_cfg = config["model"].copy()
            decision_threshold = model_cfg.pop("decision_threshold", 0.5)
            model_cfg.pop("verbose", None)
            
            # Apply class weights if specified (only for hard labels)
            class_weights = config["training"].get("class_weights")
            if class_weights is not None and not use_label_smoothing:
                model_cfg["class_weights"] = class_weights
            
            # Add early stopping
            model_cfg["early_stopping_rounds"] = self.early_stopping_rounds
            
            # Train model
            if self.verbose:
                print(f"  Trial {trial.number}: Training model...", end=" ", flush=True)

            model = CatBoostClassifier(**model_cfg)
            model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val_hard),
                verbose=False,
            )
            
            if self.verbose:
                print("done.")
            
            # Evaluate on validation set
            # CRITICAL: Always evaluate on HARD labels (y_val_hard) even if training on soft labels
            metrics = evaluate_model(
                model, X_val, y_val_hard, 
                threshold=decision_threshold,
                use_soft_labels=False, # We are passing hard labels
            )
            
            # Report intermediate values for pruning
            trial.set_user_attr("val_f1", metrics["f1"])
            trial.set_user_attr("val_precision", metrics["precision"])
            trial.set_user_attr("val_recall", metrics["recall"])
            trial.set_user_attr("val_roc_auc", metrics["roc_auc"])
            trial.set_user_attr("n_features", len(X_train.columns))
            trial.set_user_attr("features", list(X_train.columns))
            trial.set_user_attr("decision_threshold", decision_threshold)
            trial.set_user_attr("label_smoothing", use_label_smoothing)
            if use_label_smoothing:
                trial.set_user_attr("ls_temporal_sigma", ls_config.get("temporal_sigma"))
                trial.set_user_attr("ls_spatial_sigma", ls_config.get("spatial_sigma"))
            
            return metrics[self.metric]
            
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned()


def run_optimization(
    config_path: str = "config.yaml",
    n_trials: int = 100,
    study_name: str = "event_optimization",
    storage: str | None = None,
    output_path: str = "optimized_params.yaml",
    metric: str = "f1",
    train_years: List[int] | None = None,
    val_years: List[int] | None = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run Optuna hyperparameter optimization in two stages.
    
    Stage 1: Use all features, optimize model parameters.
    Stage 2: Use best model parameters, optimize feature selection and engineering.
    """
    # Load base config
    base_config = load_config(config_path)
    
    # Set default years
    if train_years is None:
        train_years = [2017, 2018, 2019, 2020, 2021]
    if val_years is None:
        val_years = [2022, 2023]  # Use both years for scoring
    
    output_path = Path(output_path)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Shared objective instance to recycle data cache
    # Initialize with default mode "model" first
    objective = OptunaObjective(
        base_config=base_config,
        train_years=train_years,
        val_years=val_years,
        metric=metric,
        verbose=verbose,
        mode="model"
    )
    
    # =========================================================================
    # STAGE 1: OPTIMIZE MODEL PARAMETERS (with ALL features)
    # =========================================================================
    print("=" * 70)
    print(f"STAGE 1: OPTIMIZE MODEL PARAMETERS (All Features)")
    print(f"Device: {base_config['model'].get('task_type', 'CPU')}")
    print("=" * 70)
    
    # Configure objective for Stage 1
    objective.mode = "model"
    objective.fixed_model_params = None
    
    study_model = optuna.create_study(
        study_name=f"{study_name}_model",
        storage=f"sqlite:///{storage}" if storage else None,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True,
    )
    
    study_model.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True,
        n_jobs=1,
    )
    
    # Save Stage 1 Logs
    log_path_model = output_dir / f"optuna_log_stage1_model.csv"
    study_model.trials_dataframe().to_csv(log_path_model, index=False)
    print(f"Stage 1 logs saved to: {log_path_model}")
    
    best_trial_model = study_model.best_trial
    print(f"Stage 1 Best {metric}: {best_trial_model.value:.4f}")
    
    # Extract best model parameters
    best_config_stage1 = create_trial_config(best_trial_model, base_config, mode="model")
    best_model_params = best_config_stage1["model"]
    
    # Capture class weights if optimized
    best_training_params = best_config_stage1["training"]
    if "class_weights" in best_training_params and best_training_params["class_weights"]:
         best_model_params["class_weights_ratio"] = best_training_params["class_weights"][1]
         best_model_params["use_class_weights"] = True
    
    # =========================================================================
    # STAGE 2: OPTIMIZE FEATURES (with Fixed Model Params)
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"STAGE 2: OPTIMIZE FEATURES (Fixed Model Params)")
    print(f"Device: {base_config['model'].get('task_type', 'CPU')}")
    print("=" * 70)
    
    # Configure objective for Stage 2
    objective.mode = "features"
    objective.fixed_model_params = best_model_params
    
    study_features = optuna.create_study(
        study_name=f"{study_name}_features",
        storage=f"sqlite:///{storage}" if storage else None,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True,
    )
    
    study_features.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True,
        n_jobs=1,
    )
    
    # Save Stage 2 Logs
    log_path_features = output_dir / f"optuna_log_stage2_features.csv"
    study_features.trials_dataframe().to_csv(log_path_features, index=False)
    print(f"Stage 2 logs saved to: {log_path_features}")
    
    best_trial_features = study_features.best_trial
    print(f"Stage 2 Best {metric}: {best_trial_features.value:.4f}")
    
    # Reconstruct final best config
    best_config = create_trial_config(
        best_trial_features, 
        base_config, 
        mode="features", 
        fixed_model_params=best_model_params 
    )
    
    # =========================================================================
    # FINAL EVALUATION
    # =========================================================================
    print("\n" + "-" * 70)
    print("FINAL MODEL TRAINING AND EVALUATION")
    print("-" * 70)
    
    best_ls_config = best_config.get("label_smoothing", {})
    best_use_label_smoothing = best_ls_config.get("enabled", False)
    
    if best_use_label_smoothing:
        print(f"Best model uses label smoothing: temporal_sigma={best_ls_config.get('temporal_sigma')}, "
              f"spatial_sigma={best_ls_config.get('spatial_sigma')}")

    # Load training data
    print("\nLoading training data...")
    X_train, y_train = load_years(
        train_years,
        config=best_config,
        verbose=True,
    )
    
    # Train final model
    model_cfg = best_config["model"].copy()
    decision_threshold = model_cfg.pop("decision_threshold", 0.5)
    model_cfg.pop("verbose", None)
    
    # Ensure class weights are consistent
    class_weights = best_config["training"].get("class_weights")
    if class_weights is not None and not best_use_label_smoothing:
        model_cfg["class_weights"] = class_weights
    
    print("\nTraining final model...")
    final_model = CatBoostClassifier(**model_cfg)
    final_model.fit(X_train, y_train, verbose=10)
    
    # Evaluation
    eval_config = deepcopy(best_config)
    eval_config["label_smoothing"] = {"enabled": False}
    
    print("\n" + "-" * 70)
    print("PER-YEAR EVALUATION RESULTS")
    print("-" * 70)
    
    per_year_metrics = {}
    all_X_val = []
    all_y_val = []
    
    for year in val_years:
        print(f"\nLoading {year} data...")
        X_year, y_year, y_year_hard = load_years(
            [year],
            config=eval_config,
            verbose=False,
            return_hard_labels=True,
        )
        
        metrics = evaluate_model(
            final_model, 
            X_year, 
            y_year_hard, 
            threshold=decision_threshold,
            use_soft_labels=False
        )
        print(f"Year {year}: F1={metrics['f1']:.4f}, AUC={metrics['roc_auc']:.4f}")
        
        per_year_metrics[year] = metrics
        all_X_val.append(X_year)
        all_y_val.append(y_year_hard)
    
    # Combined evaluation
    X_val_combined = pd.concat(all_X_val, axis=0).reset_index(drop=True)
    y_val_combined = pd.concat(all_y_val, axis=0).reset_index(drop=True)
    
    combined_metrics = evaluate_model(
        final_model, X_val_combined, y_val_combined,
        threshold=decision_threshold,
        use_soft_labels=False,
    )
    
    print(f"\n  Combined {val_years} Results:")
    print(f"    F1 Score:   {combined_metrics['f1']:.4f}")
    print(f"    ROC AUC:    {combined_metrics['roc_auc']:.4f}")
    
    # Prepare output dictionary
    output = {
        "optimization": {
            "study_name": study_name,
            "n_trials": n_trials,
            "metric": metric,
            "stage1_best_metric": float(best_trial_model.value),
            "stage2_best_metric": float(best_trial_features.value),
            "train_years": train_years,
            "val_years": val_years,
        },
        "combined_metrics": combined_metrics,
        "per_year_metrics": per_year_metrics,
        "features": best_config["features"],
        "feature_engineering": best_config["feature_engineering"],
        "label_smoothing": best_config.get("label_smoothing", {"enabled": False}),
        "model": best_config["model"],
        "training": best_config["training"]
    }
    
    # Save to file
    with open(output_path, "w") as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nSaved optimized parameters to: {output_path}")
    
    # Also save best trial parameters as JSON for reference (Stage 2 trial)
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(best_trial_features.params, f, indent=2)
    
    print(f"Saved trial parameters to: {json_path}")
    
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter optimization for heatwave prediction"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to base config YAML",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of optimization trials (default: 100)",
    )
    parser.add_argument(
        "--study-name",
        default="heatwave_optimization",
        help="Name for the Optuna study",
    )
    parser.add_argument(
        "--storage",
        default=None,
        help="SQLite storage path for study persistence (e.g., optuna_study.db)",
    )
    parser.add_argument(
        "--output",
        default="optimized_params.yaml",
        help="Path to save optimized parameters",
    )
    parser.add_argument(
        "--metric",
        choices=["f1", "precision", "recall", "roc_auc"],
        default="f1",
        help="Metric to optimize (default: f1)",
    )
    parser.add_argument(
        "--train-years",
        type=int,
        nargs="+",
        default=None,
        help="Years for training (default: 2017-2021)",
    )
    parser.add_argument(
        "--val-years",
        type=int,
        nargs="+",
        default=None,
        help="Years for validation/scoring (default: 2022, 2023)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    
    args = parser.parse_args()
    
    run_optimization(
        config_path=args.config,
        n_trials=args.n_trials,
        study_name=args.study_name,
        storage=args.storage,
        output_path=args.output,
        metric=args.metric,
        train_years=args.train_years,
        val_years=args.val_years,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()



