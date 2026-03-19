# src/models/boosting.py
"""
XGBoost - Extreme Gradient Boosting

Replaces sklearn's GradientBoosting with XGBoost (faster, better regularization).

XGBoost vs sklearn GradientBoosting:
- Same algorithm (gradient boosting)
- 2-10x faster (parallelized)
- Built-in L1 (reg_alpha) and L2 (reg_lambda) regularization
- Better handling of missing values

Key Hyperparameters:
- n_estimators: Number of boosting rounds
- learning_rate (eta): Shrinkage rate (0.01-0.3)
- max_depth: Tree depth (3-10 typical)
- subsample: Row sampling (0.5-1.0)
- colsample_bytree: Column sampling (0.5-1.0)
- reg_alpha: L1 regularization (default 0)
- reg_lambda: L2 regularization (default 1)
"""

import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBRegressor, XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def _build_preprocessing_and_model(num_cols, cat_cols, problem_type, random_state):
    """Create preprocessing + XGBoost pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    if problem_type == "classification":
        model = XGBClassifier(
            random_state=random_state,
            n_jobs=max(1, os.cpu_count() - 1),
            eval_metric='logloss',
            verbosity=0,
        )
    elif problem_type == "regression":
        model = XGBRegressor(
            random_state=random_state,
            n_jobs=max(1, os.cpu_count() - 1),
            verbosity=0,
        )
    else:
        raise ValueError(f"Unsupported problem_type: {problem_type}")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    return pipeline


def run_boosting_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictors: list,
    target_col: str,
    problem_type: str = "classification",
    random_state: int = 42,
    param_grid: dict | None = None,
    scoring: str | None = None,
    cv_folds: int | object = 5,
):
    """
    Run XGBoost experiment (classification or regression).

    XGBoost has built-in regularization:
    - reg_alpha: L1 regularization (default 0)
    - reg_lambda: L2 regularization (default 1)

    cv_folds: int for KFold/StratifiedKFold, or a cv splitter object (e.g. LeaveOneOut()).
    """
    np.random.seed(random_state)

    # =========================
    # 1. Split X / y
    # =========================
    X_train = train_df[predictors].copy()
    X_test = test_df[predictors].copy()

    y_train_raw = train_df[target_col].copy()
    y_test_raw = test_df[target_col].copy()

    # For classification, encode labels
    label_encoder = None
    if problem_type == "classification":
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train_raw)
        y_test = label_encoder.transform(y_test_raw)
    else:
        y_train = y_train_raw.values
        y_test = y_test_raw.values

    # Identify numeric & categorical columns
    num_cols = [c for c in predictors if pd.api.types.is_numeric_dtype(X_train[c])]
    cat_cols = [c for c in predictors if not pd.api.types.is_numeric_dtype(X_train[c])]

    # =========================
    # 2. Build base pipeline
    # =========================
    pipeline = _build_preprocessing_and_model(
        num_cols=num_cols,
        cat_cols=cat_cols,
        problem_type=problem_type,
        random_state=random_state,
    )

    # =========================
    # 3. Define hyperparameter grid
    # =========================
    if param_grid is None:
        if problem_type == "regression":
            param_grid = {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [3, 5],
                "model__subsample": [0.8, 1.0],
                "model__reg_alpha": [0, 0.1],
                "model__reg_lambda": [1, 10],
            }
        else:  # classification
            param_grid = {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [3, 5],
                "model__subsample": [0.8, 1.0],
                "model__scale_pos_weight": [1, 3],  # For imbalanced classes
                "model__reg_alpha": [0, 0.1],
                "model__reg_lambda": [1, 10],
            }

    # Choose scoring
    if scoring is None:
        if problem_type == "classification":
            n_classes = len(np.unique(y_train))
            if n_classes > 2:
                scoring = "f1_weighted"
            else:
                scoring = "roc_auc"
        else:
            scoring = "neg_mean_squared_error"

    # =========================
    # 4. Grid search CV
    # =========================
    if isinstance(cv_folds, int):
        if problem_type == "classification":
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    else:
        cv = cv_folds  # Pre-built cv splitter (e.g. LeaveOneOut)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=max(1, os.cpu_count() - 1),
        refit=True,
        verbose=1,
        return_train_score=True,
    )

    print(f"\n=== GRID SEARCH: XGBOOST ({problem_type}) ===")
    print(f"Training samples: {len(X_train)}")
    print("XGBoost has built-in L1 (reg_alpha) and L2 (reg_lambda) regularization")
    print("Fitting GridSearchCV...")

    grid_search.fit(X_train, y_train)

    best_estimator = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    best_index = grid_search.best_index_

    cv_train_score = grid_search.cv_results_["mean_train_score"][best_index]
    cv_val_score = grid_search.cv_results_["mean_test_score"][best_index]

    print("\n=== GRID SEARCH COMPLETE ===")
    print("Best params:", best_params)
    print(f"Best CV score ({scoring}): {best_cv_score:.4f}")

    # =========================
    # 5. Evaluate on TEST set
    # =========================
    print("\n=== EVALUATING ON TEST SET ===")
    y_pred = best_estimator.predict(X_test)

    test_metrics = {}
    y_proba = None

    if problem_type == "classification":
        # Multi-class aware metrics
        n_classes = len(np.unique(y_train))
        avg = "weighted" if n_classes > 2 else "binary"

        # Get probabilities for ROC
        if hasattr(best_estimator.named_steps["model"], "predict_proba"):
            y_proba_full = best_estimator.predict_proba(X_test)
            if n_classes > 2:
                # Multiclass: use full probability matrix
                y_proba = y_proba_full
                try:
                    roc_auc = roc_auc_score(
                        y_test,
                        y_proba,
                        multi_class="ovr",
                        average="weighted",
                    )
                except ValueError:
                    roc_auc = np.nan
            else:
                # Binary: use probabilities for positive class
                y_proba = y_proba_full[:, 1]
                try:
                    roc_auc = roc_auc_score(y_test, y_proba)
                except ValueError:
                    roc_auc = np.nan
        else:
            roc_auc = np.nan
            y_proba = None

        test_metrics["accuracy"] = accuracy_score(y_test, y_pred)
        test_metrics["precision"] = precision_score(y_test, y_pred, average=avg, zero_division=0)
        test_metrics["recall"] = recall_score(y_test, y_pred, average=avg, zero_division=0)
        test_metrics["f1"] = f1_score(y_test, y_pred, average=avg, zero_division=0)
        test_metrics["roc_auc"] = roc_auc

        print(f"Test Accuracy : {test_metrics['accuracy']:.4f}")
        print(f"Test Precision: {test_metrics['precision']:.4f}")
        print(f"Test Recall   : {test_metrics['recall']:.4f}")
        print(f"Test F1       : {test_metrics['f1']:.4f}")
        print(f"Test ROC-AUC  : {test_metrics['roc_auc']:.4f}")
    else:
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        test_metrics["mse"] = mse
        test_metrics["rmse"] = rmse
        test_metrics["mae"] = mae
        test_metrics["r2"] = r2

        print(f"Test R²  : {r2:.4f}")
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test MAE : {mae:.4f}")

    # =========================
    # 6. Evaluate on TRAIN set (for overfitting detection)
    # =========================
    y_train_pred = best_estimator.predict(X_train)

    train_metrics = {}
    if problem_type == "classification":
        n_classes = len(np.unique(y_train))
        avg = "weighted" if n_classes > 2 else "binary"
        train_metrics["accuracy"] = accuracy_score(y_train, y_train_pred)
        train_metrics["precision"] = precision_score(y_train, y_train_pred, average=avg, zero_division=0)
        train_metrics["recall"] = recall_score(y_train, y_train_pred, average=avg, zero_division=0)
        train_metrics["f1"] = f1_score(y_train, y_train_pred, average=avg, zero_division=0)
        # Train ROC-AUC
        try:
            if hasattr(best_estimator.named_steps["model"], "predict_proba"):
                y_train_proba = best_estimator.predict_proba(X_train)
                if n_classes == 2:
                    train_metrics["roc_auc"] = roc_auc_score(y_train, y_train_proba[:, 1])
                else:
                    train_metrics["roc_auc"] = roc_auc_score(y_train, y_train_proba, multi_class="ovr")
        except Exception:
            train_metrics["roc_auc"] = None
    else:
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_metrics["mse"] = train_mse
        train_metrics["rmse"] = np.sqrt(train_mse)
        train_metrics["mae"] = mean_absolute_error(y_train, y_train_pred)
        train_metrics["r2"] = r2_score(y_train, y_train_pred)

    # =========================
    # 7. Get feature importances
    # =========================
    xgb_model = best_estimator.named_steps["model"]
    feature_importances = xgb_model.feature_importances_
    
    try:
        preprocessor = best_estimator.named_steps["preprocessor"]
        feature_names = preprocessor.get_feature_names_out().tolist()
    except AttributeError:
        feature_names = None

    # =========================
    # 8. Pack results
    # =========================
    results = {
        "model": best_estimator,
        "best_params": best_params,
        "cv_scoring": scoring,
        "cv_train_score": float(cv_train_score),
        "cv_val_score": float(cv_val_score),
        "grid_search": grid_search,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "y_test_raw": y_test_raw,
        "feature_importances": feature_importances,
        "feature_names": feature_names,
        "predictors": predictors,
        "target_col": target_col,
        "problem_type": problem_type,
        "label_encoder": label_encoder,
    }

    return results
