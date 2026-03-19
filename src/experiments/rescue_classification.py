"""
Rescue Experiment - Classification (Helped vs Did Not Help)

Within-subjects design: 3 trials per participant, counterbalanced.
- Condition 0: Empty ocean (baseline, no victims) -> always PeopleRescued=0
- Condition 1: Single stranded person
- Condition 2: Multiple stranded people (4 victims)

Binary target: Helped (PeopleRescued > 0) = 1, Did Not Help = 0

PIPELINE:
Step 2 - ML on Conditions 1 & 2 only:
  Classification: predict Helped from TimeNearVictim, TimeElapsed,
  DrownCount, Condition, Age, Gender, Group
Step 3 - Condition 0 as covariate:
  Extract each participant's Condition 0 TimeElapsed and DrownCount
  as baseline navigation skill, merge onto victim-condition rows.

CV Strategy: Leave-One-Out Cross-Validation (LOOCV)
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.dont_write_bytecode = True

import numpy as np
from sklearn.model_selection import train_test_split, LeaveOneOut

from utils.load.load_rescue import load_rescue_data
from utils.clean.clean_rescue import clean_rescue, build_model_df
from utils.eda.eda_rescue_classification import eda_rescue_classification

from models.boosting import run_boosting_experiment
from models.random_forest import run_random_forest_experiment
from models.neural_net import run_neural_net_experiment

from graphs.rescue_classification_plots import (
    plot_rescue_cls_boosting_summary,
    plot_rescue_cls_random_forest_summary,
    plot_rescue_cls_neural_network_summary,
    plot_rescue_cls_model_comparison,
)


# ==========================================
# Constants
# ==========================================
RANDOM_STATE = 42
BOOSTING_NAME = "boosting"
RANDOM_FOREST_NAME = "random_forest"
NEURAL_NETWORK_NAME = "neural_network"

TARGET_COL = "Helped"


# ==========================================
# Model Generator Functions
# ==========================================
def generate_boosting(train_df, test_df, random_state, predictors, target_col, cv):
    """XGBoost classification with LOOCV."""
    param_grid = {
        "model__n_estimators": [10, 15, 25, 50],
        "model__learning_rate": [0.05, 0.1, 0.01],
        "model__max_depth": [3, 5, 7, 10],
        "model__subsample": [0.8, 1.0],
        "model__reg_lambda": [1, 10],
    }
    return run_boosting_experiment(
        train_df=train_df,
        test_df=test_df,
        predictors=predictors,
        target_col=target_col,
        problem_type="classification",
        random_state=random_state,
        param_grid=param_grid,
        scoring="f1",
        cv_folds=cv,
    )


def generate_random_forest(train_df, test_df, random_state, predictors, target_col, cv):
    """Random Forest classification with LOOCV."""
    param_grid = {
        "model__n_estimators": [10, 15, 25, 50],
        "model__max_depth": [5, 10, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 5],
        "model__max_features": ["sqrt", "log2"],
    }
    return run_random_forest_experiment(
        train_df=train_df,
        test_df=test_df,
        predictors=predictors,
        target_col=target_col,
        problem_type="classification",
        random_state=random_state,
        param_grid=param_grid,
        scoring="f1",
        cv_folds=cv,
    )


def generate_neural_network(train_df, test_df, random_state, predictors, target_col, cv):
    """Neural Network (MLP) classification with LOOCV."""
    param_grid = {
        "model__hidden_layer_sizes": [(8,), (16,), (32,), (16, 8)],
        "model__alpha": [0.01, 0.1, 0.001],
        "model__learning_rate_init": [0.01, 0.05, 0.001],
        "model__batch_size": ["auto", 16, 32],
        "model__early_stopping": [False],
    }
    return run_neural_net_experiment(
        train_df=train_df,
        test_df=test_df,
        predictors=predictors,
        target_col=target_col,
        problem_type="classification",
        random_state=random_state,
        param_grid=param_grid,
        scoring="f1",
        cv_folds=cv,
    )


# ==========================================
# Main Experiment
# ==========================================
def main():
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    print("\n" + "=" * 60)
    print("RESCUE EXPERIMENT - CLASSIFICATION (Helped vs Not Helped)")
    print("=" * 60)

    # ==========================================
    # 1. Load and Clean Data
    # ==========================================
    data = load_rescue_data(curr_dir)
    clean_df = clean_rescue(data)

    # ==========================================
    # 1b. EDA
    # ==========================================
    eda_rescue_classification(clean_df)

    # ==========================================
    # 2. Build Model DataFrame (Steps 2 & 3)
    # ==========================================
    model_df = build_model_df(clean_df)

    # ==========================================
    # 3. Create Binary Target
    # ==========================================
    model_df[TARGET_COL] = (model_df["PeopleRescued"] > 0).astype(int)

    print(f"\n--- Binary Target Distribution ---")
    print(f"Helped (1): {model_df[TARGET_COL].sum()}")
    print(f"Not Helped (0): {(model_df[TARGET_COL] == 0).sum()}")

    # ==========================================
    # 4. Define Features and Target
    # ==========================================
    # PeopleRescued: direct source of Helped label — leakage, excluded from model
    # TrialNumber: perfectly confounded with Condition within each Group
    exclude_cols = ["ParticipantID", "TrialNumber", "PeopleRescued", TARGET_COL]
    predictors = [c for c in model_df.columns if c not in exclude_cols]

    print(f"\n[rescue_classification.py] Target: {TARGET_COL}")
    print(f"[rescue_classification.py] Predictors ({len(predictors)}): {predictors}")
    print(f"[rescue_classification.py] Excluded (leakage): PeopleRescued")

    # ==========================================
    # 5. Experiment Configuration
    # ==========================================
    n_samples = len(model_df)
    loo = LeaveOneOut()

    print(f"\n[rescue_classification.py] Samples: {n_samples}")
    print(f"[rescue_classification.py] CV: Leave-One-Out ({n_samples} folds)")

    split_configs = {}
    for name, test_size in [("50_50", 0.50), ("80_20", 0.20)]:
        n_train = int(n_samples * (1 - test_size))
        if n_train >= 2:
            split_configs[name] = test_size
        else:
            print(f"[rescue_classification.py] Skipping {name} split: only {n_train} train samples")

    if not split_configs:
        print(f"[rescue_classification.py] WARNING: Dataset too small ({n_samples} rows).")
        return

    n_trials = 3

    results = {
        BOOSTING_NAME: {},
        RANDOM_FOREST_NAME: {},
        NEURAL_NETWORK_NAME: {},
    }

    # ==========================================
    # 6. Run Experiments
    # ==========================================
    for split_name, test_size in split_configs.items():
        print(f"\n{'=' * 60}")
        print(f"SPLIT {split_name} (test_size={test_size}, cv=LOOCV)")
        print(f"{'=' * 60}")

        results[BOOSTING_NAME][split_name] = []
        results[RANDOM_FOREST_NAME][split_name] = []
        results[NEURAL_NETWORK_NAME][split_name] = []

        for trial in range(n_trials):
            print(f"\n----- Trial {trial + 1}/{n_trials} -----")

            rs = RANDOM_STATE + trial

            train_df, test_df = train_test_split(
                model_df,
                test_size=test_size,
                random_state=rs,
                shuffle=True,
                stratify=model_df[TARGET_COL],
            )

            print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
            print(f"Train class balance: {dict(train_df[TARGET_COL].value_counts())}")

            # ----- XGBoost -----
            print(f"\n>>> Running XGBoost Classification (Trial {trial + 1})...")
            boosting_result = generate_boosting(
                train_df, test_df, rs, predictors, TARGET_COL, loo
            )

            boosting_record = {
                "trial": trial,
                "best_params": boosting_result["best_params"],
                "cv_train_score": boosting_result["cv_train_score"],
                "cv_val_score": boosting_result["cv_val_score"],
                "cv_scoring": boosting_result["cv_scoring"],
                "train_metrics": boosting_result.get("train_metrics", {}),
                "test_metrics": {
                    "accuracy": boosting_result["test_metrics"]["accuracy"],
                    "precision": boosting_result["test_metrics"]["precision"],
                    "recall": boosting_result["test_metrics"]["recall"],
                    "f1": boosting_result["test_metrics"]["f1"],
                    "roc_auc": float(boosting_result["test_metrics"]["roc_auc"])
                    if not np.isnan(boosting_result["test_metrics"]["roc_auc"])
                    else None,
                },
                "y_test": boosting_result["y_test"].tolist(),
                "y_pred": boosting_result["y_pred"].tolist(),
                "y_proba": boosting_result["y_proba"].tolist()
                if boosting_result.get("y_proba") is not None
                else None,
                "feature_importances": (
                    boosting_result["feature_importances"].tolist()
                    if boosting_result.get("feature_importances") is not None
                    else None
                ),
                "feature_names": boosting_result.get("feature_names"),
            }
            results[BOOSTING_NAME][split_name].append(boosting_record)

            # ----- Random Forest -----
            print(f"\n>>> Running Random Forest Classification (Trial {trial + 1})...")
            rf_result = generate_random_forest(
                train_df, test_df, rs, predictors, TARGET_COL, loo
            )

            rf_record = {
                "trial": trial,
                "best_params": rf_result["best_params"],
                "cv_train_score": rf_result["cv_train_score"],
                "cv_val_score": rf_result["cv_val_score"],
                "cv_scoring": rf_result["cv_scoring"],
                "train_metrics": rf_result.get("train_metrics", {}),
                "test_metrics": {
                    "accuracy": rf_result["test_metrics"]["accuracy"],
                    "precision": rf_result["test_metrics"]["precision"],
                    "recall": rf_result["test_metrics"]["recall"],
                    "f1": rf_result["test_metrics"]["f1"],
                    "roc_auc": float(rf_result["test_metrics"]["roc_auc"])
                    if not np.isnan(rf_result["test_metrics"]["roc_auc"])
                    else None,
                },
                "y_test": rf_result["y_test"].tolist(),
                "y_pred": rf_result["y_pred"].tolist(),
                "y_proba": rf_result["y_proba"].tolist()
                if rf_result.get("y_proba") is not None
                else None,
                "feature_importances": (
                    rf_result["feature_importances"].tolist()
                    if rf_result.get("feature_importances") is not None
                    else None
                ),
                "feature_names": rf_result.get("feature_names"),
            }
            results[RANDOM_FOREST_NAME][split_name].append(rf_record)

            # ----- Neural Network -----
            print(f"\n>>> Running Neural Network Classification (Trial {trial + 1})...")
            nn_result = generate_neural_network(
                train_df, test_df, rs, predictors, TARGET_COL, loo
            )

            nn_record = {
                "trial": trial,
                "best_params": nn_result["best_params"],
                "cv_train_score": nn_result["cv_train_score"],
                "cv_val_score": nn_result["cv_val_score"],
                "cv_scoring": nn_result["cv_scoring"],
                "train_metrics": nn_result.get("train_metrics", {}),
                "test_metrics": {
                    "accuracy": nn_result["test_metrics"]["accuracy"],
                    "precision": nn_result["test_metrics"]["precision"],
                    "recall": nn_result["test_metrics"]["recall"],
                    "f1": nn_result["test_metrics"]["f1"],
                    "roc_auc": float(nn_result["test_metrics"]["roc_auc"])
                    if not np.isnan(nn_result["test_metrics"]["roc_auc"])
                    else None,
                },
                "y_test": nn_result["y_test"].tolist(),
                "y_pred": nn_result["y_pred"].tolist(),
                "y_proba": nn_result["y_proba"].tolist()
                if nn_result.get("y_proba") is not None
                else None,
                "training_info": nn_result.get("training_info"),
            }
            results[NEURAL_NETWORK_NAME][split_name].append(nn_record)

    # ==========================================
    # 7. Save Results
    # ==========================================
    results_dir = os.path.join(curr_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    out_path = os.path.join(results_dir, "rescue_classification_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[rescue_classification.py] Saved all results to {out_path}")

    # ==========================================
    # 8. Generate Plots
    # ==========================================
    plots_base = os.path.join(curr_dir, "../..", "plots/rescue_classification_plots/results")

    boosting_plots_dir = os.path.join(plots_base, "boosting")
    plot_rescue_cls_boosting_summary(results[BOOSTING_NAME], boosting_plots_dir)

    rf_plots_dir = os.path.join(plots_base, "random_forest")
    plot_rescue_cls_random_forest_summary(results[RANDOM_FOREST_NAME], rf_plots_dir)

    nn_plots_dir = os.path.join(plots_base, "neural_network")
    plot_rescue_cls_neural_network_summary(results[NEURAL_NETWORK_NAME], nn_plots_dir)

    comparison_dir = os.path.join(plots_base, "comparison")
    plot_rescue_cls_model_comparison(results, comparison_dir)

    print("\n" + "=" * 60)
    print("CLASSIFICATION EXPERIMENT COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
