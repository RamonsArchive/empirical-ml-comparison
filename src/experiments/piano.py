"""
Piano Emotion Classification Experiment
Course: COGS 118C

Task: Binary classification - emotional/melancholic (0) vs upbeat/happy (1)
      from spectral and temporal features extracted from 30-second piano clips.

Dataset:
  18 songs x 3 random clips = 54 samples (27 per class, perfectly balanced).
  Features: Top-20 by effect size (acoustic, rhythmic, timbral).

CV Strategy: Leave-One-Song-Out Cross-Validation (LOGO-CV)
  - Each fold holds out all 3 clips of one song for testing.
  - Trains on the remaining 51 clips (17 songs).
  - 18 folds total; no random train/test split needed.
  - CRITICAL: prevents data leakage - clips from the same recording must not
    appear in both train and test sets.

Inner CV: StratifiedKFold(5) used inside GridSearchCV for hyperparameter tuning
  on the training set of each outer fold.

Trials: 3 trials per model (RANDOM_STATE + trial), averaged.
  The LOGO-CV fold structure is deterministic (always same 18 song groupings),
  but varying the random seed changes the inner CV splits and model
  initialisation, giving variance estimates across trials.

Models: XGBoost (Boosting), Random Forest, Neural Network (MLP)
Scaling: StandardScaler is applied inside each model pipeline.

WARNING:
  Labels were assigned by the researcher based on perceived musical style.
  The classifier learns to separate 'slow/sparse/sustained' from 'fast/dense/bright'
  piano playing - NOT objective emotional ground truth.
  Report results with the caveat: "These labels reflect musical genre conventions
  rather than validated emotional ground truth."
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.dont_write_bytecode = True

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

from utils.load.load_piano import load_piano_data
from utils.clean.clean_piano import clean_piano
from utils.eda.eda_piano import eda_piano

from models.boosting import run_boosting_experiment
from models.random_forest import run_random_forest_experiment
from models.neural_net import run_neural_net_experiment

from graphs.piano_plots import (
    plot_piano_boosting_summary,
    plot_piano_random_forest_summary,
    plot_piano_neural_network_summary,
    plot_piano_model_comparison,
)


# ==========================================
# Constants
# ==========================================
RANDOM_STATE = 42
BOOSTING_NAME = "boosting"
RANDOM_FOREST_NAME = "random_forest"
NEURAL_NETWORK_NAME = "neural_network"

TARGET_COL = "label"

# Top-20 most discriminative features ranked by Cohen's d effect size.
# Source: feature_report.txt from the Piano Feature Extraction Pipeline.
# rms_mean, identifiers, and zero-variance PDC columns are intentionally excluded.
TOP_20_FEATURES = [
    "onset_strength_mean",
    "stft_mid_mean",
    "stft_spectral_flux_mean",
    "note_density",
    "stft_presence_std",
    "mfcc_2_mean",
    "onset_strength_std",
    "stft_high_mid_mean",
    "stft_high_mid_std",
    "stft_brilliance_std",
    "stft_presence_mean",
    "stft_mid_std",
    "mt_spectral_rolloff_95",
    "stft_spectral_flux_std",
    "zcr_mean",
    "welch_spectral_spread",
    "stft_low_mid_mean",
    "mfcc_0_mean",
    "stft_brilliance_mean",
    "coherence_bass",
]


# ==========================================
# Train Metrics Helper
# ==========================================
def _compute_train_metrics(result, train_df, predictors, target_col):
    """
    Compute train-set classification metrics using the best fitted pipeline.
    Used to check for overfitting by comparing train vs test performance.
    """
    model = result["model"]
    label_encoder = result.get("label_encoder")

    X_train = train_df[predictors]
    y_train_raw = train_df[target_col]

    if label_encoder is not None:
        y_train = label_encoder.transform(y_train_raw)
    else:
        y_train = y_train_raw.values

    y_train_pred = model.predict(X_train)

    avg = "binary"
    train_metrics = {
        "accuracy":  accuracy_score(y_train, y_train_pred),
        "precision": precision_score(y_train, y_train_pred, average=avg, zero_division=0),
        "recall":    recall_score(y_train, y_train_pred, average=avg, zero_division=0),
        "f1":        f1_score(y_train, y_train_pred, average=avg, zero_division=0),
    }

    try:
        if hasattr(model.named_steps["model"], "predict_proba"):
            y_train_proba = model.predict_proba(X_train)[:, 1]
            train_metrics["roc_auc"] = float(roc_auc_score(y_train, y_train_proba))
        elif hasattr(model.named_steps["model"], "decision_function"):
            y_train_score = model.decision_function(X_train)
            train_metrics["roc_auc"] = float(roc_auc_score(y_train, y_train_score))
        else:
            train_metrics["roc_auc"] = float("nan")
    except ValueError:
        train_metrics["roc_auc"] = float("nan")

    return train_metrics


# ==========================================
# Aggregate LOGO-CV Folds Into One Trial Record
# ==========================================
def _aggregate_folds(fold_records, trial_idx):
    """
    Collapse 18 LOGO-CV fold records into a single trial record
    that matches the rescue experiment's trial-record structure.

    IMPORTANT: Per-fold binary metrics are misleading for LOGO-CV because each
    fold's test set contains only ONE class (all 3 clips from one song share
    the same label). A perfectly classified emotional song gets F1=0 since
    there are no positive-class samples. Therefore we compute test_metrics
    from CONCATENATED predictions across all 18 folds (54 total), where both
    classes are represented and metrics are meaningful.

    - test_metrics: computed from concatenated y_test/y_pred across all 18 folds
    - train_metrics: mean across folds (training sets always have both classes)
    - y_test / y_pred / y_proba: concatenated across folds (54 total predictions)
    - feature_importances: averaged across folds
    - best_params: taken from the fold with highest inner-CV val score
    - fold_details: preserved for per-fold plots
    """
    # Concatenate predictions across all folds (gives 54 total)
    y_test_all = []
    y_pred_all = []
    y_proba_all = []
    has_proba = True
    for f in fold_records:
        y_test_all.extend(f["y_test"])
        y_pred_all.extend(f["y_pred"])
        if f.get("y_proba") is not None:
            y_proba_all.extend(f["y_proba"])
        else:
            has_proba = False

    # Compute TEST metrics from concatenated predictions (correct for LOGO-CV)
    y_test_arr = np.array(y_test_all)
    y_pred_arr = np.array(y_pred_all)

    test_metrics = {
        "accuracy":  float(accuracy_score(y_test_arr, y_pred_arr)),
        "precision": float(precision_score(y_test_arr, y_pred_arr, average="binary", zero_division=0)),
        "recall":    float(recall_score(y_test_arr, y_pred_arr, average="binary", zero_division=0)),
        "f1":        float(f1_score(y_test_arr, y_pred_arr, average="binary", zero_division=0)),
    }
    if has_proba and y_proba_all:
        try:
            test_metrics["roc_auc"] = float(roc_auc_score(y_test_arr, np.array(y_proba_all)))
        except ValueError:
            test_metrics["roc_auc"] = None
    else:
        test_metrics["roc_auc"] = None

    # Compute TRAIN metrics as mean across folds (training sets always have both classes)
    train_metrics = {}
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        tr_vals = [
            f["train_metrics"][metric] for f in fold_records
            if f["train_metrics"].get(metric) is not None
            and not (isinstance(f["train_metrics"][metric], float)
                     and np.isnan(f["train_metrics"][metric]))
        ]
        train_metrics[metric] = float(np.mean(tr_vals)) if tr_vals else None

    # Average feature importances across folds
    imp_folds = [
        f for f in fold_records
        if f.get("feature_importances") is not None
    ]
    feature_importances = None
    feature_names = None
    if imp_folds:
        feature_importances = np.mean(
            [f["feature_importances"] for f in imp_folds], axis=0
        ).tolist()
        feature_names = imp_folds[0]["feature_names"]

    # Best params: use the fold with highest inner-CV val score
    best_cv_fold = max(fold_records, key=lambda f: f.get("cv_val_score", 0))

    return {
        "trial":          trial_idx,
        "best_params":    best_cv_fold["best_params"],
        "cv_train_score": float(np.mean([f["cv_train_score"] for f in fold_records])),
        "cv_val_score":   float(np.mean([f["cv_val_score"] for f in fold_records])),
        "cv_scoring":     fold_records[0]["cv_scoring"],
        "test_metrics":   test_metrics,
        "train_metrics":  train_metrics,
        "y_test":         y_test_all,
        "y_pred":         y_pred_all,
        "y_proba":        y_proba_all if has_proba and y_proba_all else None,
        "feature_importances": feature_importances,
        "feature_names":       feature_names,
        "fold_details":        fold_records,
    }


# ==========================================
# Model Generator Functions
# ==========================================

def generate_boosting(train_df, test_df, random_state, predictors, target_col, inner_cv):
    """
    XGBoost classification - expanded grid for small dataset.
    Grid: 108 combinations.
    """
    param_grid = {
        "model__n_estimators":   [10, 25, 50, 100],
        "model__learning_rate":  [0.01, 0.05, 0.1],
        "model__max_depth":      [2, 3, 5, None],
        "model__subsample":      [0.8, 1.0],
        "model__reg_lambda":     [1, 5, 10],
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
        cv_folds=inner_cv,
    )


def generate_random_forest(train_df, test_df, random_state, predictors, target_col, inner_cv):
    """
    Random Forest classification - expanded grid.
    Grid: 96 combinations.
    """
    param_grid = {
        "model__n_estimators":      [10, 25, 50, 100],
        "model__max_depth":         [1, 2, 3, 5, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf":  [1, 2, 4],
        "model__max_features":      ["sqrt", "log2"],
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
        cv_folds=inner_cv,
    )


def generate_neural_network(train_df, test_df, random_state, predictors, target_col, inner_cv):
    """
    MLP Neural Network classification - expanded grid.
    Grid: 36 combinations.
    """
    param_grid = {
        "model__alpha":              [0.001, 0.01, 0.1, 0.0001],
        "model__learning_rate_init": [0.001, 0.005, 0.01, 0.05],
    }
    hidden_layer_sizes_grid = [(16,), (32,), (64,), (32, 16), (64, 32)]

    return run_neural_net_experiment(
        train_df=train_df,
        test_df=test_df,
        predictors=predictors,
        target_col=target_col,
        problem_type="classification",
        random_state=random_state,
        hidden_layer_sizes_grid=hidden_layer_sizes_grid,
        param_grid=param_grid,
        scoring="f1",
        cv_folds=inner_cv,
    )


# ==========================================
# Main Experiment
# ==========================================

def main():
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    print("\n" + "=" * 70)
    print("PIANO EMOTION CLASSIFICATION EXPERIMENT")
    print("Task:     Emotional (0) vs Happy (1) piano music")
    print("CV:       Leave-One-Song-Out (18 folds) x 3 trials")
    print("Features: Top-20 by effect size")
    print("=" * 70)

    # ==========================================
    # 1. Load and Clean Data
    # ==========================================
    df_raw = load_piano_data(curr_dir)
    df = clean_piano(df_raw)

    # ==========================================
    # 2. EDA
    # ==========================================
    eda_piano(df)

    # ==========================================
    # 3. Validate Features Against Dataset
    # ==========================================
    missing_features = [f for f in TOP_20_FEATURES if f not in df.columns]
    if missing_features:
        raise ValueError(
            f"[piano.py] These TOP_20_FEATURES are not in the dataset: {missing_features}\n"
            f"Available columns: {list(df.columns)}"
        )

    predictors = TOP_20_FEATURES

    print(f"\n[piano.py] Target:    {TARGET_COL}")
    print(f"[piano.py] Predictors ({len(predictors)}): {predictors}")
    print(f"[piano.py] Samples:   {len(df)} ({(df[TARGET_COL]==0).sum()} emotional, "
          f"{(df[TARGET_COL]==1).sum()} happy)")

    # ==========================================
    # 4. Leave-One-Song-Out CV Setup
    # ==========================================
    groups = df["song_name"].values
    y = df[TARGET_COL].values
    X = df[predictors]

    logo = LeaveOneGroupOut()
    n_folds = logo.get_n_splits(X, y, groups)
    n_trials = 3

    print(f"\n[piano.py] LOGO-CV folds: {n_folds}")
    print(f"[piano.py] Trials: {n_trials} (random_state = {RANDOM_STATE} + trial)")
    print(f"[piano.py] Inner CV: StratifiedKFold(5) for GridSearchCV")
    print(f"[piano.py] Each outer fold: {n_folds - 1} songs (51 clips) train, "
          f"1 song (3 clips) test")

    # ==========================================
    # 5. Run Experiments: 3 trials x 18 LOGO-CV folds
    # ==========================================
    # Results structure matches rescue_classification convention:
    #   results[model_name]["LOGO_CV"] = [trial_0_record, trial_1_record, trial_2_record]
    results = {
        BOOSTING_NAME:        {"LOGO_CV": []},
        RANDOM_FOREST_NAME:   {"LOGO_CV": []},
        NEURAL_NETWORK_NAME:  {"LOGO_CV": []},
    }

    for trial in range(n_trials):
        rs = RANDOM_STATE + trial
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)

        print(f"\n{'#' * 70}")
        print(f"TRIAL {trial + 1}/{n_trials}  (random_state={rs})")
        print(f"{'#' * 70}")

        # Collect per-fold records for this trial
        trial_folds = {
            BOOSTING_NAME:        [],
            RANDOM_FOREST_NAME:   [],
            NEURAL_NETWORK_NAME:  [],
        }

        for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
            test_song = groups[test_idx[0]]

            print(f"\n{'=' * 60}")
            print(f"Trial {trial + 1} | Fold {fold_idx + 1}/{n_folds} - Test: {test_song}")
            print(f"{'=' * 60}")

            train_df = df.iloc[train_idx].reset_index(drop=True)
            test_df  = df.iloc[test_idx].reset_index(drop=True)

            train_balance = dict(train_df[TARGET_COL].value_counts().sort_index())
            print(f"Train: {len(train_df)} clips, balance: {train_balance}")

            # ----- XGBoost -----
            print(f"\n>>> Boosting (Trial {trial + 1}, Fold {fold_idx + 1})...")
            boost_result = generate_boosting(
                train_df, test_df, rs, predictors, TARGET_COL, inner_cv
            )
            boost_train_m = _compute_train_metrics(boost_result, train_df, predictors, TARGET_COL)

            trial_folds[BOOSTING_NAME].append({
                "fold":           fold_idx,
                "test_song":      test_song,
                "best_params":    boost_result["best_params"],
                "cv_train_score": boost_result["cv_train_score"],
                "cv_val_score":   boost_result["cv_val_score"],
                "cv_scoring":     boost_result["cv_scoring"],
                "train_metrics":  boost_train_m,
                "test_metrics": {
                    "accuracy":  boost_result["test_metrics"]["accuracy"],
                    "precision": boost_result["test_metrics"]["precision"],
                    "recall":    boost_result["test_metrics"]["recall"],
                    "f1":        boost_result["test_metrics"]["f1"],
                    "roc_auc":   float(boost_result["test_metrics"]["roc_auc"])
                                 if not np.isnan(boost_result["test_metrics"]["roc_auc"])
                                 else None,
                },
                "y_test":  boost_result["y_test"].tolist(),
                "y_pred":  boost_result["y_pred"].tolist(),
                "y_proba": boost_result["y_proba"].tolist()
                           if boost_result.get("y_proba") is not None else None,
                "feature_importances": (
                    boost_result["feature_importances"].tolist()
                    if boost_result.get("feature_importances") is not None else None
                ),
                "feature_names": boost_result.get("feature_names"),
            })

            # ----- Random Forest -----
            print(f"\n>>> Random Forest (Trial {trial + 1}, Fold {fold_idx + 1})...")
            rf_result = generate_random_forest(
                train_df, test_df, rs, predictors, TARGET_COL, inner_cv
            )
            rf_train_m = _compute_train_metrics(rf_result, train_df, predictors, TARGET_COL)

            trial_folds[RANDOM_FOREST_NAME].append({
                "fold":           fold_idx,
                "test_song":      test_song,
                "best_params":    rf_result["best_params"],
                "cv_train_score": rf_result["cv_train_score"],
                "cv_val_score":   rf_result["cv_val_score"],
                "cv_scoring":     rf_result["cv_scoring"],
                "train_metrics":  rf_train_m,
                "test_metrics": {
                    "accuracy":  rf_result["test_metrics"]["accuracy"],
                    "precision": rf_result["test_metrics"]["precision"],
                    "recall":    rf_result["test_metrics"]["recall"],
                    "f1":        rf_result["test_metrics"]["f1"],
                    "roc_auc":   float(rf_result["test_metrics"]["roc_auc"])
                                 if not np.isnan(rf_result["test_metrics"]["roc_auc"])
                                 else None,
                },
                "y_test":  rf_result["y_test"].tolist(),
                "y_pred":  rf_result["y_pred"].tolist(),
                "y_proba": rf_result["y_proba"].tolist()
                           if rf_result.get("y_proba") is not None else None,
                "feature_importances": (
                    rf_result["feature_importances"].tolist()
                    if rf_result.get("feature_importances") is not None else None
                ),
                "feature_names": rf_result.get("feature_names"),
            })

            # ----- Neural Network -----
            print(f"\n>>> Neural Network (Trial {trial + 1}, Fold {fold_idx + 1})...")
            nn_result = generate_neural_network(
                train_df, test_df, rs, predictors, TARGET_COL, inner_cv
            )
            nn_train_m = _compute_train_metrics(nn_result, train_df, predictors, TARGET_COL)

            trial_folds[NEURAL_NETWORK_NAME].append({
                "fold":           fold_idx,
                "test_song":      test_song,
                "best_params":    nn_result["best_params"],
                "cv_train_score": nn_result["cv_train_score"],
                "cv_val_score":   nn_result["cv_val_score"],
                "cv_scoring":     nn_result["cv_scoring"],
                "train_metrics":  nn_train_m,
                "test_metrics": {
                    "accuracy":  nn_result["test_metrics"]["accuracy"],
                    "precision": nn_result["test_metrics"]["precision"],
                    "recall":    nn_result["test_metrics"]["recall"],
                    "f1":        nn_result["test_metrics"]["f1"],
                    "roc_auc":   float(nn_result["test_metrics"]["roc_auc"])
                                 if not np.isnan(nn_result["test_metrics"]["roc_auc"])
                                 else None,
                },
                "y_test":  nn_result["y_test"].tolist(),
                "y_pred":  nn_result["y_pred"].tolist(),
                "y_proba": nn_result["y_proba"].tolist()
                           if nn_result.get("y_proba") is not None else None,
                "training_info": nn_result.get("training_info"),
            })

        # Aggregate 18 folds into one trial record per model
        for model_name in [BOOSTING_NAME, RANDOM_FOREST_NAME, NEURAL_NETWORK_NAME]:
            trial_record = _aggregate_folds(trial_folds[model_name], trial)
            results[model_name]["LOGO_CV"].append(trial_record)

            tr_f1 = trial_record["train_metrics"]["f1"]
            te_f1 = trial_record["test_metrics"]["f1"]
            te_acc = trial_record["test_metrics"]["accuracy"]
            print(f"\n  [{model_name}] Trial {trial + 1} aggregate: "
                  f"Train F1={tr_f1:.3f}  Test F1={te_f1:.3f}  Test Acc={te_acc:.3f}")

    # ==========================================
    # 6. Print Final Summary (averaged over 3 trials)
    # ==========================================
    print("\n\n" + "=" * 70)
    print("FINAL RESULTS - PIANO EMOTION CLASSIFICATION")
    print("(averaged over 3 trials x 18 LOGO-CV folds)")
    print("=" * 70)
    print(f"\n{'Model':<22} {'Train Acc':>10} {'Test Acc':>10} {'Train F1':>10} "
          f"{'Test F1':>10} {'Test Prec':>10} {'Test Rec':>10} {'Test AUC':>10}")
    print("-" * 105)

    for model_name in [BOOSTING_NAME, RANDOM_FOREST_NAME, NEURAL_NETWORK_NAME]:
        trials = results[model_name]["LOGO_CV"]

        tr_accs  = [t["train_metrics"]["accuracy"]  for t in trials]
        te_accs  = [t["test_metrics"]["accuracy"]   for t in trials]
        tr_f1s   = [t["train_metrics"]["f1"]        for t in trials]
        te_f1s   = [t["test_metrics"]["f1"]         for t in trials]
        te_precs = [t["test_metrics"]["precision"]  for t in trials]
        te_recs  = [t["test_metrics"]["recall"]     for t in trials]
        te_aucs  = [t["test_metrics"]["roc_auc"]    for t in trials
                    if t["test_metrics"]["roc_auc"] is not None]

        auc_str = f"{np.mean(te_aucs):.3f}±{np.std(te_aucs):.3f}" if te_aucs else "    N/A"

        print(
            f"{model_name:<22} "
            f"{np.mean(tr_accs):.3f}±{np.std(tr_accs):.3f}  "
            f"{np.mean(te_accs):.3f}±{np.std(te_accs):.3f}  "
            f"{np.mean(tr_f1s):.3f}±{np.std(tr_f1s):.3f}  "
            f"{np.mean(te_f1s):.3f}±{np.std(te_f1s):.3f}  "
            f"{np.mean(te_precs):.3f}±{np.std(te_precs):.3f}  "
            f"{np.mean(te_recs):.3f}±{np.std(te_recs):.3f}  "
            f"{auc_str}"
        )

    print(f"\n  Baseline (majority class): Acc=0.500, F1=0.500, AUC=0.500")
    print(f"  Train F1 >> Test F1 by more than 0.20 indicates overfitting.")

    # ==========================================
    # 7. Save Results
    # ==========================================
    results_dir = os.path.join(curr_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    def _to_serialisable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    out_path = os.path.join(results_dir, "piano_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=_to_serialisable)
    print(f"\n[piano.py] Saved all results to {out_path}")

    # ==========================================
    # 8. Generate Plots
    # ==========================================
    plots_base = os.path.join(curr_dir, "../..", "plots/piano_plots/results")

    plot_piano_boosting_summary(
        results[BOOSTING_NAME],
        os.path.join(plots_base, "boosting"),
    )
    plot_piano_random_forest_summary(
        results[RANDOM_FOREST_NAME],
        os.path.join(plots_base, "random_forest"),
    )
    plot_piano_neural_network_summary(
        results[NEURAL_NETWORK_NAME],
        os.path.join(plots_base, "neural_network"),
    )
    plot_piano_model_comparison(
        results,
        os.path.join(plots_base, "comparison"),
    )

    print("\n" + "=" * 70)
    print("PIANO CLASSIFICATION EXPERIMENT COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
