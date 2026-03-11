"""
Piano Emotion Classification Experiment
Course: COGS 118C

Task: Binary classification — emotional/melancholic (0) vs upbeat/happy (1)
      from spectral and temporal features extracted from 30-second piano clips.

Dataset:
  18 songs x 3 random clips = 54 samples (27 per class, perfectly balanced).
  Features: Top-20 by effect size (acoustic, rhythmic, timbral).

CV Strategy: Leave-One-Song-Out Cross-Validation (LOGO-CV)
  - Each fold holds out all 3 clips of one song for testing.
  - Trains on the remaining 51 clips (17 songs).
  - 18 folds total; no random train/test split needed.
  - CRITICAL: prevents data leakage — clips from the same recording must not
    appear in both train and test sets.

Inner CV: StratifiedKFold(5) used inside GridSearchCV for hyperparameter tuning
  on the training set of each outer fold.

Models: XGBoost (Boosting), Random Forest, Neural Network (MLP)
Scaling: StandardScaler is applied inside each model pipeline.

WARNING:
  Labels were assigned by the researcher based on perceived musical style.
  The classifier learns to separate 'slow/sparse/sustained' from 'fast/dense/bright'
  piano playing — NOT objective emotional ground truth.
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

    # ROC-AUC on training set (using predict_proba or decision_function)
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
# Model Generator Functions
# ==========================================

def generate_boosting(train_df, test_df, predictors, target_col, inner_cv):
    """
    XGBoost classification.
    Tuned for small dataset (54 samples): shallow trees, strong regularization.
    Grid: 48 combinations.
    """
    param_grid = {
        "model__n_estimators":   [50, 100, 200],
        "model__learning_rate":  [0.05, 0.1],
        "model__max_depth":      [2, 3],
        "model__subsample":      [0.8, 1.0],
        "model__reg_lambda":     [1, 5],
    }
    return run_boosting_experiment(
        train_df=train_df,
        test_df=test_df,
        predictors=predictors,
        target_col=target_col,
        problem_type="classification",
        random_state=RANDOM_STATE,
        param_grid=param_grid,
        scoring="f1",
        cv_folds=inner_cv,
    )


def generate_random_forest(train_df, test_df, predictors, target_col, inner_cv):
    """
    Random Forest classification.
    Tuned for small dataset: moderate tree depth, balanced class weights.
    Grid: 36 combinations.
    """
    param_grid = {
        "model__n_estimators":    [50, 100, 200],
        "model__max_depth":       [3, 5, None],
        "model__min_samples_split": [2, 5],
        "model__max_features":    ["sqrt", "log2"],
    }
    return run_random_forest_experiment(
        train_df=train_df,
        test_df=test_df,
        predictors=predictors,
        target_col=target_col,
        problem_type="classification",
        random_state=RANDOM_STATE,
        param_grid=param_grid,
        scoring="f1",
        cv_folds=inner_cv,
    )


def generate_neural_network(train_df, test_df, predictors, target_col, inner_cv):
    """
    MLP Neural Network classification.
    Small architectures suited to 20 input features and ~48 training samples.
    Grid: 18 combinations.
    """
    param_grid = {
        "model__alpha":              [0.001, 0.01, 0.1],
        "model__learning_rate_init": [0.001, 0.01],
    }
    hidden_layer_sizes_grid = [(32,), (64,), (32, 16)]

    return run_neural_net_experiment(
        train_df=train_df,
        test_df=test_df,
        predictors=predictors,
        target_col=target_col,
        problem_type="classification",
        random_state=RANDOM_STATE,
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
    print("CV:       Leave-One-Song-Out (18 folds)")
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

    # Inner CV for GridSearchCV hyperparameter tuning (on training set only)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print(f"\n[piano.py] LOGO-CV folds: {n_folds}")
    print(f"[piano.py] Inner CV: StratifiedKFold(5) for GridSearchCV")
    print(f"[piano.py] Each outer fold: ~{n_folds - 1} songs (51 clips) train, "
          f"1 song (3 clips) test")

    # ==========================================
    # 5. Run LOGO-CV Experiment
    # ==========================================
    results = {
        BOOSTING_NAME:        [],
        RANDOM_FOREST_NAME:   [],
        NEURAL_NETWORK_NAME:  [],
    }

    songs = df["song_name"].unique()
    song_to_label = dict(zip(df["song_name"], df["label_name"] if "label_name" in df.columns
                              else df[TARGET_COL]))

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        test_song = groups[test_idx[0]]
        train_songs = sorted(set(groups[train_idx]))

        print(f"\n{'=' * 70}")
        print(f"FOLD {fold_idx + 1}/{n_folds} — Test Song: {test_song}")
        print(f"Train songs ({len(train_songs)}): {len(train_idx)} clips")
        print(f"Test  clips: {len(test_idx)} (label = {y[test_idx[0]]})")
        print(f"{'=' * 70}")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df  = df.iloc[test_idx].reset_index(drop=True)

        # Class balance check
        train_balance = dict(train_df[TARGET_COL].value_counts().sort_index())
        print(f"Train class balance: {train_balance}")

        # ----- XGBoost -----
        print(f"\n>>> Boosting (Fold {fold_idx + 1})...")
        boost_result = generate_boosting(train_df, test_df, predictors, TARGET_COL, inner_cv)
        boost_train_metrics = _compute_train_metrics(boost_result, train_df, predictors, TARGET_COL)

        boost_record = {
            "fold":           fold_idx,
            "test_song":      test_song,
            "best_params":    boost_result["best_params"],
            "cv_train_score": boost_result["cv_train_score"],
            "cv_val_score":   boost_result["cv_val_score"],
            "cv_scoring":     boost_result["cv_scoring"],
            "train_metrics":  boost_train_metrics,
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
        }
        results[BOOSTING_NAME].append(boost_record)

        # ----- Random Forest -----
        print(f"\n>>> Random Forest (Fold {fold_idx + 1})...")
        rf_result = generate_random_forest(train_df, test_df, predictors, TARGET_COL, inner_cv)
        rf_train_metrics = _compute_train_metrics(rf_result, train_df, predictors, TARGET_COL)

        rf_record = {
            "fold":           fold_idx,
            "test_song":      test_song,
            "best_params":    rf_result["best_params"],
            "cv_train_score": rf_result["cv_train_score"],
            "cv_val_score":   rf_result["cv_val_score"],
            "cv_scoring":     rf_result["cv_scoring"],
            "train_metrics":  rf_train_metrics,
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
        }
        results[RANDOM_FOREST_NAME].append(rf_record)

        # ----- Neural Network -----
        print(f"\n>>> Neural Network (Fold {fold_idx + 1})...")
        nn_result = generate_neural_network(train_df, test_df, predictors, TARGET_COL, inner_cv)
        nn_train_metrics = _compute_train_metrics(nn_result, train_df, predictors, TARGET_COL)

        nn_record = {
            "fold":           fold_idx,
            "test_song":      test_song,
            "best_params":    nn_result["best_params"],
            "cv_train_score": nn_result["cv_train_score"],
            "cv_val_score":   nn_result["cv_val_score"],
            "cv_scoring":     nn_result["cv_scoring"],
            "train_metrics":  nn_train_metrics,
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
        }
        results[NEURAL_NETWORK_NAME].append(nn_record)

    # ==========================================
    # 6. Print Final Aggregated Summary to Console
    # ==========================================
    print("\n\n" + "=" * 70)
    print("FINAL RESULTS — PIANO EMOTION CLASSIFICATION")
    print("(mean ± std across 18 LOGO-CV folds)")
    print("=" * 70)
    print(f"\n{'Model':<22} {'Train Acc':>10} {'Test Acc':>9} {'Train F1':>9} {'Test F1':>9} "
          f"{'Test Prec':>10} {'Test Rec':>9} {'Test AUC':>9}")
    print("-" * 100)

    for model_name in [BOOSTING_NAME, RANDOM_FOREST_NAME, NEURAL_NETWORK_NAME]:
        folds = results[model_name]

        tr_accs  = [f["train_metrics"]["accuracy"]  for f in folds]
        te_accs  = [f["test_metrics"]["accuracy"]   for f in folds]
        tr_f1s   = [f["train_metrics"]["f1"]        for f in folds]
        te_f1s   = [f["test_metrics"]["f1"]         for f in folds]
        te_precs = [f["test_metrics"]["precision"]  for f in folds]
        te_recs  = [f["test_metrics"]["recall"]     for f in folds]
        te_aucs  = [f["test_metrics"]["roc_auc"]    for f in folds
                    if f["test_metrics"]["roc_auc"] is not None]

        auc_str = f"{np.mean(te_aucs):.3f}±{np.std(te_aucs):.3f}" if te_aucs else "  N/A"

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

    print("\n  Baseline (majority class): Acc=0.500, F1=0.500, AUC=0.500")
    print("\n  Train F1 >> Test F1 by more than 0.20 indicates overfitting.")

    # ==========================================
    # 7. Save Results
    # ==========================================
    results_dir = os.path.join(curr_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    # Convert any remaining numpy types for JSON serialisation
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
