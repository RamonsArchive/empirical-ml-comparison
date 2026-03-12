"""
Plotting functions for Piano Emotion Classification experiment results.

Results structure (per model):
    results[model_name] = {"LOGO_CV": [trial_0, trial_1, trial_2]}

Each trial record:
    {
        "trial":           int,
        "best_params":     dict,
        "cv_train_score":  float,
        "cv_val_score":    float,
        "cv_scoring":      str,
        "train_metrics":   {accuracy, precision, recall, f1, roc_auc},  # mean across 18 folds
        "test_metrics":    {accuracy, precision, recall, f1, roc_auc},  # mean across 18 folds
        "y_test":          list[int],   # concatenated across 18 folds (54 total)
        "y_pred":          list[int],
        "y_proba":         list[float] | None,
        "feature_importances": list[float] | None,  # averaged across 18 folds
        "feature_names":       list[str]  | None,
        "fold_details":        list[dict],           # 18 per-fold records
    }

Convention: Matches rescue_classification_plots.py structure
  results = {"LOGO_CV": [list of trial records]}
  — averaged over 3 trials, just like rescue averages over 3 trials per split.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==========================================
# Helpers
# ==========================================

def _safe(val):
    """Format float or return 'N/A'."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.4f}"


def _metric_vals(trials, key, metric):
    """Collect metric values across trials, dropping None/NaN."""
    vals = []
    for t in trials:
        v = t[key][metric]
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            vals.append(v)
    return vals


def _mean_std(vals):
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


# ==========================================
# Text report
# ==========================================

def _generate_classification_report(results, model_name, output_dir):
    """
    Text report for one model.
    results = {"LOGO_CV": [trial_0, trial_1, trial_2]}
    Returns the trial with the best test F1.
    """
    r = []
    r.append("=" * 70)
    r.append(f"PIANO EMOTION CLASSIFICATION — {model_name.upper()} RESULTS")
    r.append("=" * 70)

    split_names = list(results.keys())       # ["LOGO_CV"]
    all_best_f1 = -1.0
    all_best_trial = None
    all_best_split = None

    # ---- Performance by split (averaged over 3 trials) ----
    r.append("\n" + "-" * 70)
    r.append("PERFORMANCE (averaged over 3 trials)")
    r.append("-" * 70)
    r.append(
        f"\n{'Split':<12} {'Accuracy':<14} {'Precision':<14} {'Recall':<14} "
        f"{'F1':<14} {'ROC-AUC':<14}"
    )
    r.append("-" * 70)

    for split_name in split_names:
        trials = results[split_name]

        accs  = _metric_vals(trials, "test_metrics", "accuracy")
        precs = _metric_vals(trials, "test_metrics", "precision")
        recs  = _metric_vals(trials, "test_metrics", "recall")
        f1s   = _metric_vals(trials, "test_metrics", "f1")
        aucs  = _metric_vals(trials, "test_metrics", "roc_auc")

        acc_m, acc_s   = _mean_std(accs)
        prec_m, prec_s = _mean_std(precs)
        rec_m, rec_s   = _mean_std(recs)
        f1_m, f1_s     = _mean_std(f1s)
        auc_m, auc_s   = _mean_std(aucs)

        acc_str  = f"{acc_m:.4f}±{acc_s:.4f}"  if acc_m  is not None else "N/A"
        prec_str = f"{prec_m:.4f}±{prec_s:.4f}" if prec_m is not None else "N/A"
        rec_str  = f"{rec_m:.4f}±{rec_s:.4f}"  if rec_m  is not None else "N/A"
        f1_str   = f"{f1_m:.4f}±{f1_s:.4f}"    if f1_m   is not None else "N/A"
        auc_str  = f"{auc_m:.4f}±{auc_s:.4f}"  if auc_m  is not None else "N/A"

        r.append(
            f"{split_name:<12} {acc_str:<14} {prec_str:<14} {rec_str:<14} "
            f"{f1_str:<14} {auc_str:<14}"
        )

        for trial in trials:
            if trial["test_metrics"]["f1"] is not None and trial["test_metrics"]["f1"] > all_best_f1:
                all_best_f1 = trial["test_metrics"]["f1"]
                all_best_trial = trial
                all_best_split = split_name

    # ---- Overfitting check (train vs test, averaged over 3 trials) ----
    r.append("\n" + "-" * 70)
    r.append("OVERFITTING CHECK (train vs test, averaged over 3 trials)")
    r.append("-" * 70)

    for split_name in split_names:
        trials = results[split_name]
        r.append(f"\n  Split: {split_name}")

        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            tr_vals = _metric_vals(trials, "train_metrics", metric)
            te_vals = _metric_vals(trials, "test_metrics", metric)
            tr_m, tr_s = _mean_std(tr_vals)
            te_m, te_s = _mean_std(te_vals)

            if tr_m is not None and te_m is not None:
                gap = tr_m - te_m
                flag = "  *** OVERFIT" if gap > 0.20 else ""
                r.append(
                    f"  Train {metric:<12}: {tr_m:.4f}±{tr_s:.4f}  |  "
                    f"Test: {te_m:.4f}±{te_s:.4f}  |  Gap: {gap:+.4f}{flag}"
                )

    # ---- Best trial details ----
    if all_best_trial:
        r.append("\n" + "-" * 70)
        r.append(f"BEST TRIAL (highest mean F1 = {all_best_f1:.4f})")
        r.append("-" * 70)
        r.append(f"  Split: {all_best_split}")
        r.append(f"  Trial: {all_best_trial['trial'] + 1}")

        r.append(f"\n  Best Hyperparameters (from fold with highest inner CV):")
        for param, val in all_best_trial["best_params"].items():
            r.append(f"    {param}: {val}")

        tm = all_best_trial["test_metrics"]
        tr = all_best_trial["train_metrics"]
        r.append(f"\n  Test Metrics (mean across 18 LOGO-CV folds):")
        r.append(f"    Accuracy:  {_safe(tm.get('accuracy'))}")
        r.append(f"    Precision: {_safe(tm.get('precision'))}")
        r.append(f"    Recall:    {_safe(tm.get('recall'))}")
        r.append(f"    F1:        {_safe(tm.get('f1'))}")
        r.append(f"    ROC-AUC:   {_safe(tm.get('roc_auc'))}")
        r.append(f"\n  Train Metrics:")
        r.append(f"    Accuracy:  {_safe(tr.get('accuracy'))}")
        r.append(f"    F1:        {_safe(tr.get('f1'))}")

        # Per-fold breakdown from best trial
        fold_details = all_best_trial.get("fold_details", [])
        if fold_details:
            r.append(f"\n  Per-Fold Breakdown ({len(fold_details)} folds):")
            r.append(
                f"  {'Fold':<5} {'Song':<38} {'Acc':>6} {'F1':>6} {'Prec':>6} {'Rec':>6}"
            )
            r.append("  " + "-" * 68)
            for fd in fold_details:
                ftm = fd["test_metrics"]
                song_short = fd["test_song"][:36]
                r.append(
                    f"  {fd['fold']+1:<5} {song_short:<38} "
                    f"{ftm['accuracy']:>6.3f} {ftm['f1']:>6.3f} "
                    f"{ftm['precision']:>6.3f} {ftm['recall']:>6.3f}"
                )

    # ---- All trials: hyperparameters & metrics ----
    r.append("\n" + "-" * 70)
    r.append("ALL TRIALS — HYPERPARAMETERS & METRICS")
    r.append("-" * 70)

    for split_name in split_names:
        trials = results[split_name]
        for trial in trials:
            trial_num = trial["trial"] + 1
            tm = trial["test_metrics"]
            tr = trial["train_metrics"]
            r.append(f"\n  Trial {trial_num} (Split: {split_name})")
            r.append("  " + "." * 40)
            r.append(f"  Hyperparameters (from fold with highest inner CV):")
            for param, val in trial["best_params"].items():
                r.append(f"    {param}: {val}")
            r.append(f"  Test Metrics:")
            r.append(
                f"    Accuracy: {_safe(tm.get('accuracy'))}  |  "
                f"Precision: {_safe(tm.get('precision'))}  |  "
                f"Recall: {_safe(tm.get('recall'))}  |  "
                f"F1: {_safe(tm.get('f1'))}  |  "
                f"ROC-AUC: {_safe(tm.get('roc_auc'))}"
            )
            r.append(f"  Train Metrics:")
            r.append(
                f"    Accuracy: {_safe(tr.get('accuracy'))}  |  "
                f"F1: {_safe(tr.get('f1'))}"
            )

    # ---- Interpretation ----
    r.append("\n" + "-" * 70)
    r.append("INTERPRETATION")
    r.append("-" * 70)

    if all_best_trial:
        f1 = all_best_trial["test_metrics"]["f1"]
        if f1 is not None:
            if f1 > 0.90:   quality = "EXCELLENT"
            elif f1 > 0.75: quality = "GOOD"
            elif f1 > 0.60: quality = "MODERATE"
            elif f1 > 0.50: quality = "FAIR"
            else:           quality = "POOR (at or below majority-class baseline)"
            r.append(f"  Model Quality: {quality} (Best Mean F1 = {f1:.4f})")

    r.append(f"  Baseline: F1=0.50, Accuracy=0.50 (balanced classes)")
    r.append(
        f"\n  NOTE: Labels classify piano music by acoustic activity and style,\n"
        f"  not by validated emotional ground truth."
    )

    report_path = os.path.join(output_dir, f"piano_{model_name}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(r))

    return all_best_trial


# ==========================================
# Confusion matrix (from best trial, accumulated across folds)
# ==========================================

def _plot_confusion_matrix(best_trial, model_name, output_dir):
    """Confusion matrix from best trial's concatenated y_test/y_pred (54 samples)."""
    y_test = np.array(best_trial["y_test"])
    y_pred = np.array(best_trial["y_pred"])

    tp = int(np.sum((y_test == 1) & (y_pred == 1)))
    tn = int(np.sum((y_test == 0) & (y_pred == 0)))
    fp = int(np.sum((y_test == 0) & (y_pred == 1)))
    fn = int(np.sum((y_test == 1) & (y_pred == 0)))
    cm = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    labels = ["Emotional (0)", "Happy (1)"]
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=labels, yticklabels=labels,
        ylabel="Actual", xlabel="Predicted",
    )

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0
    ax.set_title(
        f"{model_name.replace('_', ' ').title()}: Confusion Matrix\n"
        f"(Best Trial, N={total}, Acc={acc:.3f})",
        fontsize=11,
    )

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=16,
            )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"piano_{model_name}_confusion.png"), dpi=150)
    plt.close()


# ==========================================
# Confusion matrix (averaged across all trials)
# ==========================================

def _plot_avg_confusion_matrix(trials, model_name, output_dir):
    """Confusion matrix averaged across all trials' concatenated y_test/y_pred."""
    n_trials = len(trials)
    cms = []
    for trial in trials:
        y_test = np.array(trial["y_test"])
        y_pred = np.array(trial["y_pred"])
        tp = int(np.sum((y_test == 1) & (y_pred == 1)))
        tn = int(np.sum((y_test == 0) & (y_pred == 0)))
        fp = int(np.sum((y_test == 0) & (y_pred == 1)))
        fn = int(np.sum((y_test == 1) & (y_pred == 0)))
        cms.append(np.array([[tn, fp], [fn, tp]]))

    cm_mean = np.mean(cms, axis=0)
    cm_std = np.std(cms, axis=0)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(cm_mean, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    labels = ["Emotional (0)", "Happy (1)"]
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=labels, yticklabels=labels,
        ylabel="Actual", xlabel="Predicted",
    )

    total = cm_mean.sum()
    acc = (cm_mean[0, 0] + cm_mean[1, 1]) / total if total > 0 else 0
    ax.set_title(
        f"{model_name.replace('_', ' ').title()}: Confusion Matrix\n"
        f"(Averaged over {n_trials} Trials, N={int(total)}, Acc={acc:.3f})",
        fontsize=11,
    )

    thresh = cm_mean.max() / 2.0
    for i in range(2):
        for j in range(2):
            cell_text = f"{cm_mean[i, j]:.1f}\n(\u00b1{cm_std[i, j]:.1f})"
            ax.text(
                j, i, cell_text,
                ha="center", va="center",
                color="white" if cm_mean[i, j] > thresh else "black",
                fontsize=14,
            )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"piano_{model_name}_confusion_avg.png"), dpi=150
    )
    plt.close()


# ==========================================
# Train vs Test overfitting chart
# ==========================================

def _plot_train_vs_test(results, model_name, output_dir):
    """
    Grouped bar chart: mean train vs test for accuracy, F1, precision, recall
    averaged over 3 trials.
    """
    trials = results["LOGO_CV"]
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1"]

    train_means, test_means = [], []
    train_stds, test_stds = [], []

    for metric in metrics:
        tr_vals = _metric_vals(trials, "train_metrics", metric)
        te_vals = _metric_vals(trials, "test_metrics", metric)
        tm, ts = _mean_std(tr_vals)
        em, es = _mean_std(te_vals)
        train_means.append(tm or 0)
        test_means.append(em or 0)
        train_stds.append(ts or 0)
        test_stds.append(es or 0)

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - width / 2, train_means, width, yerr=train_stds, capsize=4,
           label="Train", color="steelblue", alpha=0.85, edgecolor="black")
    ax.bar(x + width / 2, test_means, width, yerr=test_stds, capsize=4,
           label="Test (LOGO-CV)", color="coral", alpha=0.85, edgecolor="black")

    for i, (tm, em) in enumerate(zip(train_means, test_means)):
        ax.text(i, max(tm, em) + 0.05, f"gap={tm - em:+.2f}",
                ha="center", va="bottom", fontsize=9, color="dimgray")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.3)
    ax.set_ylabel("Score (mean ± std across 3 trials)", fontsize=11)
    ax.set_title(
        f"{model_name.replace('_', ' ').title()}: Train vs Test Metrics\n"
        f"(positive gap = overfitting, averaged over 3 trials x 18 folds)",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"piano_{model_name}_train_vs_test.png"), dpi=150)
    plt.close()


# ==========================================
# Per-fold metrics (from best trial)
# ==========================================

def _plot_per_fold_metrics(best_trial, model_name, output_dir):
    """Per-fold accuracy and F1 from the best trial's fold_details."""
    folds = best_trial.get("fold_details", [])
    if not folds:
        return

    songs = [f["test_song"].replace(" - ", "\n")[:28] for f in folds]
    accs = [f["test_metrics"]["accuracy"] for f in folds]
    f1s  = [f["test_metrics"]["f1"] for f in folds]

    x = np.arange(len(folds))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(14, len(folds) * 0.8), 6))
    ax.bar(x - width / 2, accs, width, label="Accuracy",
           color="steelblue", alpha=0.85, edgecolor="black")
    ax.bar(x + width / 2, f1s, width, label="F1 Score",
           color="coral", alpha=0.85, edgecolor="black")

    ax.axhline(y=np.mean(accs), color="steelblue", linestyle="--", alpha=0.6,
               label=f"Mean Acc = {np.mean(accs):.3f}")
    ax.axhline(y=np.mean(f1s), color="coral", linestyle="--", alpha=0.6,
               label=f"Mean F1  = {np.mean(f1s):.3f}")
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Baseline = 0.50")

    ax.set_xticks(x)
    ax.set_xticklabels(songs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title(
        f"{model_name.replace('_', ' ').title()}: Per-Fold (Best Trial)\n"
        f"(each fold = one held-out song, 3 test clips)",
        fontsize=12,
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"piano_{model_name}_per_fold.png"), dpi=150)
    plt.close()


# ==========================================
# Per-fold metrics (averaged across all trials)
# ==========================================

def _plot_avg_per_fold_metrics(trials, model_name, output_dir):
    """Per-fold accuracy and F1 averaged across all trials."""
    all_fold_details = [t.get("fold_details", []) for t in trials]
    if not all_fold_details or not all_fold_details[0]:
        return

    n_folds = len(all_fold_details[0])
    n_trials = len(trials)
    songs = [f["test_song"].replace(" - ", "\n")[:28] for f in all_fold_details[0]]

    fold_accs = np.zeros((n_trials, n_folds))
    fold_f1s = np.zeros((n_trials, n_folds))

    for t_idx, folds in enumerate(all_fold_details):
        for f_idx, fd in enumerate(folds):
            fold_accs[t_idx, f_idx] = fd["test_metrics"]["accuracy"]
            fold_f1s[t_idx, f_idx] = fd["test_metrics"]["f1"]

    acc_means = fold_accs.mean(axis=0)
    acc_stds = fold_accs.std(axis=0)
    f1_means = fold_f1s.mean(axis=0)
    f1_stds = fold_f1s.std(axis=0)

    x = np.arange(n_folds)
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(14, n_folds * 0.8), 6))
    ax.bar(x - width / 2, acc_means, width, yerr=acc_stds, capsize=3,
           label="Accuracy", color="steelblue", alpha=0.85, edgecolor="black")
    ax.bar(x + width / 2, f1_means, width, yerr=f1_stds, capsize=3,
           label="F1 Score", color="coral", alpha=0.85, edgecolor="black")

    ax.axhline(y=np.mean(acc_means), color="steelblue", linestyle="--", alpha=0.6,
               label=f"Mean Acc = {np.mean(acc_means):.3f}")
    ax.axhline(y=np.mean(f1_means), color="coral", linestyle="--", alpha=0.6,
               label=f"Mean F1  = {np.mean(f1_means):.3f}")
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Baseline = 0.50")

    ax.set_xticks(x)
    ax.set_xticklabels(songs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Score (mean \u00b1 std across trials)", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title(
        f"{model_name.replace('_', ' ').title()}: Per-Fold Metrics\n"
        f"(averaged over {n_trials} trials, each fold = one held-out song)",
        fontsize=12,
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"piano_{model_name}_per_fold_avg.png"), dpi=150
    )
    plt.close()


# ==========================================
# Feature importances
# ==========================================

def _plot_feature_importances(best_trial, model_name, output_dir, top_n=15):
    """Feature importances from best trial (averaged across 18 LOGO-CV folds)."""
    importances = best_trial.get("feature_importances")
    feature_names = best_trial.get("feature_names")

    if importances is None or feature_names is None:
        return

    importances = np.array(importances)
    n_features = min(top_n, len(feature_names))
    indices = np.argsort(importances)[-n_features:][::-1]
    top_names = [feature_names[i] for i in indices]
    top_imps = importances[indices]

    fig, ax = plt.subplots(figsize=(10, max(4, n_features * 0.45)))
    y_pos = np.arange(n_features)
    ax.barh(y_pos, top_imps, color="steelblue", alpha=0.85, edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (avg across 18 folds)", fontsize=11)
    ax.set_title(
        f"Top-{n_features} Feature Importances\n"
        f"({model_name.replace('_', ' ').title()}, Best Trial)",
        fontsize=12,
    )
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"piano_{model_name}_feature_importance.png"), dpi=150)
    plt.close()


# ==========================================
# Public API — per-model summaries
# ==========================================

def plot_piano_boosting_summary(results, output_dir):
    """Generate all plots and report for Boosting classification."""
    os.makedirs(output_dir, exist_ok=True)
    model_name = "boosting"
    print(f"[plot] Generating {model_name} piano plots to {output_dir}")

    best_trial = _generate_classification_report(results, model_name, output_dir)
    trials = results["LOGO_CV"]

    if best_trial:
        _plot_confusion_matrix(best_trial, model_name, output_dir)
        _plot_feature_importances(best_trial, model_name, output_dir)
        _plot_per_fold_metrics(best_trial, model_name, output_dir)

    _plot_avg_confusion_matrix(trials, model_name, output_dir)
    _plot_avg_per_fold_metrics(trials, model_name, output_dir)
    _plot_train_vs_test(results, model_name, output_dir)
    print(f"[plot] Saved {model_name} piano plots and report")


def plot_piano_random_forest_summary(results, output_dir):
    """Generate all plots and report for Random Forest classification."""
    os.makedirs(output_dir, exist_ok=True)
    model_name = "random_forest"
    print(f"[plot] Generating {model_name} piano plots to {output_dir}")

    best_trial = _generate_classification_report(results, model_name, output_dir)
    trials = results["LOGO_CV"]

    if best_trial:
        _plot_confusion_matrix(best_trial, model_name, output_dir)
        _plot_feature_importances(best_trial, model_name, output_dir)
        _plot_per_fold_metrics(best_trial, model_name, output_dir)

    _plot_avg_confusion_matrix(trials, model_name, output_dir)
    _plot_avg_per_fold_metrics(trials, model_name, output_dir)
    _plot_train_vs_test(results, model_name, output_dir)
    print(f"[plot] Saved {model_name} piano plots and report")


def plot_piano_neural_network_summary(results, output_dir):
    """Generate all plots and report for Neural Network classification."""
    os.makedirs(output_dir, exist_ok=True)
    model_name = "neural_network"
    print(f"[plot] Generating {model_name} piano plots to {output_dir}")

    best_trial = _generate_classification_report(results, model_name, output_dir)
    trials = results["LOGO_CV"]

    if best_trial:
        _plot_confusion_matrix(best_trial, model_name, output_dir)
        _plot_per_fold_metrics(best_trial, model_name, output_dir)

    _plot_avg_confusion_matrix(trials, model_name, output_dir)
    _plot_avg_per_fold_metrics(trials, model_name, output_dir)
    _plot_train_vs_test(results, model_name, output_dir)
    print(f"[plot] Saved {model_name} piano plots and report")


# ==========================================
# Model comparison
# ==========================================

def plot_piano_model_comparison(all_results, output_dir):
    """
    Cross-model comparison: bar chart of mean test metrics and summary report.

    all_results = {model_name: {"LOGO_CV": [trial_records]}}
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[plot] Generating piano model comparison to {output_dir}")

    model_names = list(all_results.keys())
    split_name = "LOGO_CV"
    colors = ["steelblue", "forestgreen", "coral"]
    metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
    metric_labels = ["Accuracy", "F1", "Precision", "Recall", "ROC-AUC"]

    # ---- Grouped bar chart ----
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, model in enumerate(model_names):
        trials = all_results[model][split_name]
        means, stds = [], []
        for metric in metrics:
            vals = _metric_vals(trials, "test_metrics", metric)
            m, s = _mean_std(vals)
            means.append(m or 0)
            stds.append(s or 0)

        offset = (i - 1) * width
        ax.bar(
            x + offset, means, width, yerr=stds, capsize=3,
            label=model.replace("_", " ").title(),
            color=colors[i % len(colors)], alpha=0.85, edgecolor="black",
        )

    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Baseline = 0.50")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Score (mean ± std, 3 trials x 18 LOGO-CV folds)", fontsize=10)
    ax.set_title(
        "Piano Emotion Classification — Model Comparison\n"
        "(Leave-One-Song-Out CV, averaged over 3 trials)",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "piano_comparison_metrics.png"), dpi=150)
    plt.close()

    # ---- Train vs Test F1 per model ----
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(model_names))
    width = 0.35

    tr_means, tr_stds = [], []
    te_means, te_stds = [], []
    for model in model_names:
        trials = all_results[model][split_name]
        tr_vals = _metric_vals(trials, "train_metrics", "f1")
        te_vals = _metric_vals(trials, "test_metrics", "f1")
        tr_m, tr_s = _mean_std(tr_vals)
        te_m, te_s = _mean_std(te_vals)
        tr_means.append(tr_m or 0)
        tr_stds.append(tr_s or 0)
        te_means.append(te_m or 0)
        te_stds.append(te_s or 0)

    ax.bar(x - width / 2, tr_means, width, yerr=tr_stds, capsize=4,
           label="Train F1", color="steelblue", alpha=0.85, edgecolor="black")
    ax.bar(x + width / 2, te_means, width, yerr=te_stds, capsize=4,
           label="Test F1", color="coral", alpha=0.85, edgecolor="black")

    for i, (tm, em) in enumerate(zip(tr_means, te_means)):
        ax.text(i, max(tm, em) + 0.04, f"gap={tm - em:+.2f}",
                ha="center", va="bottom", fontsize=9, color="dimgray")

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in model_names], fontsize=11)
    ax.set_ylim(0, 1.3)
    ax.set_ylabel("F1 Score (mean ± std, 3 trials)", fontsize=11)
    ax.set_title("Train vs Test F1 — Overfitting Check (all models)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "piano_comparison_overfit.png"), dpi=150)
    plt.close()

    # ---- Summary report ----
    r = []
    r.append("=" * 80)
    r.append("PIANO EMOTION CLASSIFICATION — MODEL COMPARISON SUMMARY")
    r.append("=" * 80)

    r.append("")
    r.append("EXPERIMENT OVERVIEW")
    r.append("-" * 80)
    r.append("Task:         Binary Classification (emotional=0 vs happy=1)")
    r.append("Dataset:      18 songs x 3 clips = 54 samples (27 per class)")
    r.append("Features:     Top-20 by effect size (acoustic, rhythmic, timbral)")
    r.append("CV:           Leave-One-Song-Out (18 folds)")
    r.append("Inner CV:     StratifiedKFold(5) for GridSearchCV hyperparameter tuning")
    r.append("Trials:       3 per model with different random seeds (averaged)")
    r.append("Baseline:     F1 = 0.50, Accuracy = 0.50 (balanced classes)")
    r.append("")
    r.append("IMPORTANT: Labels reflect musical style/acoustic activity, NOT")
    r.append("validated emotional ground truth.")

    # Metrics table
    r.append("")
    r.append("-" * 80)
    r.append("AVERAGE TEST METRICS BY MODEL (averaged over 3 trials)")
    r.append("-" * 80)

    hdr = f"{'Model':<22}"
    for ml in metric_labels:
        hdr += f"{ml:>16}"
    r.append(hdr)
    r.append("-" * 80)

    for model in model_names:
        trials = all_results[model][split_name]
        row = f"{model:<22}"
        for metric in metrics:
            vals = _metric_vals(trials, "test_metrics", metric)
            m, s = _mean_std(vals)
            if m is not None:
                row += f"{m:>8.4f}±{s:<6.4f}"
            else:
                row += f"{'N/A':>16}"
        r.append(row)

    # Overfitting table
    r.append("")
    r.append("-" * 80)
    r.append("TRAIN vs TEST F1 (overfitting check — gap > 0.20 flagged)")
    r.append("-" * 80)
    r.append(f"{'Model':<22} {'Train F1':>14} {'Test F1':>14} {'Gap':>10} {'Flag'}")
    r.append("-" * 65)

    for model in model_names:
        trials = all_results[model][split_name]
        tr_vals = _metric_vals(trials, "train_metrics", "f1")
        te_vals = _metric_vals(trials, "test_metrics", "f1")
        tr_m, tr_s = _mean_std(tr_vals)
        te_m, te_s = _mean_std(te_vals)
        if tr_m is not None and te_m is not None:
            gap = tr_m - te_m
            flag = "OVERFIT" if gap > 0.20 else "OK"
            r.append(
                f"{model:<22} {tr_m:>8.4f}±{tr_s:<4.4f} "
                f"{te_m:>8.4f}±{te_s:<4.4f} {gap:>+10.4f} {flag}"
            )

    # Best single trial
    r.append("")
    r.append("-" * 80)
    r.append("BEST SINGLE TRIAL (for reference only)")
    r.append("-" * 80)

    best_f1 = -1.0
    best_model = None
    best_trial_info = None

    for model in model_names:
        for trial in all_results[model][split_name]:
            f1 = trial["test_metrics"].get("f1")
            if f1 is not None and f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_trial_info = trial

    if best_model:
        tm = best_trial_info["test_metrics"]
        r.append(f"  Model:     {best_model}")
        r.append(f"  Trial:     {best_trial_info['trial'] + 1}")
        r.append(f"  Accuracy:  {_safe(tm.get('accuracy'))}")
        r.append(f"  Precision: {_safe(tm.get('precision'))}")
        r.append(f"  Recall:    {_safe(tm.get('recall'))}")
        r.append(f"  F1:        {_safe(tm.get('f1'))}")
        r.append(f"  ROC-AUC:   {_safe(tm.get('roc_auc'))}")

    report_path = os.path.join(output_dir, "piano_model_comparison_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(r))

    print(f"[plot] Saved piano model comparison plots and report")
