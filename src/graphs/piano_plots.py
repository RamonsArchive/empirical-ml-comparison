"""
Plotting functions for Piano Emotion Classification experiment results.

Results structure (per model):
    results[model_name] = list of fold records, one per LOGO-CV fold.

Each fold record:
    {
        "fold":            int,
        "test_song":       str,
        "best_params":     dict,
        "cv_train_score":  float,
        "cv_val_score":    float,
        "cv_scoring":      str,
        "train_metrics":   {accuracy, precision, recall, f1, roc_auc},
        "test_metrics":    {accuracy, precision, recall, f1, roc_auc},
        "y_test":          list[int],
        "y_pred":          list[int],
        "y_proba":         list[float] | None,
        "feature_importances": list[float] | None,
        "feature_names":       list[str]  | None,
    }
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ==========================================
# Helpers
# ==========================================

def _safe(val):
    """Format float or return 'N/A'."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.4f}"


def _agg(folds, key, subkey):
    """Collect metric values across folds, dropping NaN/None."""
    vals = []
    for f in folds:
        v = f[key][subkey]
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            vals.append(v)
    return vals


def _mean_std(vals):
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


# ==========================================
# Report
# ==========================================

def _generate_classification_report(folds, model_name, output_dir):
    """
    Write a text report for one model's LOGO-CV results.

    Returns the fold with the best test F1.
    """
    r = []
    r.append("=" * 70)
    r.append(f"PIANO EMOTION CLASSIFICATION — {model_name.upper()} RESULTS")
    r.append("=" * 70)

    n_folds = len(folds)
    r.append(f"\nCV Strategy:  Leave-One-Song-Out ({n_folds} folds)")
    r.append(f"Note:         Each fold tests on all 3 clips of one held-out song.")

    # ---- Per-fold table ----
    r.append("\n" + "-" * 70)
    r.append("PER-FOLD TEST METRICS")
    r.append("-" * 70)
    r.append(
        f"\n{'Fold':<5} {'Song (test)':<38} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>8}"
    )
    r.append("-" * 70)

    best_f1 = -1.0
    best_fold = None

    for fold in folds:
        tm = fold["test_metrics"]
        auc_str = _safe(tm.get("roc_auc"))
        song_short = fold["test_song"][:36]
        r.append(
            f"{fold['fold'] + 1:<5} {song_short:<38} "
            f"{tm['accuracy']:>6.3f} {tm['precision']:>6.3f} "
            f"{tm['recall']:>6.3f} {tm['f1']:>6.3f} {auc_str:>8}"
        )
        if tm["f1"] > best_f1:
            best_f1 = tm["f1"]
            best_fold = fold

    # ---- Aggregate test metrics ----
    r.append("\n" + "-" * 70)
    r.append("AGGREGATE TEST METRICS (mean ± std across folds)")
    r.append("-" * 70)

    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        vals = _agg(folds, "test_metrics", metric)
        mean, std = _mean_std(vals)
        if mean is not None:
            r.append(f"  Test  {metric:<12}: {mean:.4f} ± {std:.4f}")
        else:
            r.append(f"  Test  {metric:<12}: N/A")

    # ---- Aggregate train metrics (overfitting check) ----
    r.append("\n" + "-" * 70)
    r.append("AGGREGATE TRAIN METRICS (overfitting check)")
    r.append("-" * 70)

    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        train_vals = _agg(folds, "train_metrics", metric)
        test_vals = _agg(folds, "test_metrics", metric)
        train_m, train_s = _mean_std(train_vals)
        test_m, test_s = _mean_std(test_vals)
        if train_m is not None and test_m is not None:
            gap = train_m - test_m
            flag = "  *** OVERFIT" if gap > 0.20 else ""
            r.append(
                f"  Train {metric:<12}: {train_m:.4f} ± {train_s:.4f}  |  "
                f"Test: {test_m:.4f} ± {test_s:.4f}  |  Gap: {gap:+.4f}{flag}"
            )

    # ---- CV inner scores ----
    r.append("\n" + "-" * 70)
    r.append("INNER CV SCORES (GridSearchCV, best params per fold)")
    r.append("-" * 70)
    cv_train_vals = [f["cv_train_score"] for f in folds]
    cv_val_vals = [f["cv_val_score"] for f in folds]
    scoring = folds[0]["cv_scoring"] if folds else "?"
    r.append(f"  Scoring:       {scoring}")
    r.append(f"  CV Train Mean: {np.mean(cv_train_vals):.4f} ± {np.std(cv_train_vals):.4f}")
    r.append(f"  CV Val   Mean: {np.mean(cv_val_vals):.4f} ± {np.std(cv_val_vals):.4f}")

    # ---- Best fold details ----
    if best_fold:
        r.append("\n" + "-" * 70)
        r.append(f"BEST FOLD (highest test F1 = {best_f1:.4f})")
        r.append("-" * 70)
        r.append(f"  Song:      {best_fold['test_song']}")
        r.append(f"  Fold:      {best_fold['fold'] + 1}")
        r.append(f"\n  Best hyperparameters:")
        for param, val in best_fold["best_params"].items():
            r.append(f"    {param}: {val}")
        tm = best_fold["test_metrics"]
        r.append(f"\n  Test Accuracy:  {tm['accuracy']:.4f}")
        r.append(f"  Test Precision: {tm['precision']:.4f}")
        r.append(f"  Test Recall:    {tm['recall']:.4f}")
        r.append(f"  Test F1:        {tm['f1']:.4f}")
        r.append(f"  Test ROC-AUC:   {_safe(tm.get('roc_auc'))}")
        tr = best_fold["train_metrics"]
        r.append(f"\n  Train Accuracy: {tr['accuracy']:.4f}")
        r.append(f"  Train F1:       {tr['f1']:.4f}")

    # ---- Interpretation ----
    r.append("\n" + "-" * 70)
    r.append("INTERPRETATION")
    r.append("-" * 70)

    test_f1_vals = _agg(folds, "test_metrics", "f1")
    mean_f1, _ = _mean_std(test_f1_vals)
    if mean_f1 is not None:
        if mean_f1 > 0.90:
            quality = "EXCELLENT"
        elif mean_f1 > 0.75:
            quality = "GOOD"
        elif mean_f1 > 0.60:
            quality = "MODERATE"
        elif mean_f1 > 0.50:
            quality = "FAIR"
        else:
            quality = "POOR (at or below majority-class baseline)"
        r.append(f"  Model Quality: {quality} (Mean F1 = {mean_f1:.4f})")
        r.append(f"  Baseline F1:   0.5000 (majority class predictor on balanced data)")
        r.append(
            f"\n  NOTE: These labels classify piano music by acoustic activity and style,\n"
            f"  not by validated emotional ground truth. A slow classical piece or an\n"
            f"  atypical happy/sad piece may be misclassified."
        )

    report_path = os.path.join(output_dir, f"piano_{model_name}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(r))

    return best_fold


# ==========================================
# Per-fold metrics bar chart
# ==========================================

def _plot_per_fold_metrics(folds, model_name, output_dir):
    """Bar chart of accuracy and F1 per LOGO-CV fold (one bar per song)."""
    songs = [f["test_song"].replace(" - ", "\n")[:28] for f in folds]
    accs = [f["test_metrics"]["accuracy"] for f in folds]
    f1s = [f["test_metrics"]["f1"] for f in folds]

    x = np.arange(len(folds))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(14, len(folds) * 0.8), 6))
    ax.bar(x - width / 2, accs, width, label="Accuracy", color="steelblue", alpha=0.85, edgecolor="black")
    ax.bar(x + width / 2, f1s, width, label="F1 Score", color="coral", alpha=0.85, edgecolor="black")

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
        f"{model_name.replace('_', ' ').title()}: Per-Fold Accuracy & F1\n"
        f"(each fold = one held-out song, 3 test clips)",
        fontsize=12,
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"piano_{model_name}_per_fold.png"), dpi=150)
    plt.close()


# ==========================================
# Train vs Test overfitting chart
# ==========================================

def _plot_train_vs_test(folds, model_name, output_dir):
    """
    Grouped bar chart: mean train vs mean test for accuracy, F1, precision, recall.
    Makes overfit gaps immediately visible.
    """
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1"]

    train_means = []
    test_means = []
    train_stds = []
    test_stds = []

    for metric in metrics:
        tr_vals = _agg(folds, "train_metrics", metric)
        te_vals = _agg(folds, "test_metrics", metric)
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
                ha="center", va="bottom", fontsize=8, color="dimgray")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Score (mean ± std across 18 folds)", fontsize=11)
    ax.set_title(
        f"{model_name.replace('_', ' ').title()}: Train vs Test Metrics\n"
        f"(positive gap = overfitting)",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"piano_{model_name}_train_vs_test.png"), dpi=150)
    plt.close()


# ==========================================
# Confusion matrix (accumulated across folds)
# ==========================================

def _plot_confusion_matrix(folds, model_name, output_dir):
    """Accumulated confusion matrix across all LOGO-CV folds."""
    tp = tn = fp = fn = 0
    for fold in folds:
        y_test = np.array(fold["y_test"])
        y_pred = np.array(fold["y_pred"])
        tp += int(np.sum((y_test == 1) & (y_pred == 1)))
        tn += int(np.sum((y_test == 0) & (y_pred == 0)))
        fp += int(np.sum((y_test == 0) & (y_pred == 1)))
        fn += int(np.sum((y_test == 1) & (y_pred == 0)))

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

    total_test = tp + tn + fp + fn
    overall_acc = (tp + tn) / total_test if total_test > 0 else 0
    ax.set_title(
        f"{model_name.replace('_', ' ').title()}: Accumulated Confusion Matrix\n"
        f"(all LOGO-CV folds, N={total_test}, Acc={overall_acc:.3f})",
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
# Feature importances (RF / Boosting)
# ==========================================

def _plot_feature_importances(folds, model_name, output_dir, top_n=15):
    """
    Average feature importances across all folds (RF and Boosting only).
    Uses folds that have valid feature_importances and feature_names.
    """
    valid_folds = [
        f for f in folds
        if f.get("feature_importances") is not None and f.get("feature_names") is not None
    ]
    if not valid_folds:
        return

    # Average importance across folds
    feature_names = valid_folds[0]["feature_names"]
    importance_matrix = np.array([f["feature_importances"] for f in valid_folds])
    mean_importance = importance_matrix.mean(axis=0)
    std_importance = importance_matrix.std(axis=0)

    n_features = min(top_n, len(feature_names))
    indices = np.argsort(mean_importance)[-n_features:][::-1]
    top_names = [feature_names[i] for i in indices]
    top_means = mean_importance[indices]
    top_stds = std_importance[indices]

    fig, ax = plt.subplots(figsize=(10, max(4, n_features * 0.45)))
    y_pos = np.arange(n_features)
    ax.barh(y_pos, top_means, xerr=top_stds, color="steelblue", alpha=0.85,
            edgecolor="black", capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(f"Mean Feature Importance (± std, {len(valid_folds)} folds)", fontsize=11)
    ax.set_title(
        f"Top-{n_features} Feature Importances\n({model_name.replace('_', ' ').title()})",
        fontsize=12,
    )
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"piano_{model_name}_feature_importance.png"), dpi=150)
    plt.close()


# ==========================================
# Public API — per-model summaries
# ==========================================

def plot_piano_boosting_summary(folds, output_dir):
    """Generate all plots and report for Boosting classification."""
    os.makedirs(output_dir, exist_ok=True)
    model_name = "boosting"
    print(f"[plot] Generating {model_name} piano plots to {output_dir}")

    best_fold = _generate_classification_report(folds, model_name, output_dir)
    _plot_per_fold_metrics(folds, model_name, output_dir)
    _plot_train_vs_test(folds, model_name, output_dir)
    _plot_confusion_matrix(folds, model_name, output_dir)
    _plot_feature_importances(folds, model_name, output_dir)

    print(f"[plot] Saved {model_name} piano plots and report")


def plot_piano_random_forest_summary(folds, output_dir):
    """Generate all plots and report for Random Forest classification."""
    os.makedirs(output_dir, exist_ok=True)
    model_name = "random_forest"
    print(f"[plot] Generating {model_name} piano plots to {output_dir}")

    best_fold = _generate_classification_report(folds, model_name, output_dir)
    _plot_per_fold_metrics(folds, model_name, output_dir)
    _plot_train_vs_test(folds, model_name, output_dir)
    _plot_confusion_matrix(folds, model_name, output_dir)
    _plot_feature_importances(folds, model_name, output_dir)

    print(f"[plot] Saved {model_name} piano plots and report")


def plot_piano_neural_network_summary(folds, output_dir):
    """Generate all plots and report for Neural Network classification."""
    os.makedirs(output_dir, exist_ok=True)
    model_name = "neural_network"
    print(f"[plot] Generating {model_name} piano plots to {output_dir}")

    best_fold = _generate_classification_report(folds, model_name, output_dir)
    _plot_per_fold_metrics(folds, model_name, output_dir)
    _plot_train_vs_test(folds, model_name, output_dir)
    _plot_confusion_matrix(folds, model_name, output_dir)
    # No feature importances for MLP

    print(f"[plot] Saved {model_name} piano plots and report")


# ==========================================
# Model comparison
# ==========================================

def plot_piano_model_comparison(all_results, output_dir):
    """
    Cross-model comparison: bar chart of mean test metrics and summary report.

    Args:
        all_results: dict  {model_name: list_of_fold_records}
        output_dir:  str   path for outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[plot] Generating piano model comparison to {output_dir}")

    model_names = list(all_results.keys())
    colors = ["steelblue", "forestgreen", "coral"]
    metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
    metric_labels = ["Accuracy", "F1", "Precision", "Recall", "ROC-AUC"]

    # ---- Grouped bar chart ----
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, model in enumerate(model_names):
        folds = all_results[model]
        means, stds = [], []
        for metric in metrics:
            vals = _agg(folds, "test_metrics", metric)
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
    ax.set_ylabel("Score (mean ± std, 18 LOGO-CV folds)", fontsize=11)
    ax.set_title("Piano Emotion Classification — Model Comparison\n(Leave-One-Song-Out CV)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "piano_comparison_metrics.png"), dpi=150)
    plt.close()

    # ---- Train vs Test per model (overfitting overview) ----
    fig, axes = plt.subplots(1, len(model_names), figsize=(5 * len(model_names), 5), sharey=True)
    if len(model_names) == 1:
        axes = [axes]

    for ax, model in zip(axes, model_names):
        folds = all_results[model]
        tr_f1 = [f["train_metrics"]["f1"] for f in folds]
        te_f1 = [f["test_metrics"]["f1"] for f in folds]

        ax.scatter(
            [f["fold"] + 1 for f in folds], tr_f1,
            label="Train F1", color="steelblue", alpha=0.8, s=60, marker="o",
        )
        ax.scatter(
            [f["fold"] + 1 for f in folds], te_f1,
            label="Test F1", color="coral", alpha=0.8, s=60, marker="s",
        )
        ax.axhline(y=np.mean(tr_f1), color="steelblue", linestyle="--", alpha=0.5)
        ax.axhline(y=np.mean(te_f1), color="coral", linestyle="--", alpha=0.5)
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
        ax.set_xlabel("Fold (song index)", fontsize=10)
        ax.set_ylabel("F1 Score", fontsize=10)
        ax.set_title(model.replace("_", " ").title(), fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.15)

    plt.suptitle("Train vs Test F1 per Fold — Overfitting Check", fontsize=13, fontweight="bold")
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
    r.append("CV:           Leave-One-Song-Out (18 folds; test = 3 clips of 1 song)")
    r.append("Inner CV:     StratifiedKFold(5) for GridSearchCV hyperparameter tuning")
    r.append("Baseline:     F1 = 0.50, Accuracy = 0.50 (balanced classes)")
    r.append("")
    r.append("IMPORTANT: Labels reflect musical style/acoustic activity, NOT")
    r.append("validated emotional ground truth. The model separates 'slow/sparse'")
    r.append("from 'fast/dense' piano style, not objective human emotion states.")

    r.append("")
    r.append("-" * 80)
    r.append("TEST METRICS BY MODEL (mean ± std, 18 LOGO-CV folds)")
    r.append("-" * 80)

    hdr = f"{'Model':<22}"
    for ml in metric_labels:
        hdr += f"{ml:>16}"
    r.append(hdr)
    r.append("-" * 80)

    for model in model_names:
        folds = all_results[model]
        row = f"{model:<22}"
        for metric in metrics:
            vals = _agg(folds, "test_metrics", metric)
            m, s = _mean_std(vals)
            if m is not None:
                row += f"{m:>8.4f}±{s:<6.4f}"
            else:
                row += f"{'N/A':>16}"
        r.append(row)

    r.append("")
    r.append("-" * 80)
    r.append("TRAIN vs TEST F1 (overfitting check — large gap > 0.20 flagged)")
    r.append("-" * 80)
    r.append(f"{'Model':<22} {'Train F1':>12} {'Test F1':>12} {'Gap':>10} {'Flag'}")
    r.append("-" * 60)

    for model in model_names:
        folds = all_results[model]
        tr_vals = _agg(folds, "train_metrics", "f1")
        te_vals = _agg(folds, "test_metrics", "f1")
        tr_m, tr_s = _mean_std(tr_vals)
        te_m, te_s = _mean_std(te_vals)
        if tr_m is not None and te_m is not None:
            gap = tr_m - te_m
            flag = "OVERFIT" if gap > 0.20 else "OK"
            r.append(
                f"{model:<22} {tr_m:>8.4f}±{tr_s:<3.4f} "
                f"{te_m:>8.4f}±{te_s:<3.4f} {gap:>+10.4f} {flag}"
            )

    # Best single fold
    r.append("")
    r.append("-" * 80)
    r.append("BEST SINGLE FOLD (highest test F1 across all models)")
    r.append("-" * 80)

    best_f1 = -1.0
    best_model = best_song = None
    best_fold_info = None
    for model in model_names:
        for fold in all_results[model]:
            if fold["test_metrics"]["f1"] > best_f1:
                best_f1 = fold["test_metrics"]["f1"]
                best_model = model
                best_song = fold["test_song"]
                best_fold_info = fold

    if best_fold_info:
        tm = best_fold_info["test_metrics"]
        r.append(f"  Model:     {best_model}")
        r.append(f"  Test Song: {best_song}")
        r.append(f"  Accuracy:  {tm['accuracy']:.4f}")
        r.append(f"  Precision: {tm['precision']:.4f}")
        r.append(f"  Recall:    {tm['recall']:.4f}")
        r.append(f"  F1:        {best_f1:.4f}")
        r.append(f"  ROC-AUC:   {_safe(tm.get('roc_auc'))}")

    report_path = os.path.join(output_dir, "piano_model_comparison_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(r))

    print(f"[plot] Saved piano model comparison plots and report")
