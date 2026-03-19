"""
Plotting functions for Rescue Classification experiment results.
Generates summary reports, confusion matrices, and comparison charts.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def _safe_metric(val):
    """Return float or 'N/A' string for display."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.4f}"


def _generate_classification_report(results, model_name, output_dir):
    """Generate text report for a classification model with train+test metrics."""
    r = []
    r.append("=" * 70)
    r.append(f"RESCUE CLASSIFICATION - {model_name.upper()} RESULTS")
    r.append("=" * 70)

    split_names = []
    # Train metrics (refit on full train set)
    mean_train_acc = []
    mean_train_f1 = []
    # Test metrics
    mean_acc = []
    mean_prec = []
    mean_rec = []
    mean_f1 = []
    mean_auc = []

    best_f1 = -1
    best_split = None
    best_trial = None

    for split_name, trials in results.items():
        split_names.append(split_name)

        accs = [t["test_metrics"]["accuracy"] for t in trials]
        precs = [t["test_metrics"]["precision"] for t in trials]
        recs = [t["test_metrics"]["recall"] for t in trials]
        f1s = [t["test_metrics"]["f1"] for t in trials]
        aucs = [t["test_metrics"]["roc_auc"] for t in trials if t["test_metrics"]["roc_auc"] is not None]

        # Train metrics
        train_accs = [t["train_metrics"]["accuracy"] for t in trials if t.get("train_metrics")]
        train_f1s = [t["train_metrics"]["f1"] for t in trials if t.get("train_metrics")]
        mean_train_acc.append(np.mean(train_accs) if train_accs else None)
        mean_train_f1.append(np.mean(train_f1s) if train_f1s else None)

        mean_acc.append(np.mean(accs))
        mean_prec.append(np.mean(precs))
        mean_rec.append(np.mean(recs))
        mean_f1.append(np.mean(f1s))
        mean_auc.append(np.mean(aucs) if aucs else None)

        for trial in trials:
            if trial["test_metrics"]["f1"] > best_f1:
                best_f1 = trial["test_metrics"]["f1"]
                best_split = split_name
                best_trial = trial

    # Performance by Split - TRAIN metrics
    r.append("\n" + "-" * 70)
    r.append("TRAIN PERFORMANCE BY SPLIT (averaged over 3 trials)")
    r.append("-" * 70)
    r.append(f"\n{'Split':<10} {'Train Acc':<12} {'Train F1':<12}")
    r.append("-" * 40)

    for i, split_name in enumerate(split_names):
        t_acc = _safe_metric(mean_train_acc[i])
        t_f1 = _safe_metric(mean_train_f1[i])
        r.append(f"{split_name:<10} {t_acc:<12} {t_f1:<12}")

    # Performance by Split - TEST metrics
    r.append("\n" + "-" * 70)
    r.append("TEST PERFORMANCE BY SPLIT (averaged over 3 trials)")
    r.append("-" * 70)
    r.append(
        f"\n{'Split':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'ROC-AUC':<12}"
    )
    r.append("-" * 70)

    for i, split_name in enumerate(split_names):
        auc_str = _safe_metric(mean_auc[i])
        r.append(
            f"{split_name:<10} "
            f"{mean_acc[i]:<12.4f} "
            f"{mean_prec[i]:<12.4f} "
            f"{mean_rec[i]:<12.4f} "
            f"{mean_f1[i]:<12.4f} "
            f"{auc_str:<12}"
        )

    # Overfitting Analysis
    r.append("\n" + "-" * 70)
    r.append("OVERFITTING ANALYSIS (Train - Test gap, averaged over 3 trials)")
    r.append("-" * 70)
    r.append(f"\n{'Split':<10} {'Train F1':<12} {'Test F1':<12} {'Gap':<12} {'Status':<15}")
    r.append("-" * 60)

    for i, split_name in enumerate(split_names):
        t_f1 = mean_train_f1[i]
        te_f1 = mean_f1[i]
        if t_f1 is not None:
            gap = t_f1 - te_f1
            if gap > 0.3:
                status = "SEVERE OVERFIT"
            elif gap > 0.15:
                status = "MODERATE OVERFIT"
            elif gap > 0.05:
                status = "MILD OVERFIT"
            else:
                status = "OK"
            r.append(
                f"{split_name:<10} {t_f1:<12.4f} {te_f1:<12.4f} {gap:+<12.4f} {status:<15}"
            )
        else:
            r.append(f"{split_name:<10} {'N/A':<12} {te_f1:<12.4f} {'N/A':<12} {'N/A':<15}")

    # Best Model Details
    if best_trial:
        r.append("\n" + "-" * 70)
        r.append("BEST MODEL (highest F1)")
        r.append("-" * 70)
        r.append(f"Split: {best_split}")
        r.append(f"Trial: {best_trial['trial'] + 1}")
        r.append(f"\nBest Parameters:")
        for param, val in best_trial["best_params"].items():
            r.append(f"  {param}: {val}")

        r.append(f"\nTrain Metrics:")
        tm = best_trial.get("train_metrics", {})
        if tm:
            r.append(f"  Accuracy:  {tm.get('accuracy', 'N/A'):.4f}")
            r.append(f"  Precision: {tm.get('precision', 'N/A'):.4f}")
            r.append(f"  Recall:    {tm.get('recall', 'N/A'):.4f}")
            r.append(f"  F1:        {tm.get('f1', 'N/A'):.4f}")
        else:
            r.append("  (not available)")

        r.append(f"\nTest Metrics:")
        r.append(f"  Accuracy:  {best_trial['test_metrics']['accuracy']:.4f}")
        r.append(f"  Precision: {best_trial['test_metrics']['precision']:.4f}")
        r.append(f"  Recall:    {best_trial['test_metrics']['recall']:.4f}")
        r.append(f"  F1:        {best_trial['test_metrics']['f1']:.4f}")
        r.append(f"  ROC-AUC:   {_safe_metric(best_trial['test_metrics']['roc_auc'])}")

    # Interpretation
    r.append("\n" + "-" * 70)
    r.append("INTERPRETATION")
    r.append("-" * 70)
    if best_trial:
        f1 = best_trial["test_metrics"]["f1"]
        if f1 > 0.9:
            quality = "EXCELLENT"
        elif f1 > 0.7:
            quality = "GOOD"
        elif f1 > 0.5:
            quality = "MODERATE"
        else:
            quality = "POOR"
        r.append(f"Model Quality: {quality} (Test F1 = {f1:.4f})")

        # Overfitting warning
        tm = best_trial.get("train_metrics", {})
        if tm and "f1" in tm:
            gap = tm["f1"] - f1
            if gap > 0.15:
                r.append(f"WARNING: Train F1 ({tm['f1']:.4f}) >> Test F1 ({f1:.4f}), "
                         f"gap = {gap:+.4f}. Model is overfitting.")

    # Save report
    report_path = os.path.join(output_dir, f"rescue_cls_{model_name}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(r))

    return best_trial


def _plot_confusion_matrix(best_trial, model_name, output_dir):
    """Plot confusion matrix for best trial."""
    y_test = np.array(best_trial["y_test"])
    y_pred = np.array(best_trial["y_pred"])

    # Build confusion matrix manually
    tp = np.sum((y_test == 1) & (y_pred == 1))
    tn = np.sum((y_test == 0) & (y_pred == 0))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    fn = np.sum((y_test == 1) & (y_pred == 0))
    cm = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    labels = ["Not Helped (0)", "Helped (1)"]
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=labels,
        yticklabels=labels,
        ylabel="Actual",
        xlabel="Predicted",
    )
    ax.set_title(f"{model_name}: Confusion Matrix\nF1 = {best_trial['test_metrics']['f1']:.4f}", fontsize=14)

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"rescue_cls_{model_name}_confusion.png"), dpi=150)
    plt.close()


def _plot_metrics_by_split(results, model_name, output_dir):
    """Plot accuracy and F1 across splits."""
    split_names = list(results.keys())

    mean_acc = []
    std_acc = []
    mean_f1 = []
    std_f1 = []

    for split_name in split_names:
        trials = results[split_name]
        accs = [t["test_metrics"]["accuracy"] for t in trials]
        f1s = [t["test_metrics"]["f1"] for t in trials]
        mean_acc.append(np.mean(accs))
        std_acc.append(np.std(accs))
        mean_f1.append(np.mean(f1s))
        std_f1.append(np.std(f1s))

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(split_names))
    width = 0.35

    bars1 = ax.bar(x - width / 2, mean_acc, width, yerr=std_acc, capsize=5,
                   color="steelblue", alpha=0.8, label="Accuracy")
    bars2 = ax.bar(x + width / 2, mean_f1, width, yerr=std_f1, capsize=5,
                   color="coral", alpha=0.8, label="F1 Score")

    ax.set_xlabel("Train/Test Split", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"{model_name}: Accuracy & F1 by Split (± std over 3 trials)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(split_names)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    for bar, mean in zip(bars1, mean_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{mean:.2f}", ha="center", va="bottom", fontsize=10)
    for bar, mean in zip(bars2, mean_f1):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{mean:.2f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"rescue_cls_{model_name}_metrics_by_split.png"), dpi=150)
    plt.close()


def _plot_train_test_roc_auc(results, model_name, output_dir):
    """Plot Train vs Test ROC-AUC side-by-side by split for a single model."""
    split_names = list(results.keys())
    train_aucs = []
    test_aucs = []

    for split_name in split_names:
        trials = results[split_name]
        tr = [t["train_metrics"]["roc_auc"] for t in trials
              if t.get("train_metrics") and t["train_metrics"].get("roc_auc") is not None]
        te = [t["test_metrics"]["roc_auc"] for t in trials
              if t["test_metrics"].get("roc_auc") is not None]
        train_aucs.append(np.mean(tr) if tr else 0)
        test_aucs.append(np.mean(te) if te else 0)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(split_names))
    width = 0.35

    bars_train = ax.bar(x - width / 2, train_aucs, width, color="#4C72B0",
                        alpha=0.85, edgecolor="black", linewidth=0.8, label="Train ROC-AUC")
    bars_test = ax.bar(x + width / 2, test_aucs, width, color="#DD8452",
                       alpha=0.85, edgecolor="black", linewidth=0.8, label="Test ROC-AUC")

    for bar, val in zip(bars_train, train_aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar, val in zip(bars_test, test_aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("Train/Test Split", fontsize=12)
    ax.set_ylabel("ROC-AUC (higher is better)", fontsize=12)
    ax.set_title(f"{model_name.replace('_', ' ').title()}: Train vs Test ROC-AUC\n(averaged over 3 trials per split)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(split_names)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"rescue_cls_{model_name}_train_test_roc_auc.png"), dpi=150)
    plt.close()


def _plot_feature_importances(best_trial, model_name, output_dir):
    """Plot top feature importances."""
    importances = np.array(best_trial["feature_importances"])
    feature_names = best_trial["feature_names"]

    n_features = min(15, len(feature_names))
    indices = np.argsort(importances)[-n_features:][::-1]
    top_importances = importances[indices]
    top_names = [feature_names[i] for i in indices]

    fig, ax = plt.subplots(figsize=(10, max(4, n_features * 0.4)))

    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_importances, color="steelblue", alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title(f"Top Feature Importances ({model_name})", fontsize=14)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"rescue_cls_{model_name}_feature_importance.png"), dpi=150)
    plt.close()


# ==========================================
# Public API
# ==========================================

def plot_rescue_cls_boosting_summary(results, output_dir):
    """Generate all plots and report for Boosting classification."""
    os.makedirs(output_dir, exist_ok=True)
    model_name = "boosting"

    print(f"[plot] Generating {model_name} classification plots to {output_dir}")

    best_trial = _generate_classification_report(results, model_name, output_dir)

    if best_trial:
        _plot_confusion_matrix(best_trial, model_name, output_dir)
        if best_trial.get("feature_importances") and best_trial.get("feature_names"):
            _plot_feature_importances(best_trial, model_name, output_dir)

    _plot_metrics_by_split(results, model_name, output_dir)
    _plot_train_test_roc_auc(results, model_name, output_dir)
    print(f"[plot] Saved {model_name} classification plots and report")


def plot_rescue_cls_random_forest_summary(results, output_dir):
    """Generate all plots and report for Random Forest classification."""
    os.makedirs(output_dir, exist_ok=True)
    model_name = "random_forest"

    print(f"[plot] Generating {model_name} classification plots to {output_dir}")

    best_trial = _generate_classification_report(results, model_name, output_dir)

    if best_trial:
        _plot_confusion_matrix(best_trial, model_name, output_dir)
        if best_trial.get("feature_importances") and best_trial.get("feature_names"):
            _plot_feature_importances(best_trial, model_name, output_dir)

    _plot_metrics_by_split(results, model_name, output_dir)
    _plot_train_test_roc_auc(results, model_name, output_dir)
    print(f"[plot] Saved {model_name} classification plots and report")


def plot_rescue_cls_neural_network_summary(results, output_dir):
    """Generate all plots and report for Neural Network classification."""
    os.makedirs(output_dir, exist_ok=True)
    model_name = "neural_network"

    print(f"[plot] Generating {model_name} classification plots to {output_dir}")

    best_trial = _generate_classification_report(results, model_name, output_dir)

    if best_trial:
        _plot_confusion_matrix(best_trial, model_name, output_dir)

    _plot_metrics_by_split(results, model_name, output_dir)
    _plot_train_test_roc_auc(results, model_name, output_dir)
    print(f"[plot] Saved {model_name} classification plots and report")


def plot_rescue_cls_model_comparison(all_results, output_dir):
    """Generate comparison plots and report across all models."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"[plot] Generating classification model comparison to {output_dir}")

    model_names = list(all_results.keys())
    splits = list(all_results[model_names[0]].keys())

    colors = ["steelblue", "forestgreen", "coral"]
    width = 0.25

    # ==========================================
    # F1 Comparison
    # ==========================================
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(splits))

    for i, model in enumerate(model_names):
        f1_means = []
        f1_stds = []
        for split in splits:
            f1s = [t["test_metrics"]["f1"] for t in all_results[model][split]]
            f1_means.append(np.mean(f1s))
            f1_stds.append(np.std(f1s))

        offset = (i - 1) * width
        ax.bar(
            x + offset, f1_means, width, yerr=f1_stds, capsize=3,
            label=model.replace("_", " ").title(),
            color=colors[i % len(colors)], alpha=0.8,
        )

    ax.set_xlabel("Train/Test Split", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Model Comparison: F1 by Split (± std over 3 trials)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cls_comparison_f1.png"), dpi=150)
    plt.close()

    # ==========================================
    # Accuracy Comparison
    # ==========================================
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model in enumerate(model_names):
        acc_means = []
        acc_stds = []
        for split in splits:
            accs = [t["test_metrics"]["accuracy"] for t in all_results[model][split]]
            acc_means.append(np.mean(accs))
            acc_stds.append(np.std(accs))

        offset = (i - 1) * width
        ax.bar(
            x + offset, acc_means, width, yerr=acc_stds, capsize=3,
            label=model.replace("_", " ").title(),
            color=colors[i % len(colors)], alpha=0.8,
        )

    ax.set_xlabel("Train/Test Split", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Model Comparison: Accuracy by Split (± std over 3 trials)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cls_comparison_accuracy.png"), dpi=150)
    plt.close()

    # ==========================================
    # Train vs Test ROC-AUC Comparison (all models)
    # ==========================================
    fig, ax = plt.subplots(figsize=(12, 6))
    n_models = len(model_names)
    bar_width = 0.35
    x_models = np.arange(n_models)

    train_auc_means = []
    test_auc_means = []
    for model in model_names:
        all_train_auc = []
        all_test_auc = []
        for split in splits:
            for t in all_results[model][split]:
                if t.get("train_metrics") and t["train_metrics"].get("roc_auc") is not None:
                    all_train_auc.append(t["train_metrics"]["roc_auc"])
                if t["test_metrics"].get("roc_auc") is not None:
                    all_test_auc.append(t["test_metrics"]["roc_auc"])
        train_auc_means.append(np.mean(all_train_auc) if all_train_auc else 0)
        test_auc_means.append(np.mean(all_test_auc) if all_test_auc else 0)

    bars_tr = ax.bar(x_models - bar_width / 2, train_auc_means, bar_width, color="#4C72B0",
                     alpha=0.85, edgecolor="black", linewidth=0.8, label="Train ROC-AUC")
    bars_te = ax.bar(x_models + bar_width / 2, test_auc_means, bar_width, color="#DD8452",
                     alpha=0.85, edgecolor="black", linewidth=0.8, label="Test ROC-AUC")

    for bar, val in zip(bars_tr, train_auc_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar, val in zip(bars_te, test_auc_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("ROC-AUC (higher is better)", fontsize=12)
    ax.set_title("Train vs Test ROC-AUC: All Models\n(averaged over all trials & splits)", fontsize=14, fontweight="bold")
    ax.set_xticks(x_models)
    ax.set_xticklabels([m.replace("_", " ").title() for m in model_names])
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cls_comparison_train_test_roc_auc.png"), dpi=150)
    plt.close()

    # ==========================================
    # Summary Report
    # ==========================================
    r = []
    r.append("=" * 80)
    r.append("RESCUE CLASSIFICATION - MODEL COMPARISON SUMMARY")
    r.append("=" * 80)

    r.append("")
    r.append("EXPERIMENT OVERVIEW")
    r.append("-" * 80)
    r.append("Task:           Binary Classification (Helped vs Did Not Help)")
    r.append("Design:         Within-subjects, counterbalanced across 3 conditions")
    r.append("Modeled Data:   Conditions 1 (Single) & 2 (Multiple) only")
    r.append("                Condition 0 (Empty) excluded - no victims by design")
    r.append("Target:         Helped (PeopleRescued > 0 = 1, else 0)")
    r.append("Predictors:     Age, Gender, Group, Condition, TimeElapsed, DrownCount,")
    r.append("                TimeNearVictim, baseline_TimeElapsed, baseline_DrownCount")
    r.append("CV Strategy:    Leave-One-Out Cross-Validation (LOOCV)")
    r.append(f"Splits:         {', '.join(splits)}")
    r.append("Trials:         3 per split with different random seeds")

    # Naive baseline
    r.append("")
    r.append("-" * 80)
    r.append("NAIVE BASELINE (Majority Class Predictor)")
    r.append("-" * 80)

    # Calculate from all test sets across splits
    for split in splits:
        all_y_test = []
        for model in model_names:
            for trial in all_results[model][split]:
                all_y_test.extend(trial["y_test"])
        y_arr = np.array(all_y_test[:len(all_results[model_names[0]][split][0]["y_test"])])
        pos_rate = np.mean(y_arr)
        majority_class = 1 if pos_rate >= 0.5 else 0
        majority_acc = max(pos_rate, 1 - pos_rate)
        r.append(f"  {split}: class balance = {pos_rate:.0%} positive, "
                 f"majority class = {majority_class}, baseline accuracy = {majority_acc:.2f}")
    r.append("")
    r.append("Any useful model must beat these baselines on BOTH accuracy and F1.")
    r.append("F1 baseline for always-predict-majority: 0.00 (never predicts minority class)")

    # F1 table
    r.append("")
    r.append("-" * 80)
    r.append("AVERAGE TEST F1 BY MODEL AND SPLIT (averaged over 3 trials)")
    r.append("-" * 80)

    header = f"{'Model':<20}"
    for split in splits:
        header += f"{split:<15}"
    header += f"{'Overall':<15}"
    r.append(header)
    r.append("-" * 80)

    for model in model_names:
        row = f"{model:<20}"
        all_f1 = []
        for split in splits:
            f1s = [t["test_metrics"]["f1"] for t in all_results[model][split]]
            f1_mean = np.mean(f1s)
            all_f1.append(f1_mean)
            row += f"{f1_mean:<15.4f}"
        row += f"{np.mean(all_f1):.4f}"
        r.append(row)

    # Accuracy table
    r.append("")
    r.append("-" * 80)
    r.append("AVERAGE TEST ACCURACY BY MODEL AND SPLIT (averaged over 3 trials)")
    r.append("-" * 80)

    header = f"{'Model':<20}"
    for split in splits:
        header += f"{split:<15}"
    header += f"{'Overall':<15}"
    r.append(header)
    r.append("-" * 80)

    for model in model_names:
        row = f"{model:<20}"
        all_acc = []
        for split in splits:
            accs = [t["test_metrics"]["accuracy"] for t in all_results[model][split]]
            acc_mean = np.mean(accs)
            all_acc.append(acc_mean)
            row += f"{acc_mean:<15.4f}"
        row += f"{np.mean(all_acc):.4f}"
        r.append(row)

    # ROC-AUC table
    r.append("")
    r.append("-" * 80)
    r.append("AVERAGE TEST ROC-AUC BY MODEL AND SPLIT (averaged over 3 trials)")
    r.append("-" * 80)

    header = f"{'Model':<20}"
    for split in splits:
        header += f"{split:<15}"
    header += f"{'Overall':<15}"
    r.append(header)
    r.append("-" * 80)

    for model in model_names:
        row = f"{model:<20}"
        all_auc = []
        for split in splits:
            aucs = [t["test_metrics"]["roc_auc"] for t in all_results[model][split]
                    if t["test_metrics"]["roc_auc"] is not None]
            if aucs:
                auc_mean = np.mean(aucs)
                all_auc.append(auc_mean)
                row += f"{auc_mean:<15.4f}"
            else:
                row += f"{'N/A':<15}"
        if all_auc:
            row += f"{np.mean(all_auc):.4f}"
        else:
            row += "N/A"
        r.append(row)

    # Precision/Recall table
    r.append("")
    r.append("-" * 80)
    r.append("AVERAGE TEST PRECISION / RECALL BY MODEL AND SPLIT")
    r.append("-" * 80)

    header = f"{'Model':<20}"
    for split in splits:
        header += f"{'P(' + split + ')':<10} {'R(' + split + ')':<10}"
    r.append(header)
    r.append("-" * 80)

    for model in model_names:
        row = f"{model:<20}"
        for split in splits:
            precs = [t["test_metrics"]["precision"] for t in all_results[model][split]]
            recs = [t["test_metrics"]["recall"] for t in all_results[model][split]]
            row += f"{np.mean(precs):<10.4f} {np.mean(recs):<10.4f}"
        r.append(row)

    # Overfitting Analysis
    r.append("")
    r.append("-" * 80)
    r.append("OVERFITTING ANALYSIS: TRAIN F1 vs TEST F1 (averaged over 3 trials)")
    r.append("-" * 80)

    header = f"{'Model':<20}"
    for split in splits:
        header += f"{'Train(' + split + ')':<12} {'Test(' + split + ')':<12} {'Gap':<10}"
    r.append(header)
    r.append("-" * 80)

    for model in model_names:
        row = f"{model:<20}"
        for split in splits:
            train_f1s = [t["train_metrics"]["f1"] for t in all_results[model][split]
                         if t.get("train_metrics")]
            test_f1s = [t["test_metrics"]["f1"] for t in all_results[model][split]]
            if train_f1s:
                tr = np.mean(train_f1s)
                te = np.mean(test_f1s)
                gap = tr - te
                row += f"{tr:<12.4f} {te:<12.4f} {gap:+.4f}   "
            else:
                te = np.mean(test_f1s)
                row += f"{'N/A':<12} {te:<12.4f} {'N/A':<10}"
        r.append(row)

    # Interpretation
    r.append("")
    r.append("-" * 80)
    r.append("INTERPRETATION")
    r.append("-" * 80)

    any_good_f1 = False
    for model in model_names:
        for split in splits:
            for trial in all_results[model][split]:
                if trial["test_metrics"]["f1"] > 0.5:
                    any_good_f1 = True

    if not any_good_f1:
        r.append("All models produced F1 scores at or below 0.50, indicating")
        r.append("limited discriminative ability between Helped and Not Helped.")
        r.append("This is expected with the current sample size.")
        r.append("")
        r.append("RECOMMENDATION: Collect data from at least 30-50 participants")
        r.append("(60-100 modeling rows) before drawing conclusions.")
        r.append("The pipeline is fully functional and will scale with more data.")
    else:
        r.append("At least one model achieved F1 > 0.50, indicating some")
        r.append("discriminative ability between Helped and Not Helped classes.")

    # Best model by AVERAGED performance
    r.append("")
    r.append("-" * 80)
    r.append("BEST MODEL (by averaged F1 across ALL trials, not cherry-picked)")
    r.append("-" * 80)

    best_avg_f1 = -1
    best_avg_model = None
    for model in model_names:
        all_f1 = []
        for split in splits:
            for trial in all_results[model][split]:
                all_f1.append(trial["test_metrics"]["f1"])
        avg = np.mean(all_f1)
        if avg > best_avg_f1:
            best_avg_f1 = avg
            best_avg_model = model

    if best_avg_model:
        all_acc = []
        all_prec = []
        all_rec = []
        all_f1 = []
        all_auc = []
        all_train_f1 = []
        for split in splits:
            for trial in all_results[best_avg_model][split]:
                all_acc.append(trial["test_metrics"]["accuracy"])
                all_prec.append(trial["test_metrics"]["precision"])
                all_rec.append(trial["test_metrics"]["recall"])
                all_f1.append(trial["test_metrics"]["f1"])
                if trial["test_metrics"]["roc_auc"] is not None:
                    all_auc.append(trial["test_metrics"]["roc_auc"])
                if trial.get("train_metrics"):
                    all_train_f1.append(trial["train_metrics"]["f1"])

        r.append(f"Winner:     {best_avg_model}")
        r.append(f"")
        r.append(f"Averaged Test Metrics (mean +/- std over {len(all_f1)} trials):")
        r.append(f"  Accuracy:  {np.mean(all_acc):.4f} +/- {np.std(all_acc):.4f}")
        r.append(f"  Precision: {np.mean(all_prec):.4f} +/- {np.std(all_prec):.4f}")
        r.append(f"  Recall:    {np.mean(all_rec):.4f} +/- {np.std(all_rec):.4f}")
        r.append(f"  F1:        {np.mean(all_f1):.4f} +/- {np.std(all_f1):.4f}")
        if all_auc:
            r.append(f"  ROC-AUC:   {np.mean(all_auc):.4f} +/- {np.std(all_auc):.4f}")
        if all_train_f1:
            gap = np.mean(all_train_f1) - np.mean(all_f1)
            r.append(f"")
            r.append(f"Averaged Train F1:  {np.mean(all_train_f1):.4f}")
            r.append(f"Train-Test F1 Gap:  {gap:+.4f}")

    # Best single trial (for reference)
    r.append("")
    r.append("-" * 80)
    r.append("BEST SINGLE TRIAL (for reference only - do NOT report this)")
    r.append("-" * 80)

    best_model = None
    best_f1 = -1
    best_split = None
    best_trial_info = None

    for model in model_names:
        for split in splits:
            for trial in all_results[model][split]:
                if trial["test_metrics"]["f1"] > best_f1:
                    best_f1 = trial["test_metrics"]["f1"]
                    best_model = model
                    best_split = split
                    best_trial_info = trial

    if best_model:
        r.append(f"Model:     {best_model}")
        r.append(f"Split:     {best_split}")
        r.append(f"Test F1:   {best_f1:.4f}")
        r.append(f"NOTE: This is cherry-picked. Use averaged metrics above for reporting.")

    report_path = os.path.join(output_dir, "cls_model_comparison_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(r))

    print(f"[plot] Saved classification model comparison plots and report")
