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
    """Generate text report for a classification model."""
    r = []
    r.append("=" * 70)
    r.append(f"RESCUE CLASSIFICATION - {model_name.upper()} RESULTS")
    r.append("=" * 70)

    split_names = []
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

    # Performance by Split
    r.append("\n" + "-" * 70)
    r.append("PERFORMANCE BY SPLIT (averaged over 3 trials)")
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
        r.append(f"Model Quality: {quality} (F1 = {f1:.4f})")

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

    # Calculate from first trial
    first_model = model_names[0]
    first_split = splits[0]
    first_trial = all_results[first_model][first_split][0]
    y_test_sample = np.array(first_trial["y_test"])
    majority_class = 1 if np.mean(y_test_sample) >= 0.5 else 0
    majority_acc = max(np.mean(y_test_sample), 1 - np.mean(y_test_sample))
    r.append(f"A model that always predicts class {majority_class} achieves accuracy = {majority_acc:.2f}")
    r.append("Any useful model must achieve F1 and accuracy above this baseline.")

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
        r.append("This is expected with the current sample size (8 modeling rows).")
        r.append("")
        r.append("RECOMMENDATION: Collect data from at least 30-50 participants")
        r.append("(60-100 modeling rows) before drawing conclusions.")
        r.append("The pipeline is fully functional and will scale with more data.")
    else:
        r.append("At least one model achieved F1 > 0.50, indicating some")
        r.append("discriminative ability between Helped and Not Helped classes.")

    # Best single trial
    r.append("")
    r.append("-" * 80)
    r.append("BEST SINGLE TRIAL (for reference only)")
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
        r.append(f"Accuracy:  {best_trial_info['test_metrics']['accuracy']:.4f}")
        r.append(f"Precision: {best_trial_info['test_metrics']['precision']:.4f}")
        r.append(f"Recall:    {best_trial_info['test_metrics']['recall']:.4f}")
        r.append(f"F1:        {best_f1:.4f}")
        r.append(f"ROC-AUC:   {_safe_metric(best_trial_info['test_metrics']['roc_auc'])}")

    report_path = os.path.join(output_dir, "cls_model_comparison_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(r))

    print(f"[plot] Saved classification model comparison plots and report")
