"""
Plotting functions for Rescue experiment regression results.
Generates summary reports, scatter plots, and comparison charts.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def _generate_regression_report(results, model_name, output_dir):
    """Generate text report for a regression model with train+test metrics."""
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append(f"RESCUE EXPERIMENT - {model_name.upper()} RESULTS")
    report_lines.append("=" * 70)

    split_names = []
    mean_train_r2 = []
    mean_train_rmse = []
    mean_test_r2 = []
    mean_test_rmse = []
    mean_test_mae = []
    mean_cv_train = []
    mean_cv_val = []

    best_rmse = float("inf")
    best_split = None
    best_trial = None

    for split_name, trials in results.items():
        split_names.append(split_name)

        cv_train_scores = [t["cv_train_score"] for t in trials]
        cv_val_scores = [t["cv_val_score"] for t in trials]
        test_r2s = [t["test_metrics"]["r2"] for t in trials]
        test_rmses = [t["test_metrics"]["rmse"] for t in trials]
        test_maes = [t["test_metrics"]["mae"] for t in trials]

        # Train set metrics
        train_r2s = [t["train_metrics"]["r2"] for t in trials if t.get("train_metrics")]
        train_rmses = [t["train_metrics"]["rmse"] for t in trials if t.get("train_metrics")]
        mean_train_r2.append(np.mean(train_r2s) if train_r2s else None)
        mean_train_rmse.append(np.mean(train_rmses) if train_rmses else None)

        mean_cv_train.append(np.mean(cv_train_scores))
        mean_cv_val.append(np.mean(cv_val_scores))
        mean_test_r2.append(np.mean(test_r2s))
        mean_test_rmse.append(np.mean(test_rmses))
        mean_test_mae.append(np.mean(test_maes))

        for trial in trials:
            if trial["test_metrics"]["rmse"] < best_rmse:
                best_rmse = trial["test_metrics"]["rmse"]
                best_split = split_name
                best_trial = trial

    # Train Performance
    report_lines.append("\n" + "-" * 70)
    report_lines.append("TRAIN PERFORMANCE BY SPLIT (averaged over 3 trials)")
    report_lines.append("-" * 70)
    report_lines.append(f"\n{'Split':<10} {'Train R2':<12} {'Train RMSE':<12} {'CV Train':<12} {'CV Val':<12}")
    report_lines.append("-" * 60)

    for i, split_name in enumerate(split_names):
        tr2 = f"{mean_train_r2[i]:.4f}" if mean_train_r2[i] is not None else "N/A"
        trmse = f"{mean_train_rmse[i]:.4f}" if mean_train_rmse[i] is not None else "N/A"
        report_lines.append(
            f"{split_name:<10} {tr2:<12} {trmse:<12} "
            f"{mean_cv_train[i]:<12.4f} {mean_cv_val[i]:<12.4f}"
        )

    # Test Performance
    report_lines.append("\n" + "-" * 70)
    report_lines.append("TEST PERFORMANCE BY SPLIT (averaged over 3 trials)")
    report_lines.append("-" * 70)
    report_lines.append(
        f"\n{'Split':<10} {'Test R2':<12} {'Test RMSE':<12} {'Test MAE':<12}"
    )
    report_lines.append("-" * 50)

    for i, split_name in enumerate(split_names):
        report_lines.append(
            f"{split_name:<10} "
            f"{mean_test_r2[i]:<12.4f} "
            f"{mean_test_rmse[i]:<12.4f} "
            f"{mean_test_mae[i]:<12.4f}"
        )

    # Overfitting Analysis
    report_lines.append("\n" + "-" * 70)
    report_lines.append("OVERFITTING ANALYSIS (Train - Test gap, averaged over 3 trials)")
    report_lines.append("-" * 70)
    report_lines.append(f"\n{'Split':<10} {'Train RMSE':<12} {'Test RMSE':<12} {'Gap':<12} {'Status':<15}")
    report_lines.append("-" * 60)

    for i, split_name in enumerate(split_names):
        tr = mean_train_rmse[i]
        te = mean_test_rmse[i]
        if tr is not None:
            gap = te - tr  # positive = worse on test
            if gap > 1.0:
                status = "SEVERE OVERFIT"
            elif gap > 0.5:
                status = "MODERATE OVERFIT"
            elif gap > 0.2:
                status = "MILD OVERFIT"
            else:
                status = "OK"
            report_lines.append(
                f"{split_name:<10} {tr:<12.4f} {te:<12.4f} {gap:+<12.4f} {status:<15}"
            )
        else:
            report_lines.append(
                f"{split_name:<10} {'N/A':<12} {te:<12.4f} {'N/A':<12} {'N/A':<15}"
            )

    # Best Model Details
    if best_trial:
        report_lines.append("\n" + "-" * 70)
        report_lines.append("BEST MODEL (lowest RMSE)")
        report_lines.append("-" * 70)
        report_lines.append(f"Split: {best_split}")
        report_lines.append(f"Trial: {best_trial['trial'] + 1}")
        report_lines.append(f"\nBest Parameters:")
        for param, val in best_trial["best_params"].items():
            report_lines.append(f"  {param}: {val}")

        report_lines.append(f"\nTrain Metrics:")
        tm = best_trial.get("train_metrics", {})
        if tm:
            report_lines.append(f"  R2:   {tm.get('r2', 'N/A'):.4f}")
            report_lines.append(f"  RMSE: {tm.get('rmse', 'N/A'):.4f}")
            report_lines.append(f"  MAE:  {tm.get('mae', 'N/A'):.4f}")
        else:
            report_lines.append("  (not available)")

        report_lines.append(f"\nTest Metrics:")
        report_lines.append(f"  R2:   {best_trial['test_metrics']['r2']:.4f}")
        report_lines.append(f"  RMSE: {best_trial['test_metrics']['rmse']:.4f}")
        report_lines.append(f"  MAE:  {best_trial['test_metrics']['mae']:.4f}")

    # Interpretation
    report_lines.append("\n" + "-" * 70)
    report_lines.append("INTERPRETATION")
    report_lines.append("-" * 70)
    if best_trial:
        r2 = best_trial["test_metrics"]["r2"]
        if r2 > 0.9:
            quality = "EXCELLENT"
        elif r2 > 0.7:
            quality = "GOOD"
        elif r2 > 0.5:
            quality = "MODERATE"
        else:
            quality = "POOR"

        report_lines.append(f"Model Quality: {quality} (Test R2 = {r2:.4f})")
        report_lines.append(f"The model explains {r2 * 100:.1f}% of variance in PeopleRescued.")
        report_lines.append(f"Average prediction error: +/- {best_trial['test_metrics']['mae']:.3f} people")

        # Overfitting warning
        tm = best_trial.get("train_metrics", {})
        if tm and "r2" in tm:
            train_r2 = tm["r2"]
            if train_r2 - r2 > 0.3:
                report_lines.append(
                    f"WARNING: Train R2 ({train_r2:.4f}) >> Test R2 ({r2:.4f}). "
                    f"Model is overfitting."
                )

    # Save report
    report_path = os.path.join(output_dir, f"rescue_{model_name}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    return best_trial


def _plot_predictions_scatter(best_trial, model_name, output_dir):
    """Plot predicted vs actual scatter plot."""
    y_test = np.array(best_trial["y_test"])
    y_pred = np.array(best_trial["y_pred"])

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_test, y_pred, alpha=0.6, s=60, c="steelblue", edgecolors="black", linewidth=0.5)

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")

    ax.set_xlabel("Actual PeopleRescued", fontsize=12)
    ax.set_ylabel("Predicted PeopleRescued", fontsize=12)
    ax.set_title(
        f"{model_name}: Predicted vs Actual\nR² = {best_trial['test_metrics']['r2']:.4f}",
        fontsize=14,
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"rescue_{model_name}_scatter.png"), dpi=150)
    plt.close()


def _plot_train_test_rmse(results, model_name, output_dir):
    """Plot Train vs Test RMSE side-by-side by split for a single model."""
    split_names = list(results.keys())
    train_rmses = []
    test_rmses = []

    for split_name in split_names:
        trials = results[split_name]
        tr = [t["train_metrics"]["rmse"] for t in trials if t.get("train_metrics")]
        te = [t["test_metrics"]["rmse"] for t in trials]
        train_rmses.append(np.mean(tr) if tr else 0)
        test_rmses.append(np.mean(te))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(split_names))
    width = 0.35

    bars_train = ax.bar(x - width / 2, train_rmses, width, color="#4C72B0",
                        alpha=0.85, edgecolor="black", linewidth=0.8, label="Train RMSE")
    bars_test = ax.bar(x + width / 2, test_rmses, width, color="#DD8452",
                       alpha=0.85, edgecolor="black", linewidth=0.8, label="Test RMSE")

    for bar, val in zip(bars_train, train_rmses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar, val in zip(bars_test, test_rmses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("Train/Test Split", fontsize=12)
    ax.set_ylabel("RMSE (lower is better)", fontsize=12)
    ax.set_title(f"{model_name.replace('_', ' ').title()}: Train vs Test RMSE\n(averaged over 3 trials per split)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(split_names)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"rescue_{model_name}_train_test_rmse.png"), dpi=150)
    plt.close()


def _plot_r2_by_split(results, model_name, output_dir):
    """Plot R2 scores across splits."""
    split_names = []
    mean_test_r2 = []
    std_test_r2 = []

    for split_name, trials in results.items():
        split_names.append(split_name)
        test_r2s = [t["test_metrics"]["r2"] for t in trials]
        mean_test_r2.append(np.mean(test_r2s))
        std_test_r2.append(np.std(test_r2s))

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(split_names))
    bars = ax.bar(x, mean_test_r2, yerr=std_test_r2, capsize=5, color="steelblue", alpha=0.8)

    ax.set_xlabel("Train/Test Split", fontsize=12)
    ax.set_ylabel("R² Score", fontsize=12)
    ax.set_title(f"{model_name}: R² by Split (± std over 3 trials)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(split_names)
    ax.grid(True, axis="y", alpha=0.3)

    for bar, mean in zip(bars, mean_test_r2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"rescue_{model_name}_r2_by_split.png"), dpi=150)
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
    plt.savefig(os.path.join(output_dir, f"rescue_{model_name}_feature_importance.png"), dpi=150)
    plt.close()


# ==========================================
# Public API
# ==========================================

def plot_rescue_boosting_summary(results, output_dir):
    """Generate all plots and report for Boosting (XGBoost) results."""
    os.makedirs(output_dir, exist_ok=True)
    model_name = "boosting"

    print(f"[plot] Generating {model_name} (XGBoost) plots to {output_dir}")

    best_trial = _generate_regression_report(results, model_name, output_dir)

    if best_trial:
        _plot_predictions_scatter(best_trial, model_name, output_dir)
        if best_trial.get("feature_importances") and best_trial.get("feature_names"):
            _plot_feature_importances(best_trial, model_name, output_dir)

    _plot_train_test_rmse(results, model_name, output_dir)
    _plot_r2_by_split(results, model_name, output_dir)
    print(f"[plot] Saved {model_name} plots and report")


def plot_rescue_random_forest_summary(results, output_dir):
    """Generate all plots and report for Random Forest results."""
    os.makedirs(output_dir, exist_ok=True)
    model_name = "random_forest"

    print(f"[plot] Generating {model_name} plots to {output_dir}")

    best_trial = _generate_regression_report(results, model_name, output_dir)

    if best_trial:
        _plot_predictions_scatter(best_trial, model_name, output_dir)
        if best_trial.get("feature_importances") and best_trial.get("feature_names"):
            _plot_feature_importances(best_trial, model_name, output_dir)

    _plot_train_test_rmse(results, model_name, output_dir)
    _plot_r2_by_split(results, model_name, output_dir)
    print(f"[plot] Saved {model_name} plots and report")


def plot_rescue_neural_network_summary(results, output_dir):
    """Generate all plots and report for Neural Network results."""
    os.makedirs(output_dir, exist_ok=True)
    model_name = "neural_network"

    print(f"[plot] Generating {model_name} plots to {output_dir}")

    best_trial = _generate_regression_report(results, model_name, output_dir)

    if best_trial:
        _plot_predictions_scatter(best_trial, model_name, output_dir)

    _plot_train_test_rmse(results, model_name, output_dir)
    _plot_r2_by_split(results, model_name, output_dir)
    print(f"[plot] Saved {model_name} plots and report")


def plot_rescue_model_comparison(all_results, output_dir):
    """Generate comparison plots across all models."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"[plot] Generating model comparison plots to {output_dir}")

    model_names = list(all_results.keys())
    splits = list(all_results[model_names[0]].keys())

    colors = ["steelblue", "forestgreen", "coral"]
    width = 0.25

    # ==========================================
    # R2 Comparison
    # ==========================================
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(splits))

    for i, model in enumerate(model_names):
        r2_means = []
        r2_stds = []
        for split in splits:
            test_r2s = [t["test_metrics"]["r2"] for t in all_results[model][split]]
            r2_means.append(np.mean(test_r2s))
            r2_stds.append(np.std(test_r2s))

        offset = (i - 1) * width
        ax.bar(
            x + offset, r2_means, width, yerr=r2_stds, capsize=3,
            label=model.replace("_", " ").title(),
            color=colors[i % len(colors)], alpha=0.8,
        )

    ax.set_xlabel("Train/Test Split", fontsize=12)
    ax.set_ylabel("R² Score", fontsize=12)
    ax.set_title("Model Comparison: R² by Split (± std over 3 trials)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison_r2.png"), dpi=150)
    plt.close()

    # ==========================================
    # RMSE Comparison
    # ==========================================
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model in enumerate(model_names):
        rmse_means = []
        rmse_stds = []
        for split in splits:
            test_rmses = [t["test_metrics"]["rmse"] for t in all_results[model][split]]
            rmse_means.append(np.mean(test_rmses))
            rmse_stds.append(np.std(test_rmses))

        offset = (i - 1) * width
        ax.bar(
            x + offset, rmse_means, width, yerr=rmse_stds, capsize=3,
            label=model.replace("_", " ").title(),
            color=colors[i % len(colors)], alpha=0.8,
        )

    ax.set_xlabel("Train/Test Split", fontsize=12)
    ax.set_ylabel("RMSE (people)", fontsize=12)
    ax.set_title("Model Comparison: RMSE by Split (lower is better)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison_rmse.png"), dpi=150)
    plt.close()

    # ==========================================
    # Train vs Test RMSE Comparison (all models)
    # ==========================================
    fig, ax = plt.subplots(figsize=(12, 6))
    n_models = len(model_names)
    bar_width = 0.35
    x = np.arange(n_models)

    train_means = []
    test_means = []
    for model in model_names:
        all_train = []
        all_test = []
        for split in splits:
            for t in all_results[model][split]:
                if t.get("train_metrics"):
                    all_train.append(t["train_metrics"]["rmse"])
                all_test.append(t["test_metrics"]["rmse"])
        train_means.append(np.mean(all_train) if all_train else 0)
        test_means.append(np.mean(all_test))

    bars_tr = ax.bar(x - bar_width / 2, train_means, bar_width, color="#4C72B0",
                     alpha=0.85, edgecolor="black", linewidth=0.8, label="Train RMSE")
    bars_te = ax.bar(x + bar_width / 2, test_means, bar_width, color="#DD8452",
                     alpha=0.85, edgecolor="black", linewidth=0.8, label="Test RMSE")

    for bar, val in zip(bars_tr, train_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar, val in zip(bars_te, test_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("RMSE (lower is better)", fontsize=12)
    ax.set_title("Train vs Test RMSE: All Models\n(averaged over all trials & splits)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in model_names])
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison_train_test_rmse.png"), dpi=150)
    plt.close()

    # ==========================================
    # Summary Report
    # ==========================================
    r = []
    r.append("=" * 80)
    r.append("RESCUE EXPERIMENT - MODEL COMPARISON SUMMARY")
    r.append("=" * 80)

    # Count total observations
    first_model = model_names[0]
    first_split = splits[0]
    n_test_example = len(all_results[first_model][first_split][0]["y_test"])
    n_obs = sum(
        len(all_results[first_model][s][0]["y_test"])
        + len(all_results[first_model][s][0]["y_pred"])
        for s in splits[:1]
    ) // 2  # rough count from one split

    r.append("")
    r.append("EXPERIMENT OVERVIEW")
    r.append("-" * 80)
    r.append("Task:           Regression (predict PeopleRescued from behavioral features)")
    r.append("Design:         Within-subjects, counterbalanced across 3 conditions")
    r.append("Modeled Data:   Conditions 1 (Single) & 2 (Multiple) only")
    r.append("                Condition 0 (Empty) excluded - no victims by design")
    r.append("Predictors:     Age, Gender, Group, Condition, TimeElapsed, DrownCount,")
    r.append("                TimeNearVictim, baseline_TimeElapsed, baseline_DrownCount")
    r.append("Target:         PeopleRescued (integer 0-4)")
    r.append(f"Splits:         {', '.join(splits)}")
    r.append("Trials:         3 per split with different random seeds")

    # Naive baseline - compute actual values
    r.append("")
    r.append("-" * 80)
    r.append("NAIVE BASELINE (Mean Predictor)")
    r.append("-" * 80)

    for split in splits:
        y_tests = []
        for trial in all_results[first_model][split]:
            y_tests.extend(trial["y_test"])
        y_arr = np.array(y_tests[:len(all_results[first_model][split][0]["y_test"])])
        mean_pred = np.mean(y_arr)
        baseline_rmse = np.sqrt(np.mean((y_arr - mean_pred) ** 2))
        r.append(f"  {split}: mean = {mean_pred:.2f}, baseline RMSE = {baseline_rmse:.4f}")
    r.append("")
    r.append("A model that always predicts the mean PeopleRescued.")
    r.append("Any useful model must achieve RMSE below baseline and R2 > 0.")

    # RMSE table
    r.append("")
    r.append("-" * 80)
    r.append("AVERAGE TEST RMSE BY MODEL AND SPLIT (averaged over 3 trials)")
    r.append("-" * 80)

    header = f"{'Model':<20}"
    for split in splits:
        header += f"{split:<15}"
    header += f"{'Overall':<15}"
    r.append(header)
    r.append("-" * 80)

    for model in model_names:
        row = f"{model:<20}"
        all_rmse = []
        for split in splits:
            test_rmses = [t["test_metrics"]["rmse"] for t in all_results[model][split]]
            rmse = np.mean(test_rmses)
            all_rmse.append(rmse)
            row += f"{rmse:<15.4f}"
        row += f"{np.mean(all_rmse):.4f}"
        r.append(row)

    # R2 table
    r.append("")
    r.append("-" * 80)
    r.append("AVERAGE TEST R2 BY MODEL AND SPLIT (averaged over 3 trials)")
    r.append("-" * 80)

    header = f"{'Model':<20}"
    for split in splits:
        header += f"{split:<15}"
    header += f"{'Overall':<15}"
    r.append(header)
    r.append("-" * 80)

    for model in model_names:
        row = f"{model:<20}"
        all_r2 = []
        for split in splits:
            test_r2s = [t["test_metrics"]["r2"] for t in all_results[model][split]]
            r2 = np.mean(test_r2s)
            all_r2.append(r2)
            row += f"{r2:<15.4f}"
        row += f"{np.mean(all_r2):.4f}"
        r.append(row)

    # Overfitting Analysis
    r.append("")
    r.append("-" * 80)
    r.append("OVERFITTING ANALYSIS: TRAIN RMSE vs TEST RMSE (averaged over 3 trials)")
    r.append("-" * 80)

    header = f"{'Model':<20}"
    for split in splits:
        header += f"{'Train(' + split + ')':<12} {'Test(' + split + ')':<12} {'Gap':<10}"
    r.append(header)
    r.append("-" * 80)

    for model in model_names:
        row = f"{model:<20}"
        for split in splits:
            train_rmses = [t["train_metrics"]["rmse"] for t in all_results[model][split]
                           if t.get("train_metrics")]
            test_rmses = [t["test_metrics"]["rmse"] for t in all_results[model][split]]
            if train_rmses:
                tr = np.mean(train_rmses)
                te = np.mean(test_rmses)
                gap = te - tr
                row += f"{tr:<12.4f} {te:<12.4f} {gap:+.4f}   "
            else:
                te = np.mean(test_rmses)
                row += f"{'N/A':<12} {te:<12.4f} {'N/A':<10}"
        r.append(row)

    # Interpretation
    r.append("")
    r.append("-" * 80)
    r.append("INTERPRETATION")
    r.append("-" * 80)

    any_positive_r2 = False
    for model in model_names:
        for split in splits:
            for trial in all_results[model][split]:
                if trial["test_metrics"]["r2"] > 0.05:
                    any_positive_r2 = True

    if not any_positive_r2:
        r.append("All models produced negative or near-zero R2 values, indicating")
        r.append("performance worse than the naive mean predictor. This is expected")
        r.append("with the current sample size - ML models cannot learn generalizable")
        r.append("patterns when the number of features approaches or exceeds the")
        r.append("number of observations.")
        r.append("")
        r.append("RECOMMENDATION: Collect data from at least 30-50 participants")
        r.append("(60-100 modeling rows) before drawing conclusions from ML results.")
        r.append("The pipeline is fully functional and will scale with more data.")
    else:
        r.append("At least one model achieved positive R2, indicating predictive")
        r.append("value above the naive mean predictor.")

    # Best model by averaged RMSE
    r.append("")
    r.append("-" * 80)
    r.append("BEST MODEL (by averaged RMSE across ALL trials, not cherry-picked)")
    r.append("-" * 80)

    best_avg_rmse = float("inf")
    best_avg_model = None
    for model in model_names:
        all_rmse = []
        for split in splits:
            for trial in all_results[model][split]:
                all_rmse.append(trial["test_metrics"]["rmse"])
        avg = np.mean(all_rmse)
        if avg < best_avg_rmse:
            best_avg_rmse = avg
            best_avg_model = model

    if best_avg_model:
        all_r2 = []
        all_rmse = []
        all_mae = []
        all_train_rmse = []
        for split in splits:
            for trial in all_results[best_avg_model][split]:
                all_r2.append(trial["test_metrics"]["r2"])
                all_rmse.append(trial["test_metrics"]["rmse"])
                all_mae.append(trial["test_metrics"]["mae"])
                if trial.get("train_metrics"):
                    all_train_rmse.append(trial["train_metrics"]["rmse"])

        r.append(f"Winner:     {best_avg_model}")
        r.append(f"")
        r.append(f"Averaged Test Metrics (mean +/- std over {len(all_rmse)} trials):")
        r.append(f"  R2:   {np.mean(all_r2):.4f} +/- {np.std(all_r2):.4f}")
        r.append(f"  RMSE: {np.mean(all_rmse):.4f} +/- {np.std(all_rmse):.4f}")
        r.append(f"  MAE:  {np.mean(all_mae):.4f} +/- {np.std(all_mae):.4f}")
        if all_train_rmse:
            gap = np.mean(all_rmse) - np.mean(all_train_rmse)
            r.append(f"")
            r.append(f"Averaged Train RMSE: {np.mean(all_train_rmse):.4f}")
            r.append(f"Train-Test RMSE Gap: {gap:+.4f}")

    # Best single trial
    r.append("")
    r.append("-" * 80)
    r.append("BEST SINGLE TRIAL (for reference only - do NOT report this)")
    r.append("-" * 80)

    best_model = None
    best_rmse = float("inf")
    best_split = None
    best_trial_info = None

    for model in model_names:
        for split in splits:
            for trial in all_results[model][split]:
                if trial["test_metrics"]["rmse"] < best_rmse:
                    best_rmse = trial["test_metrics"]["rmse"]
                    best_model = model
                    best_split = split
                    best_trial_info = trial

    if best_model:
        r.append(f"Model:    {best_model}")
        r.append(f"Split:    {best_split}")
        r.append(f"Test RMSE: {best_rmse:.4f}")
        r.append(f"NOTE: This is cherry-picked. Use averaged metrics above for reporting.")

    report_path = os.path.join(output_dir, "model_comparison_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(r))

    print(f"[plot] Saved model comparison plots and report")
