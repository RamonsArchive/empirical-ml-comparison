import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error


def eda_rescue(data):
    """
    Exploratory Data Analysis for the Rescue experiment.

    Task: REGRESSION - Predict PeopleRescued from behavioral/experimental features.
    Within-subjects design: 3 trials per participant, counterbalanced.

    Generates:
    - Text report with naive baselines
    - Distribution plots for DVs
    - Correlation heatmap
    - Condition comparison plots
    """
    output_dir = "plots/rescue_plots/eda"
    os.makedirs(output_dir, exist_ok=True)

    df = data.copy()
    target_col = "PeopleRescued"

    print(f"\n[EDA] Generating Rescue experiment EDA...")
    print(f"[EDA] Output directory: {output_dir}")

    # ==========================================
    # 1. TEXT REPORT
    # ==========================================
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("RESCUE EXPERIMENT - EXPLORATORY DATA ANALYSIS")
    report_lines.append("=" * 70)
    report_lines.append(f"\nTask: REGRESSION (Predict PeopleRescued from behavioral features)")
    report_lines.append(f"Design: Within-subjects, 3 trials per participant")
    report_lines.append(f"Hypothesis: Diffusion of responsibility in Multiple vs Single condition")

    # Dataset Overview
    report_lines.append("\n" + "-" * 50)
    report_lines.append("DATASET OVERVIEW")
    report_lines.append("-" * 50)
    report_lines.append(f"Total rows: {len(df)}")
    report_lines.append(f"Participants: {df['ParticipantID'].nunique()}")
    report_lines.append(f"Trials per participant: {len(df) // max(df['ParticipantID'].nunique(), 1)}")
    report_lines.append(f"Conditions: 0=Empty, 1=Single, 2=Multiple")
    report_lines.append(f"Groups: 0 (odd ID), 1 (even ID)")
    report_lines.append(f"Columns: {list(df.columns)}")

    # Demographics
    report_lines.append("\n" + "-" * 50)
    report_lines.append("DEMOGRAPHICS")
    report_lines.append("-" * 50)
    report_lines.append(f"Age: mean={df['Age'].mean():.1f}, range=[{df['Age'].min()}, {df['Age'].max()}]")
    gender_counts = df.drop_duplicates("ParticipantID")["Gender"].value_counts()
    for g, count in gender_counts.items():
        report_lines.append(f"  {g}: {count} participants")

    # Target Statistics by Condition
    report_lines.append("\n" + "-" * 50)
    report_lines.append("TARGET VARIABLE BY CONDITION")
    report_lines.append("-" * 50)
    for cond in sorted(df["Condition"].unique()):
        subset = df[df["Condition"] == cond]
        cond_label = {0: "Empty", 1: "Single", 2: "Multiple"}.get(cond, str(cond))
        report_lines.append(f"\nCondition {cond} ({cond_label}):")
        report_lines.append(f"  PeopleRescued: mean={subset[target_col].mean():.3f}, "
                            f"std={subset[target_col].std():.3f}, "
                            f"range=[{subset[target_col].min()}, {subset[target_col].max()}]")
        report_lines.append(f"  TimeNearVictim: mean={subset['TimeNearVictim'].mean():.3f}")
        report_lines.append(f"  TimeElapsed: mean={subset['TimeElapsed'].mean():.3f}")

    # Naive Baseline
    report_lines.append("\n" + "-" * 50)
    report_lines.append("NAIVE BASELINE PERFORMANCE")
    report_lines.append("-" * 50)

    # Only conditions with victims for baseline
    victim_df = df[df["Condition"] > 0]
    y = victim_df[target_col]
    y_pred_mean = np.full_like(y, y.mean(), dtype=float)
    rmse_mean = np.sqrt(mean_squared_error(y, y_pred_mean))
    mae_mean = mean_absolute_error(y, y_pred_mean)

    report_lines.append(f"\nMean Predictor (Conditions 1 & 2 only):")
    report_lines.append(f"  Mean PeopleRescued: {y.mean():.4f}")
    report_lines.append(f"  RMSE: {rmse_mean:.4f}")
    report_lines.append(f"  MAE:  {mae_mean:.4f}")
    report_lines.append(f"  R2:   0.0000 (by definition)")
    report_lines.append(f"\nA good model should achieve RMSE < {rmse_mean:.2f}")

    # Feature Correlations (victim conditions only)
    report_lines.append("\n" + "-" * 50)
    report_lines.append("FEATURE CORRELATIONS WITH PeopleRescued (Conditions 1 & 2)")
    report_lines.append("-" * 50)

    numeric_cols = victim_df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        corr_series = victim_df[numeric_cols].corr()[target_col].drop(target_col).sort_values(key=abs, ascending=False)
        for feat, corr in corr_series.items():
            direction = "+" if corr > 0 else "-"
            report_lines.append(f"  {feat}: {direction}{abs(corr):.4f}")

    # Save report
    report_path = os.path.join(output_dir, "eda_rescue_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"[EDA] Saved report to {report_path}")

    # ==========================================
    # 2. PLOTS
    # ==========================================

    # --- PeopleRescued by Condition ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cond_labels = {0: "Empty", 1: "Single", 2: "Multiple"}
    df["Condition_Label"] = df["Condition"].map(cond_labels)

    sns.boxplot(x="Condition_Label", y="PeopleRescued", data=df,
                order=["Empty", "Single", "Multiple"], ax=axes[0])
    axes[0].set_title("PeopleRescued by Condition")
    axes[0].set_xlabel("Condition")

    sns.boxplot(x="Condition_Label", y="TimeNearVictim", data=df,
                order=["Empty", "Single", "Multiple"], ax=axes[1])
    axes[1].set_title("TimeNearVictim by Condition")
    axes[1].set_xlabel("Condition")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "condition_comparison.png"), dpi=150)
    plt.close()
    df.drop(columns=["Condition_Label"], inplace=True)
    print(f"[EDA] Saved condition_comparison.png")

    # --- Correlation Heatmap (numeric features, victim conditions) ---
    numeric_victim = victim_df.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_victim.corr(), annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5)
    plt.title("Correlation Heatmap (Conditions 1 & 2)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=150)
    plt.close()
    print(f"[EDA] Saved correlation_heatmap.png")

    # --- TimeElapsed vs PeopleRescued scatter ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, cond in enumerate([1, 2]):
        subset = df[df["Condition"] == cond]
        cond_label = cond_labels[cond]
        axes[i].scatter(subset["TimeElapsed"], subset["PeopleRescued"],
                        alpha=0.7, s=60, edgecolors="black", linewidth=0.5)
        axes[i].set_xlabel("TimeElapsed (s)")
        axes[i].set_ylabel("PeopleRescued")
        axes[i].set_title(f"Condition {cond} ({cond_label})")
        axes[i].grid(True, alpha=0.3)

    plt.suptitle("TimeElapsed vs PeopleRescued")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "time_vs_rescued.png"), dpi=150)
    plt.close()
    print(f"[EDA] Saved time_vs_rescued.png")

    # --- Gender comparison ---
    victim_only = df[df["Condition"] > 0].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x="Gender", y="PeopleRescued", hue="Condition",
                data=victim_only, ax=ax)
    ax.set_title("PeopleRescued by Gender and Condition")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gender_condition.png"), dpi=150)
    plt.close()
    print(f"[EDA] Saved gender_condition.png")

    print(f"\n[EDA] Complete! All outputs saved to {output_dir}/")

    return df
