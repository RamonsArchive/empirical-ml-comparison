import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def eda_rescue_classification(data):
    """
    Exploratory Data Analysis for the Rescue Classification experiment.

    Task: CLASSIFICATION - Predict Helped (PeopleRescued > 0) from behavioral features.
    Within-subjects design: 3 trials per participant, counterbalanced.

    Generates:
    - Text report with class distribution and naive baseline
    - Binary target distribution by condition
    - Correlation heatmap (victim conditions)
    - TimeNearVictim vs Helped scatter
    - Gender/Condition breakdown
    """
    output_dir = "plots/rescue_classification_plots/eda"
    os.makedirs(output_dir, exist_ok=True)

    df = data.copy()
    target_col = "PeopleRescued"

    print(f"\n[EDA] Generating Rescue Classification EDA...")
    print(f"[EDA] Output directory: {output_dir}")

    # ==========================================
    # 1. TEXT REPORT
    # ==========================================
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("RESCUE CLASSIFICATION - EXPLORATORY DATA ANALYSIS")
    report_lines.append("=" * 70)
    report_lines.append(f"\nTask: CLASSIFICATION (Predict Helped vs Not Helped from behavioral features)")
    report_lines.append(f"Design: Within-subjects, {len(df) // max(df['ParticipantID'].nunique(), 1)} trials per participant")
    report_lines.append(f"Hypothesis: Diffusion of responsibility reduces helping in Multiple vs Single condition")
    report_lines.append(f"Target:     Helped = (PeopleRescued > 0)")

    # Dataset Overview
    n_participants = df["ParticipantID"].nunique()
    report_lines.append("\n" + "-" * 50)
    report_lines.append("DATASET OVERVIEW")
    report_lines.append("-" * 50)
    report_lines.append(f"Total rows: {len(df)}")
    report_lines.append(f"Participants: {n_participants}")
    report_lines.append(f"Trials per participant: {len(df) // max(n_participants, 1)}")
    report_lines.append(f"Conditions: 0=Empty, 1=Single, 2=Multiple")
    report_lines.append(f"Groups: 0 (odd ID), 1 (even ID)")
    report_lines.append(f"Columns: {list(df.columns)}")

    # Demographics
    report_lines.append("\n" + "-" * 50)
    report_lines.append("DEMOGRAPHICS")
    report_lines.append("-" * 50)
    report_lines.append(f"Age: mean={df['Age'].mean():.1f}, range=[{int(df['Age'].min())}, {int(df['Age'].max())}]")
    gender_counts = df.drop_duplicates("ParticipantID")["Gender"].value_counts()
    for g, count in gender_counts.items():
        report_lines.append(f"  {g}: {count} participants")

    # Binary target construction
    victim_df = df[df["Condition"] > 0].copy()
    victim_df["Helped"] = (victim_df[target_col] > 0).astype(int)
    n_helped = victim_df["Helped"].sum()
    n_not_helped = (victim_df["Helped"] == 0).sum()
    n_modeling = len(victim_df)

    # Class Distribution
    report_lines.append("\n" + "-" * 50)
    report_lines.append("BINARY TARGET: Helped (PeopleRescued > 0)")
    report_lines.append("-" * 50)
    report_lines.append(f"Modeling rows (Conditions 1 & 2): {n_modeling}")
    report_lines.append(f"Helped     (1): {n_helped}  ({n_helped / n_modeling * 100:.1f}%)")
    report_lines.append(f"Not Helped (0): {n_not_helped}  ({n_not_helped / n_modeling * 100:.1f}%)")
    report_lines.append(f"Class imbalance ratio: {max(n_helped, n_not_helped) / max(min(n_helped, n_not_helped), 1):.2f}:1")

    # By Condition
    report_lines.append("\n" + "-" * 50)
    report_lines.append("HELPED RATE BY CONDITION")
    report_lines.append("-" * 50)
    for cond in sorted(victim_df["Condition"].unique()):
        subset = victim_df[victim_df["Condition"] == cond]
        cond_label = {1: "Single", 2: "Multiple"}.get(cond, str(cond))
        helped_rate = subset["Helped"].mean()
        report_lines.append(f"\nCondition {cond} ({cond_label}):")
        report_lines.append(f"  Helped: {subset['Helped'].sum()} / {len(subset)} ({helped_rate * 100:.1f}%)")
        report_lines.append(f"  TimeNearVictim: mean={subset['TimeNearVictim'].mean():.3f}")
        report_lines.append(f"  TimeElapsed: mean={subset['TimeElapsed'].mean():.3f}")

    # Naive Baseline
    report_lines.append("\n" + "-" * 50)
    report_lines.append("NAIVE BASELINE PERFORMANCE (Conditions 1 & 2)")
    report_lines.append("-" * 50)

    majority_class = int(victim_df["Helped"].mode()[0])
    majority_acc = victim_df["Helped"].value_counts(normalize=True).max()
    minority_rate = victim_df["Helped"].value_counts(normalize=True).min()

    report_lines.append(f"\nMajority Class Predictor (always predict {majority_class}):")
    report_lines.append(f"  Majority class: {majority_class} ({'Helped' if majority_class == 1 else 'Not Helped'})")
    report_lines.append(f"  Accuracy:  {majority_acc:.4f}")
    report_lines.append(f"  F1 Score:  {majority_acc if majority_class == 1 else 0.0:.4f}  (zero for negative-predicting majority)")
    report_lines.append(f"  Minority class rate: {minority_rate:.4f}")
    report_lines.append(f"\nA good model must achieve F1 and ROC-AUC above {majority_acc:.2f} baseline accuracy.")

    # Feature Correlations with binary target
    # PeopleRescued shown for reference but excluded from modeling (direct leakage)
    report_lines.append("\n" + "-" * 50)
    report_lines.append("FEATURE CORRELATIONS WITH Helped (Conditions 1 & 2)")
    report_lines.append("-" * 50)
    report_lines.append("  * PeopleRescued shown for reference only — excluded from model (leakage)")

    numeric_cols = [
        c for c in victim_df.select_dtypes(include=[np.number]).columns
        if c != "Helped"
    ]
    corr_series = victim_df[numeric_cols + ["Helped"]].corr()["Helped"].drop("Helped").sort_values(key=abs, ascending=False)
    for feat, corr in corr_series.items():
        direction = "+" if corr > 0 else "-"
        note = "  [EXCLUDED FROM MODEL]" if feat == "PeopleRescued" else ""
        report_lines.append(f"  {feat}: {direction}{abs(corr):.4f}{note}")

    # Save report
    report_path = os.path.join(output_dir, "eda_rescue_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"[EDA] Saved report to {report_path}")

    # ==========================================
    # 2. PLOTS
    # ==========================================

    cond_labels = {0: "Empty", 1: "Single", 2: "Multiple"}
    df["Condition_Label"] = df["Condition"].map(cond_labels)
    victim_df["Condition_Label"] = victim_df["Condition"].map(cond_labels)

    # --- Helped Rate by Condition ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    helped_by_cond = victim_df.groupby("Condition_Label")["Helped"].mean().reindex(["Single", "Multiple"])
    axes[0].bar(helped_by_cond.index, helped_by_cond.values, color=["steelblue", "coral"], alpha=0.8, edgecolor="black")
    for i, (label, val) in enumerate(helped_by_cond.items()):
        axes[0].text(i, val + 0.02, f"{val:.0%}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    axes[0].set_ylim(0, 1.2)
    axes[0].set_title("Rate of Helping by Condition")
    axes[0].set_xlabel("Condition")
    axes[0].set_ylabel("Proportion Helped")
    axes[0].grid(True, axis="y", alpha=0.3)

    sns.boxplot(x="Condition_Label", y="TimeNearVictim", data=victim_df,
                order=["Single", "Multiple"], ax=axes[1])
    axes[1].set_title("TimeNearVictim by Condition")
    axes[1].set_xlabel("Condition")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cls_condition_comparison.png"), dpi=150)
    plt.close()
    print(f"[EDA] Saved cls_condition_comparison.png")

    # --- Binary class distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Overall pie
    class_counts = victim_df["Helped"].value_counts()
    axes[0].pie(class_counts.values, labels=["Not Helped (0)", "Helped (1)"],
                autopct="%1.1f%%", colors=["coral", "steelblue"], startangle=90)
    axes[0].set_title(f"Class Distribution\n(Conditions 1 & 2, N={n_modeling})")

    # By condition stacked bar
    helped_counts = victim_df.groupby(["Condition_Label", "Helped"]).size().unstack(fill_value=0)
    helped_counts.plot(kind="bar", stacked=True, ax=axes[1], color=["coral", "steelblue"],
                       edgecolor="black", alpha=0.8)
    axes[1].set_title("Helped vs Not Helped by Condition")
    axes[1].set_xlabel("Condition")
    axes[1].set_ylabel("Count")
    axes[1].legend(["Not Helped (0)", "Helped (1)"])
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cls_class_distribution.png"), dpi=150)
    plt.close()
    print(f"[EDA] Saved cls_class_distribution.png")

    # --- Correlation Heatmap (numeric features, victim conditions) ---
    numeric_victim = victim_df.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_victim.corr(), annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5)
    plt.title("Correlation Heatmap (Conditions 1 & 2)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cls_correlation_heatmap.png"), dpi=150)
    plt.close()
    print(f"[EDA] Saved cls_correlation_heatmap.png")

    # --- TimeNearVictim vs Helped ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, cond in enumerate([1, 2]):
        subset = victim_df[victim_df["Condition"] == cond]
        cond_label = cond_labels[cond]
        colors_map = {0: "coral", 1: "steelblue"}
        for label, grp in subset.groupby("Helped"):
            axes[i].scatter(
                grp["TimeNearVictim"], grp["TimeElapsed"],
                label=f"{'Helped' if label == 1 else 'Not Helped'}",
                color=colors_map[label], alpha=0.8, s=80, edgecolors="black", linewidth=0.5
            )
        axes[i].set_xlabel("TimeNearVictim (s)")
        axes[i].set_ylabel("TimeElapsed (s)")
        axes[i].set_title(f"Condition {cond} ({cond_label})")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.suptitle("TimeNearVictim vs TimeElapsed (colored by Helped)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cls_time_vs_helped.png"), dpi=150)
    plt.close()
    print(f"[EDA] Saved cls_time_vs_helped.png")

    # --- Gender comparison ---
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x="Gender", hue="Helped", data=victim_df, ax=ax,
                  palette={0: "coral", 1: "steelblue"})
    ax.set_title("Helped vs Not Helped by Gender")
    ax.legend(title="Helped", labels=["No (0)", "Yes (1)"])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cls_gender_helped.png"), dpi=150)
    plt.close()
    print(f"[EDA] Saved cls_gender_helped.png")

    # Cleanup temp columns
    df.drop(columns=["Condition_Label"], inplace=True)

    print(f"\n[EDA] Complete! All outputs saved to {output_dir}/")

    return df
