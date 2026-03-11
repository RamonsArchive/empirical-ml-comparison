"""
Exploratory Data Analysis for the Piano Emotion Classification experiment.

Task: CLASSIFICATION — predict emotional (0) vs happy (1) from acoustic features.
Design: 18 songs x 3 clips = 54 samples; perfectly balanced (27 per class).

Generates:
  - Text report: class distribution, naive baseline, per-feature effect sizes
  - Class distribution bar chart
  - Top-20 feature violin plots (emotional vs happy)
  - Effect size bar chart (Cohen's d for top 20 features)
  - Correlation heatmap of top-20 features
  - Per-song clip summary scatter (onset_strength_mean vs note_density)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# Top 20 most discriminative features (ranked by effect size from feature report)
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


def _cohens_d(group0, group1):
    """Cohen's d effect size (pooled std)."""
    n0, n1 = len(group0), len(group1)
    var0, var1 = np.var(group0, ddof=1), np.var(group1, ddof=1)
    pooled_std = np.sqrt(((n0 - 1) * var0 + (n1 - 1) * var1) / (n0 + n1 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group0)) / pooled_std


def eda_piano(data):
    """
    Full EDA for the piano classification experiment.

    Args:
        data: Cleaned DataFrame from clean_piano().
    """
    output_dir = "plots/piano_plots/eda"
    os.makedirs(output_dir, exist_ok=True)

    df = data.copy()

    print(f"\n[EDA] Generating Piano Classification EDA...")
    print(f"[EDA] Output directory: {output_dir}")

    emotional = df[df["label"] == 0]
    happy = df[df["label"] == 1]

    # ==========================================
    # 1. TEXT REPORT
    # ==========================================
    r = []
    r.append("=" * 70)
    r.append("PIANO EMOTION CLASSIFICATION — EXPLORATORY DATA ANALYSIS")
    r.append("=" * 70)
    r.append(f"\nTask:   Binary classification (emotional=0 vs happy=1)")
    r.append(f"Design: 18 songs x 3 random 30-second clips = 54 samples")
    r.append(f"CV:     Leave-One-Song-Out (LOGO-CV) — 18 folds")
    r.append(f"Note:   Labels reflect musical style/activity, not validated emotion ground truth.")

    # Dataset Overview
    r.append("\n" + "-" * 50)
    r.append("DATASET OVERVIEW")
    r.append("-" * 50)
    r.append(f"Total rows:           {len(df)}")
    r.append(f"Total songs:          {df['song_name'].nunique()}")
    r.append(f"Clips per song:       3")
    r.append(f"Emotional clips (0):  {len(emotional)}  ({len(emotional) / len(df) * 100:.1f}%)")
    r.append(f"Happy clips (1):      {len(happy)}  ({len(happy) / len(df) * 100:.1f}%)")
    r.append(f"Class balance:        PERFECT (27/27) — no resampling needed")

    # Song list
    r.append("\n" + "-" * 50)
    r.append("EMOTIONAL SONGS (label = 0)")
    r.append("-" * 50)
    for song in sorted(emotional["song_name"].unique()):
        r.append(f"  {song}")

    r.append("\n" + "-" * 50)
    r.append("HAPPY SONGS (label = 1)")
    r.append("-" * 50)
    for song in sorted(happy["song_name"].unique()):
        r.append(f"  {song}")

    # Naive Baseline
    r.append("\n" + "-" * 50)
    r.append("NAIVE BASELINE")
    r.append("-" * 50)
    r.append(f"Majority class predictor accuracy: 0.5000 (always predicts either class)")
    r.append(f"Majority class F1:                 0.5000")
    r.append(f"A useful model must exceed F1 = 0.50 and ROC-AUC = 0.50.")

    # Effect sizes for top 20 features
    r.append("\n" + "-" * 50)
    r.append("TOP-20 FEATURES — CLASS STATISTICS")
    r.append("-" * 50)
    r.append(f"\n{'Feature':<30} {'Emo Mean':>10} {'Happy Mean':>11} {'Cohen d':>9} {'Direction'}")
    r.append("-" * 75)

    feature_effects = []
    for feat in TOP_20_FEATURES:
        if feat not in df.columns:
            continue
        emo_vals = emotional[feat].values
        happy_vals = happy[feat].values
        d = _cohens_d(emo_vals, happy_vals)
        direction = "Happy > Emo" if d > 0 else "Emo > Happy"
        feature_effects.append((feat, np.mean(emo_vals), np.mean(happy_vals), abs(d), direction))
        r.append(
            f"{feat:<30} {np.mean(emo_vals):>10.4f} {np.mean(happy_vals):>11.4f} "
            f"{abs(d):>9.3f} {direction}"
        )

    # All features sorted by absolute effect size
    r.append("\n" + "-" * 50)
    r.append("ALL FEATURES — RANKED BY EFFECT SIZE (|Cohen's d|)")
    r.append("-" * 50)
    r.append(f"\n{'Rank':<6} {'Feature':<35} {'|Cohen d|':>10} {'Direction'}")
    r.append("-" * 65)

    feat_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in {"label", "clip_index"}
    ]
    all_effects = []
    for feat in feat_cols:
        emo_vals = emotional[feat].values
        happy_vals = happy[feat].values
        d = abs(_cohens_d(emo_vals, happy_vals))
        direction = "Happy > Emo" if (np.mean(happy_vals) > np.mean(emo_vals)) else "Emo > Happy"
        all_effects.append((feat, d, direction))

    all_effects.sort(key=lambda x: x[1], reverse=True)
    for rank, (feat, d, direction) in enumerate(all_effects[:30], 1):
        r.append(f"{rank:<6} {feat:<35} {d:>10.3f} {direction}")

    # Correlation among top 20
    r.append("\n" + "-" * 50)
    r.append("INTER-FEATURE CORRELATIONS (Top 20)")
    r.append("-" * 50)
    available_top20 = [f for f in TOP_20_FEATURES if f in df.columns]
    corr_matrix = df[available_top20].corr()
    high_corr_pairs = []
    for i in range(len(available_top20)):
        for j in range(i + 1, len(available_top20)):
            c = corr_matrix.iloc[i, j]
            if abs(c) >= 0.80:
                high_corr_pairs.append((available_top20[i], available_top20[j], c))
    if high_corr_pairs:
        r.append(f"\nHighly correlated pairs (|r| >= 0.80) — potential redundancy:")
        for f1, f2, c in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
            r.append(f"  {f1} <-> {f2}: r = {c:.3f}")
    else:
        r.append("\nNo pairs with |r| >= 0.80 among top-20 features.")

    report_path = os.path.join(output_dir, "eda_piano_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(r))
    print(f"[EDA] Saved report to {report_path}")

    # ==========================================
    # 2. CLASS DISTRIBUTION
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    class_counts = df["label"].value_counts().sort_index()
    axes[0].bar(
        ["Emotional (0)", "Happy (1)"],
        class_counts.values,
        color=["#4878CF", "#D65F5F"],
        alpha=0.85,
        edgecolor="black",
    )
    for i, v in enumerate(class_counts.values):
        axes[0].text(i, v + 0.3, str(v), ha="center", va="bottom", fontsize=13, fontweight="bold")
    axes[0].set_ylim(0, 35)
    axes[0].set_title("Class Distribution (N=54)", fontsize=13)
    axes[0].set_ylabel("Number of Clips")
    axes[0].grid(True, axis="y", alpha=0.3)

    # Clips per song
    clips_per_song = df.groupby("song_name")["label"].first().sort_values()
    song_labels = [s.replace(" - ", "\n")[:30] for s in clips_per_song.index]
    colors = ["#4878CF" if l == 0 else "#D65F5F" for l in clips_per_song.values]
    axes[1].barh(range(len(song_labels)), [3] * len(song_labels), color=colors, alpha=0.85, edgecolor="black")
    axes[1].set_yticks(range(len(song_labels)))
    axes[1].set_yticklabels(song_labels, fontsize=7)
    axes[1].set_xlabel("Clips (3 per song)")
    axes[1].set_title("Songs by Class (blue=emotional, red=happy)", fontsize=11)
    axes[1].set_xlim(0, 4)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "piano_class_distribution.png"), dpi=150)
    plt.close()
    print(f"[EDA] Saved piano_class_distribution.png")

    # ==========================================
    # 3. EFFECT SIZE BAR CHART (Top 20)
    # ==========================================
    feat_names = [f[0] for f in feature_effects]
    effect_sizes = [f[3] for f in feature_effects]
    directions = [f[4] for f in feature_effects]
    bar_colors = ["#D65F5F" if d == "Happy > Emo" else "#4878CF" for d in directions]

    sorted_idx = np.argsort(effect_sizes)[::-1]
    feat_names_s = [feat_names[i] for i in sorted_idx]
    effect_sizes_s = [effect_sizes[i] for i in sorted_idx]
    bar_colors_s = [bar_colors[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(feat_names_s))
    ax.barh(y_pos, effect_sizes_s, color=bar_colors_s, alpha=0.85, edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_names_s, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="|d| = 0.5 (medium)")
    ax.axvline(x=0.8, color="gray", linestyle="-.", alpha=0.5, label="|d| = 0.8 (large)")
    ax.set_xlabel("Effect Size |Cohen's d|", fontsize=11)
    ax.set_title(
        "Top-20 Feature Discriminability\n(red = Happy > Emo, blue = Emo > Happy)",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "piano_effect_sizes.png"), dpi=150)
    plt.close()
    print(f"[EDA] Saved piano_effect_sizes.png")

    # ==========================================
    # 4. VIOLIN PLOTS — Top 8 Features
    # ==========================================
    top8 = [f[0] for f in sorted(feature_effects, key=lambda x: x[3], reverse=True)[:8]]
    top8 = [f for f in top8 if f in df.columns]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    plot_df = df[top8 + ["label"]].copy()
    plot_df["Class"] = plot_df["label"].map({0: "Emotional", 1: "Happy"})

    for i, feat in enumerate(top8):
        sns.violinplot(
            x="Class", y=feat, data=plot_df,
            palette={"Emotional": "#4878CF", "Happy": "#D65F5F"},
            ax=axes[i], inner="box", cut=0,
        )
        axes[i].set_title(feat, fontsize=9, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
        axes[i].grid(True, axis="y", alpha=0.3)

    plt.suptitle("Top-8 Feature Distributions by Class", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "piano_top8_violin.png"), dpi=150)
    plt.close()
    print(f"[EDA] Saved piano_top8_violin.png")

    # ==========================================
    # 5. CORRELATION HEATMAP — Top 20
    # ==========================================
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        corr_matrix,
        annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, square=True, linewidths=0.3,
        annot_kws={"size": 7},
    )
    plt.title("Feature Correlation Heatmap (Top-20 Features)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "piano_correlation_heatmap.png"), dpi=150)
    plt.close()
    print(f"[EDA] Saved piano_correlation_heatmap.png")

    # ==========================================
    # 6. ONSET STRENGTH vs NOTE DENSITY SCATTER (per clip, colored by class)
    # ==========================================
    if "onset_strength_mean" in df.columns and "note_density" in df.columns:
        fig, ax = plt.subplots(figsize=(9, 6))
        for label_val, label_name, color in [(0, "Emotional", "#4878CF"), (1, "Happy", "#D65F5F")]:
            subset = df[df["label"] == label_val]
            ax.scatter(
                subset["onset_strength_mean"],
                subset["note_density"],
                label=label_name,
                color=color,
                alpha=0.75,
                s=80,
                edgecolors="black",
                linewidth=0.5,
            )
        ax.set_xlabel("Onset Strength Mean (attack intensity)", fontsize=11)
        ax.set_ylabel("Note Density (notes/sec)", fontsize=11)
        ax.set_title("Onset Strength vs Note Density by Class\n(top 2 discriminative features)", fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "piano_onset_vs_density.png"), dpi=150)
        plt.close()
        print(f"[EDA] Saved piano_onset_vs_density.png")

    print(f"\n[EDA] Complete! All outputs saved to {output_dir}/")
