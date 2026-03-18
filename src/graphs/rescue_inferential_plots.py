"""
Visualizations for Rescue Experiment Inferential Statistics

Publication-quality plots for the within-subjects VR rescue experiment:
1. condition_outcomes.png      - Violin+strip of outcomes by Condition
2. experimenter_effect.png     - Outcomes by ExperimenterPresent
3. interaction_effects.png     - Condition x ExperimenterPresent interaction
4. participant_trajectories.png - Spaghetti plot of individual participants
5. model_coefficients.png      - Forest plot of model coefficients
6. effect_sizes.png            - Bar chart of effect sizes
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D


# ==========================================
# Color Palette
# ==========================================
COND_PALETTE = ["#4C72B0", "#DD8452"]
EXP_PALETTE = ["#55A868", "#C44E52"]
COND_ORDER = ["Single", "Multiple"]
EXP_ORDER = ["Absent", "Present"]

TERM_LABELS = {
    "Intercept": "Intercept\n(Single, Exp Absent)",
    "IsMultiple": "Multiple vs Single",
    "ExperimenterPresent": "Experimenter Present\nvs Absent",
    "IsMultiple:ExperimenterPresent": "Condition ×\nExperimenter",
}


def _apply_style():
    """Apply consistent publication style."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })


def _sig_stars(p):
    """Return significance stars for a p-value."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."


def _add_sig_bracket(ax, x1, x2, y, p, height=0.03):
    """Add significance bracket with stars between two x positions."""
    stars = _sig_stars(p)
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y],
            lw=1.5, color="black")
    ax.text((x1 + x2) / 2, y + height + 0.01, stars,
            ha="center", va="bottom", fontsize=12, fontweight="bold")


# ==========================================
# Plot 1: Condition Outcomes
# ==========================================
def plot_condition_outcomes(df, test_results, output_dir):
    """
    Two-panel figure: PeopleRescued violin+strip and Helping Rate bars by Condition.
    Includes significance annotations from non-parametric tests.
    """
    _apply_style()
    os.makedirs(output_dir, exist_ok=True)

    plot_df = df.copy()
    plot_df["Cond"] = plot_df["Condition"].map({1: "Single", 2: "Multiple"})

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Panel 1: PeopleRescued violin + strip ---
    sns.violinplot(
        x="Cond", y="PeopleRescued", data=plot_df,
        order=COND_ORDER, ax=axes[0],
        palette=COND_PALETTE, alpha=0.6, inner=None, cut=0,
    )
    sns.stripplot(
        x="Cond", y="PeopleRescued", data=plot_df,
        order=COND_ORDER, ax=axes[0],
        color="black", alpha=0.7, size=7, jitter=0.15,
    )

    for i, cond in enumerate(COND_ORDER):
        subset = plot_df[plot_df["Cond"] == cond]["PeopleRescued"]
        m, s = subset.mean(), subset.std()
        axes[0].plot(i, m, "D", color="red", markersize=10, zorder=5,
                     markeredgecolor="darkred")
        axes[0].text(i + 0.25, m, f"M={m:.2f}\nSD={s:.2f}",
                     fontsize=9, color="red", fontweight="bold", va="center")

    if "wilcoxon_rescued" in test_results and "p_value" in test_results["wilcoxon_rescued"]:
        p = test_results["wilcoxon_rescued"]["p_value"]
        max_y = plot_df["PeopleRescued"].max()
        _add_sig_bracket(axes[0], 0, 1, max_y + 0.3, p)

    axes[0].set_title("People Rescued by Condition", fontweight="bold")
    axes[0].set_xlabel("Condition")
    axes[0].set_ylabel("People Rescued")

    # --- Panel 2: Helping Rate bars ---
    rate = plot_df.groupby("Cond")["Helped"].agg(["mean", "sem", "count"]).reindex(COND_ORDER)
    # Handle sem=0 or NaN
    rate["sem"] = rate["sem"].fillna(0)

    axes[1].bar(
        range(len(rate)), rate["mean"],
        yerr=rate["sem"] * 1.96,
        color=COND_PALETTE, alpha=0.8, edgecolor="black", linewidth=1.2,
        capsize=6, error_kw={"linewidth": 2},
    )
    axes[1].set_xticks(range(len(rate)))
    axes[1].set_xticklabels(rate.index)

    for i, (idx, row) in enumerate(rate.iterrows()):
        n = int(row["count"])
        ci_top = row["mean"] + row["sem"] * 1.96
        axes[1].text(i, ci_top + 0.05,
                     f'{row["mean"]:.0%}\n(n={n})',
                     ha="center", fontsize=11, fontweight="bold")

    if "mcnemar_helped" in test_results and "p_value" in test_results["mcnemar_helped"]:
        p = test_results["mcnemar_helped"]["p_value"]
        top = max(rate["mean"] + rate["sem"] * 1.96) + 0.18
        _add_sig_bracket(axes[1], 0, 1, top, p)

    axes[1].set_ylim(0, 1.5)
    axes[1].set_title("Helping Rate by Condition", fontweight="bold")
    axes[1].set_xlabel("Condition")
    axes[1].set_ylabel("Proportion Who Helped")
    axes[1].axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout(pad=2.5)
    fig.suptitle("RQ1: Does Victim Count Affect Rescue Behavior?",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.savefig(os.path.join(output_dir, "condition_outcomes.png"),
                dpi=200, bbox_inches="tight")
    plt.close()
    print("[Plots] Saved condition_outcomes.png")


# ==========================================
# Plot 2: Experimenter Effect
# ==========================================
def plot_experimenter_effect(df, test_results, output_dir):
    """Outcomes by ExperimenterPresent (between-subjects)."""
    _apply_style()
    os.makedirs(output_dir, exist_ok=True)

    plot_df = df.copy()
    plot_df["Exp"] = plot_df["ExperimenterPresent"].map({0: "Absent", 1: "Present"})

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Panel 1: PeopleRescued ---
    sns.boxplot(
        x="Exp", y="PeopleRescued", data=plot_df,
        order=EXP_ORDER, ax=axes[0], palette=EXP_PALETTE, width=0.5,
    )
    sns.stripplot(
        x="Exp", y="PeopleRescued", data=plot_df,
        order=EXP_ORDER, ax=axes[0],
        color="black", alpha=0.7, size=7, jitter=0.15,
    )

    for i, exp in enumerate(EXP_ORDER):
        subset = plot_df[plot_df["Exp"] == exp]["PeopleRescued"]
        m, s = subset.mean(), subset.std()
        axes[0].plot(i, m, "D", color="red", markersize=10, zorder=5,
                     markeredgecolor="darkred")
        axes[0].text(i + 0.25, m, f"M={m:.2f}\nSD={s:.2f}",
                     fontsize=9, color="red", fontweight="bold", va="center")

    if "mannwhitney_rescued" in test_results and "p_value" in test_results["mannwhitney_rescued"]:
        p = test_results["mannwhitney_rescued"]["p_value"]
        max_y = plot_df["PeopleRescued"].max()
        _add_sig_bracket(axes[0], 0, 1, max_y + 0.3, p)

    axes[0].set_title("People Rescued by Experimenter Presence", fontweight="bold")
    axes[0].set_xlabel("Experimenter")
    axes[0].set_ylabel("People Rescued")

    # --- Panel 2: Helping Rate ---
    rate = plot_df.groupby("Exp")["Helped"].agg(["mean", "sem", "count"]).reindex(EXP_ORDER)
    rate["sem"] = rate["sem"].fillna(0)

    axes[1].bar(
        range(len(rate)), rate["mean"],
        yerr=rate["sem"] * 1.96,
        color=EXP_PALETTE, alpha=0.8, edgecolor="black", linewidth=1.2,
        capsize=6, error_kw={"linewidth": 2},
    )
    axes[1].set_xticks(range(len(rate)))
    axes[1].set_xticklabels(rate.index)

    for i, (idx, row) in enumerate(rate.iterrows()):
        n = int(row["count"])
        ci_top = row["mean"] + row["sem"] * 1.96
        axes[1].text(i, ci_top + 0.05,
                     f'{row["mean"]:.0%}\n(n={n})',
                     ha="center", fontsize=11, fontweight="bold")

    if "fisher_helped" in test_results and "p_value" in test_results["fisher_helped"]:
        p = test_results["fisher_helped"]["p_value"]
        top = max(rate["mean"] + rate["sem"] * 1.96) + 0.18
        _add_sig_bracket(axes[1], 0, 1, top, p)

    axes[1].set_ylim(0, 1.5)
    axes[1].set_title("Helping Rate by Experimenter Presence", fontweight="bold")
    axes[1].set_xlabel("Experimenter")
    axes[1].set_ylabel("Proportion Who Helped")

    plt.tight_layout(pad=2.5)
    fig.suptitle("RQ2: Does Experimenter Presence Affect Rescue Behavior?",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.savefig(os.path.join(output_dir, "experimenter_effect.png"),
                dpi=200, bbox_inches="tight")
    plt.close()
    print("[Plots] Saved experimenter_effect.png")


# ==========================================
# Plot 3: Interaction Effects
# ==========================================
def plot_interaction_effects(df, output_dir):
    """Condition x ExperimenterPresent interaction plot (means + CI)."""
    _apply_style()
    os.makedirs(output_dir, exist_ok=True)

    plot_df = df.copy()
    plot_df["Cond"] = plot_df["Condition"].map({1: "Single", 2: "Multiple"})
    plot_df["Exp"] = plot_df["ExperimenterPresent"].map({0: "Absent", 1: "Present"})

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for panel, (outcome, ylabel, title) in enumerate([
        ("PeopleRescued", "Mean People Rescued (±95% CI)", "People Rescued"),
        ("Helped", "Proportion Helped (±95% CI)", "Helping Rate"),
    ]):
        for i, exp in enumerate(EXP_ORDER):
            subset = plot_df[plot_df["Exp"] == exp]
            means = subset.groupby("Cond")[outcome].mean().reindex(COND_ORDER)
            sems = subset.groupby("Cond")[outcome].sem().reindex(COND_ORDER).fillna(0)

            x_pos = np.arange(len(COND_ORDER)) + (i - 0.5) * 0.15
            axes[panel].errorbar(
                x_pos, means, yerr=sems * 1.96,
                marker="o", markersize=10, capsize=5, linewidth=2.5,
                color=EXP_PALETTE[i], label=f"Experimenter {exp}",
                markeredgecolor="black", markeredgewidth=0.5,
            )

        axes[panel].set_xticks(range(len(COND_ORDER)))
        axes[panel].set_xticklabels(COND_ORDER)
        axes[panel].set_title(title, fontweight="bold")
        axes[panel].set_xlabel("Condition")
        axes[panel].set_ylabel(ylabel)
        axes[panel].legend(fontsize=10, framealpha=0.9)

        if outcome == "Helped":
            axes[panel].set_ylim(-0.1, 1.3)

    plt.tight_layout(pad=2.5)
    fig.suptitle("Condition × Experimenter Presence Interaction",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.savefig(os.path.join(output_dir, "interaction_effects.png"),
                dpi=200, bbox_inches="tight")
    plt.close()
    print("[Plots] Saved interaction_effects.png")


# ==========================================
# Plot 4: Participant Trajectories
# ==========================================
def plot_participant_trajectories(df, output_dir):
    """Spaghetti plot showing individual participant responses across conditions."""
    _apply_style()
    os.makedirs(output_dir, exist_ok=True)

    plot_df = df.copy()
    plot_df["Exp"] = plot_df["ExperimenterPresent"].map({0: "Absent", 1: "Present"})

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Panel 1: PeopleRescued trajectories ---
    for pid in plot_df["ParticipantID"].unique():
        p_data = plot_df[plot_df["ParticipantID"] == pid].sort_values("Condition")
        if len(p_data) < 2:
            continue
        exp = p_data["Exp"].iloc[0]
        color = EXP_PALETTE[EXP_ORDER.index(exp)]
        axes[0].plot(
            p_data["Condition"], p_data["PeopleRescued"],
            "o-", color=color, alpha=0.5, linewidth=1.5, markersize=7,
            markeredgecolor="black", markeredgewidth=0.5,
        )

    # Group means
    for exp_val, exp_label in [(0, "Absent"), (1, "Present")]:
        subset = plot_df[plot_df["ExperimenterPresent"] == exp_val]
        means = subset.groupby("Condition")["PeopleRescued"].mean()
        color = EXP_PALETTE[EXP_ORDER.index(exp_label)]
        axes[0].plot(
            means.index, means.values, "D--", color=color, linewidth=3,
            markersize=12, markeredgecolor="black", markeredgewidth=1.5,
            alpha=0.9, zorder=5,
        )

    legend_elements = [
        Line2D([0], [0], color=EXP_PALETTE[0], linewidth=2,
               label="Experimenter Absent"),
        Line2D([0], [0], color=EXP_PALETTE[1], linewidth=2,
               label="Experimenter Present"),
        Line2D([0], [0], marker="D", color="gray", linestyle="--",
               linewidth=2, markersize=8, label="Group Mean"),
    ]
    axes[0].legend(handles=legend_elements, fontsize=9, loc="upper left",
                   framealpha=0.9)
    axes[0].set_xticks([1, 2])
    axes[0].set_xticklabels(["Single (1)", "Multiple (2)"])
    axes[0].set_title("Individual Trajectories: People Rescued", fontweight="bold")
    axes[0].set_xlabel("Condition")
    axes[0].set_ylabel("People Rescued")

    # --- Panel 2: Helped trajectories ---
    np.random.seed(42)
    for pid in plot_df["ParticipantID"].unique():
        p_data = plot_df[plot_df["ParticipantID"] == pid].sort_values("Condition")
        if len(p_data) < 2:
            continue
        exp = p_data["Exp"].iloc[0]
        color = EXP_PALETTE[EXP_ORDER.index(exp)]
        jitter = np.random.uniform(-0.03, 0.03, size=len(p_data))
        axes[1].plot(
            p_data["Condition"], p_data["Helped"] + jitter,
            "o-", color=color, alpha=0.5, linewidth=1.5, markersize=7,
            markeredgecolor="black", markeredgewidth=0.5,
        )

    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["Not Helped", "Helped"])
    axes[1].set_xticks([1, 2])
    axes[1].set_xticklabels(["Single (1)", "Multiple (2)"])
    axes[1].set_title("Individual Trajectories: Helped (Binary)", fontweight="bold")
    axes[1].set_xlabel("Condition")
    axes[1].set_ylabel("Outcome")
    axes[1].set_ylim(-0.15, 1.15)
    axes[1].legend(handles=legend_elements[:2], fontsize=9, loc="upper left",
                   framealpha=0.9)

    plt.tight_layout(pad=2.5)
    fig.suptitle("Within-Subject Response Patterns Across Conditions",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.savefig(os.path.join(output_dir, "participant_trajectories.png"),
                dpi=200, bbox_inches="tight")
    plt.close()
    print("[Plots] Saved participant_trajectories.png")


# ==========================================
# Plot 5: Forest Plot of Model Coefficients
# ==========================================
def plot_forest_coefficients(coef_data_list, output_dir):
    """
    Forest plot of model coefficients with 95% CIs.

    coef_data_list: list of dicts, each with keys:
        title, terms, coefs, lower_ci, upper_ci, pvalues
    """
    _apply_style()
    os.makedirs(output_dir, exist_ok=True)

    valid = [d for d in coef_data_list if d is not None]
    if not valid:
        print("[Plots] No model results for forest plot - skipping")
        return

    n_panels = len(valid)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, max(5, len(valid[0]["terms"]) * 0.8 + 2)))
    if n_panels == 1:
        axes = [axes]

    for ax, data in zip(axes, valid):
        terms = data["terms"]
        coefs = np.array(data["coefs"])
        lower = np.array(data["lower_ci"])
        upper = np.array(data["upper_ci"])
        pvals = np.array(data["pvalues"])

        # Pretty labels
        labels = [TERM_LABELS.get(t, t) for t in terms]

        y_pos = np.arange(len(terms))

        # Color by significance
        colors = ["#C44E52" if p < 0.05 else "#4C72B0" for p in pvals]

        ax.axvline(x=0, color="black", linestyle="-", linewidth=1, alpha=0.7)

        for i in range(len(terms)):
            ax.plot([lower[i], upper[i]], [y_pos[i], y_pos[i]],
                    color=colors[i], linewidth=2.5, solid_capstyle="round")
            ax.plot(coefs[i], y_pos[i], "o", color=colors[i],
                    markersize=10, markeredgecolor="black", markeredgewidth=0.5)

            # Annotation with coefficient and p-value
            stars = _sig_stars(pvals[i])
            ax.text(max(upper[i], coefs[i]) + 0.05 * (ax.get_xlim()[1] - ax.get_xlim()[0] + 1),
                    y_pos[i],
                    f"β={coefs[i]:.3f} {stars}",
                    va="center", fontsize=9, fontweight="bold")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Coefficient (β)")
        ax.set_title(data["title"], fontweight="bold")
        ax.invert_yaxis()

        # Legend
        legend_elements = [
            Line2D([0], [0], marker="o", color="#C44E52", linestyle="None",
                   markersize=8, label="p < 0.05"),
            Line2D([0], [0], marker="o", color="#4C72B0", linestyle="None",
                   markersize=8, label="p ≥ 0.05"),
        ]
        ax.legend(handles=legend_elements, fontsize=9, loc="lower right",
                  framealpha=0.9)

    plt.tight_layout(pad=2.5)
    fig.suptitle("Mixed-Effects Model Coefficients (95% CI)",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.savefig(os.path.join(output_dir, "model_coefficients.png"),
                dpi=200, bbox_inches="tight")
    plt.close()
    print("[Plots] Saved model_coefficients.png")


# ==========================================
# Plot 6: Effect Sizes
# ==========================================
def plot_effect_sizes(effects, output_dir):
    """Horizontal bar chart of effect sizes (Cohen's d)."""
    _apply_style()
    os.makedirs(output_dir, exist_ok=True)

    # Filter to entries with a "value" key (Cohen's d effects)
    d_effects = {k: v for k, v in effects.items() if "value" in v}
    if not d_effects:
        print("[Plots] No effect sizes to plot - skipping")
        return

    labels = [v["label"] for v in d_effects.values()]
    values = [v["value"] for v in d_effects.values()]
    interps = [v["interpretation"] for v in d_effects.values()]

    # Color by magnitude
    color_map = {
        "negligible": "#CCCCCC",
        "small": "#A8D8EA",
        "medium": "#FFD93D",
        "large": "#FF6B6B",
    }
    colors = [color_map.get(i, "#CCCCCC") for i in interps]

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 1.2)))

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors, edgecolor="black",
                   linewidth=1, height=0.6, alpha=0.85)

    # Threshold lines
    for thresh, label_text in [(0.2, "Small"), (0.5, "Medium"), (0.8, "Large")]:
        ax.axvline(x=thresh, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(x=-thresh, color="gray", linestyle=":", alpha=0.5)
        ax.text(thresh, len(labels) - 0.5, label_text, fontsize=8,
                color="gray", ha="center", va="bottom")

    ax.axvline(x=0, color="black", linewidth=1)

    # Annotations
    for i, (val, interp) in enumerate(zip(values, interps)):
        offset = 0.05 if val >= 0 else -0.05
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, i, f"d={val:.2f} ({interp})",
                va="center", ha=ha, fontsize=10, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Cohen's d", fontsize=12)
    ax.set_title("Effect Sizes Summary", fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "effect_sizes.png"),
                dpi=200, bbox_inches="tight")
    plt.close()
    print("[Plots] Saved effect_sizes.png")
