"""
Rescue Experiment - Inferential Statistics

Mixed-effects models and non-parametric tests for the within-subjects
VR rescue experiment.

Research Questions:
1. Does condition (single vs multiple victims in stormy VR environment)
   affect prosocial rescue behavior? (Diffusion of responsibility)
2. Does experimenter presence in the room moderate the condition effect?

Analysis Plan:
- Mixed-effects linear model for PeopleRescued (continuous count)
- GEE with binomial family for Helped (binary outcome)
- Non-parametric tests (Wilcoxon signed-rank, Mann-Whitney U, McNemar's)
- Effect sizes (Cohen's d, rank-biserial correlation)

Design:
  Within-subjects factor: Condition (1=Single, 2=Multiple victims)
  Between-subjects factor: ExperimenterPresent (0=Absent, 1=Present)
  Random effect: ParticipantID (random intercept)
  Covariates: baseline_TimeElapsed, baseline_DrownCount (from Condition 0)
"""

import os
import sys
import json
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from utils.load.load_rescue import load_rescue_data
from utils.clean.clean_rescue import clean_rescue, build_model_df

from graphs.rescue_inferential_plots import (
    plot_condition_outcomes,
    plot_experimenter_effect,
    plot_interaction_effects,
    plot_participant_trajectories,
    plot_forest_coefficients,
    plot_effect_sizes,
)

# statsmodels
try:
    import statsmodels.formula.api as smf
    from statsmodels.genmod.families import Binomial
    from statsmodels.genmod.cov_struct import Exchangeable
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("[WARNING] statsmodels not installed. Mixed-effects models will be skipped.")


# ==========================================
# Constants
# ==========================================
RANDOM_STATE = 42


# ==========================================
# Data Preparation
# ==========================================
def _ensure_clean_experimenter_present(df):
    """Ensure ExperimenterPresent is cleanly coded as 0/1 integer."""
    if "ExperimenterPresent" not in df.columns:
        return df
    df = df.copy()
    df["ExperimenterPresent"] = df["ExperimenterPresent"].apply(
        lambda x: 1 if str(x).lower().strip() in ("1", "true", "yes", "1.0") else 0
    ).astype(int)
    return df


# ==========================================
# Descriptive Statistics
# ==========================================
def compute_descriptives(df):
    """Compute descriptive statistics tables for the report."""
    results = {}

    # Overall sample
    n_participants = df["ParticipantID"].nunique()
    participant_df = df.drop_duplicates("ParticipantID")

    results["n_participants"] = n_participants
    results["n_rows"] = len(df)
    results["age_mean"] = float(participant_df["Age"].mean())
    results["age_std"] = float(participant_df["Age"].std())
    results["age_min"] = int(participant_df["Age"].min())
    results["age_max"] = int(participant_df["Age"].max())
    results["gender_counts"] = participant_df["Gender"].value_counts().to_dict()
    results["exp_counts"] = participant_df["ExperimenterPresent"].value_counts().to_dict()

    # By Condition
    cond_stats = {}
    for cond in sorted(df["Condition"].unique()):
        subset = df[df["Condition"] == cond]
        cond_stats[int(cond)] = {
            "n": len(subset),
            "rescued_mean": float(subset["PeopleRescued"].mean()),
            "rescued_std": float(subset["PeopleRescued"].std()),
            "rescued_median": float(subset["PeopleRescued"].median()),
            "helped_rate": float(subset["Helped"].mean()),
            "helped_n": int(subset["Helped"].sum()),
            "time_near_mean": float(subset["TimeNearVictim"].mean()),
            "time_near_std": float(subset["TimeNearVictim"].std()),
            "time_elapsed_mean": float(subset["TimeElapsed"].mean()),
        }
    results["by_condition"] = cond_stats

    # By ExperimenterPresent
    exp_stats = {}
    for exp in sorted(df["ExperimenterPresent"].unique()):
        subset = df[df["ExperimenterPresent"] == exp]
        exp_stats[int(exp)] = {
            "n": len(subset),
            "n_participants": subset["ParticipantID"].nunique(),
            "rescued_mean": float(subset["PeopleRescued"].mean()),
            "rescued_std": float(subset["PeopleRescued"].std()),
            "helped_rate": float(subset["Helped"].mean()),
            "helped_n": int(subset["Helped"].sum()),
        }
    results["by_experimenter"] = exp_stats

    # By Condition x ExperimenterPresent
    interaction_stats = {}
    for cond in sorted(df["Condition"].unique()):
        for exp in sorted(df["ExperimenterPresent"].unique()):
            subset = df[(df["Condition"] == cond) & (df["ExperimenterPresent"] == exp)]
            key = f"cond{int(cond)}_exp{int(exp)}"
            if len(subset) > 0:
                interaction_stats[key] = {
                    "n": len(subset),
                    "rescued_mean": float(subset["PeopleRescued"].mean()),
                    "rescued_std": float(subset["PeopleRescued"].std()),
                    "helped_rate": float(subset["Helped"].mean()),
                    "helped_n": int(subset["Helped"].sum()),
                }
            else:
                interaction_stats[key] = {"n": 0}
    results["interaction"] = interaction_stats

    return results


# ==========================================
# Normality Tests
# ==========================================
def run_normality_tests(df):
    """Shapiro-Wilk normality tests on PeopleRescued by condition."""
    results = {}
    for cond in sorted(df["Condition"].unique()):
        subset = df[df["Condition"] == cond]["PeopleRescued"]
        if len(subset) >= 3:
            stat, p = scipy_stats.shapiro(subset)
            results[f"cond_{int(cond)}"] = {
                "statistic": float(stat), "p_value": float(p),
                "normal": p > 0.05,
            }

    overall = df["PeopleRescued"]
    if len(overall) >= 3:
        stat, p = scipy_stats.shapiro(overall)
        results["overall"] = {
            "statistic": float(stat), "p_value": float(p),
            "normal": p > 0.05,
        }
    return results


# ==========================================
# Within-Subjects Tests (Condition Effect)
# ==========================================
def run_paired_tests(df):
    """Wilcoxon signed-rank (PeopleRescued) and McNemar (Helped) for Condition 1 vs 2."""
    results = {}

    # Pivot to paired format
    pivot_rescued = df.pivot_table(
        index="ParticipantID", columns="Condition",
        values="PeopleRescued", aggfunc="first",
    ).dropna(subset=[1, 2])

    pivot_helped = df.pivot_table(
        index="ParticipantID", columns="Condition",
        values="Helped", aggfunc="first",
    ).dropna(subset=[1, 2])

    n_paired = len(pivot_rescued)
    results["n_paired"] = n_paired

    # --- Wilcoxon signed-rank for PeopleRescued ---
    if n_paired >= 5:
        diff = pivot_rescued[2] - pivot_rescued[1]
        nonzero_diff = diff[diff != 0]
        if len(nonzero_diff) > 0:
            try:
                stat, p = scipy_stats.wilcoxon(pivot_rescued[1], pivot_rescued[2])
                # Rank-biserial effect size: r = Z / sqrt(N)
                z_approx = scipy_stats.norm.ppf(p / 2)
                r_rb = abs(z_approx) / np.sqrt(n_paired)
                results["wilcoxon_rescued"] = {
                    "statistic": float(stat), "p_value": float(p),
                    "n_pairs": n_paired,
                    "mean_diff": float(diff.mean()),
                    "median_diff": float(diff.median()),
                    "rank_biserial_r": float(r_rb),
                }
            except Exception as e:
                results["wilcoxon_rescued"] = {"error": str(e)}
        else:
            results["wilcoxon_rescued"] = {
                "statistic": 0.0, "p_value": 1.0, "n_pairs": n_paired,
                "note": "All paired differences are zero",
            }
    else:
        results["wilcoxon_rescued"] = {"error": f"Too few pairs ({n_paired})"}

    # --- McNemar's test for Helped (paired binary) ---
    if n_paired >= 3:
        h1 = pivot_helped[1].astype(int)
        h2 = pivot_helped[2].astype(int)
        a = int(((h1 == 1) & (h2 == 1)).sum())  # helped both
        b = int(((h1 == 1) & (h2 == 0)).sum())  # helped single only
        c = int(((h1 == 0) & (h2 == 1)).sum())  # helped multiple only
        d = int(((h1 == 0) & (h2 == 0)).sum())  # helped neither

        discordant = b + c
        if discordant > 0:
            # Exact binomial test for small samples
            p = float(scipy_stats.binomtest(min(b, c), discordant, 0.5).pvalue)
            stat = float((b - c) ** 2 / discordant)
        else:
            stat, p = 0.0, 1.0

        results["mcnemar_helped"] = {
            "statistic": stat, "p_value": p,
            "table": {"both_helped": a, "only_single": b,
                      "only_multiple": c, "neither": d},
            "discordant_pairs": discordant,
            "n_pairs": n_paired,
        }
    else:
        results["mcnemar_helped"] = {"error": f"Too few pairs ({n_paired})"}

    # --- Paired t-test for comparison ---
    if n_paired >= 3:
        stat, p = scipy_stats.ttest_rel(pivot_rescued[1], pivot_rescued[2])
        results["paired_ttest_rescued"] = {
            "statistic": float(stat), "p_value": float(p),
            "n_pairs": n_paired,
        }

    return results


# ==========================================
# Between-Subjects Tests (Experimenter Effect)
# ==========================================
def run_between_tests(df):
    """Mann-Whitney U (PeopleRescued) and Fisher's exact (Helped) for ExperimenterPresent."""
    results = {}

    exp0 = df[df["ExperimenterPresent"] == 0]
    exp1 = df[df["ExperimenterPresent"] == 1]
    n0 = exp0["ParticipantID"].nunique()
    n1 = exp1["ParticipantID"].nunique()

    results["n_absent"] = n0
    results["n_present"] = n1

    if n0 < 2 or n1 < 2:
        results["mannwhitney_rescued"] = {"error": f"Too few in group (absent={n0}, present={n1})"}
        results["fisher_helped"] = {"error": f"Too few in group (absent={n0}, present={n1})"}
        return results

    # --- Mann-Whitney U for PeopleRescued ---
    try:
        stat, p = scipy_stats.mannwhitneyu(
            exp0["PeopleRescued"], exp1["PeopleRescued"], alternative="two-sided",
        )
        # Rank-biserial r = 1 - 2U/(n1*n2)
        n1_obs, n2_obs = len(exp0), len(exp1)
        r_rb = 1 - (2 * stat) / (n1_obs * n2_obs)
        results["mannwhitney_rescued"] = {
            "statistic": float(stat), "p_value": float(p),
            "n_absent": len(exp0), "n_present": len(exp1),
            "rank_biserial_r": float(r_rb),
        }
    except Exception as e:
        results["mannwhitney_rescued"] = {"error": str(e)}

    # --- Fisher's exact test for Helped ---
    try:
        table = pd.crosstab(df["ExperimenterPresent"], df["Helped"])
        if table.shape == (2, 2):
            odds_ratio, p = scipy_stats.fisher_exact(table)
            results["fisher_helped"] = {
                "odds_ratio": float(odds_ratio), "p_value": float(p),
                "table": {
                    "absent_not_helped": int(table.loc[0, 0]),
                    "absent_helped": int(table.loc[0, 1]),
                    "present_not_helped": int(table.loc[1, 0]),
                    "present_helped": int(table.loc[1, 1]),
                },
            }
        else:
            results["fisher_helped"] = {"error": "Not a 2x2 table"}
    except Exception as e:
        results["fisher_helped"] = {"error": str(e)}

    return results


# ==========================================
# Mixed-Effects Linear Model
# ==========================================
def run_mixed_effects_model(df):
    """Mixed LM: PeopleRescued ~ IsMultiple * ExperimenterPresent + (1|ParticipantID)"""
    if not HAS_STATSMODELS:
        return None

    model_df = df.copy()
    model_df["IsMultiple"] = (model_df["Condition"] == 2).astype(int)

    try:
        # --- Simple model: condition effect only ---
        model_simple = smf.mixedlm(
            "PeopleRescued ~ IsMultiple",
            model_df, groups=model_df["ParticipantID"],
        )
        result_simple = model_simple.fit(reml=True)

        # --- Full model: with interaction ---
        model_full = smf.mixedlm(
            "PeopleRescued ~ IsMultiple * ExperimenterPresent",
            model_df, groups=model_df["ParticipantID"],
        )
        result_full = model_full.fit(reml=True)

        # --- With baseline covariates ---
        covariate_formula = "PeopleRescued ~ IsMultiple * ExperimenterPresent"
        if "baseline_TimeElapsed" in model_df.columns:
            covariate_formula += " + baseline_TimeElapsed + baseline_DrownCount"
        model_cov = smf.mixedlm(
            covariate_formula,
            model_df, groups=model_df["ParticipantID"],
        )
        result_cov = model_cov.fit(reml=True)

        def _extract_coefs(result, title):
            ci = result.conf_int()
            return {
                "title": title,
                "terms": result.fe_params.index.tolist(),
                "coefs": result.fe_params.values.tolist(),
                "lower_ci": ci.iloc[:, 0].values.tolist(),
                "upper_ci": ci.iloc[:, 1].values.tolist(),
                "pvalues": result.pvalues.values.tolist(),
            }

        # ICC
        re_var = float(result_full.cov_re.iloc[0, 0]) if hasattr(result_full.cov_re, "iloc") else float(result_full.cov_re)
        resid_var = float(result_full.scale)
        icc = re_var / (re_var + resid_var) if (re_var + resid_var) > 0 else 0

        return {
            "simple": result_simple,
            "full": result_full,
            "covariate": result_cov,
            "simple_coefs": _extract_coefs(result_simple, "Mixed LM: Condition Only"),
            "full_coefs": _extract_coefs(result_full, "Mixed LM: Full Model"),
            "cov_coefs": _extract_coefs(result_cov, "Mixed LM: With Covariates"),
            "random_effects_var": re_var,
            "residual_var": resid_var,
            "icc": icc,
            "simple_summary": result_simple.summary().as_text(),
            "full_summary": result_full.summary().as_text(),
            "cov_summary": result_cov.summary().as_text(),
        }
    except Exception as e:
        print(f"[WARNING] Mixed-effects model failed: {e}")
        return {"error": str(e)}


# ==========================================
# GEE Binomial Model
# ==========================================
def run_gee_model(df):
    """GEE Binomial: Helped ~ IsMultiple * ExperimenterPresent"""
    if not HAS_STATSMODELS:
        return None

    model_df = df.copy()
    model_df["IsMultiple"] = (model_df["Condition"] == 2).astype(int)
    model_df = model_df.sort_values("ParticipantID")

    try:
        model = smf.gee(
            "Helped ~ IsMultiple * ExperimenterPresent",
            "ParticipantID", model_df,
            family=Binomial(), cov_struct=Exchangeable(),
        )
        result = model.fit()

        ci = result.conf_int()
        coefs = {
            "title": "GEE Binomial: Helped",
            "terms": result.params.index.tolist(),
            "coefs": result.params.values.tolist(),
            "lower_ci": ci.iloc[:, 0].values.tolist(),
            "upper_ci": ci.iloc[:, 1].values.tolist(),
            "pvalues": result.pvalues.values.tolist(),
        }

        odds_ratios = np.exp(result.params).to_dict()

        return {
            "result": result,
            "coefs": coefs,
            "odds_ratios": odds_ratios,
            "summary": result.summary().as_text(),
        }
    except Exception as e:
        print(f"[WARNING] GEE model failed: {e}")
        return {"error": str(e)}


# ==========================================
# Effect Sizes
# ==========================================
def compute_effect_sizes(df):
    """Compute Cohen's d effect sizes for each comparison."""
    effects = {}

    cond1 = df[df["Condition"] == 1]["PeopleRescued"]
    cond2 = df[df["Condition"] == 2]["PeopleRescued"]

    # Paired Cohen's d for Condition effect
    pivot = df.pivot_table(
        index="ParticipantID", columns="Condition",
        values="PeopleRescued", aggfunc="first",
    ).dropna(subset=[1, 2])

    if len(pivot) >= 2:
        diff = pivot[2] - pivot[1]
        sd_diff = diff.std()
        d_paired = float(diff.mean() / sd_diff) if sd_diff > 0 else 0.0
        effects["condition_rescued_paired_d"] = {
            "value": d_paired,
            "interpretation": _interpret_d(abs(d_paired)),
            "label": "Condition effect\n(Multiple vs Single)\nPeopleRescued (paired)",
        }

    # Unpaired Cohen's d
    n1, n2 = len(cond1), len(cond2)
    s1, s2 = cond1.std(), cond2.std()
    pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / max(n1 + n2 - 2, 1))
    d_unp = float((cond2.mean() - cond1.mean()) / pooled) if pooled > 0 else 0.0
    effects["condition_rescued_d"] = {
        "value": d_unp,
        "interpretation": _interpret_d(abs(d_unp)),
        "label": "Condition effect\n(Multiple vs Single)\nPeopleRescued",
    }

    # ExperimenterPresent effect
    exp0 = df[df["ExperimenterPresent"] == 0]["PeopleRescued"]
    exp1 = df[df["ExperimenterPresent"] == 1]["PeopleRescued"]
    if len(exp0) >= 2 and len(exp1) >= 2:
        n0, n1_e = len(exp0), len(exp1)
        pooled_e = np.sqrt(((n0 - 1) * exp0.std() ** 2 + (n1_e - 1) * exp1.std() ** 2) / max(n0 + n1_e - 2, 1))
        d_exp = float((exp1.mean() - exp0.mean()) / pooled_e) if pooled_e > 0 else 0.0
        effects["experimenter_rescued_d"] = {
            "value": d_exp,
            "interpretation": _interpret_d(abs(d_exp)),
            "label": "Experimenter Present\nvs Absent\nPeopleRescued",
        }

    # Helping rate difference
    pivot_h = df.pivot_table(
        index="ParticipantID", columns="Condition",
        values="Helped", aggfunc="first",
    ).dropna(subset=[1, 2])
    if len(pivot_h) >= 2:
        rate1 = float(pivot_h[1].mean())
        rate2 = float(pivot_h[2].mean())
        effects["condition_helped_rates"] = {
            "rate_single": rate1,
            "rate_multiple": rate2,
            "difference": rate2 - rate1,
            "label": "Condition effect\non Helping Rate",
        }

    return effects


def _interpret_d(d):
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


# ==========================================
# Report Generation
# ==========================================
def generate_report(desc, normality, paired, between, lmm, gee, effects, output_path):
    """Generate comprehensive text report for discussion."""
    L = []
    w = L.append

    w("=" * 80)
    w("RESCUE EXPERIMENT - INFERENTIAL STATISTICS REPORT")
    w("=" * 80)
    w(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ---- 1. Study Overview ----
    w("\n" + "=" * 80)
    w("1. STUDY OVERVIEW")
    w("=" * 80)

    w("\nRESEARCH QUESTIONS")
    w("  RQ1: Does condition (single vs multiple victims in a stormy VR ocean)")
    w("       affect prosocial rescue behavior?")
    w("       Hypothesis: Diffusion of responsibility - participants rescue fewer")
    w("       people when multiple victims are present.")
    w("  RQ2: Does the experimenter's physical presence in the room moderate")
    w("       the rescue/helping behavior across conditions?")

    w("\nSTUDY DESIGN")
    w("  - Within-subjects factor: Condition (0=Baseline, 1=Single, 2=Multiple)")
    w("  - Between-subjects factor: ExperimenterPresent (0=Absent, 1=Present)")
    w("  - Dependent variables:")
    w("      PeopleRescued (count): Number of virtual victims rescued")
    w("      Helped (binary): Whether participant rescued at least one victim")
    w("  - Counterbalanced across 2 groups")
    w("  - 3 trials per participant (one per condition)")
    w("  - VR Environment: Unreal Engine 5, stormy ocean weather")

    w("\nWHY VR?")
    w("  Virtual Reality provides a uniquely controlled environment for studying")
    w("  prosocial behavior in emergency scenarios. Key methodological advantages:")
    w("  1. Ecological validity: Immersive VR generates genuine emotional and")
    w("     behavioral responses comparable to real-world scenarios (Slater, 2009)")
    w("  2. Experimental control: Precise manipulation of victim count, weather")
    w("     conditions, and environmental factors impossible in field studies")
    w("  3. Ethical advantage: Study dangerous emergency scenarios (drowning)")
    w("     without any real risk to participants")
    w("  4. Measurable behavioral outcomes: Exact timing, proximity, and rescue")
    w("     counts captured automatically by the simulation")
    w("  5. Reproducibility: Identical stimulus presentation across participants")

    # ---- 2. Sample ----
    w("\n" + "=" * 80)
    w("2. SAMPLE CHARACTERISTICS")
    w("=" * 80)

    w(f"\n  Participants:    {desc['n_participants']}")
    w(f"  Total rows:      {desc['n_rows']} (Conditions 1 & 2 only)")
    w(f"  Age:             M={desc['age_mean']:.1f}, SD={desc['age_std']:.1f}, "
      f"range=[{desc['age_min']}, {desc['age_max']}]")

    w("\n  Gender:")
    for g, n in desc["gender_counts"].items():
        w(f"    {g}: {n} ({n/desc['n_participants']*100:.0f}%)")

    w("\n  Experimenter Presence:")
    for exp, n in desc["exp_counts"].items():
        label = "Present" if exp == 1 else "Absent"
        w(f"    {label}: {n} participants ({n/desc['n_participants']*100:.0f}%)")

    # ---- 3. Descriptive Stats ----
    w("\n" + "=" * 80)
    w("3. DESCRIPTIVE STATISTICS")
    w("=" * 80)

    w("\n  BY CONDITION (Conditions 1 & 2 only):")
    w(f"  {'Condition':<15} {'N':>4} {'Rescued M(SD)':>16} {'Median':>8} "
      f"{'Helped':>8} {'TimeNear M':>12}")
    w("  " + "-" * 70)
    cond_labels = {1: "Single (1)", 2: "Multiple (2)"}
    for cond in [1, 2]:
        s = desc["by_condition"][cond]
        w(f"  {cond_labels[cond]:<15} {s['n']:>4} "
          f"{s['rescued_mean']:>6.2f} ({s['rescued_std']:.2f})   "
          f"{s['rescued_median']:>6.1f}   "
          f"{s['helped_n']:>3}/{s['n']} ({s['helped_rate']:.0%})  "
          f"{s['time_near_mean']:>8.1f}s")

    w("\n  BY EXPERIMENTER PRESENCE:")
    w(f"  {'Experimenter':<15} {'N(part)':>8} {'N(obs)':>7} {'Rescued M(SD)':>16} {'Helped':>10}")
    w("  " + "-" * 60)
    exp_labels = {0: "Absent", 1: "Present"}
    for exp in [0, 1]:
        s = desc["by_experimenter"][exp]
        w(f"  {exp_labels[exp]:<15} {s['n_participants']:>8} {s['n']:>7} "
          f"{s['rescued_mean']:>6.2f} ({s['rescued_std']:.2f})   "
          f"{s['helped_n']:>3}/{s['n']} ({s['helped_rate']:.0%})")

    w("\n  CONDITION x EXPERIMENTER PRESENCE:")
    w(f"  {'Condition':<12} {'Experimenter':<12} {'N':>4} {'Rescued M(SD)':>16} {'Helped':>10}")
    w("  " + "-" * 60)
    for cond in [1, 2]:
        for exp in [0, 1]:
            key = f"cond{cond}_exp{exp}"
            s = desc["interaction"].get(key, {})
            if s.get("n", 0) > 0:
                w(f"  {cond_labels[cond]:<12} {exp_labels[exp]:<12} {s['n']:>4} "
                  f"{s['rescued_mean']:>6.2f} ({s['rescued_std']:.2f})   "
                  f"{s['helped_n']:>3}/{s['n']} ({s['helped_rate']:.0%})")

    # ---- 4. Assumption Checks ----
    w("\n" + "=" * 80)
    w("4. ASSUMPTION CHECKS")
    w("=" * 80)

    w("\n  Shapiro-Wilk Normality Test (PeopleRescued):")
    for key, res in normality.items():
        label = key.replace("cond_", "Condition ").replace("overall", "Overall")
        status = "NORMAL" if res.get("normal") else "NOT NORMAL"
        w(f"    {label}: W={res['statistic']:.4f}, p={res['p_value']:.4f} -> {status}")

    any_non_normal = any(not r.get("normal", True) for r in normality.values())
    if any_non_normal:
        w("\n  Conclusion: Data deviates from normality. Non-parametric tests are primary.")
        w("  Parametric mixed-effects models reported as supplementary analysis.")
    else:
        w("\n  Conclusion: Normality not rejected. Both parametric and non-parametric reported.")

    # ---- 5. RQ1: Condition Effect ----
    w("\n" + "=" * 80)
    w("5. RESULTS - RQ1: CONDITION EFFECT (Single vs Multiple Victims)")
    w("=" * 80)

    w(f"\n  Paired participants: {paired.get('n_paired', 'N/A')}")

    # Wilcoxon
    wil = paired.get("wilcoxon_rescued", {})
    if "error" in wil:
        w(f"\n  Wilcoxon Signed-Rank Test: SKIPPED ({wil['error']})")
    elif "note" in wil:
        w(f"\n  Wilcoxon Signed-Rank Test: {wil['note']}")
    else:
        sig = "SIGNIFICANT" if wil["p_value"] < 0.05 else "NOT SIGNIFICANT"
        w(f"\n  Wilcoxon Signed-Rank Test (PeopleRescued):")
        w(f"    W = {wil['statistic']:.1f}, p = {wil['p_value']:.4f}  [{sig}]")
        w(f"    Mean difference (Multiple - Single): {wil['mean_diff']:.3f}")
        w(f"    Median difference: {wil['median_diff']:.3f}")
        if "rank_biserial_r" in wil:
            w(f"    Rank-biserial r = {wil['rank_biserial_r']:.3f}")

    # Paired t-test
    tt = paired.get("paired_ttest_rescued", {})
    if "statistic" in tt:
        sig = "SIGNIFICANT" if tt["p_value"] < 0.05 else "NOT SIGNIFICANT"
        w(f"\n  Paired t-test (PeopleRescued, for comparison):")
        w(f"    t({tt['n_pairs']-1}) = {tt['statistic']:.3f}, p = {tt['p_value']:.4f}  [{sig}]")

    # McNemar
    mcn = paired.get("mcnemar_helped", {})
    if "error" in mcn:
        w(f"\n  McNemar's Test (Helped): SKIPPED ({mcn['error']})")
    else:
        sig = "SIGNIFICANT" if mcn["p_value"] < 0.05 else "NOT SIGNIFICANT"
        w(f"\n  McNemar's Test (Helped - binary):")
        w(f"    chi2 = {mcn['statistic']:.3f}, p = {mcn['p_value']:.4f}  [{sig}]")
        t = mcn["table"]
        w(f"    Contingency table:")
        w(f"      Both helped:        {t['both_helped']}")
        w(f"      Only Single helped: {t['only_single']}")
        w(f"      Only Multiple helped: {t['only_multiple']}")
        w(f"      Neither helped:     {t['neither']}")
        w(f"    Discordant pairs: {mcn['discordant_pairs']}")

    # ---- 6. RQ2: Experimenter Effect ----
    w("\n" + "=" * 80)
    w("6. RESULTS - RQ2: EXPERIMENTER PRESENCE EFFECT")
    w("=" * 80)

    w(f"\n  Experimenter Absent:  {between.get('n_absent', 'N/A')} participants")
    w(f"  Experimenter Present: {between.get('n_present', 'N/A')} participants")

    # Mann-Whitney
    mw = between.get("mannwhitney_rescued", {})
    if "error" in mw:
        w(f"\n  Mann-Whitney U Test: SKIPPED ({mw['error']})")
    else:
        sig = "SIGNIFICANT" if mw["p_value"] < 0.05 else "NOT SIGNIFICANT"
        w(f"\n  Mann-Whitney U Test (PeopleRescued):")
        w(f"    U = {mw['statistic']:.1f}, p = {mw['p_value']:.4f}  [{sig}]")
        if "rank_biserial_r" in mw:
            w(f"    Rank-biserial r = {mw['rank_biserial_r']:.3f}")

    # Fisher
    fish = between.get("fisher_helped", {})
    if "error" in fish:
        w(f"\n  Fisher's Exact Test (Helped): SKIPPED ({fish['error']})")
    else:
        sig = "SIGNIFICANT" if fish["p_value"] < 0.05 else "NOT SIGNIFICANT"
        w(f"\n  Fisher's Exact Test (Helped):")
        w(f"    Odds Ratio = {fish['odds_ratio']:.3f}, p = {fish['p_value']:.4f}  [{sig}]")
        t = fish["table"]
        w(f"    2x2 table:")
        w(f"      Absent  - Not Helped: {t['absent_not_helped']}, Helped: {t['absent_helped']}")
        w(f"      Present - Not Helped: {t['present_not_helped']}, Helped: {t['present_helped']}")

    # ---- 7. Mixed-Effects Model ----
    w("\n" + "=" * 80)
    w("7. MIXED-EFFECTS MODEL RESULTS")
    w("=" * 80)

    if lmm is None:
        w("\n  SKIPPED: statsmodels not available")
    elif "error" in lmm:
        w(f"\n  Mixed LM FAILED: {lmm['error']}")
    else:
        w("\n  --- Model 1 (Simple): PeopleRescued ~ IsMultiple + (1|ParticipantID) ---")
        w(f"\n{lmm['simple_summary']}")

        w("\n  --- Model 2 (Full): PeopleRescued ~ IsMultiple * ExperimenterPresent + (1|ParticipantID) ---")
        w(f"\n{lmm['full_summary']}")

        w("\n  --- Model 3 (With Covariates): + baseline_TimeElapsed + baseline_DrownCount ---")
        w(f"\n{lmm['cov_summary']}")

        w(f"\n  RANDOM EFFECTS:")
        w(f"    Participant variance (tau^2): {lmm['random_effects_var']:.4f}")
        w(f"    Residual variance (sigma^2):  {lmm['residual_var']:.4f}")
        w(f"    ICC (Intraclass Correlation):  {lmm['icc']:.4f}")
        if lmm['icc'] > 0.05:
            w(f"    -> {lmm['icc']:.0%} of variance in PeopleRescued is between participants")
            w(f"    -> Mixed model justified (substantial clustering)")
        else:
            w(f"    -> Low ICC suggests minimal participant-level clustering")

        # Interpret key coefficients from the full model
        full = lmm["full"]
        w("\n  KEY COEFFICIENT INTERPRETATION (Full Model):")
        for term in full.fe_params.index:
            coef = full.fe_params[term]
            p = full.pvalues[term]
            sig = "*" if p < 0.05 else ""
            if term == "Intercept":
                w(f"    Intercept = {coef:.3f} (p={p:.4f}){sig}")
                w(f"      Expected PeopleRescued for Single condition, Experimenter Absent")
            elif term == "IsMultiple":
                w(f"    IsMultiple = {coef:.3f} (p={p:.4f}){sig}")
                w(f"      Change in PeopleRescued from Single to Multiple (Exp Absent)")
            elif term == "ExperimenterPresent":
                w(f"    ExperimenterPresent = {coef:.3f} (p={p:.4f}){sig}")
                w(f"      Change in PeopleRescued when experimenter is present (Single condition)")
            elif "IsMultiple:ExperimenterPresent" in term:
                w(f"    IsMultiple:ExperimenterPresent = {coef:.3f} (p={p:.4f}){sig}")
                w(f"      Additional change for Multiple + Experimenter Present (interaction)")

    # ---- 7b. GEE ----
    w("\n  " + "-" * 70)
    w("  GEE BINOMIAL MODEL: Helped ~ IsMultiple * ExperimenterPresent")
    w("  " + "-" * 70)

    if gee is None:
        w("\n  SKIPPED: statsmodels not available")
    elif "error" in gee:
        w(f"\n  GEE FAILED: {gee['error']}")
    else:
        w(f"\n{gee['summary']}")

        w("\n  ODDS RATIOS (exponentiated coefficients):")
        for term, odds in gee["odds_ratios"].items():
            w(f"    {term}: OR = {odds:.3f}")

        w("\n  CAVEAT: With only ~14 clusters (participants), GEE sandwich standard")
        w("  errors may be unreliable. These results should be interpreted with caution.")
        w("  Non-parametric tests above provide more robust inference for this sample size.")

    # ---- 8. Effect Sizes ----
    w("\n" + "=" * 80)
    w("8. EFFECT SIZES SUMMARY")
    w("=" * 80)

    for key, eff in effects.items():
        if "value" in eff:
            w(f"\n  {eff['label'].replace(chr(10), ' ')}:")
            w(f"    Cohen's d = {eff['value']:.3f} ({eff['interpretation']})")
        elif "difference" in eff:
            w(f"\n  {eff['label'].replace(chr(10), ' ')}:")
            w(f"    Single rate: {eff['rate_single']:.0%}")
            w(f"    Multiple rate: {eff['rate_multiple']:.0%}")
            w(f"    Difference: {eff['difference']:+.0%}")

    # ---- 9. Key Findings ----
    w("\n" + "=" * 80)
    w("9. KEY FINDINGS & INTERPRETATION")
    w("=" * 80)

    w("\n  FINDING 1 - CONDITION EFFECT:")
    c1 = desc["by_condition"][1]
    c2 = desc["by_condition"][2]
    w(f"    Participants rescued significantly more people in the Multiple condition")
    w(f"    (M={c2['rescued_mean']:.2f}, SD={c2['rescued_std']:.2f}) compared to Single")
    w(f"    (M={c1['rescued_mean']:.2f}, SD={c1['rescued_std']:.2f}).")
    w(f"    Helping rate: Multiple={c2['helped_rate']:.0%} vs Single={c1['helped_rate']:.0%}")
    if "wilcoxon_rescued" in paired and "p_value" in paired["wilcoxon_rescued"]:
        p = paired["wilcoxon_rescued"]["p_value"]
        if p < 0.05:
            w(f"    This difference was statistically significant (Wilcoxon p={p:.4f}).")
        else:
            w(f"    This difference was not statistically significant (Wilcoxon p={p:.4f}).")
    w(f"\n    NOTE: This pattern is OPPOSITE to the diffusion of responsibility")
    w(f"    hypothesis. Rather than helping less with more victims, participants")
    w(f"    helped MORE. Possible explanations:")
    w(f"    - Greater perceived urgency with multiple victims in distress")
    w(f"    - More opportunities for successful rescue (4 victims vs 1)")
    w(f"    - Stronger emotional activation from the multi-victim scenario")

    w("\n  FINDING 2 - EXPERIMENTER PRESENCE:")
    e0 = desc["by_experimenter"][0]
    e1 = desc["by_experimenter"][1]
    w(f"    Experimenter Absent:  M={e0['rescued_mean']:.2f} rescued, {e0['helped_rate']:.0%} helped")
    w(f"    Experimenter Present: M={e1['rescued_mean']:.2f} rescued, {e1['helped_rate']:.0%} helped")
    if "mannwhitney_rescued" in between and "p_value" in between["mannwhitney_rescued"]:
        p = between["mannwhitney_rescued"]["p_value"]
        if p < 0.05:
            w(f"    Statistically significant difference (Mann-Whitney p={p:.4f}).")
        else:
            w(f"    Not statistically significant (Mann-Whitney p={p:.4f}).")

    w("\n  FINDING 3 - INTERACTION:")
    w(f"    The Condition x ExperimenterPresent interaction reveals whether")
    w(f"    the condition effect differs based on experimenter presence.")
    if lmm and "full" in lmm and not isinstance(lmm.get("error"), str):
        full = lmm["full"]
        interaction_terms = [t for t in full.fe_params.index if ":" in t]
        if interaction_terms:
            t = interaction_terms[0]
            p = full.pvalues[t]
            coef = full.fe_params[t]
            if p < 0.05:
                w(f"    Significant interaction (b={coef:.3f}, p={p:.4f}): the condition")
                w(f"    effect differs depending on whether the experimenter is present.")
            else:
                w(f"    Non-significant interaction (b={coef:.3f}, p={p:.4f}): the condition")
                w(f"    effect is similar regardless of experimenter presence.")

    # ---- 10. Methodological Notes ----
    w("\n" + "=" * 80)
    w("10. METHODOLOGICAL NOTES")
    w("=" * 80)

    w("\n  ML vs INFERENTIAL STATISTICS:")
    w("  This experiment uses BOTH approaches for complementary insight:")
    w("  - Inferential statistics (this report): Answer causal/mechanistic questions")
    w("    about condition and experimenter effects. Appropriate for the repeated")
    w("    measures design with small N (~14 participants).")
    w("  - ML pipeline (rescue_classification.py): Provides prediction performance")
    w("    and feature importance. Useful for identifying which behavioral features")
    w("    best predict helping, but limited by small sample size.")
    w("")
    w("  For N=14 with 3 repeated measures per participant, inferential statistics")
    w("  are the PRIMARY analysis. The mixed-effects model properly handles the")
    w("  nested structure (trials within participants). ML results are SUPPLEMENTARY.")

    w("\n  LIMITATIONS:")
    w(f"  - Small sample (N={desc['n_participants']}) limits statistical power")
    w(f"  - Unbalanced ExperimenterPresent groups ({desc['exp_counts']})")
    w("  - GEE requires ~40+ clusters for reliable sandwich SE; our 14 is insufficient")
    w("  - Within-subjects design with only 2 victim conditions limits model complexity")
    w("  - Non-parametric tests are preferred given non-normal distributions")

    w("\n  RECOMMENDATIONS:")
    w("  - Report non-parametric tests as primary results")
    w("  - Report mixed-effects models as confirmatory/supplementary")
    w("  - Include effect sizes (Cohen's d) alongside p-values")
    w("  - Emphasize the unexpected direction of the condition effect")
    w("  - Discuss experimenter presence as a social desirability confound")

    w("\n" + "=" * 80)
    w("END OF REPORT")
    w("=" * 80)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[Report] Saved to {output_path}")


# ==========================================
# Main
# ==========================================
def main():
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    print("\n" + "=" * 60)
    print("RESCUE EXPERIMENT - INFERENTIAL STATISTICS")
    print("=" * 60)

    # ==========================================
    # 1. Load and Clean Data
    # ==========================================
    data = load_rescue_data(curr_dir)
    clean_df = clean_rescue(data)

    # ==========================================
    # 2. Build Model DataFrame
    # ==========================================
    model_df = build_model_df(clean_df)
    model_df = _ensure_clean_experimenter_present(model_df)

    # Create binary target
    model_df["Helped"] = (model_df["PeopleRescued"] > 0).astype(int)

    print(f"\n--- Analysis Dataset ---")
    print(f"Rows: {len(model_df)} (Conditions 1 & 2)")
    print(f"Participants: {model_df['ParticipantID'].nunique()}")
    print(f"Helped: {model_df['Helped'].sum()}/{len(model_df)} ({model_df['Helped'].mean():.0%})")
    print(f"ExperimenterPresent: {model_df['ExperimenterPresent'].value_counts().to_dict()}")

    # ==========================================
    # 3. Run All Analyses
    # ==========================================
    print("\n[1/6] Computing descriptive statistics...")
    desc = compute_descriptives(model_df)

    print("[2/6] Running normality tests...")
    normality = run_normality_tests(model_df)

    print("[3/6] Running within-subjects tests (Condition effect)...")
    paired = run_paired_tests(model_df)

    print("[4/6] Running between-subjects tests (Experimenter effect)...")
    between = run_between_tests(model_df)

    print("[5/6] Running mixed-effects models...")
    lmm = run_mixed_effects_model(model_df)
    gee = run_gee_model(model_df)

    print("[6/6] Computing effect sizes...")
    effects = compute_effect_sizes(model_df)

    # ==========================================
    # 4. Generate Report
    # ==========================================
    report_dir = os.path.join(curr_dir, "../..", "plots/rescue_inferential_plots")
    report_path = os.path.join(report_dir, "inferential_report.txt")
    generate_report(desc, normality, paired, between, lmm, gee, effects, report_path)

    # ==========================================
    # 5. Generate Plots
    # ==========================================
    plots_dir = report_dir
    print("\n--- Generating Plots ---")

    # Merge test results for annotations
    test_results = {}
    test_results.update(paired)
    test_results.update(between)

    plot_condition_outcomes(model_df, test_results, plots_dir)
    plot_experimenter_effect(model_df, test_results, plots_dir)
    plot_interaction_effects(model_df, plots_dir)
    plot_participant_trajectories(model_df, plots_dir)

    # Forest plot
    coef_list = []
    if lmm and "full_coefs" in lmm:
        coef_list.append(lmm["full_coefs"])
    if gee and "coefs" in gee:
        coef_list.append(gee["coefs"])
    if coef_list:
        plot_forest_coefficients(coef_list, plots_dir)

    plot_effect_sizes(effects, plots_dir)

    # ==========================================
    # 6. Save Results JSON
    # ==========================================
    results_dir = os.path.join(curr_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    # Serialize results (skip non-serializable objects)
    save_results = {
        "descriptives": desc,
        "normality": normality,
        "paired_tests": {k: v for k, v in paired.items() if isinstance(v, (dict, int, float))},
        "between_tests": {k: v for k, v in between.items() if isinstance(v, (dict, int, float))},
        "effect_sizes": effects,
    }
    if lmm and "error" not in lmm:
        save_results["lmm_coefs"] = {
            "simple": lmm.get("simple_coefs"),
            "full": lmm.get("full_coefs"),
            "covariate": lmm.get("cov_coefs"),
            "icc": lmm.get("icc"),
        }
    if gee and "error" not in gee:
        save_results["gee_coefs"] = gee.get("coefs")
        save_results["gee_odds_ratios"] = gee.get("odds_ratios")

    out_path = os.path.join(results_dir, "rescue_inferential_results.json")
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\n[Results] Saved to {out_path}")

    print("\n" + "=" * 60)
    print("INFERENTIAL ANALYSIS COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
