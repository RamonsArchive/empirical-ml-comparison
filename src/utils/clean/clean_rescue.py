import pandas as pd
import numpy as np


def clean_rescue(data):
    """
    Clean rescue experiment data following multicollinearity recommendations.

    Drops:
    - DidDrown: fully derivable from DrownCount (DrownCount > 0)
    - HealthRemaining: near-perfect inverse correlation with TimeElapsed
    - AttemptedRescue: highly correlated with TimeNearVictim > 0

    Keeps ParticipantID for within-subjects aggregation (dropped at experiment time).
    Encodes Gender as categorical for OneHotEncoder in the pipeline.
    Converts boolean columns to int.
    """
    print("\n" + "=" * 60)
    print("CLEANING RESCUE EXPERIMENT DATA")
    print("=" * 60)

    df = data.copy()

    # ==========================================
    # 1. Null Check
    # ==========================================
    print("\n--- Null Value Check ---")
    total_nulls = df.isnull().sum().sum()
    if total_nulls == 0:
        print("No missing values")
    else:
        print(f"Found {total_nulls} null values:")
        print(df.isnull().sum()[df.isnull().sum() > 0])
        df = df.dropna()
        print(f"Dropped rows with nulls. New shape: {df.shape}")

    # ==========================================
    # 2. Drop Redundant Columns (Multicollinearity)
    # ==========================================
    drop_cols = ["DidDrown", "HealthRemaining", "AttemptedRescue"]
    existing_drops = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing_drops)
    print(f"\nDropped redundant columns: {existing_drops}")
    print("  DidDrown -> derivable from DrownCount")
    print("  HealthRemaining -> near-perfect inverse of TimeElapsed")
    print("  AttemptedRescue -> captured by TimeNearVictim > 0")

    # ==========================================
    # 3. Convert Boolean Columns to Int
    # ==========================================
    for col in df.columns:
        if df[col].dtype == object and set(df[col].unique()).issubset({"true", "false", True, False}):
            df[col] = df[col].map({"true": 1, "false": 0, True: 1, False: 0}).astype(int)

    # ==========================================
    # 4. Data Overview
    # ==========================================
    print(f"\n--- Dataset Overview ---")
    print(f"Total rows: {len(df)}")
    print(f"Participants: {df['ParticipantID'].nunique()}")
    print(f"Trials per participant: {len(df) // df['ParticipantID'].nunique()}")
    print(f"Conditions: {sorted(df['Condition'].unique())}")
    print(f"Columns: {list(df.columns)}")

    return df


def build_model_df(clean_df):
    """
    Build the modeling DataFrame: Conditions 1 & 2 only, with Condition 0
    baseline metrics merged as per-participant covariates (Step 3).

    Condition 0 (Empty ocean) has no victims so PeopleRescued=0 by design.
    We extract each participant's Condition 0 TimeElapsed and DrownCount
    as navigation skill covariates to control for baseline ability.

    Returns:
        model_df: DataFrame with Conditions 1 & 2 rows + baseline covariates
    """
    # Step 3: Extract baseline (Condition 0) navigation skill per participant
    cond0 = clean_df[clean_df["Condition"] == 0][["ParticipantID", "TimeElapsed", "DrownCount"]]
    cond0 = cond0.rename(columns={
        "TimeElapsed": "baseline_TimeElapsed",
        "DrownCount": "baseline_DrownCount",
    })

    # Filter to victim conditions only (Step 2)
    model_df = clean_df[clean_df["Condition"] > 0].copy()

    # Merge baseline covariates onto each participant's victim-condition rows
    model_df = model_df.merge(cond0, on="ParticipantID", how="left")

    # Fill NaN baselines with 0 for participants missing Condition 0
    model_df["baseline_TimeElapsed"] = model_df["baseline_TimeElapsed"].fillna(0)
    model_df["baseline_DrownCount"] = model_df["baseline_DrownCount"].fillna(0)

    print(f"\n--- Model DataFrame (Conditions 1 & 2 + baseline covariates) ---")
    print(f"Rows: {len(model_df)}")
    print(f"Participants: {model_df['ParticipantID'].nunique()}")
    print(f"Columns: {list(model_df.columns)}")

    return model_df
