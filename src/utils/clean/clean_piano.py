import pandas as pd
import numpy as np


def clean_piano(data):
    """
    Clean piano features dataset for downstream ML.

    Drops:
    - pdc_lr_brilliance, pdc_rl_brilliance, pdc_lr_presence, pdc_rl_presence:
      All-zero columns — these frequency bands are above the Nyquist frequency
      after downsampling, so PDC cannot be estimated there.
    - label_name: String column redundant with the integer 'label' target.
    - Any remaining zero-variance columns detected automatically.

    Keeps:
    - song_name:   Required for Leave-One-Song-Out CV grouping (dropped at experiment time).
    - clip_index:  Identifier column (dropped at experiment time).
    - label:       Binary target  (0 = emotional, 1 = happy).
    - All remaining numeric feature columns (~109 features after cleaning).

    Note on rms_mean:
      After RMS normalization rms_mean shows only a 2.7% class difference.
      It is kept here but intentionally excluded from the TOP_20_FEATURES
      predictor set defined in the experiment file.
    """
    print("\n" + "=" * 60)
    print("CLEANING PIANO FEATURES DATASET")
    print("=" * 60)

    df = data.copy()

    # ==========================================
    # 1. Null Check
    # ==========================================
    print("\n--- Null Value Check ---")
    total_nulls = df.isnull().sum().sum()
    if total_nulls == 0:
        print("No missing values found")
    else:
        print(f"Found {total_nulls} null values:")
        print(df.isnull().sum()[df.isnull().sum() > 0])
        df = df.dropna()
        print(f"Dropped rows with nulls. New shape: {df.shape}")

    # ==========================================
    # 2. Drop Known Zero-Variance PDC Columns
    # ==========================================
    zero_var_cols = [
        "pdc_lr_brilliance",
        "pdc_rl_brilliance",
        "pdc_lr_presence",
        "pdc_rl_presence",
    ]
    existing_zero_var = [c for c in zero_var_cols if c in df.columns]
    df = df.drop(columns=existing_zero_var)
    print(f"\nDropped zero-variance PDC columns (above Nyquist): {existing_zero_var}")

    # ==========================================
    # 3. Drop String Redundancy
    # ==========================================
    drop_str = ["label_name"]
    existing_str = [c for c in drop_str if c in df.columns]
    df = df.drop(columns=existing_str)
    print(f"Dropped string-redundant columns: {existing_str}")

    # ==========================================
    # 4. Auto-detect Any Remaining Zero-Variance Numeric Features
    # ==========================================
    id_cols = {"song_name", "clip_index", "label"}
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in id_cols
    ]
    variances = df[numeric_cols].var()
    extra_zero_var = variances[variances == 0].index.tolist()
    if extra_zero_var:
        df = df.drop(columns=extra_zero_var)
        print(f"\nDropped additional zero-variance features: {extra_zero_var}")
    else:
        print("\nNo additional zero-variance features found")

    # ==========================================
    # 5. Dataset Summary
    # ==========================================
    feature_cols = [c for c in df.columns if c not in id_cols]
    print(f"\n--- Cleaned Dataset Overview ---")
    print(f"Total rows:         {len(df)}")
    print(f"Songs:              {df['song_name'].nunique()}")
    print(f"Clips per class:    {dict(df['label'].value_counts().sort_index())}")
    print(f"Total columns:      {len(df.columns)}")
    print(f"Feature columns:    {len(feature_cols)}")
    print(f"Identifier columns: song_name, clip_index, label")

    return df
