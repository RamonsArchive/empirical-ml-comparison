import pandas as pd
import os
import glob


def load_rescue_data(curr_dir):
    """
    Load all individual participant CSVs from RescueSets/ and concatenate.
    Each CSV is one participant with 3 trials (within-subjects design).
    """
    data_dir = os.path.join(curr_dir, "../../datasets/RescueSets")
    csv_files = glob.glob(os.path.join(data_dir, "*_data.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    dfs = []
    for f in sorted(csv_files):
        df = pd.read_csv(f, encoding="utf-8-sig")
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    n_participants = data["ParticipantID"].nunique()
    print(f"[load_rescue] Loaded {len(data)} rows from {len(csv_files)} files ({n_participants} participants)")
    print(f"[load_rescue] Columns: {list(data.columns)}")

    return data
