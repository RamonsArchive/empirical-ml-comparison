import pandas as pd
import os


def load_piano_data(curr_dir):
    """
    Load piano_features.csv from datasets/PianoSets/.

    Dataset: 54 rows (18 songs x 3 clips each), 117 columns.
    Labels:  0 = emotional/melancholic, 1 = upbeat/happy
    Groups:  song_name — 3 clips per song; must not be split across train/test
             (use Leave-One-Song-Out CV to avoid data leakage).
    """
    data_dir = os.path.join(curr_dir, "../../datasets/PianoSets")
    csv_path = os.path.join(data_dir, "piano_features.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"piano_features.csv not found at {csv_path}")

    df = pd.read_csv(csv_path)

    n_songs = df["song_name"].nunique()
    n_emotional = (df["label"] == 0).sum()
    n_happy = (df["label"] == 1).sum()

    print(f"[load_piano] Loaded {len(df)} rows x {len(df.columns)} columns")
    print(f"[load_piano] Songs: {n_songs} ({n_songs // 2} emotional, {n_songs // 2} happy)")
    print(f"[load_piano] Clips: {n_emotional} emotional (label=0), {n_happy} happy (label=1)")
    print(f"[load_piano] Columns (first 10): {list(df.columns)[:10]} ...")

    return df
