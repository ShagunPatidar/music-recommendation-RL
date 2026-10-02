import os
import pandas as pd

REQUIRED = ["song_id", "song", "artist", "genre", "mood", "duration"]
HISTORY_COLS = ["Time", "Song", "Artist", "Genre", "Mood", "Action", "Reward",
                "UserMood", "Policy", "song_id"]


def load_songs(path):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"songs.csv is missing columns: {missing}")
    return df


def load_history(path):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            if all(c in df.columns for c in HISTORY_COLS):
                return df[HISTORY_COLS].to_dict("records")
        except Exception:
            pass
    return []


def save_history(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(records, columns=HISTORY_COLS).to_csv(path, index=False)
