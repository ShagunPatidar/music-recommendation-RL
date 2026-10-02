"""Maps the agent's action (a genre) to a concrete song from the dataset."""
import numpy as np


class Recommender:
    def __init__(self, songs_df, seed=None):
        self.df = songs_df
        self.rng = np.random.default_rng(seed)

    def pick_song(self, genre, mood, recent_ids):
        """Prefer a song of this genre that matches the mood and was not shown recently."""
        g = self.df[self.df["genre"] == genre]
        for pool in (g[g["mood"] == mood], g):
            fresh = pool[~pool["song_id"].isin(recent_ids)]
            if len(fresh):
                return fresh.iloc[int(self.rng.integers(len(fresh)))].to_dict()
        return g.iloc[int(self.rng.integers(len(g)))].to_dict()   # everything was recent
