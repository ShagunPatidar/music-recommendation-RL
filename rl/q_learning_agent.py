"""Tabular Q-learning agent (NumPy only, no RL library)."""
import json
import os
import numpy as np

# ----------------------------------------------------------------- config
MOODS = ["Happy", "Sad", "Relaxed", "Energetic", "Focus", "Romantic"]
GENRES = ["Pop", "Rock", "Lo-fi", "Classical", "Hip-Hop", "EDM", "Bollywood"]  # = actions
# Second part of the state: how the user reacted to the PREVIOUS recommendation.
FEEDBACK_STATES = ["Start", "Liked", "Completed", "Skipped", "Disliked"]

# Reward values (configurable)
REWARDS = {"like": 10, "complete": 5, "skip": -3, "dislike": -10, "repeat": -2}

# Hyper-parameters (configurable)
ALPHA = 0.3            # learning rate
GAMMA = 0.6            # discount factor
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.9    # epsilon <- max(EPSILON_MIN, epsilon * EPSILON_DECAY) per interaction


class QLearningAgent:
    """State = (mood, last_feedback); Action = genre index."""

    def __init__(self, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON_START,
                 epsilon_min=EPSILON_MIN, epsilon_decay=EPSILON_DECAY, seed=None):
        self.alpha, self.gamma = alpha, gamma
        self.epsilon, self.epsilon_min, self.epsilon_decay = epsilon, epsilon_min, epsilon_decay
        self.rng = np.random.default_rng(seed)
        self.states = [(m, f) for m in MOODS for f in FEEDBACK_STATES]
        self.state_index = {s: i for i, s in enumerate(self.states)}
        self.n_actions = len(GENRES)
        self.reset_table()

    # ------------------------------------------------------------ table
    def reset_table(self):
        self.q = np.zeros((len(self.states), self.n_actions))      # the Q-table
        self.visits = np.zeros((len(self.states), self.n_actions), dtype=int)
        self.steps = 0
        self.epsilon = EPSILON_START

    def seed_preference(self, mood, genre, bonus=2.0):
        """Optional prior: the user's declared favourite genre starts with a small Q-bonus."""
        a = GENRES.index(genre)
        for fb in FEEDBACK_STATES:
            i = self.state_index[(mood, fb)]
            self.q[i, a] = max(self.q[i, a], bonus)

    # ----------------------------------------------------------- policy
    def choose_action(self, state):
        """Epsilon-greedy. Returns (action_index, 'Explore' | 'Exploit')."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions)), "Explore"
        row = self.q[self.state_index[state]]
        best = np.flatnonzero(row == row.max())          # random tie-break
        return int(self.rng.choice(best)), "Exploit"

    # ----------------------------------------------------------- learning
    def update(self, state, action, reward, next_state):
        """Q(s,a) <- Q(s,a) + alpha * [R + gamma * max_a' Q(s',a') - Q(s,a)]"""
        s, s2 = self.state_index[state], self.state_index[next_state]
        old = self.q[s, action]
        max_next = self.q[s2].max()
        td_target = reward + self.gamma * max_next
        new = old + self.alpha * (td_target - old)
        self.q[s, action] = new
        self.visits[s, action] += 1
        return {"old": float(old), "new": float(new), "max_next": float(max_next),
                "td_target": float(td_target), "reward": reward}

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.steps += 1

    # ----------------------------------------------------------- helpers
    def q_dataframe(self, visited_only=False):
        import pandas as pd
        idx = [f"{m} | {f}" for m, f in self.states]
        df = pd.DataFrame(self.q, index=idx, columns=GENRES)
        if visited_only:
            df = df[self.visits.sum(axis=1) > 0]
        return df

    def mood_q(self, mood):
        """Mean Q-value per genre over all states of one mood."""
        rows = [self.state_index[(mood, f)] for f in FEEDBACK_STATES]
        return self.q[rows].mean(axis=0)

    # -------------------------------------------------------- persistence
    def save(self, path):
        data = {"alpha": self.alpha, "gamma": self.gamma, "epsilon": self.epsilon,
                "steps": self.steps,
                "q": {f"{m}|{f}": {g: float(self.q[i, j]) for j, g in enumerate(GENRES)}
                      for i, (m, f) in enumerate(self.states)},
                "visits": {f"{m}|{f}": {g: int(self.visits[i, j]) for j, g in enumerate(GENRES)}
                           for i, (m, f) in enumerate(self.states)}}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, path)

    def load(self, path):
        """Load a saved table; silently keep a fresh table if file is missing/invalid."""
        try:
            with open(path) as fh:
                data = json.load(fh)
            q = np.zeros_like(self.q)
            v = np.zeros_like(self.visits)
            for i, (m, f) in enumerate(self.states):
                for j, g in enumerate(GENRES):
                    q[i, j] = data["q"][f"{m}|{f}"][g]
                    v[i, j] = data.get("visits", {}).get(f"{m}|{f}", {}).get(g, 0)
            self.q, self.visits = q, v
            self.epsilon = float(data["epsilon"])
            self.steps = int(data["steps"])
            return True
        except (OSError, ValueError, KeyError, TypeError):
            return False
