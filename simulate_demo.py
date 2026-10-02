"""Headless demo: a simulated user who loves Lo-fi (Relaxed mood). Proves Q-values really change."""
import numpy as np
from rl.q_learning_agent import GENRES, REWARDS, QLearningAgent
from utils.data_loader import load_songs
from utils.recommender import Recommender

LIKES = {"Lo-fi": "like", "Classical": "complete"}          # everything else: dislike/skip
N, MOOD = 120, "Relaxed"


def run(seed):
    rng = np.random.default_rng(seed)
    agent, rec = QLearningAgent(seed=seed), Recommender(load_songs("data/songs.csv"), seed=seed)
    agent.seed_preference(MOOD, "Lo-fi")
    fb, recent, rewards, genres = "Start", [], [], []
    for _ in range(N):
        s = (MOOD, fb)
        a, _p = agent.choose_action(s)
        song = rec.pick_song(GENRES[a], MOOD, recent[-5:]); recent.append(song["song_id"])
        kind = LIKES.get(GENRES[a]) or ("skip" if rng.random() < .5 else "dislike")
        fb = {"like": "Liked", "complete": "Completed", "skip": "Skipped", "dislike": "Disliked"}[kind]
        agent.update(s, a, REWARDS[kind], (MOOD, fb)); agent.decay_epsilon()
        rewards.append(REWARDS[kind]); genres.append(GENRES[a])
    return agent, rewards, genres


if __name__ == "__main__":
    first, last, lofi_last = [], [], []
    for seed in range(20):
        agent, r, g = run(seed)
        first.append(np.mean(r[:20])); last.append(np.mean(r[-20:]))
        lofi_last.append(np.mean([x == "Lo-fi" for x in g[-20:]]))
    print(f"Avg reward, first 20 interactions : {np.mean(first):6.2f}")
    print(f"Avg reward, last 20 interactions  : {np.mean(last):6.2f}")
    print(f"Lo-fi share of last 20 picks      : {np.mean(lofi_last):6.1%}  (random baseline 14.3%)")
    agent, r, g = run(0)
    print("\nMean Q per genre (mood=Relaxed), seed 0:")
    for name, v in zip(GENRES, agent.mood_q(MOOD)):
        print(f"  {name:10s} {v:7.2f}")
