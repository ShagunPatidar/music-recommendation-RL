# 🎵 Adaptive Music Recommendation Agent Using Q-Learning

## 1. Project Title
**Adaptive Music Recommendation Agent Using Q-Learning**

## 2. Project Description
A software-only Reinforcement Learning project. A tabular Q-learning agent (written from scratch with NumPy) recommends songs from a local CSV. The user answers with **Like / Listened / Skip / Dislike**; each answer becomes a reward and updates the Q-table, so recommendations adapt to the user's taste. Everything runs offline in a Streamlit web app. No audio, no APIs, no keys.

## 3. Problem Statement
Traditional music recommendation systems often rely primarily on predefined preferences, similarity measures, or historical data. Such systems may not continuously adapt to a user's changing preferences during interaction. The proposed system uses Reinforcement Learning to create an adaptive music recommendation agent that learns from user feedback. By treating music recommendation as a sequential decision-making problem, the agent updates its Q-values based on user interactions and gradually improves personalized recommendations.

## 4. Objectives
1. Develop a software-based music recommendation system.
2. Apply Reinforcement Learning to personalization.
3. Implement Q-Learning from scratch.
4. Learn user preferences from feedback.
5. Use rewards to improve future recommendations.
6. Demonstrate exploration and exploitation.
7. Provide analytics showing the learning process.
8. Build a simple and user-friendly web interface.

## 5. Features
- Mood + optional favourite genre selection
- Song card (name, artist, genre, mood, duration) with Like / Listened / Skip / Dislike buttons
- Modern dark UI with stat cards, animated now-playing card, colour-coded feedback buttons and a live "learned taste" bar chart
- **Auto-play:** each recommended song plays automatically in an embedded YouTube player (song looked up with `yt-dlp` search, cached in `data/youtube_ids.json`; no API key, nothing downloaded). Needs internet; switch off with the sidebar toggle "Auto-play in app" to run fully offline. A "🔎 Open on YouTube" button is the fallback.
- Live "what the agent just learned" panel showing the exact Q-update arithmetic
- Q-table heat-map, reward curve, likes/recommendations by genre, Q-value comparison
- Metrics: like rate, skip rate, average reward, reward improvement, diversity, favourite genre
- History table with CSV download
- Q-table saved to `models/q_table.json`, history to `data/history.csv` (survives restarts)

## 6. Reinforcement Learning Explanation
The agent interacts with the user (the environment) in a loop: observe state → choose action (genre) → receive reward (feedback) → move to next state → update Q-value. Over time it learns which genre earns the highest long-term reward in each state.

## 7. State
`state = (mood, last_feedback)`
- mood ∈ {Happy, Sad, Relaxed, Energetic, Focus, Romantic} (6)
- last_feedback ∈ {Start, Liked, Completed, Skipped, Disliked} (5) – the user's reaction to the previous song

→ **30 states**. Including recent feedback makes the state sequential: the best next genre can differ after a Like versus after a Dislike. The *initial favourite genre* is not part of the state; it only gives that genre a small starting Q-bonus (+2) for the chosen mood.

## 8. Actions
7 actions = the genre to recommend: Pop, Rock, Lo-fi, Classical, Hip-Hop, EDM, Bollywood. The recommender then picks a concrete song of that genre (preferring songs matching the mood, avoiding the last 5 songs).

## 9. Rewards
| Feedback | Reward |
|---|---|
| ❤️ Like | +10 |
| ✅ Listened fully | +5 |
| ⏭️ Skip | −3 |
| 👎 Dislike | −10 |
| Repeated recent song (extra) | −2 |

Configurable in `rl/q_learning_agent.py` (`REWARDS`). The repeat penalty only triggers if every song in a genre was recently shown, so it is rare with this dataset.

## 10. Q-Learning Formula
```
Q(s,a) ← Q(s,a) + α [ R + γ · max_a' Q(s',a') − Q(s,a) ]
```
α = 0.3 (learning rate), γ = 0.6 (discount factor). Exploration: ε-greedy, ε = 1.0 → ×0.9 per interaction → minimum 0.1.

## 11. Architecture
```
User → Streamlit UI → Preference Manager → Recommendation Engine → Q-Learning Agent
     → Q-Table → Song Dataset → Recommended Song → User Feedback
     → Reward Calculation → Q-Table Update → Improved Recommendation (loop)
```

## 12. Technologies
Python 3.9+, Streamlit, Pandas, NumPy, Matplotlib; yt-dlp (only for optional song lookup/auto-play). No deep learning, no API keys.

## 13. Project Structure
```
music-recommendation-rl/
├── app.py                    # Streamlit UI (5 pages) + feedback loop
├── simulate_demo.py          # headless demo proving learning (no UI)
├── requirements.txt
├── README.md
├── REPORT_AND_VIVA.md        # report chapters, pseudocode, viva Q&A
├── data/songs.csv            # 56 songs (metadata only)
├── rl/q_learning_agent.py    # Q-table, ε-greedy, Q-update, save/load
├── utils/data_loader.py      # CSV + history loading/saving
├── utils/recommender.py      # genre (action) -> concrete song
├── utils/youtube.py          # optional: finds YouTube video id for auto-play
├── .streamlit/config.toml    # dark theme
├── components/charts.py      # Matplotlib charts
└── models/q_table.json       # persisted learned Q-values
```

## 14. Installation
```bash
cd music-recommendation-rl
python -m venv venv
venv\Scripts\activate          # Windows   (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
```

## 15. How to Run
```bash
streamlit run app.py           # opens http://localhost:8501
python simulate_demo.py        # optional: command-line proof of learning
```
To start from scratch use *Reset learning* in the sidebar (or delete `models/q_table.json` content → `{}` and `data/history.csv`).

## 16. Example Usage (viva demo)
1. Dashboard → Mood **Relaxed**, Genre **Lo-fi** → *Start Recommendation*.
2. 🎧 Recommendation → click **Like** on Lo-fi songs, **Skip/Dislike** on others. Watch the learning panel: `Q_new = old + 0.3 × [...]`.
3. 📊 Learning Analytics → Q-table: Lo-fi column grows; "Mean Q-value per Genre" shows Lo-fi highest.
4. After ~15–20 interactions ε ≈ 0.1–0.2, so the badge mostly says 🎯 Exploiting and Lo-fi dominates.
5. Show the reward curve and history.

Tip: ε starts at 1.0 (fully random), so the first ~10 recommendations are exploratory by design — say this out loud in the viva.

**Simulation result** (`python simulate_demo.py`, 20 runs × 120 steps, simulated Lo-fi lover): Lo-fi made up ~87 % of the last 20 recommendations versus a 14.3 % random baseline; average reward rose from ≈4.8 to ≈8.3. This is a simulation with a scripted user, not a real-world accuracy claim.

## 17. Screenshots
- `[Dashboard screenshot]`
- `[Recommendation page screenshot]`
- `[Q-table / analytics screenshot]`
- `[History screenshot]`

## 18. Future Scope
Spotify/YouTube integration, larger datasets, user accounts, collaborative filtering, Deep Q-Networks, context-aware recommendations, real listening-duration feedback, mood detection, voice interaction, mobile app, cloud deployment, multi-user personalization.

## 19. Conclusion
The project shows a genuine RL loop — states, actions, rewards, a Q-table, ε-greedy exploration and the Bellman-style Q-update — applied to music personalization, with a UI that makes the learning visible and easy to explain.

## 20. Live Demo
https://music-recommendation-rl.onrender.com
