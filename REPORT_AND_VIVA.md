# Adaptive Music Recommendation Agent Using Q-Learning — Report Content

## Architecture Diagram
```
 ┌──────┐   ┌──────────────┐   ┌───────────────────┐   ┌──────────────────────┐   ┌──────────────────┐
 │ User │──▶│ Streamlit UI │──▶│ Preference Manager│──▶│ Recommendation Engine│──▶│ Q-Learning Agent │
 └──────┘   └──────────────┘   │ (mood, genre)     │   │ (utils/recommender)  │   │ (ε-greedy)       │
     ▲                         └───────────────────┘   └──────────────────────┘   └────────┬─────────┘
     │                                                                                     ▼
 ┌───┴────────────┐   ┌─────────────────┐   ┌──────────────┐   ┌──────────────┐   ┌────────────────┐
 │ User Feedback  │◀──│ Recommended Song│◀──│ Song Dataset │◀──│   Q-Table    │◀──│ action = genre │
 │ Like/Skip/Dis. │   └─────────────────┘   │ (songs.csv)  │   │ 30 × 7       │   └────────────────┘
 └───┬────────────┘                         └──────────────┘   └──────▲───────┘
     ▼                                                                │
 ┌──────────────────┐   ┌─────────────────────────────────────────────┴──┐
 │Reward Calculation│──▶│ Q-Table Update → Improved Recommendation (loop)│
 └──────────────────┘   └────────────────────────────────────────────────┘
```

## Algorithm / Pseudocode
```
Initialise Q[s][a] = 0 for all 30 states and 7 actions; ε = 1.0
Load Q from models/q_table.json if it exists
User selects mood (and optional genre → Q[(mood,*)][genre] = max(Q, 2))
fb ← "Start";  s ← (mood, fb)
LOOP for every interaction:
    with probability ε:  a ← random genre            (explore)
    otherwise:           a ← argmax_a Q[s][a]         (exploit, random tie-break)
    song ← pick a song of genre a (prefer mood match, avoid last 5 songs)
    show song; wait for user feedback
    R ← reward(feedback)  (+ repeat penalty −2 if song recently shown)
    fb ← feedback label;  s' ← (mood, fb)
    Q[s][a] ← Q[s][a] + α · (R + γ · max_a' Q[s'][a'] − Q[s][a])
    ε ← max(0.1, ε · 0.9)
    store interaction in history; save Q-table; s ← s'
```

---

## CHAPTER 1 — INTRODUCTION
Music streaming offers millions of tracks, so personalization is essential. Reinforcement Learning (RL) is a branch of machine learning in which an agent learns by trial and error from rewards. This project applies Q-learning, a classical RL algorithm, to learn a user's music taste from direct feedback (Like, Listened, Skip, Dislike) in a simple offline web application.

## CHAPTER 2 — PROBLEM STATEMENT
Traditional music recommendation systems often rely primarily on predefined preferences, similarity measures, or historical data. Such systems may not continuously adapt to a user's changing preferences during interaction. The proposed system uses Reinforcement Learning to create an adaptive music recommendation agent that learns from user feedback. By treating music recommendation as a sequential decision-making problem, the agent updates its Q-values based on user interactions and gradually improves personalized recommendations.

## CHAPTER 3 — OBJECTIVES
1. Develop a software-based music recommendation system. 2. Apply Reinforcement Learning to personalization. 3. Implement Q-Learning from scratch. 4. Learn user preferences from feedback. 5. Use rewards to improve future recommendations. 6. Demonstrate exploration and exploitation. 7. Provide analytics showing the learning process. 8. Build a simple and user-friendly web interface.

## CHAPTER 4 — EXISTING SYSTEM
Existing approaches include content-based filtering (matching song attributes to a profile), collaborative filtering (using similar users' histories) and fixed playlists/rules. They depend on large historical data, are mostly trained offline, and adapt slowly to a user's in-session reactions. Commercial platforms also combine many more signals and models than this project attempts.

## CHAPTER 5 — PROPOSED SYSTEM
A Q-learning agent treats each recommendation as an action and each piece of feedback as a reward. It adapts after every single interaction, needs no historical dataset, runs offline, and exposes its Q-table and learning curves so the learning process can be inspected and explained.

## CHAPTER 6 — SYSTEM REQUIREMENTS
**Software:** Python 3.9+, Streamlit, Pandas, NumPy, Matplotlib; any modern browser; Windows/Linux/macOS.
**Hardware:** any ordinary laptop (≥4 GB RAM). No GPU, sensors or internet needed after installation.
**Functional:** choose mood/genre; recommend songs; accept feedback; update Q-table; show analytics and history; persist learning.

## CHAPTER 7 — SYSTEM ARCHITECTURE
See the architecture diagram above. Layers: (1) Streamlit UI, (2) preference manager (mood/genre in session state), (3) recommendation engine (`utils/recommender.py`: genre → song), (4) Q-learning agent (`rl/q_learning_agent.py`), (5) data layer (`songs.csv`, `q_table.json`, `history.csv`).

## CHAPTER 8 — REINFORCEMENT LEARNING METHODOLOGY
- **Agent:** the recommender. **Environment:** the user.
- **State:** (mood, last feedback) — 6 × 5 = 30 states.
- **Action:** genre to recommend — 7 actions.
- **Reward:** Like +10, Listened +5, Skip −3, Dislike −10, repeated song −2 extra.
- **Next state:** same mood, with last feedback = the user's latest reaction.
- **Policy:** ε-greedy; ε = 1.0 decays ×0.9 per interaction to a minimum of 0.1.
- **Task type:** continuing (no terminal state); γ = 0.6 makes the agent value follow-up rewards, e.g. a genre that leads to another Like.
The optional favourite genre only seeds a small prior (+2) so the agent starts with a hint, then feedback overrides it.

## CHAPTER 9 — Q-LEARNING ALGORITHM
Q-learning is a model-free, off-policy temporal-difference method that learns the action-value function Q(s,a):

`Q(s,a) ← Q(s,a) + α [R + γ max_a' Q(s',a') − Q(s,a)]`

R + γ·max Q(s',a') is the TD target; the bracket is the TD error. α = 0.3 controls how fast new evidence replaces old; γ = 0.6 weighs future rewards. Worked example: Q = 0, user Likes (R = 10), max Q(s') = 0 → Q = 0 + 0.3 × (10 + 0 − 0) = **3.0**. Another Like in the same state with max Q(s') = 3 → Q = 3 + 0.3 × (10 + 1.8 − 3) = **5.64**. See the pseudocode above.

## CHAPTER 10 — MODULE DESCRIPTION
| Module | Responsibility |
|---|---|
| `app.py` | UI pages, session state, feedback loop (reward → update → history → next song) |
| `rl/q_learning_agent.py` | Q-table, ε-greedy `choose_action`, `update`, `decay_epsilon`, save/load |
| `utils/recommender.py` | Converts a genre action into a specific song |
| `utils/data_loader.py` | Loads/validates `songs.csv`, loads/saves history |
| `components/charts.py` | Reward, likes, recommendations and Q-value charts |
| `simulate_demo.py` | Headless demonstration with a simulated user |

## CHAPTER 11 — IMPLEMENTATION
Implemented in Python with NumPy (`q` is a 30×7 array). `choose_action` draws a random number and compares it with ε. `update` applies the equation exactly as written, using the Q-row of the *next* state. Streamlit button callbacks (`on_click`) run `handle_feedback`, which computes the reward, updates the Q-table, decays ε, appends to history, saves JSON/CSV and selects the next song. The UI shows the numeric Q-update after every click so learning is visible.

## CHAPTER 12 — RESULTS
**Evaluation metrics** (shown in the Analytics page): interaction count, like rate, skip rate, average reward, reward improvement (last 10 vs first 10), recommendation diversity, most liked/recommended genre, and Q-value growth. No accuracy percentage is reported because this is an interactive RL task, not supervised classification.

**Simulation** (`simulate_demo.py`: mood Relaxed, scripted user who likes Lo-fi, mildly accepts Classical, skips/dislikes the rest; 20 runs × 120 interactions):

| Measure | Result |
|---|---|
| Average reward, first 20 interactions | ≈ 4.8 |
| Average reward, last 20 interactions | ≈ 8.3 |
| Lo-fi share of last 20 recommendations | ≈ 87 % (random baseline 14.3 %) |
| Mean Q (Relaxed) — Lo-fi vs others | ≈ 14.4 vs ≈ −0.6 … +0.5 (seed 0) |

These numbers come from a scripted simulated user and will vary with random seeds; they demonstrate that the Q-values and policy genuinely change with feedback, not real-world performance. (Insert your own screenshots from the live demo here.)

## CHAPTER 13 — ADVANTAGES
Personalized recommendations; learns from user feedback; adaptive behaviour; simple RL implementation; interactive UI; easy to demonstrate; no external API dependency; works offline; provides learning analytics.

## CHAPTER 14 — LIMITATIONS
Small local dataset (56 songs); feedback is user-provided/simulated; no actual music streaming; Q-learning suits only small state/action spaces; recommendations are limited to the dataset; no large-scale collaborative filtering; actions are genres, so the agent cannot distinguish individual songs within a genre; a real commercial platform would need far larger datasets and more sophisticated models.

## CHAPTER 15 — FUTURE SCOPE
Spotify/YouTube integration; larger datasets; user accounts; collaborative filtering; Deep Q-Networks; context-aware recommendations; real listening-duration feedback; mood detection; voice interaction; mobile application; cloud deployment; multi-user personalization.

## CHAPTER 16 — CONCLUSION
The project delivers a working, explainable RL system. The agent maintains a Q-table over (mood, last feedback) states and genre actions, explores with ε-greedy, and updates values from user rewards, so its recommendations measurably shift towards the user's preferred genres. It is intentionally small and honest about its limits, and it forms a base for richer RL-based recommenders.

## CHAPTER 17 — REFERENCES
1. R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018.
2. C. J. C. H. Watkins and P. Dayan, "Q-learning," *Machine Learning*, vol. 8, pp. 279–292, 1992.
3. Streamlit documentation — https://docs.streamlit.io
4. NumPy documentation — https://numpy.org/doc
5. Pandas documentation — https://pandas.pydata.org/docs
6. Matplotlib documentation — https://matplotlib.org/stable

---

# Viva Questions and Answers

**1. What is Reinforcement Learning?** A type of machine learning where an agent learns to choose actions by interacting with an environment and receiving rewards or penalties, aiming to maximize cumulative reward.

**2. Why did you use RL?** Recommendation is a sequential decision problem and the only training signal is user feedback, which arrives one step at a time. RL learns online from exactly that signal without needing a labelled dataset.

**3. Why Q-learning?** It is simple, model-free, learns from a table, and the problem is small (30 states × 7 actions), so a table is enough and easy to display and explain. Deep networks would be unnecessary.

**4. What is a state?** `(mood, last feedback)`, e.g. (Relaxed, Liked): the user's current mood and how they reacted to the previous song. 30 states total.

**5. What is an action?** Choosing which genre to recommend (7 genres); a song of that genre is then picked from the CSV.

**6. What is the reward?** Numeric feedback: Like +10, Listened +5, Skip −3, Dislike −10, plus −2 for a repeated song. Values are configurable.

**7. What is the Q-table?** A 30 × 7 table storing Q(s,a), the expected long-term reward of recommending genre *a* in state *s*. It starts at zero and is updated after every feedback.

**8. What is epsilon-greedy?** With probability ε choose a random action; otherwise choose the action with the highest Q-value. ε starts at 1.0 and decays ×0.9 per interaction to 0.1.

**9. What is exploration?** Trying genres at random to discover how the user reacts, even if they are not currently the best-known choice.

**10. What is exploitation?** Choosing the genre with the highest learned Q-value in the current state to earn reward from existing knowledge.

**11. How does the agent learn?** After each feedback it computes the TD target R + γ·max Q(s',·) and moves Q(s,a) a fraction α towards it.

**12. How does feedback affect recommendations?** Likes raise the Q-value of that genre in that state, making it more likely to be the argmax; Skips/Dislikes lower it, so other genres win. Exploitation then follows the updated values.

**13. Why not Deep Learning?** The state/action space is tiny, there is little data, and a table is transparent and quick to run offline. Deep Q-Networks are listed as future scope for large or continuous state spaces.

**14. What happens on a Dislike?** Reward −10 gives a negative TD error, so that genre's Q-value in that state drops, and the agent is less likely to recommend it there. The next state becomes (mood, Disliked).

**15. How can it be improved?** Larger song catalogue, song-level or feature-based actions, real listening-time feedback, user accounts, mood detection, collaborative filtering, and DQN for bigger state spaces.

**16. Limitations?** Small dataset, simulated/user-given feedback, no streaming, genre-level actions only, a table that does not scale to huge state spaces, and far simpler than commercial systems.

**17. Difference from a normal recommender?** A normal recommender typically computes similarity or fixed rules from stored data. This agent keeps a value function and changes its own policy after every reward, balancing exploration and exploitation.

**18. Q-learning equation?** `Q(s,a) ← Q(s,a) + α[R + γ max_a' Q(s',a') − Q(s,a)]`.

**19. Alpha and gamma?** α (0.3) is the learning rate — how strongly new feedback overrides the old estimate. γ (0.6) is the discount factor — how much future reward matters compared with immediate reward.

**20. How do you evaluate?** Interaction count, like/skip rate, average reward, reward improvement over time, recommendation diversity, most preferred genre, and growth of the preferred genre's Q-values. No accuracy % is claimed because it is an interactive RL simulation.
