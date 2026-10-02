"""Adaptive Music Recommendation Agent using Q-Learning  (Streamlit UI)."""
from datetime import datetime
from html import escape
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from components import charts
from rl.q_learning_agent import (ALPHA, EPSILON_DECAY, EPSILON_MIN, GAMMA, GENRES, MOODS,
                                 REWARDS, QLearningAgent)
from utils.data_loader import load_history, load_songs, save_history
from utils.recommender import Recommender
from utils.youtube import get_video_id

BASE = Path(__file__).parent
SONGS_PATH = str(BASE / "data" / "songs.csv")
MODEL_PATH = str(BASE / "models" / "q_table.json")
HISTORY_PATH = str(BASE / "data" / "history.csv")
YT_CACHE_PATH = str(BASE / "data" / "youtube_ids.json")

# Make sure the dark theme exists (takes effect the next time the app is started).
_cfg = BASE / ".streamlit" / "config.toml"
if not _cfg.exists():
    _cfg.parent.mkdir(exist_ok=True)
    _cfg.write_text('[theme]\nbase = "dark"\nprimaryColor = "#8B5CF6"\n'
                    'backgroundColor = "#0B0B14"\nsecondaryBackgroundColor = "#151526"\n'
                    'textColor = "#EDEDF7"\n')

MOOD_LABELS = {"Happy": "😊 Happy", "Sad": "😢 Sad", "Relaxed": "😌 Relaxed",
               "Energetic": "⚡ Energetic", "Focus": "📚 Focus", "Romantic": "❤️ Romantic"}
ACTION_NAMES = {"like": "Like", "complete": "Listened", "skip": "Skip", "dislike": "Dislike"}
FEEDBACK_STATE = {"like": "Liked", "complete": "Completed", "skip": "Skipped", "dislike": "Disliked"}
GENRE_STYLE = {"Pop": ("🎤", "#f472b6", "#8b5cf6"), "Rock": ("🎸", "#fb923c", "#dc2626"),
               "Lo-fi": ("🌙", "#38bdf8", "#6366f1"), "Classical": ("🎻", "#fbbf24", "#b45309"),
               "Hip-Hop": ("🎧", "#a3e635", "#16a34a"), "EDM": ("⚡", "#22d3ee", "#d946ef"),
               "Bollywood": ("🪔", "#fb7185", "#f59e0b")}

st.set_page_config(page_title="Music RL Agent", page_icon="🎵", layout="wide")

CSS = """
<style>
#MainMenu, footer {visibility:hidden;}
header[data-testid="stHeader"] {background:transparent;}
.stApp {background: radial-gradient(900px 500px at 8% -5%, rgba(139,92,246,.28), transparent 60%),
        radial-gradient(800px 500px at 100% 0%, rgba(236,72,153,.18), transparent 55%), #0B0B14;
        color:#EDEDF7;}
.block-container {padding-top:2.2rem; max-width:1200px;}
h1,h2,h3 {letter-spacing:-.02em;}
/* sidebar */
[data-testid="stSidebar"] {background:linear-gradient(180deg,#12122a,#0d0d1c);
        border-right:1px solid rgba(255,255,255,.06);}
.brand {display:flex; align-items:center; gap:12px; margin:4px 0 18px 0;}
.brand .logo {width:44px;height:44px;border-radius:14px;display:flex;align-items:center;
        justify-content:center;font-size:22px;background:linear-gradient(135deg,#8b5cf6,#ec4899);
        box-shadow:0 6px 20px rgba(139,92,246,.45);}
.brand b {font-size:1.05rem;color:#fff;} .brand small {display:block;color:#9a9ab8;}
[data-testid="stSidebar"] [role="radiogroup"] {gap:4px;}
[data-testid="stSidebar"] [role="radiogroup"] label {padding:10px 14px;border-radius:12px;width:100%;
        transition:.15s; border:1px solid transparent;}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {background:rgba(255,255,255,.06);}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background:linear-gradient(135deg,rgba(139,92,246,.35),rgba(236,72,153,.25));
        border-color:rgba(167,139,250,.5);}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {display:none;}
.side-chip {background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.08);
        border-radius:12px;padding:10px 14px;margin:6px 0;color:#cfcfe8;font-size:.88rem;}
.side-chip b {color:#fff;}
/* hero */
.hero {position:relative; padding:30px 34px; border-radius:24px; margin-bottom:22px; overflow:hidden;
       background:linear-gradient(135deg,rgba(139,92,246,.30),rgba(236,72,153,.18) 60%,rgba(34,211,238,.12));
       border:1px solid rgba(255,255,255,.10); backdrop-filter:blur(8px);}
.hero h1 {margin:0 0 6px 0; font-size:2.3rem; font-weight:800;
       background:linear-gradient(90deg,#fff,#c4b5fd 45%,#f0abfc); -webkit-background-clip:text;
       background-clip:text; color:transparent;}
.hero p {margin:0; color:#cfcfe8; font-size:1.05rem;}
.steps {display:flex; gap:10px; margin-top:18px; flex-wrap:wrap;}
.step {background:rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.10); border-radius:30px;
       padding:6px 14px; font-size:.85rem; color:#e5e5ff;}
/* stat cards */
.stats {display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:14px; margin:6px 0 22px 0;}
.stat {background:rgba(255,255,255,.045); border:1px solid rgba(255,255,255,.09); border-radius:18px;
       padding:16px 18px; transition:.2s;}
.stat:hover {transform:translateY(-3px); border-color:rgba(167,139,250,.55);
       box-shadow:0 10px 28px rgba(139,92,246,.18);}
.stat .ic {font-size:1.35rem;} .stat .lb {color:#9a9ab8; font-size:.78rem; text-transform:uppercase;
       letter-spacing:.08em; margin-top:6px;} .stat .vl {font-size:1.55rem; font-weight:700; color:#fff;}
.panel {background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.09); border-radius:20px;
       padding:20px 22px; margin-bottom:16px;}
.panel h3 {margin:0 0 12px 0; font-size:1.05rem; color:#fff;}
.sub {color:#9a9ab8; font-size:.85rem;}
/* now playing */
.np {display:flex; gap:24px; align-items:center; padding:24px; border-radius:24px; margin-bottom:14px;
     background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.10);
     box-shadow:0 18px 50px rgba(0,0,0,.45);}
.art {flex:0 0 150px; height:150px; border-radius:22px; display:flex; align-items:center;
      justify-content:center; font-size:64px; box-shadow:0 14px 34px rgba(0,0,0,.5);}
.np-label {color:#a78bfa; font-size:.75rem; letter-spacing:.18em; font-weight:700; display:flex;
      align-items:center; gap:10px;}
.np-title {font-size:2rem; font-weight:800; color:#fff; line-height:1.15; margin-top:6px;}
.np-artist {color:#cfcfe8; font-size:1.1rem; margin-bottom:12px;}
.chip {display:inline-block; padding:5px 13px; margin:0 6px 6px 0; border-radius:20px; font-size:.82rem;
      background:rgba(255,255,255,.09); border:1px solid rgba(255,255,255,.12); color:#f1f1ff;}
.pol {display:inline-block; padding:5px 13px; border-radius:20px; font-size:.8rem; font-weight:600;}
.pol.ex {background:rgba(251,191,36,.15); color:#fcd34d; border:1px solid rgba(251,191,36,.4);}
.pol.ep {background:rgba(52,211,153,.15); color:#6ee7b7; border:1px solid rgba(52,211,153,.4);}
.eq {display:inline-flex; gap:3px; align-items:flex-end; height:14px;}
.eq i {display:block; width:3px; background:#a78bfa; border-radius:2px; animation:eq 1s infinite ease-in-out;}
.eq i:nth-child(1){animation-delay:-.9s}.eq i:nth-child(2){animation-delay:-.6s}
.eq i:nth-child(3){animation-delay:-.3s}.eq i:nth-child(4){animation-delay:-.75s}
.eq i:nth-child(5){animation-delay:-.45s}
@keyframes eq {0%,100%{height:3px} 50%{height:14px}}
/* taste bars */
.tb {display:flex; align-items:center; gap:10px; margin:9px 0;}
.tb-name {width:84px; color:#e5e5ff; font-size:.88rem;}
.tb-track {flex:1; height:10px; background:rgba(255,255,255,.08); border-radius:8px; overflow:hidden;}
.tb-fill {height:100%; border-radius:8px; transition:width .6s;}
.tb-val {width:52px; text-align:right; color:#cfcfe8; font-size:.85rem; font-variant-numeric:tabular-nums;}
.meter {height:10px; border-radius:8px; background:rgba(255,255,255,.08); overflow:hidden; margin:8px 0 4px;}
.meter div {height:100%; background:linear-gradient(90deg,#34d399,#fbbf24,#f472b6);}
.qrow {display:flex; align-items:baseline; gap:10px;} .qold {color:#9a9ab8; font-size:1.2rem;}
.qnew {font-size:2.2rem; font-weight:800; color:#fff;}
.rw {display:inline-block; padding:3px 12px; border-radius:20px; font-weight:700; font-size:.9rem;}
.rw.pos {background:rgba(52,211,153,.18); color:#6ee7b7;} .rw.neg {background:rgba(248,113,113,.18); color:#fca5a5;}
/* buttons */
.stButton>button, .stLinkButton>a {border-radius:14px; font-weight:600; padding:.65rem 1rem;
      border:1px solid rgba(255,255,255,.14); background:rgba(255,255,255,.06); color:#fff; transition:.15s;}
.stButton>button:hover, .stLinkButton>a:hover {transform:translateY(-2px); border-color:#a78bfa;
      box-shadow:0 8px 22px rgba(139,92,246,.30); color:#fff;}
.stButton>button[kind="primary"] {background:linear-gradient(135deg,#8b5cf6,#ec4899); border:0;
      box-shadow:0 8px 24px rgba(139,92,246,.45);}
.st-key-btn_like button {background:linear-gradient(135deg,#10b981,#34d399); color:#04281c; border:0;}
.st-key-btn_complete button {background:linear-gradient(135deg,#0ea5e9,#22d3ee); color:#03212b; border:0;}
.st-key-btn_skip button {background:linear-gradient(135deg,#f59e0b,#fbbf24); color:#2b1c00; border:0;}
.st-key-btn_dislike button {background:linear-gradient(135deg,#ef4444,#f472b6); color:#fff; border:0;}
.st-key-btn_like button:hover,.st-key-btn_complete button:hover,.st-key-btn_skip button:hover,
.st-key-btn_dislike button:hover {color:inherit; filter:brightness(1.1);}
[data-testid="stDataFrame"] {border-radius:16px; overflow:hidden; border:1px solid rgba(255,255,255,.09);}
@media (max-width:800px){.np{flex-direction:column;align-items:flex-start}.art{flex-basis:auto;width:100%}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def md(s):
    """Render HTML safely through markdown (strip indentation so it isn't treated as code)."""
    st.markdown("\n".join(line.strip() for line in s.splitlines() if line.strip()),
                unsafe_allow_html=True)


# ------------------------------------------------------------------ state
def init_session():
    ss = st.session_state
    if "initialised" in ss:
        return
    ss.songs = load_songs(SONGS_PATH)
    ss.recommender = Recommender(ss.songs)
    ss.agent = QLearningAgent()
    ss.agent.load(MODEL_PATH)                 # resume learned Q-table if present
    ss.history = load_history(HISTORY_PATH)
    ss.mood = None
    ss.last_fb = "Start"
    ss.pending = None                         # the recommendation currently on screen
    ss.last_update = None
    ss.video_ids = {}                         # song_id -> youtube id ("" = not found)
    ss.initialised = True


def new_recommendation():
    """Agent picks an action (genre) with epsilon-greedy; recommender maps it to a song."""
    ss = st.session_state
    state = (ss.mood, ss.last_fb)
    action, policy = ss.agent.choose_action(state)
    recent = [h["song_id"] for h in ss.history[-5:]]
    song = ss.recommender.pick_song(GENRES[action], ss.mood, recent)
    ss.pending = {"state": state, "action": action, "policy": policy, "song": song,
                  "repeat": song["song_id"] in recent}


def start_recommendation(mood, genre):
    ss = st.session_state
    ss.mood, ss.last_fb, ss.last_update = mood, "Start", None
    if genre != "No preference":
        ss.agent.seed_preference(mood, genre)
    new_recommendation()


def handle_feedback(kind):
    """reward -> Q-update -> new state -> store history -> next recommendation."""
    ss = st.session_state
    p = ss.pending
    if p is None:
        return
    reward = REWARDS[kind] + (REWARDS["repeat"] if p["repeat"] else 0)
    next_state = (ss.mood, FEEDBACK_STATE[kind])
    info = ss.agent.update(p["state"], p["action"], reward, next_state)
    ss.agent.decay_epsilon()
    s = p["song"]
    ss.history.append({"Time": datetime.now().strftime("%I:%M:%S %p"), "Song": s["song"],
                       "Artist": s["artist"], "Genre": s["genre"], "Mood": s["mood"],
                       "Action": ACTION_NAMES[kind], "Reward": reward, "UserMood": ss.mood,
                       "Policy": p["policy"], "song_id": s["song_id"]})
    info.update(state=p["state"], genre=GENRES[p["action"]])
    ss.last_update = info
    ss.last_fb = FEEDBACK_STATE[kind]
    ss.agent.save(MODEL_PATH)
    save_history(ss.history, HISTORY_PATH)
    new_recommendation()


def reset_all():
    ss = st.session_state
    ss.agent.reset_table()
    ss.history, ss.pending, ss.last_update, ss.last_fb = [], None, None, "Start"
    ss.agent.save(MODEL_PATH)
    save_history(ss.history, HISTORY_PATH)
    if ss.mood:
        new_recommendation()


def hist_df():
    return pd.DataFrame(st.session_state.history)


def favourite_genre():
    ss = st.session_state
    if not ss.history or not ss.mood:
        return "—"
    q = ss.agent.mood_q(ss.mood)
    return GENRES[int(q.argmax())] if q.max() > 0 else "—"


# ---------------------------------------------------------------- widgets
def stat_cards(items):
    cards = "".join(f'<div class="stat"><div class="ic">{ic}</div><div class="lb">{lb}</div>'
                    f'<div class="vl">{escape(str(vl))}</div></div>' for ic, lb, vl in items)
    md(f'<div class="stats">{cards}</div>')


def taste_profile(mood):
    """Horizontal bars of the agent's learned value for each genre (current mood)."""
    q = st.session_state.agent.mood_q(mood)
    if not st.session_state.history or abs(q).max() < 1e-9:
        body = '<div class="sub">The agent has not learned anything yet. Give it some feedback!</div>'
    else:
        top = max(q.max(), 1e-9)
        rows = []
        for i in sorted(range(len(GENRES)), key=lambda k: -q[k]):
            g = GENRES[i]
            _, c1, c2 = GENRE_STYLE[g]
            w = max(0.0, q[i]) / top * 100
            crown = " 👑" if i == int(q.argmax()) and q[i] > 0 else ""
            rows.append(f'<div class="tb"><span class="tb-name">{g}{crown}</span><div class="tb-track">'
                        f'<div class="tb-fill" style="width:{w:.0f}%;background:linear-gradient(90deg,{c1},{c2})">'
                        f'</div></div><span class="tb-val">{q[i]:+.1f}</span></div>')
        body = "".join(rows)
    md(f'<div class="panel"><h3>🧠 Learned taste · {MOOD_LABELS[mood]}</h3>{body}'
       f'<div class="sub">Average Q-value of each genre for this mood</div></div>')


def player(song):
    """Embedded YouTube player (autoplay) + fallback search link."""
    ss = st.session_state
    sid = int(song["song_id"])
    if ss.get("autoplay", True):
        if sid not in ss.video_ids:
            with st.spinner("Finding the song on YouTube…"):
                ss.video_ids[sid] = get_video_id(sid, song["song"], song["artist"], YT_CACHE_PATH)
        vid = ss.video_ids[sid]
        if vid:
            src = f"https://www.youtube.com/embed/{vid}?autoplay=1&rel=0&modestbranding=1"
            if hasattr(st, "iframe"):                      # newer Streamlit
                st.iframe(src, height=250)
            else:                                           # older Streamlit
                components.html(f'<iframe width="100%" height="250" src="{src}" frameborder="0" '
                                f'allow="autoplay; encrypted-media" allowfullscreen '
                                f'style="border:0;border-radius:16px;"></iframe>', height=256)
        else:
            st.caption("🌐 Couldn't load the in-app player (offline?). Use the button below.")
    url = "https://www.youtube.com/results?search_query=" + quote_plus(f"{song['song']} {song['artist']}")
    st.link_button("🔎 Open on YouTube", url)


# ------------------------------------------------------------------ pages
def page_dashboard():
    ss = st.session_state
    h = hist_df()
    md("""<div class="hero"><h1>Adaptive Music Recommendation Agent</h1>
    <p>A Q-Learning agent that learns your taste from every Like, Skip and Dislike.</p>
    <div class="steps"><span class="step">① Pick a mood</span><span class="step">② Agent recommends</span>
    <span class="step">③ You react</span><span class="step">④ Agent learns</span></div></div>""")
    stat_cards([("🎭", "Current mood", MOOD_LABELS[ss.mood] if ss.mood else "Not set"),
                ("🔁", "Interactions", len(h)),
                ("🏆", "Total reward", int(h["Reward"].sum()) if len(h) else 0),
                ("💜", "Favourite genre", favourite_genre()),
                ("🎵", "Recommendations", len(h)),
                ("🎲", "Exploration ε", f"{ss.agent.epsilon:.2f}")])
    left, right = st.columns([1.1, 1])
    with left:
        md('<div class="panel"><h3>🎛️ Start a listening session</h3>'
           '<div class="sub">Choose how you feel right now.</div></div>')
        mood = st.pills("Mood", MOODS, format_func=lambda m: MOOD_LABELS[m],
                        default=ss.mood or "Relaxed", key="mood_pick", label_visibility="collapsed")
        mood = mood or ss.mood or "Relaxed"
        genre = st.selectbox("Initial favourite genre (optional)", ["No preference"] + GENRES)
        if st.button("🚀 Start Recommendation", type="primary"):
            start_recommendation(mood, genre)
            st.success("Agent ready! Open **🎧 Recommendation** from the sidebar.")
    with right:
        taste_profile(ss.mood or "Relaxed")


def page_recommendation():
    ss = st.session_state
    md('<h2 style="margin-top:0">🎧 Now Recommending</h2>')
    if ss.mood is None or ss.pending is None:
        st.info("Pick a mood on the **🏠 Dashboard** and press *Start Recommendation* first.")
        return
    p, s = ss.pending, ss.pending["song"]
    emoji, c1, c2 = GENRE_STYLE[s["genre"]]
    left, right = st.columns([3, 2], gap="large")
    with left:
        pol = ('<span class="pol ex">🔍 Exploring a new genre</span>' if p["policy"] == "Explore"
               else '<span class="pol ep">🎯 Using what it learned</span>')
        md(f"""<div class="np"><div class="art" style="background:linear-gradient(135deg,{c1},{c2})">{emoji}</div>
        <div><div class="np-label">RECOMMENDED FOR YOU <span class="eq"><i></i><i></i><i></i><i></i><i></i></span></div>
        <div class="np-title">{escape(str(s['song']))}</div><div class="np-artist">{escape(str(s['artist']))}</div>
        <span class="chip">🎼 {s['genre']}</span><span class="chip">🎭 {s['mood']}</span>
        <span class="chip">⏱ {s['duration']}</span><br>{pol}</div></div>""")
        b = st.columns(4)
        b[0].button("❤️ Like", key="btn_like", on_click=handle_feedback, args=("like",), width="stretch")
        b[1].button("✅ Listened", key="btn_complete", on_click=handle_feedback, args=("complete",),
                    width="stretch")
        b[2].button("⏭️ Skip", key="btn_skip", on_click=handle_feedback, args=("skip",), width="stretch")
        b[3].button("👎 Dislike", key="btn_dislike", on_click=handle_feedback, args=("dislike",),
                    width="stretch")
        st.caption(f"Rewards: Like {REWARDS['like']:+d} · Listened {REWARDS['complete']:+d} · "
                   f"Skip {REWARDS['skip']:+d} · Dislike {REWARDS['dislike']:+d}")
        player(s)
    with right:
        eps = ss.agent.epsilon
        md(f"""<div class="panel"><h3>🎲 Explore ↔ Exploit</h3>
        <div class="meter"><div style="width:{eps*100:.0f}%"></div></div>
        <div class="sub">ε = {eps:.2f} → {eps:.0%} random exploring, {1-eps:.0%} using learned values</div></div>""")
        u = ss.last_update
        if u is None:
            md('<div class="panel"><h3>🧠 What the agent just learned</h3>'
               '<div class="sub">React to the song and the Q-update will appear here.</div></div>')
        else:
            cls = "pos" if u["reward"] > 0 else "neg"
            md(f"""<div class="panel"><h3>🧠 What the agent just learned</h3>
            <div class="sub">State <b>{u['state'][0]} | {u['state'][1]}</b> · Action <b>{u['genre']}</b>
            · <span class="rw {cls}">reward {u['reward']:+d}</span></div>
            <div class="qrow"><span class="qold">{u['old']:.2f} →</span><span class="qnew">{u['new']:.2f}</span>
            <span class="sub">({u['new']-u['old']:+.2f})</span></div></div>""")
            st.code(f"Q_new = {u['old']:.2f} + {ALPHA} × [{u['reward']} + {GAMMA} × "
                    f"{u['max_next']:.2f} − {u['old']:.2f}] = {u['new']:.2f}")
        taste_profile(ss.mood)


def page_analytics():
    ss = st.session_state
    md('<h2 style="margin-top:0">📊 Learning Analytics</h2>')
    h = hist_df()
    if h.empty:
        st.info("No interactions yet — give some feedback on the Recommendation page.")
        return
    n = len(h)
    likes, skips = int((h["Action"] == "Like").sum()), int((h["Action"] == "Skip").sum())
    dislikes, listened = int((h["Action"] == "Dislike").sum()), int((h["Action"] == "Listened").sum())
    liked = h.loc[h["Action"] == "Like", "Genre"]
    k = min(10, n // 2)
    improve = (h["Reward"].tail(k).mean() - h["Reward"].head(k).mean()) if k else 0.0
    stat_cards([("🔁", "Interactions", n), ("❤️", "Likes", likes), ("⏭️", "Skips", skips),
                ("👎", "Dislikes", dislikes), ("✅", "Listened", listened),
                ("⭐", "Avg reward", f"{h['Reward'].mean():.2f}")])
    stat_cards([("📈", "Most recommended", h["Genre"].value_counts().idxmax()),
                ("💖", "Most liked", liked.value_counts().idxmax() if len(liked) else "—"),
                ("👍", "Like rate", f"{likes / n:.0%}"), ("🙅", "Skip rate", f"{skips / n:.0%}"),
                ("🚀", "Reward improvement", f"{improve:+.2f}"),
                ("🌈", "Diversity (last 20)", f"{h.tail(20)['Genre'].nunique()}/{len(GENRES)}")])
    st.caption("Reward improvement = average reward of the last 10 interactions minus the first 10 "
               "(needs at least 2 interactions). Diversity = distinct genres in the last 20 picks.")
    md('<h3>Q-table</h3>')
    only = st.checkbox("Show only visited states", value=True)
    qdf = ss.agent.q_dataframe(visited_only=only)
    if qdf.empty:
        qdf = ss.agent.q_dataframe()
    st.dataframe(qdf.style.background_gradient(cmap="RdYlGn", axis=None).format("{:.2f}"),
                 width="stretch")
    md('<h3>Charts</h3>')
    c1, c2 = st.columns(2)
    for col, fig in ((c1, charts.reward_chart(h)),
                     (c2, charts.q_comparison(ss.agent, ss.mood or "Relaxed")),
                     (c1, charts.likes_by_genre(h)), (c2, charts.recs_by_genre(h))):
        with col:
            st.pyplot(fig)
            charts.plt.close(fig)


def page_history():
    md('<h2 style="margin-top:0">📜 Recommendation History</h2>')
    h = hist_df()
    if h.empty:
        st.info("No history yet.")
        return
    view = h[["Time", "Song", "Artist", "Genre", "Mood", "Action", "Reward"]].iloc[::-1].copy()
    icons = {"Like": "❤️ Like", "Listened": "✅ Listened", "Skip": "⏭️ Skip", "Dislike": "👎 Dislike"}
    view["Action"] = view["Action"].map(icons)
    view["Reward"] = view["Reward"].map(lambda r: f"{r:+d}")
    st.dataframe(view, width="stretch", hide_index=True)
    st.download_button("⬇️ Download CSV", h.drop(columns="song_id").to_csv(index=False),
                       "history.csv", "text/csv")


def page_about():
    md('<h2 style="margin-top:0">ℹ️ About Project</h2>')
    st.markdown(f"""
**Adaptive Music Recommendation Agent Using Q-Learning** is a software-only reinforcement
learning project using a local CSV of {len(st.session_state.songs)} songs. Songs are played through
YouTube's own embedded player; the app itself stores no audio.

| RL concept | In this project |
|---|---|
| **State** | `(mood, last feedback)` — 6 moods × 5 feedback values = 30 states |
| **Action** | Genre to recommend: {', '.join(GENRES)} (7 actions) |
| **Reward** | Like {REWARDS['like']:+d}, Listened {REWARDS['complete']:+d}, Skip {REWARDS['skip']:+d}, Dislike {REWARDS['dislike']:+d}, repeated song {REWARDS['repeat']:+d} |
| **Next state** | Same mood, last feedback = what the user just did |
| **Policy** | ε-greedy: ε starts at 1.0, ×{EPSILON_DECAY} per interaction, minimum {EPSILON_MIN} |
""")
    st.latex(r"Q(s,a) \leftarrow Q(s,a) + \alpha\,[\,R + \gamma \max_{a'} Q(s',a') - Q(s,a)\,]")
    st.write(f"α (learning rate) = **{ALPHA}**, γ (discount factor) = **{GAMMA}**")


# ------------------------------------------------------------------- main
init_session()
ss = st.session_state
with st.sidebar:
    md('<div class="brand"><div class="logo">🎵</div><div><b>Music RL Agent</b>'
       '<small>Q-Learning recommender</small></div></div>')
    PAGES = {"🏠 Dashboard": page_dashboard, "🎧 Recommendation": page_recommendation,
             "📊 Learning Analytics": page_analytics, "📜 History": page_history,
             "ℹ️ About Project": page_about}
    choice = st.radio("Navigation", list(PAGES), label_visibility="collapsed")
    st.toggle("▶️ Auto-play in app", value=True, key="autoplay",
              help="Plays each recommended song automatically in an embedded YouTube player "
                   "(needs internet). Turn off to work fully offline.")
    md(f'<div class="side-chip">Mood: <b>{MOOD_LABELS.get(ss.mood, "not set")}</b></div>'
       f'<div class="side-chip">ε = <b>{ss.agent.epsilon:.2f}</b> · steps = <b>{ss.agent.steps}</b></div>')
    with st.expander("⚠️ Reset learning"):
        if st.button("Reset Q-table & history"):
            reset_all()
            st.success("Agent reset.")
PAGES[choice]()
