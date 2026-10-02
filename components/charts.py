import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from rl.q_learning_agent import GENRES

COLORS = ["#f472b6", "#fb923c", "#60a5fa", "#fbbf24", "#a3e635", "#22d3ee", "#fb7185"]
TEXT, GRID, GOOD, BAD = "#CFCFE8", "#3a3a58", "#34d399", "#f87171"


def _fig():
    fig, ax = plt.subplots(figsize=(6, 3.4))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.yaxis.label.set_color(TEXT); ax.xaxis.label.set_color(TEXT)
    ax.grid(axis="y", color=GRID, alpha=0.5, lw=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def _title(ax, text):
    ax.set_title(text, color="white", fontsize=11, fontweight="bold", loc="left", pad=10)


def reward_chart(hist: pd.DataFrame):
    fig, ax = _fig()
    x = range(1, len(hist) + 1)
    ax.bar(x, hist["Reward"], color=[GOOD if r > 0 else BAD for r in hist["Reward"]], alpha=0.5,
           label="Reward")
    ax.plot(x, hist["Reward"].rolling(5, min_periods=1).mean(), color="#c4b5fd", lw=2.4,
            label="Moving avg (5)")
    ax.set_xlabel("Interaction"); ax.set_ylabel("Reward"); _title(ax, "Reward vs Interaction")
    leg = ax.legend(frameon=False)
    for t in leg.get_texts():
        t.set_color(TEXT)
    return fig


def _genre_counts(series):
    return series.value_counts().reindex(GENRES, fill_value=0)


def likes_by_genre(hist: pd.DataFrame):
    fig, ax = _fig()
    c = _genre_counts(hist.loc[hist["Action"] == "Like", "Genre"])
    ax.bar(c.index, c.values, color=COLORS)
    _title(ax, "Likes by Genre"); ax.set_ylabel("Likes"); plt.xticks(rotation=30)
    return fig


def recs_by_genre(hist: pd.DataFrame):
    fig, ax = _fig()
    c = _genre_counts(hist["Genre"])
    ax.bar(c.index, c.values, color=COLORS)
    _title(ax, "Recommendations by Genre"); ax.set_ylabel("Times recommended")
    plt.xticks(rotation=30)
    return fig


def q_comparison(agent, mood):
    fig, ax = _fig()
    vals = agent.mood_q(mood)
    ax.bar(GENRES, vals, color=[GOOD if v >= 0 else BAD for v in vals])
    ax.axhline(0, color=TEXT, lw=0.8, alpha=0.6)
    _title(ax, f"Mean Q-value per Genre  ·  mood: {mood}"); ax.set_ylabel("Q-value")
    plt.xticks(rotation=30)
    return fig
