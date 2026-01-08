# app.py
# Streamlit dashboard for Bulls subreddit game threads (deterministic rules only)

import re
from collections import Counter
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Bulls Reddit Narrative Dashboard", layout="wide")


# -----------------------------
# Deterministic dictionaries
# -----------------------------
# Customize these whenever you want. This is the whole point: deterministic rules.
PLAYERS: Dict[str, List[str]] = {
    "Nikola Vucevic": [r"\bvooch\b", r"\bvucevic\b", r"\bvuc\b"],
    "Patrick Williams": [r"\bpwill\b", r"\bpatrick williams\b", r"\bpat\b"],
    "Coby White": [r"\bcoby\b", r"\bcoby white\b"],
    "Josh Giddey": [r"\bgiddey\b", r"\bjosh giddey\b"],
    "Lonzo Ball": [r"\blonzo\b", r"\blonzo ball\b", r"\bball\b"],
    "Ayo Dosunmu": [r"\bayo\b", r"\bdosunmu\b", r"\bayo dosunmu\b"],
    "Matas Buzelis": [r"\bmatas\b", r"\bbuzelis\b", r"\bmatas buzelis\b"],
    "Billy Donovan": [r"\bdonovan\b", r"\bbilly\b", r"\bbilly donovan\b"],
    # add more
}

THEMES: Dict[str, List[str]] = {
    "injury": [r"\binjur", r"\bconcussion\b", r"\bprotocol\b", r"\bout\b", r"\bquestionable\b"],
    "coaching": [r"\bcoach\b", r"\bcoaching\b", r"\blineup\b", r"\brotation\b", r"\btimeouts?\b", r"\bdonovan\b"],
    "shooting": [r"\bshoot", r"\b3s\b", r"\bthrees\b", r"\bthree\b", r"\bbrick", r"\bfg\b", r"\bpercent\b"],
    "refs": [r"\bref", r"\bwhistle\b", r"\bfoul\b", r"\bfree throw\b", r"\bft\b"],
    "front_office": [r"\bfront office\b", r"\bfo\b", r"\bakme\b", r"\bkarnisovas\b", r"\btrade\b", r"\bdeadline\b"],
    "effort_identity": [r"\beffort\b", r"\bsoft\b", r"\bheart\b", r"\bidentity\b", r"\bvibes\b"],
    "tanking": [r"\btank\b", r"\blottery\b", r"\bpicks?\b", r"\btop pick\b"],
    # add more
}

# super simple sentiment heuristic
NEG_WORDS = [
    r"\btrash\b", r"\bembarrass", r"\bsoft\b", r"\bawful\b", r"\bbad\b", r"\bworst\b",
    r"\bpissed\b", r"\bfuck\b", r"\blose\b", r"\bloser\b", r"\bpathetic\b",
]
POS_WORDS = [
    r"\bgreat\b", r"\bgood\b", r"\bamazing\b", r"\blove\b", r"\bwin\b", r"\bsolid\b",
    r"\bproud\b", r"\bnice\b",
]


# -----------------------------
# Helpers
# -----------------------------
FILENAME_RE = re.compile(
    r"(?P<game_date>\d{4}-\d{2}-\d{2})_(?P<thread_type>pregame|live_game|postgame)_(?P<thread_id>[a-z0-9]+)\.csv",
    re.I
)

def normalize_thread_type(x: str) -> str:
    x = (x or "").strip().lower()
    if x in ["game", "live", "livegame", "live_game"]:
        return "live_game"
    if x in ["post", "postgame", "post_game"]:
        return "postgame"
    if x in ["pre", "pregame", "pre_game"]:
        return "pregame"
    return x or "unknown"

def parse_filename_meta(name: str) -> Tuple[Optional[str], str, Optional[str]]:
    """
    Returns (game_date_str, thread_type, thread_id) from filename if possible.
    Example: 2026-01-03_live_game_1q3aeit.csv
    """
    m = FILENAME_RE.search(name)
    if not m:
        return None, "unknown", None
    gd = m.group("game_date")
    tt = normalize_thread_type(m.group("thread_type"))
    tid = m.group("thread_id")
    return gd, tt, tid

def safe_text(x) -> str:
    return "" if pd.isna(x) else str(x)

def detect_mentions(text: str, patterns: Dict[str, List[str]]) -> Dict[str, int]:
    txt = text.lower()
    hits = {}
    for label, pats in patterns.items():
        c = 0
        for p in pats:
            c += len(re.findall(p, txt, flags=re.I))
        if c > 0:
            hits[label] = c
    return hits

def classify_sentiment(text: str) -> str:
    txt = text.lower()
    neg = any(re.search(p, txt, flags=re.I) for p in NEG_WORDS)
    pos = any(re.search(p, txt, flags=re.I) for p in POS_WORDS)
    if neg and pos:
        return "mixed"
    if neg:
        return "negative"
    if pos:
        return "positive"
    return "neutral"

def pct(part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(100.0 * part / total, 1)

def theme_counts_for_df(df_subset: pd.DataFrame) -> Counter:
    c = Counter()
    for body in df_subset["body"].astype(str).tolist():
        hits = detect_mentions(body, THEMES)
        for theme in hits.keys():
            c[theme] += 1
    return c

def player_counts_for_df(df_subset: pd.DataFrame) -> Counter:
    c = Counter()
    for body in df_subset["body"].astype(str).tolist():
        hits = detect_mentions(body, PLAYERS)
        for p, cnt in hits.items():
            c[p] += cnt
    return c

def deterministic_game_narrative(g: pd.DataFrame) -> Dict:
    bullets: List[str] = []

    total_comments = len(g)
    by_type = g.groupby("thread_type").size().to_dict()
    live_ct = int(by_type.get("live_game", 0))
    post_ct = int(by_type.get("postgame", 0))
    pre_ct = int(by_type.get("pregame", 0))

    s_all = Counter(g["sentiment"].astype(str).tolist())
    neg = int(s_all.get("negative", 0))
    pos = int(s_all.get("positive", 0))
    neu = int(s_all.get("neutral", 0))
    mix = int(s_all.get("mixed", 0))

    bullets.append(f"Engagement: {total_comments} total comments (live_game {live_ct}, postgame {post_ct}, pregame {pre_ct}).")
    bullets.append(
        f"Tone mix (heuristic): {pct(neg, total_comments)}% negative, "
        f"{pct(mix, total_comments)}% mixed, {pct(neu, total_comments)}% neutral, {pct(pos, total_comments)}% positive."
    )

    if live_ct > 0 and post_ct > 0:
        if post_ct > live_ct * 1.25:
            bullets.append("Conversation intensified after the final: postgame volume was meaningfully higher than live_game.")
        elif live_ct > post_ct * 1.25:
            bullets.append("Conversation peaked during the game: live_game volume was meaningfully higher than postgame.")
        else:
            bullets.append("Engagement was steady: live_game and postgame volume were in a similar range.")

    themes_all = theme_counts_for_df(g)
    top_themes = [k for k, _ in themes_all.most_common(4)]
    if top_themes:
        bullets.append("Top narratives: " + ", ".join(top_themes) + ".")

    theme_spikes_df = pd.DataFrame()
    if live_ct > 0 and post_ct > 0:
        themes_live = theme_counts_for_df(g[g["thread_type"] == "live_game"])
        themes_post = theme_counts_for_df(g[g["thread_type"] == "postgame"])

        rows = []
        for theme in sorted(set(list(themes_live.keys()) + list(themes_post.keys()))):
            lv = int(themes_live.get(theme, 0))
            pg = int(themes_post.get(theme, 0))
            rows.append({"theme": theme, "live_game": lv, "postgame": pg, "post_minus_live": pg - lv})

        theme_spikes_df = pd.DataFrame(rows).sort_values("post_minus_live", ascending=False)

        if len(theme_spikes_df) > 0:
            biggest_post = theme_spikes_df.iloc[0]
            biggest_live = theme_spikes_df.sort_values("post_minus_live", ascending=True).iloc[0]

            if int(biggest_post["post_minus_live"]) >= 10:
                bullets.append(
                    f"Postgame spike: '{biggest_post['theme']}' surged after the final "
                    f"(postgame +{int(biggest_post['post_minus_live'])} vs live_game)."
                )
            if int(biggest_live["post_minus_live"]) <= -10:
                bullets.append(
                    f"Live-game spike: '{biggest_live['theme']}' dominated during the game "
                    f"(live_game +{abs(int(biggest_live['post_minus_live']))} vs postgame)."
                )

    players_all = player_counts_for_df(g)
    top_players = players_all.most_common(6)
    player_focus_df = pd.DataFrame(top_players, columns=["player", "mentions"])

    if top_players:
        bullets.append("Most discussed players: " + ", ".join([f"{p} ({c})" for p, c in top_players[:3]]) + ".")

    heat_score = round((neg / max(total_comments, 1)) * 100 + (total_comments / 50), 1)
    if heat_score >= 70:
        bullets.append("Heat level: HIGH. High volume plus heavy negativity suggests elevated frustration / reputational risk.")
    elif heat_score >= 45:
        bullets.append("Heat level: MODERATE. Noticeable criticism, but not an all-out meltdown.")
    else:
        bullets.append("Heat level: LOW. Conversation leaned neutral-to-positive or stayed relatively calm.")

    fan_quotes = g.sort_values("score", ascending=False).head(10)
    cols = [c for c in ["game_date", "thread_type", "author", "score", "body"] if c in fan_quotes.columns]
    fan_quotes = fan_quotes[cols]

    return {
        "bullets": bullets,
        "heat_score": heat_score,
        "theme_spikes": theme_spikes_df,
        "player_focus": player_focus_df,
        "fan_quotes": fan_quotes,
        "themes_all": themes_all,
        "players_all": players_all,
        "sent_counts": s_all,
    }


def load_uploaded_csvs(files) -> pd.DataFrame:
    all_rows = []
    for f in files:
        game_date_str, thread_type, thread_id = parse_filename_meta(f.name)

        df = pd.read_csv(f)
        # Ensure required columns exist
        for col in ["body", "author", "score", "created_utc", "comment_id", "thread_id", "thread_type"]:
            if col not in df.columns:
                df[col] = None

        if "thread_type" in df.columns:
            df["thread_type"] = df["thread_type"].apply(normalize_thread_type)

        # Prefer filename thread_type/game_date if present
        if game_date_str:
            df["game_date"] = game_date_str
        else:
            if "game_date" not in df.columns:
                df["game_date"] = None

        df["thread_type"] = thread_type if thread_type != "unknown" else df["thread_type"]
        df["thread_id"] = thread_id if thread_id else df["thread_id"]

        # Sentiment
        df["body"] = df["body"].apply(safe_text)
        df["sentiment"] = df["body"].apply(classify_sentiment)

        # Themes + players (counts per comment, later we aggregate)
        df["themes_hit"] = df["body"].apply(lambda t: list(detect_mentions(t, THEMES).keys()))
        df["players_hit"] = df["body"].apply(lambda t: list(detect_mentions(t, PLAYERS).keys()))

        all_rows.append(df)

    if not all_rows:
        return pd.DataFrame()

    out = pd.concat(all_rows, ignore_index=True)
    # Normalize game_date to string YYYY-MM-DD when possible
    out["game_date"] = out["game_date"].astype(str).str.slice(0, 10)
    return out


# -----------------------------
# UI
# -----------------------------
st.title("Bulls Reddit Narrative Dashboard (Deterministic)")

st.sidebar.header("Upload your comment CSVs")
uploaded = st.sidebar.file_uploader(
    "Upload one or more files (comments_by_thread/*.csv)",
    type=["csv"],
    accept_multiple_files=True,
)

if not uploaded:
    st.info("Upload your `comments_by_thread/*.csv` files to start.")
    st.stop()

df = load_uploaded_csvs(uploaded)

if df.empty:
    st.error("No rows found. Make sure the uploaded CSVs contain comment rows.")
    st.stop()

# Filters
st.sidebar.header("Filters")
all_dates = sorted([d for d in df["game_date"].unique().tolist() if d and d != "None"])
default_date = all_dates[-1] if all_dates else None

game_date = st.sidebar.selectbox("Game date", options=all_dates, index=len(all_dates) - 1 if all_dates else 0)
thread_types = ["pregame", "live_game", "postgame"]
type_filter = st.sidebar.multiselect("Thread types", options=thread_types, default=thread_types)

f = df[(df["game_date"] == game_date) & (df["thread_type"].isin(type_filter))].copy()

tabs = st.tabs(["Dashboard", "Game-by-Game Report", "Weekly Report", "Raw Data"])


# -----------------------------
# Dashboard
# -----------------------------
with tabs[0]:
    st.subheader("Game Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Comments (filtered)", len(f))
    col2.metric("Unique authors", f["author"].nunique(dropna=True))
    col3.metric("Live game comments", int((f["thread_type"] == "live_game").sum()))
    col4.metric("Postgame comments", int((f["thread_type"] == "postgame").sum()))

    st.markdown("### Sentiment mix (filtered)")
    sent = f["sentiment"].value_counts()
    sent_df = sent.reset_index()
    sent_df.columns = ["sentiment", "count"]
    st.dataframe(sent_df, use_container_width=True)

    st.markdown("### Player mention leaderboard (filtered)")
    pc = player_counts_for_df(f)
    pc_df = pd.DataFrame(pc.most_common(20), columns=["player", "mentions"])
    st.dataframe(pc_df, use_container_width=True)

    st.markdown("### Theme tracking (filtered)")
    tc = theme_counts_for_df(f)
    tc_df = pd.DataFrame(tc.most_common(20), columns=["theme", "hits"])
    st.dataframe(tc_df, use_container_width=True)


# -----------------------------
# Game-by-Game Report
# -----------------------------
with tabs[1]:
    st.subheader(f"Game-by-Game Report: {game_date}")

    out = deterministic_game_narrative(df[df["game_date"] == game_date].copy())

    st.markdown("### Narrative Summary (deterministic rules)")
    for b in out["bullets"]:
        st.write(f"- {b}")

    st.metric("Heat score (deterministic)", out["heat_score"])

    if isinstance(out["theme_spikes"], pd.DataFrame) and len(out["theme_spikes"]) > 0:
        st.markdown("### Theme spikes (postgame vs live_game)")
        st.dataframe(out["theme_spikes"].head(12), use_container_width=True)

    st.markdown("### Player focus (mentions)")
    st.dataframe(out["player_focus"], use_container_width=True)

    st.markdown("### Fan quotes (top upvoted comments)")
    st.dataframe(out["fan_quotes"], use_container_width=True)


# -----------------------------
# Weekly Report (date range)
# -----------------------------
with tabs[2]:
    st.subheader("Weekly Report (date range)")

    # date range selection
    date_objs = []
    for d in all_dates:
        try:
            date_objs.append(datetime.strptime(d, "%Y-%m-%d").date())
        except Exception:
            pass

    if not date_objs:
        st.warning("Could not parse game_date values. Make sure filenames look like YYYY-MM-DD_live_game_THREADID.csv")
        st.stop()

    min_d, max_d = min(date_objs), max(date_objs)
    start_d, end_d = st.date_input("Select date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)

    if isinstance(start_d, tuple) or isinstance(start_d, list):
        # Streamlit sometimes returns tuple
        start_d, end_d = start_d

    weekly = df.copy()
    weekly["game_date_obj"] = pd.to_datetime(weekly["game_date"], errors="coerce").dt.date
    weekly = weekly[(weekly["game_date_obj"] >= start_d) & (weekly["game_date_obj"] <= end_d)]

    games_included = sorted([d for d in weekly["game_date"].unique().tolist() if d and d != "None"])
    st.write(f"- Games included: **{len(games_included)}**")
    st.write(f"- Total comments: **{len(weekly)}**")

    sent_all = weekly["sentiment"].value_counts()
    st.markdown("### Overall tone (heuristic)")
    st.dataframe(sent_all.reset_index().rename(columns={"index": "sentiment", "sentiment": "count"}), use_container_width=True)

    st.markdown("### Top themes (overall)")
    tc = theme_counts_for_df(weekly)
    st.dataframe(pd.DataFrame(tc.most_common(20), columns=["theme", "hits"]), use_container_width=True)

    st.markdown("### Top player mentions (overall)")
    pc = player_counts_for_df(weekly)
    st.dataframe(pd.DataFrame(pc.most_common(20), columns=["player", "mentions"]), use_container_width=True)

    st.markdown("### Game summaries (deterministic bullets)")
    for gd in games_included:
        st.markdown(f"#### {gd}")
        out = deterministic_game_narrative(weekly[weekly["game_date"] == gd].copy())
        for b in out["bullets"]:
            st.write(f"- {b}")


# -----------------------------
# Raw Data
# -----------------------------
with tabs[3]:
    st.subheader("Raw Data (filtered)")
    st.dataframe(f, use_container_width=True)
