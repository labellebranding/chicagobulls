# app.py
# Streamlit dashboard for Bulls subreddit game threads (deterministic rules only)

import re
from collections import Counter
from datetime import datetime, timezone, date
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Bulls Reddit Narrative Dashboard", layout="wide")


# -----------------------------
# Deterministic dictionaries (edit freely)
# -----------------------------
PLAYERS: Dict[str, List[str]] = {
    "Nikola Vucevic": [r"\bvooch\b", r"\bvucevic\b", r"\bvuc\b"],
    "Patrick Williams": [r"\bpwill\b", r"\bpatrick williams\b", r"\bpat\b"],
    "Coby White": [r"\bcoby\b", r"\bcoby white\b"],
    "Josh Giddey": [r"\bgiddey\b", r"\bjosh giddey\b"],
    "Lonzo Ball": [r"\blonzo\b", r"\blonzo ball\b"],
    "Ayo Dosunmu": [r"\bayo\b", r"\bdosunmu\b", r"\bayo dosunmu\b"],
    "Matas Buzelis": [r"\bmatas\b", r"\bbuzelis\b", r"\bmatas buzelis\b"],
    "Billy Donovan": [r"\bbilly donovan\b", r"\bdonovan\b"],
    # add more
}

THEMES: Dict[str, List[str]] = {
    "injury": [r"\binjur", r"\bconcussion\b", r"\bprotocol\b", r"\bout\b", r"\bquestionable\b"],
    "coaching": [r"\bcoach\b", r"\bcoaching\b", r"\blineup\b", r"\brotation\b", r"\btimeouts?\b", r"\bdonovan\b"],
    "shooting": [r"\bshoot", r"\b3s\b", r"\bthrees\b", r"\bthree\b", r"\bbrick", r"\bfg\b", r"\bpercent\b"],
    "refs": [r"\bref", r"\bwhistle\b", r"\bfoul\b", r"\bfree throw\b", r"\bft\b"],
    "front_office": [r"\bfront office\b", r"\bakme\b", r"\bkarnisovas\b", r"\btrade\b", r"\bdeadline\b"],
    "effort_identity": [r"\beffort\b", r"\bsoft\b", r"\bheart\b", r"\bidentity\b", r"\bvibes\b"],
    "tanking": [r"\btank\b", r"\blottery\b", r"\bpicks?\b", r"\btop pick\b"],
    # add more
}

NEG_WORDS = [
    r"\btrash\b", r"\bembarrass", r"\bawful\b", r"\bworst\b", r"\bpathetic\b",
    r"\bpissed\b", r"\bfuck\b", r"\bgarbage\b", r"\bchoke\b", r"\bsucks?\b",
]
POS_WORDS = [
    r"\bgreat\b", r"\bamazing\b", r"\blove\b", r"\bwin\b", r"\bsolid\b",
    r"\bproud\b", r"\bnice\b", r"\bclutch\b", r"\bballing\b",
]


# -----------------------------
# Filename parsing
# -----------------------------
FILENAME_RE = re.compile(
    r"(?P<game_date>\d{4}-\d{2}-\d{2})_(?P<thread_type>pregame|live_game|postgame|game)_(?P<thread_id>[a-z0-9]+)\.csv$",
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


# -----------------------------
# Helpers
# -----------------------------
def safe_text(x) -> str:
    return "" if pd.isna(x) else str(x)


def classify_sentiment(text: str) -> str:
    txt = (text or "").lower()
    neg = any(re.search(p, txt, flags=re.I) for p in NEG_WORDS)
    pos = any(re.search(p, txt, flags=re.I) for p in POS_WORDS)
    if neg and pos:
        return "mixed"
    if neg:
        return "negative"
    if pos:
        return "positive"
    return "neutral"


def comment_hits_any_patterns(text: str, pats: List[str]) -> bool:
    txt = text or ""
    return any(re.search(p, txt, flags=re.I) for p in pats)


def comment_hits_theme(body: str, theme: str) -> bool:
    pats = THEMES.get(theme, [])
    return comment_hits_any_patterns(body, pats)


def comment_hits_player(body: str, player: str) -> bool:
    pats = PLAYERS.get(player, [])
    return comment_hits_any_patterns(body, pats)


def pct(part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(100.0 * part / total, 1)


def player_counts_for_df(df_subset: pd.DataFrame) -> Counter:
    c = Counter()
    bodies = df_subset["body"].astype(str).tolist()
    for body in bodies:
        for player, pats in PLAYERS.items():
            hits = 0
            for p in pats:
                hits += len(re.findall(p, body, flags=re.I))
            if hits:
                c[player] += hits
    return c


def theme_counts_for_df(df_subset: pd.DataFrame) -> Counter:
    c = Counter()
    bodies = df_subset["body"].astype(str).tolist()
    for body in bodies:
        for theme, pats in THEMES.items():
            if comment_hits_any_patterns(body, pats):
                c[theme] += 1
    return c


def theme_kpi_table(df_subset: pd.DataFrame) -> pd.DataFrame:
    total = max(len(df_subset), 1)
    rows = []
    for theme in THEMES.keys():
        hit_mask = df_subset["body"].apply(lambda t: comment_hits_theme(t, theme))
        hits = int(hit_mask.sum())
        if hits == 0:
            continue

        sub = df_subset[hit_mask].copy()
        neg_pct = round(100.0 * (sub["sentiment"] == "negative").mean(), 1)

        live_hits = int((sub["thread_type"] == "live_game").sum())
        post_hits = int((sub["thread_type"] == "postgame").sum())
        spike = post_hits - live_hits

        rows.append(
            {
                "theme": theme,
                "hits": hits,
                "share_%": round(100.0 * hits / total, 1),
                "negative_%": neg_pct,
                "live_hits": live_hits,
                "post_hits": post_hits,
                "post_minus_live": spike,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["hits", "post_minus_live"], ascending=[False, False])


def top_comments_for_theme(df_subset: pd.DataFrame, theme: str, limit: int = 25) -> pd.DataFrame:
    x = df_subset.copy()
    x["hits_theme"] = x["body"].apply(lambda t: comment_hits_theme(t, theme))
    x = x[x["hits_theme"] == True].copy()

    x["score_num"] = pd.to_numeric(x["score"], errors="coerce").fillna(0).astype(int)
    cols = [c for c in ["game_date", "thread_type", "author", "score_num", "sentiment", "body"] if c in x.columns]
    x = x.sort_values("score_num", ascending=False)[cols].head(limit)
    return x.rename(columns={"score_num": "score"})


def most_negative_for_theme(df_subset: pd.DataFrame, theme: str, limit: int = 25) -> pd.DataFrame:
    x = df_subset.copy()
    x["hits_theme"] = x["body"].apply(lambda t: comment_hits_theme(t, theme))
    x = x[x["hits_theme"] == True].copy()
    x = x[x["sentiment"].isin(["negative", "mixed"])].copy()

    x["score_num"] = pd.to_numeric(x["score"], errors="coerce").fillna(0).astype(int)
    cols = [c for c in ["game_date", "thread_type", "author", "score_num", "sentiment", "body"] if c in x.columns]
    x = x.sort_values("score_num", ascending=False)[cols].head(limit)
    return x.rename(columns={"score_num": "score"})


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

    players_all = player_counts_for_df(g)
    top_players = players_all.most_common(6)
    if top_players:
        bullets.append("Most discussed: " + ", ".join([f"{p} ({c})" for p, c in top_players[:3]]) + ".")

    heat_score = round((neg / max(total_comments, 1)) * 100 + (total_comments / 60), 1)
    if heat_score >= 70:
        bullets.append("Heat level: HIGH. High volume plus heavy negativity suggests elevated frustration / reputational risk.")
    elif heat_score >= 45:
        bullets.append("Heat level: MODERATE. Noticeable criticism, but not an all-out meltdown.")
    else:
        bullets.append("Heat level: LOW. Conversation leaned neutral-to-positive or stayed relatively calm.")

    fan_quotes = g.copy()
    fan_quotes["score_num"] = pd.to_numeric(fan_quotes["score"], errors="coerce").fillna(0).astype(int)
    fan_quotes = fan_quotes.sort_values("score_num", ascending=False).head(15)
    cols = [c for c in ["game_date", "thread_type", "author", "score_num", "sentiment", "body"] if c in fan_quotes.columns]
    fan_quotes = fan_quotes[cols].rename(columns={"score_num": "score"})

    return {
        "bullets": bullets,
        "heat_score": heat_score,
        "themes_all": themes_all,
        "players_all": players_all,
        "sent_counts": s_all,
        "fan_quotes": fan_quotes,
    }


def _ensure_no_duplicate_columns(df_in: pd.DataFrame) -> pd.DataFrame:
    # Streamlit + pyarrow can error on duplicate column names
    cols = list(df_in.columns)
    seen = {}
    new_cols = []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            new_cols.append(c)
        else:
            seen[c] += 1
            new_cols.append(f"{c}__dup{seen[c]}")
    df_in.columns = new_cols
    return df_in


def load_uploaded_csvs(files) -> pd.DataFrame:
    all_rows = []

    for f in files:
        game_date_str, thread_type_from_name, thread_id_from_name = parse_filename_meta(f.name)

        df = pd.read_csv(f)
        df = _ensure_no_duplicate_columns(df)

        # Ensure standard columns exist
        for col in ["body", "author", "score", "created_utc", "comment_id", "thread_id", "thread_type"]:
            if col not in df.columns:
                df[col] = None

        # Normalize thread_type from file contents
        df["thread_type"] = df["thread_type"].apply(normalize_thread_type)

        # Override with filename meta if present
        if game_date_str:
            df["game_date"] = game_date_str
        else:
            if "game_date" not in df.columns:
                df["game_date"] = None

        if thread_type_from_name != "unknown":
            df["thread_type"] = thread_type_from_name

        if thread_id_from_name:
            df["thread_id"] = thread_id_from_name

        # Clean + enrich
        df["body"] = df["body"].apply(safe_text)
        df["sentiment"] = df["body"].apply(classify_sentiment)

        all_rows.append(df)

    if not all_rows:
        return pd.DataFrame()

    out = pd.concat(all_rows, ignore_index=True)
    out["game_date"] = out["game_date"].astype(str).str.slice(0, 10)
    out["thread_type"] = out["thread_type"].apply(normalize_thread_type)
    return out


# -----------------------------
# UI
# -----------------------------
st.title("Bulls Reddit Narrative Dashboard (Deterministic Rules)")

st.sidebar.header("Upload your comment CSVs")
uploaded = st.sidebar.file_uploader(
    "Upload one or more files (data/comments_by_thread/*.csv)",
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

all_dates = sorted([d for d in df["game_date"].dropna().unique().tolist() if d and d != "None"])
if not all_dates:
    st.error("No game_date values detected. Filenames should look like YYYY-MM-DD_live_game_THREADID.csv")
    st.stop()

default_idx = len(all_dates) - 1
game_date = st.sidebar.selectbox("Game date", options=all_dates, index=default_idx)

thread_types = ["pregame", "live_game", "postgame"]
type_filter = st.sidebar.multiselect("Thread types", options=thread_types, default=thread_types)

st.sidebar.markdown("### Search")
q = st.sidebar.text_input("Search comment text (contains)", value="").strip().lower()

f = df[(df["game_date"] == game_date) & (df["thread_type"].isin(type_filter))].copy()
if q:
    f = f[f["body"].str.lower().str.contains(re.escape(q), na=False)].copy()

tabs = st.tabs(["Dashboard", "Game-by-Game Report", "Weekly Report", "Raw Data"])


# -----------------------------
# Dashboard tab
# -----------------------------
with tabs[0]:
    st.subheader("Dashboard")

    total_comments = len(f)
    unique_authors = int(f["author"].nunique(dropna=True))
    live_ct = int((f["thread_type"] == "live_game").sum())
    post_ct = int((f["thread_type"] == "postgame").sum())

    sent_counts = f["sentiment"].value_counts()
    neg_ct = int(sent_counts.get("negative", 0))
    pos_ct = int(sent_counts.get("positive", 0))

    # Exec KPI strip
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    heat = round((neg_ct / max(total_comments, 1)) * 100 + (total_comments / 60), 1)
    col1.metric("Heat score", heat)
    col2.metric("Comments", total_comments)
    col3.metric("Unique authors", unique_authors)
    col4.metric("Neg %", f"{pct(neg_ct, total_comments)}%")
    col5.metric("Live game", live_ct)
    col6.metric("Postgame", post_ct)

    st.markdown("### Sentiment mix (filtered)")
    sent_df = sent_counts.rename_axis("sentiment").reset_index(name="count")
    st.dataframe(sent_df, use_container_width=True)

    st.markdown("### Player mention leaderboard (filtered)")
    pc = player_counts_for_df(f)
    pc_df = pd.DataFrame(pc.most_common(25), columns=["player", "mentions"])
    st.dataframe(pc_df, use_container_width=True)

    st.markdown("### Theme tracking (filtered)")
    theme_table = theme_kpi_table(f)
    if theme_table.empty:
        st.info("No theme hits found with current deterministic rules.")
    else:
        st.dataframe(theme_table, use_container_width=True)

        st.markdown("### Theme drill-down")
        theme_pick = st.selectbox("Select a theme to inspect", options=theme_table["theme"].tolist())

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Top upvoted comments for this theme")
            st.dataframe(top_comments_for_theme(f, theme_pick, limit=25), use_container_width=True)

        with c2:
            st.markdown("#### Most negative comments for this theme (sorted by score)")
            st.dataframe(most_negative_for_theme(f, theme_pick, limit=25), use_container_width=True)


# -----------------------------
# Game-by-Game Report tab
# -----------------------------
with tabs[1]:
    st.subheader(f"Game-by-Game Report: {game_date}")

    g = df[df["game_date"] == game_date].copy()
    out = deterministic_game_narrative(g)

    st.markdown("### Narrative summary (deterministic)")
    for b in out["bullets"]:
        st.write(f"- {b}")

    st.metric("Heat score (deterministic)", out["heat_score"])

    st.markdown("### Top themes (game)")
    tc = out["themes_all"]
    st.dataframe(pd.DataFrame(tc.most_common(20), columns=["theme", "hits"]), use_container_width=True)

    st.markdown("### Top player mentions (game)")
    pc = out["players_all"]
    st.dataframe(pd.DataFrame(pc.most_common(20), columns=["player", "mentions"]), use_container_width=True)

    st.markdown("### Fan quotes (top upvoted)")
    st.dataframe(out["fan_quotes"], use_container_width=True)


# -----------------------------
# Weekly Report tab
# -----------------------------
with tabs[2]:
    st.subheader("Weekly Report (date range)")

    # Parse available dates
    date_objs: List[date] = []
    for d in all_dates:
        try:
            date_objs.append(datetime.strptime(d, "%Y-%m-%d").date())
        except Exception:
            pass

    if not date_objs:
        st.warning("Could not parse game_date values. Filenames should look like YYYY-MM-DD_live_game_THREADID.csv")
        st.stop()

    min_d, max_d = min(date_objs), max(date_objs)

    # Streamlit can return a tuple here depending on version
    picked = st.date_input(
        "Select date range",
        value=(min_d, max_d),
        min_value=min_d,
        max_value=max_d,
    )

    if isinstance(picked, (tuple, list)) and len(picked) == 2:
        start_d, end_d = picked
    else:
        # fallback if streamlit returns a single date
        start_d, end_d = min_d, max_d

    weekly = df.copy()
    weekly["game_date_obj"] = pd.to_datetime(weekly["game_date"], errors="coerce").dt.date
    weekly = weekly[(weekly["game_date_obj"] >= start_d) & (weekly["game_date_obj"] <= end_d)].copy()

    games_included = sorted([d for d in weekly["game_date"].dropna().unique().tolist() if d and d != "None"])

    st.write(f"- Games included: **{len(games_included)}**")
    st.write(f"- Total comments: **{len(weekly)}**")

    st.markdown("### Overall tone (heuristic)")
    sent_all = weekly["sentiment"].value_counts()
    sent_all_df = sent_all.rename_axis("sentiment").reset_index(name="count")
    st.dataframe(sent_all_df, use_container_width=True)

    st.markdown("### Top themes (overall)")
    tc = theme_counts_for_df(weekly)
    st.dataframe(pd.DataFrame(tc.most_common(20), columns=["theme", "hits"]), use_container_width=True)

    st.markdown("### Top player mentions (overall)")
    pc = player_counts_for_df(weekly)
    st.dataframe(pd.DataFrame(pc.most_common(20), columns=["player", "mentions"]), use_container_width=True)

    st.markdown("### Theme drill-down (weekly)")
    theme_table_w = theme_kpi_table(weekly)
    if not theme_table_w.empty:
        st.dataframe(theme_table_w, use_container_width=True)
        theme_pick_w = st.selectbox("Select a theme (weekly)", options=theme_table_w["theme"].tolist(), key="theme_weekly")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Top upvoted comments (weekly, theme)")
            st.dataframe(top_comments_for_theme(weekly, theme_pick_w, limit=25), use_container_width=True)
        with c2:
            st.markdown("#### Most negative comments (weekly, theme)")
            st.dataframe(most_negative_for_theme(weekly, theme_pick_w, limit=25), use_container_width=True)

    st.markdown("### Game summaries (deterministic bullets)")
    for gd in games_included:
        st.markdown(f"#### {gd}")
        out = deterministic_game_narrative(weekly[weekly["game_date"] == gd].copy())
        for b in out["bullets"]:
            st.write(f"- {b}")


# -----------------------------
# Raw Data tab
# -----------------------------
with tabs[3]:
    st.subheader("Raw Data (filtered)")
    # Avoid pyarrow issues with duplicate col names
    safe_f = f.copy()
    safe_f = _ensure_no_duplicate_columns(safe_f)
    st.dataframe(safe_f, use_container_width=True)

