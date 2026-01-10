# app.py
# Bulls Fan Sentiment Intelligence (Streamlit) — Demo-ready, deterministic rules
# - One-page executive scroll (no confusing tabs)
# - Game switcher (All games + any game_date)
# - Exec KPIs: heat, negativity, engagement, postgame-vs-live shifts
# - Drilldowns: click a theme/player and see top upvoted + most negative comments
# - Privacy: NO usernames anywhere except optional Raw Data expander (off by default)
#
# Expected input: your per-thread CSVs from scraper:
#   data/comments_by_thread/YYYY-MM-DD_{pregame|live_game|postgame}_THREADID.csv
#
# Required column: body
# Recommended: score, created_utc, thread_type, thread_id (scraper already provides)
#
# NOTE: This app uses deterministic regex rules ONLY (no API, no ML required).

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Optional charting
try:
    import altair as alt  # type: ignore
    ALTAIR_OK = True
except Exception:
    alt = None
    ALTAIR_OK = False


# -----------------------------
# Page config + styling (Bulls-inspired, exec-friendly)
# -----------------------------
st.set_page_config(page_title="Bulls Fan Sentiment Intelligence", layout="wide")

BULLS_RED = "#CE1141"
BULLS_BLACK = "#0B0B0B"
TEXT = "#111827"
MUTED = "#6B7280"
BORDER = "#E5E7EB"
BG = "#FFFFFF"
SOFT_BG = "#F9FAFB"
CARD_SHADOW = "0 10px 18px rgba(17,24,39,0.06)"

st.markdown(
    f"""
<style>
.stApp {{
  background: {BG};
  color: {TEXT};
}}
.block-container {{
  padding-top: 1.1rem;
  padding-bottom: 2rem;
  max-width: 1500px;
}}
h1, h2, h3 {{
  letter-spacing: -0.02em;
  color: {BULLS_BLACK};
}}
h1 {{
  font-weight: 900;
  line-height: 1.05;
}}
h1::after {{
  content: "";
  display: block;
  height: 4px;
  width: 110px;
  margin-top: 10px;
  border-radius: 999px;
  background: {BULLS_RED};
  opacity: 0.95;
}}
.stCaption {{
  color: {MUTED};
}}

section[data-testid="stSidebar"] > div {{
  background: {SOFT_BG};
  border-right: 1px solid {BORDER};
}}

.hr {{
  height: 1px;
  background: {BORDER};
  margin: 16px 0;
}}

.kpi-grid {{
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 12px;
}}
@media (max-width: 1200px) {{
  .kpi-grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
}}
@media (max-width: 700px) {{
  .kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
}}
.kpi-card {{
  background: {BG};
  border: 1px solid {BORDER};
  border-radius: 16px;
  padding: 12px 14px;
  box-shadow: {CARD_SHADOW};
}}
.kpi-label {{
  font-size: 0.88rem;
  color: {MUTED};
  display: flex;
  gap: 8px;
  align-items: center;
}}
.kpi-value {{
  font-size: 1.60rem;
  font-weight: 900;
  margin-top: 4px;
  color: {BULLS_BLACK};
}}
.kpi-sub {{
  font-size: 0.82rem;
  color: {MUTED};
  margin-top: 4px;
  line-height: 1.25;
}}

.section-card {{
  background: {SOFT_BG};
  border: 1px solid {BORDER};
  border-radius: 16px;
  padding: 14px 14px;
}}

.badge {{
  display: inline-block;
  padding: 2px 10px;
  border-radius: 999px;
  border: 1px solid rgba(206,17,65,0.25);
  background: rgba(206,17,65,0.08);
  font-size: 0.78rem;
  color: {BULLS_BLACK};
}}

[data-testid="stDataFrame"] {{
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid {BORDER};
}}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Deterministic dictionaries (EDIT THESE)
# -----------------------------
# Players / coaches / key broadcast names you care about.
# Add/remove freely. Regex aliases allow nicknames + shorthand.
PLAYERS: Dict[str, List[str]] = {
    # Core
    "Matas Buzelis": [r"\bmatas\b", r"\bbuzelis\b", r"\bmatas buzelis\b"],
    "Coby White": [r"\bcoby\b", r"\bcoby white\b"],
    "Nikola Vucevic": [r"\bvooch\b", r"\bvucevic\b", r"\bvuc\b", r"\bvuce\b"],
    "Patrick Williams": [r"\bpatrick williams\b", r"\bpwill\b", r"\bp will\b", r"\bpat will\b"],
    "Ayo Dosunmu": [r"\bayo\b", r"\bdosunmu\b", r"\bayo dosunmu\b"],
    "Josh Giddey": [r"\bgiddey\b", r"\bjosh giddey\b"],
    "Lonzo Ball": [r"\blonzo\b", r"\blonzo ball\b"],
    "Kevin Huerter": [r"\bhuerter\b", r"\bkevin huerter\b"],
    "Tre Jones": [r"\btre jones\b", r"\bjones\b"],
    "Jevon Carter": [r"\bjevon\b", r"\bjevon carter\b"],
    "Julian Phillips": [r"\bjulian phillips\b", r"\bphillips\b"],
    "Dalen Terry": [r"\bdalen\b", r"\bdalen terry\b"],
    "Jalen Smith": [r"\bjalen smith\b", r"\bj smith\b"],
    "Zach Collins": [r"\bzach collins\b", r"\bcollins\b"],
    "Isaac Okoro": [r"\bisaac okoro\b", r"\bokoro\b"],
    "Noa Essengue": [r"\bessengue\b", r"\bnoa essengue\b"],
    "Yuki Kawamura": [r"\bkawamura\b", r"\byuki\b", r"\byuki kawamura\b"],
    # Coach / org
    "Billy Donovan": [r"\bbilly donovan\b", r"\bdonovan\b", r"\bcoach donovan\b"],
    # Broadcast / media (example)
    "Stacey King": [r"\bstacey king\b", r"\bstacey\b", r"\bking\b"],
}

THEMES: Dict[str, List[str]] = {
    "injury": [r"\binjur", r"\bconcussion\b", r"\bprotocol\b", r"\bout\b", r"\bquestionable\b", r"\bday[- ]to[- ]day\b"],
    "coaching": [r"\bcoach\b", r"\bcoaching\b", r"\blineup\b", r"\brotation\b", r"\btimeouts?\b", r"\bdonovan\b"],
    "shooting": [r"\bshoot", r"\b3s\b", r"\bthrees\b", r"\bthree\b", r"\bbrick", r"\bfg\b", r"\bpercent\b", r"\bwide open\b"],
    "refs": [r"\bref", r"\bwhistle\b", r"\bfoul\b", r"\bfree throws?\b", r"\bft\b", r"\bcall\b"],
    "front_office": [r"\bfront office\b", r"\bakme\b", r"\bkarnisovas\b", r"\btrade\b", r"\bdeadline\b", r"\brebuild\b"],
    "effort_identity": [r"\beffort\b", r"\bsoft\b", r"\bheart\b", r"\bidentity\b", r"\bvibes\b", r"\benergy\b"],
    "tanking": [r"\btank\b", r"\blottery\b", r"\bpicks?\b", r"\btop pick\b"],
    "defense": [r"\bdefen", r"\bprotect the rim\b", r"\bcloseouts?\b", r"\btransition d\b"],
    "turnovers": [r"\bturnover", r"\bgiveaways?\b", r"\bcareless\b"],
    "announcers": [r"\bstacey king\b", r"\bcommentary\b", r"\bannounc", r"\bbroadcast\b"],
}

# Simple heuristic sentiment (deterministic keyword rules)
NEG_WORDS = [
    r"\btrash\b", r"\bembarrass", r"\bawful\b", r"\bworst\b", r"\bpathetic\b",
    r"\bpissed\b", r"\bfuck\b", r"\bgarbage\b", r"\bchoke\b", r"\bsucks?\b",
    r"\bfire\b", r"\bcut\b", r"\bwaive\b", r"\btrade him\b", r"\btrade her\b",
]
POS_WORDS = [
    r"\bgreat\b", r"\bamazing\b", r"\blove\b", r"\bwin\b", r"\bsolid\b",
    r"\bproud\b", r"\bnice\b", r"\bclutch\b", r"\bballing\b", r"\belite\b",
]

THREAD_TYPES = ["pregame", "live_game", "postgame"]

# If you know opponents/home/away, add them here (optional).
# This is the cleanest deterministic way, without using APIs.
# Example:
# GAME_CONTEXT = {
#   "2026-01-05": {"opponent": "Hornets", "venue": "Away"},
# }
GAME_CONTEXT: Dict[str, Dict[str, str]] = {}


# -----------------------------
# Filename parsing (expects YYYY-MM-DD_type_threadid.csv)
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
    m = FILENAME_RE.search(name or "")
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


def pct(part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(100.0 * part / total, 1)


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


def _ensure_no_duplicate_columns(df_in: pd.DataFrame) -> pd.DataFrame:
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


def comment_hits_any(text: str, pats: List[str]) -> bool:
    t = text or ""
    return any(re.search(p, t, flags=re.I) for p in pats)


def theme_counts_for_df(df_subset: pd.DataFrame) -> Counter:
    c = Counter()
    bodies = df_subset["body"].astype(str).tolist()
    for body in bodies:
        for theme, pats in THEMES.items():
            if comment_hits_any(body, pats):
                c[theme] += 1
    return c


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


def heat_score(df_subset: pd.DataFrame) -> float:
    total = max(len(df_subset), 1)
    neg_ct = int((df_subset["sentiment"] == "negative").sum())
    # Simple and stable across games
    return round((neg_ct / total) * 100.0 + (len(df_subset) / 60.0), 1)


def theme_kpi_table(df_subset: pd.DataFrame) -> pd.DataFrame:
    total = max(len(df_subset), 1)
    rows = []
    for theme in THEMES.keys():
        hit_mask = df_subset["body"].apply(lambda t: comment_hits_any(t, THEMES.get(theme, [])))
        hits = int(hit_mask.sum())
        if hits == 0:
            continue

        sub = df_subset[hit_mask].copy()
        neg_pct = round(100.0 * (sub["sentiment"] == "negative").mean(), 1)
        mix_pct = round(100.0 * (sub["sentiment"] == "mixed").mean(), 1)

        live_hits = int((sub["thread_type"] == "live_game").sum())
        post_hits = int((sub["thread_type"] == "postgame").sum())
        pre_hits = int((sub["thread_type"] == "pregame").sum())
        spike = post_hits - live_hits

        rows.append(
            {
                "theme": theme,
                "hits": hits,
                "share_%": round(100.0 * hits / total, 1),
                "negative_%": neg_pct,
                "mixed_%": mix_pct,
                "pregame_hits": pre_hits,
                "live_hits": live_hits,
                "post_hits": post_hits,
                "post_minus_live": spike,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["hits", "negative_%", "post_minus_live"], ascending=[False, False, False])


def player_kpi_table(df_subset: pd.DataFrame) -> pd.DataFrame:
    pc = player_counts_for_df(df_subset)
    if not pc:
        return pd.DataFrame()
    out = pd.DataFrame(pc.most_common(30), columns=["entity", "mentions"])
    out.rename(columns={"entity": "player_or_staff"}, inplace=True)
    return out


def top_comments(df_subset: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    x = df_subset.copy()
    cols = [c for c in ["game_date", "thread_type", "score_num", "sentiment", "body"] if c in x.columns]
    x = x.sort_values("score_num", ascending=False)[cols].head(limit)
    return x.rename(columns={"score_num": "score (upvotes)"})


def top_comments_for_theme(df_subset: pd.DataFrame, theme: str, limit: int = 25) -> pd.DataFrame:
    pats = THEMES.get(theme, [])
    x = df_subset.copy()
    x = x[x["body"].apply(lambda t: comment_hits_any(t, pats))].copy()
    cols = [c for c in ["game_date", "thread_type", "score_num", "sentiment", "body"] if c in x.columns]
    x = x.sort_values("score_num", ascending=False)[cols].head(limit)
    return x.rename(columns={"score_num": "score (upvotes)"})


def most_negative_for_theme(df_subset: pd.DataFrame, theme: str, limit: int = 25) -> pd.DataFrame:
    pats = THEMES.get(theme, [])
    x = df_subset.copy()
    x = x[x["body"].apply(lambda t: comment_hits_any(t, pats))].copy()
    x = x[x["sentiment"].isin(["negative", "mixed"])].copy()
    cols = [c for c in ["game_date", "thread_type", "score_num", "sentiment", "body"] if c in x.columns]
    x = x.sort_values("score_num", ascending=False)[cols].head(limit)
    return x.rename(columns={"score_num": "score (upvotes)"})


def top_comments_for_player(df_subset: pd.DataFrame, player: str, limit: int = 25) -> pd.DataFrame:
    pats = PLAYERS.get(player, [])
    x = df_subset.copy()
    x = x[x["body"].apply(lambda t: comment_hits_any(t, pats))].copy()
    cols = [c for c in ["game_date", "thread_type", "score_num", "sentiment", "body"] if c in x.columns]
    x = x.sort_values("score_num", ascending=False)[cols].head(limit)
    return x.rename(columns={"score_num": "score (upvotes)"})


def narrative_bullets(df_slice: pd.DataFrame) -> List[str]:
    bullets: List[str] = []
    total = len(df_slice)

    by_type = df_slice.groupby("thread_type").size().to_dict()
    pre_ct = int(by_type.get("pregame", 0))
    live_ct = int(by_type.get("live_game", 0))
    post_ct = int(by_type.get("postgame", 0))

    sent = df_slice["sentiment"].value_counts()
    neg = int(sent.get("negative", 0))
    pos = int(sent.get("positive", 0))
    neu = int(sent.get("neutral", 0))
    mix = int(sent.get("mixed", 0))

    bullets.append(f"Engagement: {total} comments (Pregame {pre_ct}, Live {live_ct}, Post {post_ct}).")
    bullets.append(
        f"Tone (heuristic): {pct(neg, total)}% negative, {pct(mix, total)}% mixed, "
        f"{pct(neu, total)}% neutral, {pct(pos, total)}% positive."
    )

    if live_ct > 0 and post_ct > 0:
        if post_ct > live_ct * 1.25:
            bullets.append("Postgame volume rose vs live, suggesting stronger reflection/blame assignment after the final.")
        elif live_ct > post_ct * 1.25:
            bullets.append("Live thread dominated volume, suggesting in-game reactions were the primary narrative driver.")
        else:
            bullets.append("Live and postgame volume were similar, suggesting steady engagement before/after the final.")

    tc = theme_counts_for_df(df_slice)
    if tc:
        top = [k for k, _ in tc.most_common(4)]
        bullets.append("Top narratives: " + ", ".join(top) + ".")

    pc = player_counts_for_df(df_slice)
    if pc:
        top_p = pc.most_common(3)
        bullets.append("Most discussed: " + ", ".join([f"{p} ({c})" for p, c in top_p]) + ".")

    hs = heat_score(df_slice)
    if hs >= 70:
        bullets.append("Heat: HIGH. Elevated frustration / reputational risk signal.")
    elif hs >= 45:
        bullets.append("Heat: MODERATE. Noticeable criticism, not a full meltdown.")
    else:
        bullets.append("Heat: LOW. Conversation relatively calm / neutral-to-positive.")

    return bullets


# -----------------------------
# Data loading (Upload CSVs)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_uploaded_csvs(files) -> pd.DataFrame:
    all_rows = []
    for f in files:
        game_date_str, tt_from_name, tid_from_name = parse_filename_meta(getattr(f, "name", ""))

        try:
            df = pd.read_csv(f)
        except Exception:
            f.seek(0)
            df = pd.read_csv(f, encoding_errors="ignore")

        df = _ensure_no_duplicate_columns(df)

        # Ensure standard columns exist
        for col in ["body", "author", "score", "created_utc", "comment_id", "thread_id", "thread_type", "game_date"]:
            if col not in df.columns:
                df[col] = None

        # Normalize
        df["body"] = df["body"].apply(safe_text)
        df["thread_type"] = df["thread_type"].apply(normalize_thread_type)

        # Prefer filename metadata
        if game_date_str:
            df["game_date"] = game_date_str
        df["game_date"] = df["game_date"].astype(str).str.slice(0, 10)

        if tt_from_name != "unknown":
            df["thread_type"] = tt_from_name

        if tid_from_name:
            df["thread_id"] = tid_from_name

        # Enrich
        df["sentiment"] = df["body"].apply(classify_sentiment)
        df["score_num"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)

        all_rows.append(df)

    if not all_rows:
        return pd.DataFrame()

    out = pd.concat(all_rows, ignore_index=True)
    out["game_date"] = out["game_date"].astype(str).str.slice(0, 10)
    out["thread_type"] = out["thread_type"].apply(normalize_thread_type)

    # Hard-clean: keep only expected contexts
    out = out[out["thread_type"].isin(THREAD_TYPES)].copy()
    return out


def safe_df_for_display(df_in: pd.DataFrame, allow_author: bool) -> pd.DataFrame:
    df_out = df_in.copy()
    # Remove potentially identifying fields unless explicitly allowed
    drop_cols = []
    if not allow_author and "author" in df_out.columns:
        drop_cols.append("author")
    if "thread_id" in df_out.columns:
        drop_cols.append("thread_id")
    if "comment_id" in df_out.columns:
        drop_cols.append("comment_id")
    if "permalink" in df_out.columns:
        drop_cols.append("permalink")
    df_out = df_out.drop(columns=[c for c in drop_cols if c in df_out.columns], errors="ignore")
    return df_out


# -----------------------------
# Header
# -----------------------------
st.title("Bulls Fan Sentiment Intelligence")
st.caption("Deterministic sentiment + narrative drivers from Reddit live & postgame threads. Score = upvotes.")

# -----------------------------
# Sidebar controls (simple + reliable)
# -----------------------------
st.sidebar.header("1) Upload thread CSVs")
uploaded = st.sidebar.file_uploader(
    "Upload one or more per-thread comment CSV files",
    type=["csv"],
    accept_multiple_files=True,
    key="uploader",
)

st.sidebar.markdown("---")
st.sidebar.header("2) View controls")

show_author_raw = st.sidebar.toggle(
    "Show usernames (Raw Data only)",
    value=False,
    help="Leave off for demos. Only enable if you need to validate raw rows.",
    key="show_author_raw",
)

if not uploaded:
    st.info("Upload your `comments_by_thread/*.csv` files to start.")
    st.stop()

df = load_uploaded_csvs(uploaded)

if df.empty:
    st.error("No rows found. Make sure your CSVs include a 'body' column and filenames include YYYY-MM-DD_.")
    st.stop()

# Game options
all_dates = sorted(
    [d for d in df["game_date"].dropna().unique().tolist() if re.match(r"^\d{4}-\d{2}-\d{2}$", str(d))],
    reverse=True,
)
GAME_ALL = "All games (weekly view)"
game_options = [GAME_ALL] + all_dates

selected_game = st.sidebar.selectbox(
    "Select game",
    options=game_options,
    index=0,
    key="selected_game_date",
)

contexts = st.sidebar.multiselect(
    "Thread contexts",
    options=THREAD_TYPES,
    default=THREAD_TYPES,
    key="selected_contexts",
)

q = st.sidebar.text_input("Search comment text (contains)", value="", key="search_q").strip().lower()

# Slice
if selected_game == GAME_ALL:
    slice_df = df[df["thread_type"].isin(contexts)].copy()
else:
    slice_df = df[(df["game_date"] == selected_game) & (df["thread_type"].isin(contexts))].copy()

if q:
    slice_df = slice_df[slice_df["body"].str.lower().str.contains(re.escape(q), na=False)].copy()

if slice_df.empty:
    st.warning("No comments match your filters.")
    st.stop()

# Context panel (opponent/home-away)
def game_context_line(game_date: str) -> str:
    meta = GAME_CONTEXT.get(game_date, {})
    opp = meta.get("opponent", "Unknown")
    venue = meta.get("venue", "Unknown")  # Home/Away
    return f"Opponent: {opp} • Venue: {venue}"

# -----------------------------
# Executive summary (top)
# -----------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)

if selected_game == GAME_ALL:
    st.markdown(f"### Executive Summary <span class='badge'>Weekly view</span>", unsafe_allow_html=True)
    st.write("This dashboard summarizes fan sentiment and narrative drivers across all uploaded games.")
else:
    st.markdown(f"### Executive Summary <span class='badge'>{selected_game}</span>", unsafe_allow_html=True)
    st.caption(game_context_line(selected_game))

bullets = narrative_bullets(slice_df)
for b in bullets:
    st.write(f"- {b}")

st.markdown("</div>", unsafe_allow_html=True)
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# -----------------------------
# KPI row (exec-friendly)
# -----------------------------
total_comments = len(slice_df)
unique_commenters = int(slice_df["author"].nunique(dropna=True)) if "author" in slice_df.columns else 0
by_type = slice_df.groupby("thread_type").size().to_dict()
pre_ct = int(by_type.get("pregame", 0))
live_ct = int(by_type.get("live_game", 0))
post_ct = int(by_type.get("postgame", 0))

sent_counts = slice_df["sentiment"].value_counts()
neg_ct = int(sent_counts.get("negative", 0))
mix_ct = int(sent_counts.get("mixed", 0))
pos_ct = int(sent_counts.get("positive", 0))
neu_ct = int(sent_counts.get("neutral", 0))

hs = heat_score(slice_df)
neg_pct = pct(neg_ct, max(total_comments, 1))

st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
st.markdown(f"""
<div class="kpi-card">
  <div class="kpi-label">Heat score</div>
  <div class="kpi-value">{hs}</div>
  <div class="kpi-sub">Higher = more volatility (negativity + volume).</div>
</div>
<div class="kpi-card">
  <div class="kpi-label">Negative %</div>
  <div class="kpi-value">{neg_pct}%</div>
  <div class="kpi-sub">Heuristic keyword sentiment.</div>
</div>
<div class="kpi-card">
  <div class="kpi-label">Comments analyzed</div>
  <div class="kpi-value">{total_comments}</div>
  <div class="kpi-sub">Based on uploaded thread CSVs.</div>
</div>
<div class="kpi-card">
  <div class="kpi-label">Unique commenters</div>
  <div class="kpi-value">{unique_commenters}</div>
  <div class="kpi-sub">Breadth signal (usernames hidden by default).</div>
</div>
<div class="kpi-card">
  <div class="kpi-label">Live vs Post volume</div>
  <div class="kpi-value">{live_ct} / {post_ct}</div>
  <div class="kpi-sub">Live game vs postgame comments.</div>
</div>
<div class="kpi-card">
  <div class="kpi-label">Pregame volume</div>
  <div class="kpi-value">{pre_ct}</div>
  <div class="kpi-sub">Pregame anticipation / expectations.</div>
</div>
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# -----------------------------
# Trend context (exec: weekly + comparison)
# -----------------------------
st.subheader("Trend Context")

trend = df[df["thread_type"].isin(contexts)].copy()
trend["neg"] = (trend["sentiment"] == "negative").astype(int)
trend["pos"] = (trend["sentiment"] == "positive").astype(int)

trend_by_game = (
    trend.groupby("game_date")
    .agg(
        comments=("body", "size"),
        negative_pct=("neg", "mean"),
        positive_pct=("pos", "mean"),
        pre=("thread_type", lambda s: (s == "pregame").sum()),
        live=("thread_type", lambda s: (s == "live_game").sum()),
        post=("thread_type", lambda s: (s == "postgame").sum()),
    )
    .reset_index()
    .sort_values("game_date")
)

trend_by_game["negative_pct"] = (trend_by_game["negative_pct"] * 100).round(1)
trend_by_game["positive_pct"] = (trend_by_game["positive_pct"] * 100).round(1)
trend_by_game["post_minus_live"] = trend_by_game["post"] - trend_by_game["live"]

c1, c2 = st.columns([1.15, 0.85])

with c1:
    st.markdown("### Game-by-game overview")
    st.dataframe(
        trend_by_game.sort_values("game_date", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

with c2:
    st.markdown("### Latest vs weekly baseline")
    # baseline = average across games
    baseline_neg = float(trend_by_game["negative_pct"].mean()) if len(trend_by_game) else 0.0
    baseline_comments = float(trend_by_game["comments"].mean()) if len(trend_by_game) else 0.0

    if selected_game != GAME_ALL and selected_game in set(trend_by_game["game_date"].tolist()):
        row = trend_by_game[trend_by_game["game_date"] == selected_game].iloc[0]
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.write(f"**Selected game:** {selected_game}")
        st.write(f"- Negative %: **{row['negative_pct']}%** (weekly avg {baseline_neg:.1f}%)")
        st.write(f"- Comments: **{int(row['comments'])}** (weekly avg {baseline_comments:.0f})")
        st.write(f"- Post minus Live: **{int(row['post_minus_live'])}** (postgame reaction shift)")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.write("**Weekly baseline (all uploaded games)**")
        st.write(f"- Avg negative %: **{baseline_neg:.1f}%**")
        st.write(f"- Avg comments per game: **{baseline_comments:.0f}**")
        st.markdown("</div>", unsafe_allow_html=True)

if ALTAIR_OK and len(trend_by_game) >= 2:
    st.markdown("### Visual trend (Negative % and Volume)")
    t = trend_by_game.copy()
    t["game_date"] = pd.to_datetime(t["game_date"], errors="coerce")
    t = t.dropna(subset=["game_date"]).copy()

    line1 = (
        alt.Chart(t)
        .mark_line(point=True, color=BULLS_RED)
        .encode(
            x=alt.X("game_date:T", title="Game date"),
            y=alt.Y("negative_pct:Q", title="Negative %"),
            tooltip=["game_date:T", "negative_pct:Q", "comments:Q"],
        )
        .properties(height=220)
    )
    bar = (
        alt.Chart(t)
        .mark_bar(opacity=0.20, color=BULLS_BLACK)
        .encode(
            x=alt.X("game_date:T", title=""),
            y=alt.Y("comments:Q", title="Comments"),
            tooltip=["game_date:T", "comments:Q"],
        )
        .properties(height=220)
    )
    st.altair_chart(line1 & bar, use_container_width=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# -----------------------------
# Narrative drivers (themes)
# -----------------------------
st.subheader("Narrative Drivers (Themes)")

theme_table = theme_kpi_table(slice_df)
if theme_table.empty:
    st.info("No theme hits detected with the current deterministic theme rules.")
else:
    c1, c2 = st.columns([1.05, 0.95])

    with c1:
        st.markdown("### Theme KPI table")
        st.dataframe(theme_table, use_container_width=True, hide_index=True)

    with c2:
        st.markdown("### Theme drilldown (evidence)")
        theme_pick = st.selectbox(
            "Select a theme to view supporting comments",
            options=theme_table["theme"].tolist(),
            key="theme_pick",
        )
        st.caption("Top comments are ranked by score (upvotes).")

        a, b = st.columns(2)
        with a:
            st.markdown("**Top upvoted (theme)**")
            st.dataframe(
                safe_df_for_display(top_comments_for_theme(slice_df, theme_pick, 25), allow_author=False),
                use_container_width=True,
                hide_index=True,
            )
        with b:
            st.markdown("**Most negative (theme)**")
            st.dataframe(
                safe_df_for_display(most_negative_for_theme(slice_df, theme_pick, 25), allow_author=False),
                use_container_width=True,
                hide_index=True,
            )

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# -----------------------------
# Player / staff focus (mentions)
# -----------------------------
st.subheader("Player + Staff Focus (Mentions)")

player_table = player_kpi_table(slice_df)
if player_table.empty:
    st.info("No player/staff mentions detected with the current dictionary. Add aliases to PLAYERS.")
else:
    c1, c2 = st.columns([1.0, 1.0])

    with c1:
        st.markdown("### Mentions leaderboard")
        st.dataframe(player_table, use_container_width=True, hide_index=True)

    with c2:
        st.markdown("### Player drilldown (evidence)")
        pick = st.selectbox(
            "Select a player/staff to view top comments",
            options=player_table["player_or_staff"].tolist(),
            key="player_pick",
        )
        st.dataframe(
            safe_df_for_display(top_comments_for_player(slice_df, pick, 25), allow_author=False),
            use_container_width=True,
            hide_index=True,
        )

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# -----------------------------
# Top fan quotes (overall)
# -----------------------------
st.subheader("Top Fan Quotes (Highest Upvoted)")

st.caption("These are the most visible / agreed-with comments in the selected slice.")
st.dataframe(
    safe_df_for_display(top_comments(slice_df, limit=25), allow_author=False),
    use_container_width=True,
    hide_index=True,
)

# -----------------------------
# Optional raw data (hidden behind expander + toggle)
# -----------------------------
with st.expander("Raw Data (for validation only)", expanded=False):
    st.caption("Thread IDs and permalinks are hidden. Usernames appear only if you enable the toggle in the sidebar.")
    raw_view = safe_df_for_display(slice_df, allow_author=show_author_raw)
    st.dataframe(raw_view, use_container_width=True)
