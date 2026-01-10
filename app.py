# app.py
# Bulls Fan Sentiment Intelligence (Exec-ready Streamlit)
# Deterministic rules only. No usernames shown except inside "Raw validation" expander.

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Optional charts
try:
    import altair as alt  # type: ignore
    ALTAIR_OK = True
except Exception:
    alt = None
    ALTAIR_OK = False


# -----------------------------
# Page config + premium Bulls styling
# -----------------------------
st.set_page_config(page_title="Bulls Fan Sentiment Intelligence", layout="wide")

BULLS_RED = "#CE1141"
BULLS_BLACK = "#0B0B0B"
TEXT = "#0F172A"
MUTED = "#64748B"
BORDER = "#E2E8F0"
BG = "#FFFFFF"
SOFT_BG = "#F8FAFC"
CARD_SHADOW = "0 18px 32px rgba(2,6,23,0.08)"

st.markdown(
    f"""
<style>
.stApp {{
  background: {BG};
  color: {TEXT};
}}
.block-container {{
  padding-top: 1.0rem;
  padding-bottom: 2.0rem;
  max-width: 1450px;
}}
h1,h2,h3 {{
  letter-spacing: -0.02em;
  color: {BULLS_BLACK};
}}
h1 {{
  font-weight: 900;
  margin-bottom: 0.1rem;
}}
.header-rule {{
  height: 4px;
  width: 92px;
  border-radius: 999px;
  background: {BULLS_RED};
  opacity: 0.95;
  margin-top: 8px;
  margin-bottom: 10px;
}}
.stCaption {{ color: {MUTED}; }}

section[data-testid="stSidebar"] > div {{
  background: {SOFT_BG};
  border-right: 1px solid {BORDER};
}}

.card {{
  background: {BG};
  border: 1px solid {BORDER};
  border-radius: 16px;
  padding: 14px 14px;
  box-shadow: {CARD_SHADOW};
}}
.card-soft {{
  background: {SOFT_BG};
  border: 1px solid {BORDER};
  border-radius: 16px;
  padding: 14px 14px;
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
@media (max-width: 720px) {{
  .kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
}}

.kpi {{
  background: {BG};
  border: 1px solid {BORDER};
  border-radius: 16px;
  padding: 12px 14px;
  box-shadow: {CARD_SHADOW};
}}
.kpi-label {{
  font-size: 0.85rem;
  color: {MUTED};
}}
.kpi-value {{
  font-size: 1.65rem;
  font-weight: 900;
  margin-top: 4px;
  color: {BULLS_BLACK};
}}
.kpi-sub {{
  font-size: 0.82rem;
  color: {MUTED};
  margin-top: 6px;
  line-height: 1.2;
}}

.pill {{
  display: inline-block;
  padding: 2px 10px;
  border-radius: 999px;
  border: 1px solid rgba(206,17,65,0.25);
  background: rgba(206,17,65,0.06);
  font-size: 0.78rem;
  color: {BULLS_BLACK};
  font-weight: 650;
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
# Deterministic dictionaries (expand freely)
# -----------------------------
PLAYERS: Dict[str, List[str]] = {
    "Matas Buzelis": [r"\bmatas\b", r"\bbuzelis\b", r"\bmatas buzelis\b"],
    "Coby White": [r"\bcoby\b", r"\bcoby white\b"],
    "Nikola Vucevic": [r"\bvooch\b", r"\bvucevic\b", r"\bvuc\b"],
    "Patrick Williams": [r"\bpwill\b", r"\bpatrick williams\b", r"\bpat will\b"],
    "Josh Giddey": [r"\bgiddey\b", r"\bjosh giddey\b"],
    "Ayo Dosunmu": [r"\bayo\b", r"\bdosunmu\b", r"\bayo dosunmu\b"],
    "Jevon Carter": [r"\bjevon\b", r"\bjevon carter\b"],
    "Kevin Huerter": [r"\bhuerter\b", r"\bkevin huerter\b"],
    "Tre Jones": [r"\btre jones\b"],
    "Julian Phillips": [r"\bjulian phillips\b", r"\bjulian\b"],
    "Dalen Terry": [r"\bdalen terry\b", r"\bdalen\b"],
    "Jalen Smith": [r"\bjalen smith\b"],
    "Zach Collins": [r"\bzach collins\b"],
    "Isaac Okoro": [r"\bisaac okoro\b", r"\bokoro\b"],
    "Noa Essengue": [r"\bnoa essengue\b", r"\bessengue\b"],
    "Yuki Kawamura": [r"\byuki kawamura\b", r"\bkawamura\b", r"\byuki\b"],

    "Billy Donovan": [r"\bbilly donovan\b", r"\bdonovan\b"],

    "Stacey King": [r"\bstacey king\b", r"\bstacey\b"],
    "Adam Amin": [r"\badam amin\b", r"\bamin\b"],
}

THEMES: Dict[str, List[str]] = {
    "coaching": [r"\bcoach\b", r"\bcoaching\b", r"\blineup\b", r"\brotation\b", r"\btimeouts?\b", r"\bdonovan\b"],
    "front office": [r"\bfront office\b", r"\bakme\b", r"\bkarnisovas\b", r"\btrade\b", r"\bdeadline\b"],
    "shooting": [r"\bshoot", r"\b3s\b", r"\bthrees\b", r"\bthree\b", r"\bbrick", r"\bfg\b", r"\bpercent\b"],
    "effort / identity": [r"\beffort\b", r"\bsoft\b", r"\bheart\b", r"\bidentity\b", r"\bvibes\b"],
    "injury": [r"\binjur", r"\bconcussion\b", r"\bprotocol\b", r"\bout\b", r"\bquestionable\b"],
    "refs": [r"\bref", r"\bwhistle\b", r"\bfoul\b", r"\bfree throw\b", r"\bft\b"],
    "tanking": [r"\btank\b", r"\blottery\b", r"\bpicks?\b", r"\btop pick\b"],
    "announcers": [r"\bstacey\b", r"\bstacey king\b", r"\badam amin\b", r"\bamin\b", r"\bbroadcast\b", r"\bcommentary\b"],
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
# Filename parsing for comments CSVs
# Expected: YYYY-MM-DD_live_game_THREADID.csv, etc.
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
    m = FILENAME_RE.search(name)
    if not m:
        return None, "unknown", None
    return m.group("game_date"), normalize_thread_type(m.group("thread_type")), m.group("thread_id")


# -----------------------------
# Helpers
# -----------------------------
def safe_text(x) -> str:
    return "" if pd.isna(x) else str(x)

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

def pct(part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(100.0 * part / total, 1)

def heat_score(df_subset: pd.DataFrame) -> float:
    total = max(len(df_subset), 1)
    neg_ct = int((df_subset["sentiment"] == "negative").sum())
    return round((neg_ct / total) * 100.0 + (len(df_subset) / 75.0), 1)

def comment_hits_any_patterns(text: str, pats: List[str]) -> bool:
    txt = text or ""
    return any(re.search(p, txt, flags=re.I) for p in pats)

def player_counts_for_df(df_subset: pd.DataFrame) -> Counter:
    c = Counter()
    bodies = df_subset["body"].astype(str).tolist()
    for body in bodies:
        for name, pats in PLAYERS.items():
            hits = 0
            for p in pats:
                hits += len(re.findall(p, body, flags=re.I))
            if hits:
                c[name] += hits
    return c

def theme_counts_for_df(df_subset: pd.DataFrame) -> Counter:
    c = Counter()
    bodies = df_subset["body"].astype(str).tolist()
    for body in bodies:
        for theme, pats in THEMES.items():
            if comment_hits_any_patterns(body, pats):
                c[theme] += 1
    return c

def safe_top_label(counter: Counter, default: str = "—") -> str:
    if not counter:
        return default
    items = counter.most_common(1)
    if not items:
        return default
    return str(items[0][0]) if items[0] and items[0][0] else default

def theme_kpi_table(df_subset: pd.DataFrame) -> pd.DataFrame:
    total = max(len(df_subset), 1)
    rows = []
    for theme, pats in THEMES.items():
        hit_mask = df_subset["body"].apply(lambda t: comment_hits_any_patterns(t, pats))
        hits = int(hit_mask.sum())
        if hits == 0:
            continue
        sub = df_subset[hit_mask].copy()
        neg_pct = round(100.0 * (sub["sentiment"] == "negative").mean(), 1)
        live = int((sub["thread_type"] == "live_game").sum())
        post = int((sub["thread_type"] == "postgame").sum())
        pre = int((sub["thread_type"] == "pregame").sum())
        rows.append({
            "Theme": theme,
            "Hits": hits,
            "Share %": round(100.0 * hits / total, 1),
            "Negative %": neg_pct,
            "Pregame": pre,
            "Live": live,
            "Postgame": post,
            "Post - Live": post - live
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["Hits", "Post - Live"], ascending=[False, False])

def top_comments_for_theme(df_subset: pd.DataFrame, theme: str, limit: int = 15) -> pd.DataFrame:
    pats = THEMES.get(theme, [])
    x = df_subset.copy()
    x["hits_theme"] = x["body"].apply(lambda t: comment_hits_any_patterns(t, pats))
    x = x[x["hits_theme"] == True].copy()
    cols = [c for c in ["thread_type", "score_num", "sentiment", "body"] if c in x.columns]
    x = x.sort_values("score_num", ascending=False)[cols].head(limit)
    return x.rename(columns={"score_num": "score (upvotes)", "thread_type": "context"})

def most_negative_for_theme(df_subset: pd.DataFrame, theme: str, limit: int = 15) -> pd.DataFrame:
    pats = THEMES.get(theme, [])
    x = df_subset.copy()
    x["hits_theme"] = x["body"].apply(lambda t: comment_hits_any_patterns(t, pats))
    x = x[(x["hits_theme"] == True) & (x["sentiment"].isin(["negative", "mixed"]))].copy()
    cols = [c for c in ["thread_type", "score_num", "sentiment", "body"] if c in x.columns]
    x = x.sort_values("score_num", ascending=False)[cols].head(limit)
    return x.rename(columns={"score_num": "score (upvotes)", "thread_type": "context"})

def build_exec_bullets(slice_df: pd.DataFrame) -> List[str]:
    total = len(slice_df)
    by_type = slice_df.groupby("thread_type").size().to_dict()
    pre = int(by_type.get("pregame", 0))
    live = int(by_type.get("live_game", 0))
    post = int(by_type.get("postgame", 0))

    sent = slice_df["sentiment"].value_counts()
    neg = int(sent.get("negative", 0))
    pos = int(sent.get("positive", 0))
    neu = int(sent.get("neutral", 0))
    mix = int(sent.get("mixed", 0))

    bullets = []
    bullets.append(f"Engagement: {total} comments (pregame {pre}, live {live}, postgame {post}).")
    bullets.append(
        f"Tone (heuristic): {pct(neg, total)}% negative, {pct(mix, total)}% mixed, "
        f"{pct(neu, total)}% neutral, {pct(pos, total)}% positive."
    )

    if live and post:
        if post > live * 1.25:
            bullets.append("Conversation intensified after the final (postgame > live).")
        elif live > post * 1.25:
            bullets.append("Conversation peaked during the game (live > postgame).")
        else:
            bullets.append("Engagement was steady (live and postgame similar).")

    top_themes = theme_counts_for_df(slice_df).most_common(4)
    if top_themes:
        bullets.append("Top narratives: " + ", ".join([t for t, _ in top_themes]) + ".")

    top_people = player_counts_for_df(slice_df).most_common(4)
    if top_people:
        bullets.append("Most discussed: " + ", ".join([n for n, _ in top_people]) + ".")

    hs = heat_score(slice_df)
    if hs >= 65:
        bullets.append("Risk pulse: HIGH. Elevated negativity at meaningful volume.")
    elif hs >= 40:
        bullets.append("Risk pulse: MODERATE. Criticism present but not a meltdown.")
    else:
        bullets.append("Risk pulse: LOW. Conversation mostly neutral-to-positive.")

    return bullets


# -----------------------------
# Optional threads.csv enrichment (opponent + home/away)
# -----------------------------
def load_threads_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df = _ensure_no_duplicate_columns(df)
    for col in ["thread_id", "title", "url", "created_utc", "thread_type"]:
        if col not in df.columns:
            df[col] = None
    df["thread_type"] = df["thread_type"].apply(normalize_thread_type)
    df["thread_id"] = df["thread_id"].astype(str)
    df["title"] = df["title"].astype(str)
    return df

def infer_opponent_home_away(title: str) -> Tuple[str, str]:
    t = (title or "").strip()

    if re.search(r"\b@\b", t):
        m = re.search(r"@\s*([A-Za-z .'-]+)", t)
        if m:
            opp = re.split(r"[\(\-–—|]", m.group(1).strip())[0].strip()
            return opp, "Away"

    if re.search(r"\bvs\.?\b", t, flags=re.I):
        m = re.search(r"\bvs\.?\s*([A-Za-z .'-]+)", t, flags=re.I)
        if m:
            opp = re.split(r"[\(\-–—|]", m.group(1).strip())[0].strip()
            return opp, "Home"

    m = re.search(r"\b(defeat|beat|lose to|lost to)\s+the\s+([A-Za-z .'-]+)", t, flags=re.I)
    if m:
        opp = re.split(r"[\(\-–—|]", m.group(2).strip())[0].strip()
        return opp, "Unknown"

    return "Unknown", "Unknown"


# -----------------------------
# Load comment CSVs
# -----------------------------
def load_comment_csvs(files) -> pd.DataFrame:
    all_rows = []
    for f in files:
        gd, tt, tid = parse_filename_meta(f.name)
        try:
            df = pd.read_csv(f)
        except Exception:
            f.seek(0)
            df = pd.read_csv(f, encoding_errors="ignore")

        df = _ensure_no_duplicate_columns(df)

        for col in ["body", "score", "thread_type", "thread_id", "game_date", "author", "created_utc", "comment_id"]:
            if col not in df.columns:
                df[col] = None

        if gd:
            df["game_date"] = gd
        df["game_date"] = df["game_date"].astype(str).str.slice(0, 10)

        if tt and tt != "unknown":
            df["thread_type"] = tt
        else:
            df["thread_type"] = df["thread_type"].apply(normalize_thread_type)

        if tid:
            df["thread_id"] = tid

        df["thread_type"] = df["thread_type"].apply(normalize_thread_type)
        df["thread_id"] = df["thread_id"].astype(str)

        df["body"] = df["body"].apply(safe_text)
        df["sentiment"] = df["body"].apply(classify_sentiment)
        df["score_num"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)

        # File coverage (for debugging selection issues)
        df["_source_file"] = f.name

        all_rows.append(df)

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


# -----------------------------
# UI
# -----------------------------
st.title("Bulls Fan Sentiment Intelligence")
st.markdown('<div class="header-rule"></div>', unsafe_allow_html=True)
st.caption("Exec view: risk, drivers, and evidence. Score = upvotes. Usernames are hidden.")

st.sidebar.header("Data Input")
comments_files = st.sidebar.file_uploader(
    "Upload comment CSVs (one per thread)",
    type=["csv"],
    accept_multiple_files=True
)

threads_file = st.sidebar.file_uploader(
    "Optional: upload threads.csv for opponent + home/away",
    type=["csv"],
    accept_multiple_files=False
)

if not comments_files:
    st.info("Upload your thread comment CSVs to begin.")
    st.stop()

df = load_comment_csvs(comments_files)
if df.empty:
    st.error("No comment rows found. Confirm the CSVs contain comment data and include a 'body' column.")
    st.stop()

threads_df = None
if threads_file is not None:
    threads_df = load_threads_csv(threads_file)

if threads_df is not None:
    meta = threads_df[["thread_id", "title", "url", "thread_type", "created_utc"]].copy()
    meta["thread_id"] = meta["thread_id"].astype(str)
    df = df.merge(meta, on=["thread_id"], how="left", suffixes=("", "_meta"))
    if "thread_type_meta" in df.columns:
        df["thread_type"] = df["thread_type"].where(df["thread_type"].notna(), df["thread_type_meta"])
        df["thread_type"] = df["thread_type"].apply(normalize_thread_type)
    df["title"] = df.get("title", "").astype(str)

all_dates = sorted([d for d in df["game_date"].dropna().unique().tolist() if re.match(r"^\d{4}-\d{2}-\d{2}$", str(d))])
if not all_dates:
    st.error("No valid game_date values. Ensure filenames are YYYY-MM-DD_live_game_THREADID.csv.")
    st.stop()

# Build games table (with coverage)
games_rows = []
for gd in all_dates:
    gdf = df[df["game_date"] == gd].copy()

    opp, ha = "Unknown", "Unknown"
    if "title" in gdf.columns and gdf["title"].notna().any():
        title_mode = gdf["title"].astype(str).replace("nan", "").replace("None", "")
        title_mode = title_mode[title_mode.str.len() > 0]
        if len(title_mode) > 0:
            t = title_mode.value_counts().index[0]
            opp, ha = infer_opponent_home_away(t)

    hs = heat_score(gdf)
    neg_pct = pct(int((gdf["sentiment"] == "negative").sum()), max(len(gdf), 1))
    vol = len(gdf)

    by_type = gdf.groupby("thread_type").size().to_dict()
    live = int(by_type.get("live_game", 0))
    post = int(by_type.get("postgame", 0))
    pre = int(by_type.get("pregame", 0))

    # coverage: which files are present?
    files = sorted(gdf["_source_file"].dropna().unique().tolist())
    games_rows.append({
        "game_date": gd,
        "opponent": opp,
        "home_away": ha,
        "heat_score": hs,
        "neg_%": neg_pct,
        "comments": vol,
        "pregame": pre,
        "live": live,
        "postgame": post,
        "files_count": len(files),
    })

games = pd.DataFrame(games_rows).sort_values("game_date", ascending=False)

st.sidebar.header("Game Selection")
games["label"] = games.apply(
    lambda r: f"{r['game_date']} • vs {r['opponent']} • {r['home_away']} • {int(r['comments'])} comments",
    axis=1
)
selected_label = st.sidebar.selectbox("Select game", options=games["label"].tolist(), index=0)
selected_game_date = games.loc[games["label"] == selected_label, "game_date"].iloc[0]

context_types = st.sidebar.multiselect(
    "Include contexts",
    options=["pregame", "live_game", "postgame"],
    default=["pregame", "live_game", "postgame"]
)

view_mode = st.sidebar.radio("View mode", options=["Game view", "Weekly view"], index=0)
search_text = st.sidebar.text_input("Search (contains)", value="").strip().lower()

slice_df = df[(df["game_date"] == selected_game_date) & (df["thread_type"].isin(context_types))].copy()
if search_text:
    slice_df = slice_df[slice_df["body"].str.lower().str.contains(re.escape(search_text), na=False)].copy()

if slice_df.empty:
    st.warning("No comments match the current filters.")
    st.stop()

# -----------------------------
# Coverage banner (prevents the “why only 12 comments?” confusion)
# -----------------------------
coverage = slice_df.groupby(["thread_type", "_source_file"]).size().reset_index(name="rows")
st.markdown('<div class="card-soft">', unsafe_allow_html=True)
st.markdown("<div style='font-weight:850; font-size:1.02rem;'>Upload Coverage (what is included in this view)</div>", unsafe_allow_html=True)
st.caption("If you only see 12 comments, it means only one CSV (or only one context) exists for that date.")
st.dataframe(coverage.rename(columns={"thread_type": "context", "_source_file": "file"}), use_container_width=True, hide_index=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# -----------------------------
# Executive Summary
# -----------------------------
st.subheader("Executive Summary")
sel_meta = games[games["game_date"] == selected_game_date].iloc[0].to_dict()

headline_left, headline_right = st.columns([2.2, 1.0])
with headline_left:
    st.markdown(
        f"""
<div class="card">
  <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
    <span class="pill">{selected_game_date}</span>
    <span class="pill">Opponent: {sel_meta.get('opponent','Unknown')}</span>
    <span class="pill">{sel_meta.get('home_away','Unknown')}</span>
  </div>
  <div style="margin-top:10px; color:{MUTED}; font-size:0.92rem;">
    This dashboard summarizes fan sentiment and narrative drivers from Reddit threads.
  </div>
</div>
""",
        unsafe_allow_html=True
    )
with headline_right:
    st.markdown(
        f"""
<div class="card-soft">
  <div style="color:{MUTED}; font-size:0.85rem; font-weight:700;">Data coverage</div>
  <div style="margin-top:6px; font-size:0.92rem;">
    Comments analyzed: <b>{int(sel_meta.get('comments',0))}</b><br/>
    Pregame: <b>{int(sel_meta.get('pregame',0))}</b> • Live: <b>{int(sel_meta.get('live',0))}</b> • Post: <b>{int(sel_meta.get('postgame',0))}</b><br/>
    Files uploaded for date: <b>{int(sel_meta.get('files_count',0))}</b>
  </div>
</div>
""",
        unsafe_allow_html=True
    )

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# KPIs
total = len(slice_df)
sent = slice_df["sentiment"].value_counts()
neg_ct = int(sent.get("negative", 0))
pos_ct = int(sent.get("positive", 0))

hs = heat_score(slice_df)
neg_pct = pct(neg_ct, max(total, 1))
pos_pct = pct(pos_ct, max(total, 1))

by_type = slice_df.groupby("thread_type").size().to_dict()
live_ct = int(by_type.get("live_game", 0))
post_ct = int(by_type.get("postgame", 0))

top_theme = safe_top_label(theme_counts_for_df(slice_df), default="None detected")
top_mention = safe_top_label(player_counts_for_df(slice_df), default="None detected")

st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
st.markdown(
    f"""
<div class="kpi">
  <div class="kpi-label">Risk Pulse (Heat)</div>
  <div class="kpi-value">{hs}</div>
  <div class="kpi-sub">Higher = more reputational volatility</div>
</div>
<div class="kpi">
  <div class="kpi-label">Negative Share</div>
  <div class="kpi-value">{neg_pct}%</div>
  <div class="kpi-sub">Heuristic sentiment (keyword-based)</div>
</div>
<div class="kpi">
  <div class="kpi-label">Positive Share</div>
  <div class="kpi-value">{pos_pct}%</div>
  <div class="kpi-sub">Momentum / praise context</div>
</div>
<div class="kpi">
  <div class="kpi-label">Engagement</div>
  <div class="kpi-value">{total}</div>
  <div class="kpi-sub">Comment volume in selected slice</div>
</div>
<div class="kpi">
  <div class="kpi-label">Live → Post Shift</div>
  <div class="kpi-value">{post_ct - live_ct}</div>
  <div class="kpi-sub">Postgame comments minus live</div>
</div>
<div class="kpi">
  <div class="kpi-label">Top Drivers</div>
  <div class="kpi-value">{top_theme}</div>
  <div class="kpi-sub">Top mention: {top_mention}</div>
</div>
""",
    unsafe_allow_html=True
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# Exec bullets
bullets = build_exec_bullets(slice_df)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown(f"<div style='font-weight:850; font-size:1.05rem; margin-bottom:8px;'>What leadership should know</div>", unsafe_allow_html=True)
for b in bullets:
    st.write(f"- {b}")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# -----------------------------
# Themes + drilldown
# -----------------------------
st.subheader("Narrative Drivers (Themes)")
theme_tbl = theme_kpi_table(slice_df)
if theme_tbl.empty:
    st.info("No theme hits detected with current rules. Expand THEME patterns to match more language.")
else:
    left, right = st.columns([1.25, 1.0])
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<div style='font-weight:850; font-size:1.02rem;'>Theme leaderboard</div>", unsafe_allow_html=True)
        st.caption("Hits = number of comments matching the theme. Negative % = risk within that theme.")
        st.dataframe(theme_tbl, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<div style='font-weight:850; font-size:1.02rem;'>Theme drill-down</div>", unsafe_allow_html=True)
        theme_pick = st.selectbox("Select a theme", options=theme_tbl["Theme"].tolist())
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top upvoted comments**")
            st.dataframe(top_comments_for_theme(slice_df, theme_pick, limit=12), use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**Most negative (by upvotes)**")
            st.dataframe(most_negative_for_theme(slice_df, theme_pick, limit=12), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# -----------------------------
# Mentions + evidence
# -----------------------------
st.subheader("Who the Conversation Centers On")
pc = player_counts_for_df(slice_df)
pc_df = pd.DataFrame(pc.most_common(15), columns=["Name", "Mentions"])

colA, colB = st.columns([1.1, 1.0])
with colA:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<div style='font-weight:850; font-size:1.02rem;'>Mention leaderboard</div>", unsafe_allow_html=True)
    st.caption("Mentions reflect narrative concentration, not performance quality.")
    if pc_df.empty:
        st.info("No mentions detected yet. Expand your patterns (PLAYERS) and add nicknames.")
    else:
        st.dataframe(pc_df, use_container_width=True, hide_index=True)
        if ALTAIR_OK:
            top = pc_df.head(10)
            chart = (
                alt.Chart(top)
                .mark_bar(color=BULLS_RED)
                .encode(
                    x=alt.X("Mentions:Q", title="Mentions"),
                    y=alt.Y("Name:N", sort="-x", title=""),
                    tooltip=["Name", "Mentions"],
                )
                .properties(height=320)
            )
            st.altair_chart(chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with colB:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<div style='font-weight:850; font-size:1.02rem;'>Evidence: top upvoted comments (overall)</div>", unsafe_allow_html=True)
    st.caption("Usernames hidden. Score = upvotes.")
    top_overall = slice_df.sort_values("score_num", ascending=False).copy()
    cols = [c for c in ["thread_type", "score_num", "sentiment", "body"] if c in top_overall.columns]
    top_overall = top_overall[cols].head(18).rename(columns={"score_num": "score (upvotes)", "thread_type": "context"})
    st.dataframe(top_overall, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# -----------------------------
# Recent games overview
# -----------------------------
st.subheader("Recent Games Overview")
st.caption("Compare risk and engagement across games at a glance.")
overview = games[["game_date", "opponent", "home_away", "heat_score", "neg_%", "comments", "pregame", "live", "postgame", "files_count"]].copy()
overview = overview.sort_values("game_date", ascending=False).rename(columns={
    "game_date": "Date",
    "opponent": "Opponent",
    "home_away": "Home/Away",
    "heat_score": "Heat",
    "neg_%": "Neg %",
    "comments": "Comments",
    "pregame": "Pregame",
    "live": "Live",
    "postgame": "Postgame",
    "files_count": "Files"
})
st.dataframe(overview, use_container_width=True, hide_index=True)

# -----------------------------
# Weekly view
# -----------------------------
if view_mode == "Weekly view":
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.subheader("Weekly Summary (date window)")

    parsed_dates: List[date] = []
    for d in all_dates:
        try:
            parsed_dates.append(datetime.strptime(d, "%Y-%m-%d").date())
        except Exception:
            pass
    if parsed_dates:
        min_d, max_d = min(parsed_dates), max(parsed_dates)
        c1, c2 = st.columns(2)
        with c1:
            start_d = st.date_input("Start date", value=min_d, min_value=min_d, max_value=max_d, key="wk_start")
        with c2:
            end_d = st.date_input("End date", value=max_d, min_value=min_d, max_value=max_d, key="wk_end")
        if start_d > end_d:
            start_d, end_d = end_d, start_d

        weekly = df.copy()
        weekly["game_date_obj"] = pd.to_datetime(weekly["game_date"], errors="coerce").dt.date
        weekly = weekly[(weekly["game_date_obj"] >= start_d) & (weekly["game_date_obj"] <= end_d)].copy()

        sent_all = weekly["sentiment"].value_counts()
        total_w = len(weekly)

        st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
        st.markdown(
            f"""
<div class="kpi">
  <div class="kpi-label">Games Included</div>
  <div class="kpi-value">{weekly["game_date"].nunique()}</div>
  <div class="kpi-sub">Distinct game dates in window</div>
</div>
<div class="kpi">
  <div class="kpi-label">Total Comments</div>
  <div class="kpi-value">{total_w}</div>
  <div class="kpi-sub">Engagement volume</div>
</div>
<div class="kpi">
  <div class="kpi-label">Negative Share</div>
  <div class="kpi-value">{pct(int(sent_all.get("negative",0)), max(total_w,1))}%</div>
  <div class="kpi-sub">Heuristic sentiment</div>
</div>
<div class="kpi">
  <div class="kpi-label">Top Theme</div>
  <div class="kpi-value">{safe_top_label(theme_counts_for_df(weekly), default="None detected")}</div>
  <div class="kpi-sub">Most frequent narrative</div>
</div>
<div class="kpi">
  <div class="kpi-label">Top Mention</div>
  <div class="kpi-value">{safe_top_label(player_counts_for_df(weekly), default="None detected")}</div>
  <div class="kpi-sub">Most discussed figure</div>
</div>
<div class="kpi">
  <div class="kpi-label">Risk Pulse (Heat)</div>
  <div class="kpi-value">{heat_score(weekly)}</div>
  <div class="kpi-sub">Negativity + volume</div>
</div>
""",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Raw validation (author only here)
# -----------------------------
with st.expander("Raw data (validation only — usernames may appear here)", expanded=False):
    safe_df = _ensure_no_duplicate_columns(slice_df.copy())
    st.dataframe(safe_df, use_container_width=True)

