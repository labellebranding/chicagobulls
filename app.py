# app.py
# Bulls Fan Sentiment Intelligence (Exec-ready)
# Deterministic rules only
# - One-page scroll UI (no confusing tabs)
# - Game switcher (works even if filenames are inconsistent)
# - Data Health section to show exactly what the app loaded
# - Theme drill-down: top comments per theme (score = upvotes)
# - No usernames shown anywhere (except optional Raw Validation expander)

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
# Page config + styling
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
.stApp {{ background:{BG}; color:{TEXT}; }}
.block-container {{ padding-top: 1.0rem; padding-bottom: 2.0rem; max-width: 1450px; }}
h1,h2,h3 {{ letter-spacing:-0.02em; color:{BULLS_BLACK}; }}
h1 {{ font-weight: 900; margin-bottom: 0.1rem; }}
.header-rule {{ height:4px; width:92px; border-radius:999px; background:{BULLS_RED}; opacity:0.95; margin:8px 0 12px 0; }}
.stCaption {{ color:{MUTED}; }}

section[data-testid="stSidebar"] > div {{
  background:{SOFT_BG};
  border-right: 1px solid {BORDER};
}}

.card {{
  background:{BG};
  border:1px solid {BORDER};
  border-radius:16px;
  padding:14px;
  box-shadow:{CARD_SHADOW};
}}
.card-soft {{
  background:{SOFT_BG};
  border:1px solid {BORDER};
  border-radius:16px;
  padding:14px;
}}
.hr {{ height:1px; background:{BORDER}; margin:16px 0; }}

.kpi-grid {{
  display:grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap:12px;
}}
@media (max-width: 1200px) {{
  .kpi-grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
}}
@media (max-width: 720px) {{
  .kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
}}

.kpi {{
  background:{BG};
  border:1px solid {BORDER};
  border-radius:16px;
  padding:12px 14px;
  box-shadow:{CARD_SHADOW};
}}
.kpi-label {{ font-size:0.85rem; color:{MUTED}; }}
.kpi-value {{ font-size:1.65rem; font-weight:900; margin-top:4px; color:{BULLS_BLACK}; }}
.kpi-sub {{ font-size:0.82rem; color:{MUTED}; margin-top:6px; line-height:1.2; }}

.pill {{
  display:inline-block;
  padding:2px 10px;
  border-radius:999px;
  border:1px solid rgba(206,17,65,0.25);
  background: rgba(206,17,65,0.06);
  font-size:0.78rem;
  color:{BULLS_BLACK};
  font-weight:650;
}}

[data-testid="stDataFrame"] {{
  border-radius:16px;
  overflow:hidden;
  border:1px solid {BORDER};
}}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Dictionaries (edit freely)
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
    m = FILENAME_RE.search(name or "")
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
    # simple, stable: tone + volume
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
    top = counter.most_common(1)
    if not top:
        return default
    return str(top[0][0]) if top[0][0] else default

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
        pre = int((sub["thread_type"] == "pregame").sum())
        live = int((sub["thread_type"] == "live_game").sum())
        post = int((sub["thread_type"] == "postgame").sum())
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

def top_comments_for_theme(df_subset: pd.DataFrame, theme: str, limit: int = 12) -> pd.DataFrame:
    pats = THEMES.get(theme, [])
    x = df_subset.copy()
    x["hits_theme"] = x["body"].apply(lambda t: comment_hits_any_patterns(t, pats))
    x = x[x["hits_theme"] == True].copy()
    cols = [c for c in ["thread_type", "score_num", "sentiment", "body"] if c in x.columns]
    x = x.sort_values("score_num", ascending=False)[cols].head(limit)
    return x.rename(columns={"score_num": "score (upvotes)", "thread_type": "context"})

def most_negative_for_theme(df_subset: pd.DataFrame, theme: str, limit: int = 12) -> pd.DataFrame:
    pats = THEMES.get(theme, [])
    x = df_subset.copy()
    x["hits_theme"] = x["body"].apply(lambda t: comment_hits_any_patterns(t, pats))
    x = x[(x["hits_theme"] == True) & (x["sentiment"].isin(["negative", "mixed"]))].copy()
    cols = [c for c in ["thread_type", "score_num", "sentiment", "body"] if c in x.columns]
    x = x.sort_values("score_num", ascending=False)[cols].head(limit)
    return x.rename(columns={"score_num": "score (upvotes)", "thread_type": "context"})


# -----------------------------
# threads.csv enrichment (optional): opponent + home/away
# -----------------------------
def load_threads_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df = _ensure_no_duplicate_columns(df)
    for col in ["thread_id", "title", "url", "thread_type", "created_utc"]:
        if col not in df.columns:
            df[col] = None
    df["thread_type"] = df["thread_type"].apply(normalize_thread_type)
    df["thread_id"] = df["thread_id"].astype(str)
    df["title"] = df["title"].astype(str)
    return df

def infer_opponent_home_away(title: str) -> Tuple[str, str]:
    t = (title or "").strip()

    # Away: "Bulls @ Opponent"
    if re.search(r"\b@\b", t):
        m = re.search(r"@\s*([A-Za-z .'-]+)", t)
        if m:
            opp = re.split(r"[\(\-–—|]", m.group(1).strip())[0].strip()
            return opp, "Away"

    # Home: "Bulls vs Opponent"
    if re.search(r"\bvs\.?\b", t, flags=re.I):
        m = re.search(r"\bvs\.?\s*([A-Za-z .'-]+)", t, flags=re.I)
        if m:
            opp = re.split(r"[\(\-–—|]", m.group(1).strip())[0].strip()
            return opp, "Home"

    return "Unknown", "Unknown"


# -----------------------------
# Load comment CSVs
# Strategy:
# 1) Use filename game_date if present
# 2) Else use CSV column game_date if present
# 3) Else fallback to created_utc date (ONLY to enable switching; not perfect)
# -----------------------------
def load_comment_csvs(files, date_source: str) -> pd.DataFrame:
    all_rows = []

    for f in files:
        gd_from_name, tt_from_name, tid_from_name = parse_filename_meta(f.name)

        try:
            d = pd.read_csv(f)
        except Exception:
            f.seek(0)
            d = pd.read_csv(f, encoding_errors="ignore")

        d = _ensure_no_duplicate_columns(d)

        for col in ["body", "score", "thread_type", "thread_id", "game_date", "author", "created_utc", "comment_id"]:
            if col not in d.columns:
                d[col] = None

        # thread_type
        if tt_from_name != "unknown":
            d["thread_type"] = tt_from_name
        d["thread_type"] = d["thread_type"].apply(normalize_thread_type)

        # thread_id
        if tid_from_name:
            d["thread_id"] = tid_from_name
        d["thread_id"] = d["thread_id"].astype(str)

        # choose game_date source
        if date_source == "filename":
            if gd_from_name:
                d["game_date"] = gd_from_name
        elif date_source == "csv":
            # keep CSV game_date if valid; else fall back to filename
            if gd_from_name:
                d["game_date"] = d["game_date"].where(d["game_date"].notna(), gd_from_name)
        else:  # created_utc
            # last resort: use created_utc date for switchability
            if "created_utc" in d.columns:
                dt = pd.to_datetime(d["created_utc"], errors="coerce", utc=True)
                d["game_date"] = dt.dt.date.astype(str)

        # normalize game_date to YYYY-MM-DD where possible
        d["game_date"] = pd.to_datetime(d["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")

        # body + sentiment + score
        d["body"] = d["body"].apply(safe_text)
        d["sentiment"] = d["body"].apply(classify_sentiment)
        d["score_num"] = pd.to_numeric(d["score"], errors="coerce").fillna(0).astype(int)

        # file coverage
        d["_source_file"] = f.name
        d["_filename_date"] = gd_from_name
        d["_filename_type"] = tt_from_name
        d["_filename_tid"] = tid_from_name

        all_rows.append(d)

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)


# -----------------------------
# UI
# -----------------------------
st.title("Bulls Fan Sentiment Intelligence")
st.markdown('<div class="header-rule"></div>', unsafe_allow_html=True)
st.caption("Sentiment + narrative drivers from Reddit threads. Score = upvotes. Usernames hidden.")

st.sidebar.header("Upload")
comments_files = st.sidebar.file_uploader(
    "Upload comment CSVs (multiple)",
    type=["csv"],
    accept_multiple_files=True
)

threads_file = st.sidebar.file_uploader(
    "Optional: threads.csv (adds opponent + home/away)",
    type=["csv"],
    accept_multiple_files=False
)

if not comments_files:
    st.info("Upload your thread comment CSVs to begin.")
    st.stop()

st.sidebar.header("Date Source (important)")
date_source = st.sidebar.radio(
    "How should the app group games?",
    options=["filename", "csv", "created_utc"],
    index=0,
    help=(
        "If you only see one date, your filenames or CSV game_date are probably wrong. "
        "Try switching this to see what your data actually contains."
    )
)

df = load_comment_csvs(comments_files, date_source=date_source)
df = df[df["game_date"].notna()].copy()

threads_df = None
if threads_file is not None:
    threads_df = load_threads_csv(threads_file)
    meta = threads_df[["thread_id", "title", "thread_type"]].copy()
    meta["thread_id"] = meta["thread_id"].astype(str)
    df = df.merge(meta, on="thread_id", how="left", suffixes=("", "_meta"))
    if "thread_type_meta" in df.columns:
        # prefer existing, else meta
        df["thread_type"] = df["thread_type"].where(df["thread_type"].notna(), df["thread_type_meta"])
        df["thread_type"] = df["thread_type"].apply(normalize_thread_type)
    if "title" in df.columns:
        df["title"] = df["title"].astype(str)

if df.empty:
    st.error("Loaded 0 rows with a valid date. Try changing Date Source to 'csv' or 'created_utc'.")
    st.stop()

# ---------- Data Health (THIS will reveal why you only see one game) ----------
with st.expander("Data Health (shows exactly what was loaded)", expanded=True):
    health = (
        df.groupby(["game_date", "thread_type", "_source_file"])
          .size()
          .reset_index(name="rows")
          .sort_values(["game_date", "thread_type", "rows"], ascending=[False, True, False])
    )
    st.caption("If you only see one game_date here, then your uploaded files truly only map to one date.")
    st.dataframe(health, use_container_width=True, hide_index=True)

    dates_summary = df.groupby("game_date").size().reset_index(name="total_rows").sort_values("game_date", ascending=False)
    st.caption("Game dates detected (and how many total comment rows exist per date):")
    st.dataframe(dates_summary, use_container_width=True, hide_index=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# ---------- Game selector ----------
all_dates = sorted(df["game_date"].dropna().unique().tolist(), reverse=True)
if not all_dates:
    st.error("No valid game dates found after parsing.")
    st.stop()

# Build labels with comment volume to help exec navigation
labels = []
for gd in all_dates:
    sub = df[df["game_date"] == gd]
    opp, ha = "Unknown", "Unknown"
    if "title" in sub.columns:
        t = sub["title"].replace("nan", "").replace("None", "")
        t = t[t.str.len() > 0]
        if len(t) > 0:
            opp, ha = infer_opponent_home_away(t.value_counts().index[0])
    labels.append(f"{gd} • vs {opp} • {ha} • {len(sub)} comments")

selected_label = st.sidebar.selectbox("Select game", options=labels, index=0)
selected_date = selected_label.split(" • ")[0].strip()

st.sidebar.header("Filters")
contexts = st.sidebar.multiselect(
    "Thread contexts",
    options=["pregame", "live_game", "postgame"],
    default=["pregame", "live_game", "postgame"]
)
q = st.sidebar.text_input("Search (contains)", value="").strip().lower()

slice_df = df[(df["game_date"] == selected_date) & (df["thread_type"].isin(contexts))].copy()
if q:
    slice_df = slice_df[slice_df["body"].str.lower().str.contains(re.escape(q), na=False)].copy()

if slice_df.empty:
    st.warning("No comments match your filters.")
    st.stop()

# ---------- Header summary ----------
opp, ha = "Unknown", "Unknown"
if "title" in slice_df.columns:
    t = slice_df["title"].replace("nan", "").replace("None", "")
    t = t[t.str.len() > 0]
    if len(t) > 0:
        opp, ha = infer_opponent_home_away(t.value_counts().index[0])

st.markdown(
    f"""
<div class="card">
  <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
    <span class="pill">{selected_date}</span>
    <span class="pill">Opponent: {opp}</span>
    <span class="pill">{ha}</span>
    <span class="pill">Contexts: {", ".join(contexts) if contexts else "None"}</span>
  </div>
  <div style="margin-top:10px; color:{MUTED}; font-size:0.92rem;">
    Executive summary: sentiment, narrative drivers, and evidence from Reddit threads.
  </div>
</div>
""",
    unsafe_allow_html=True
)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# ---------- KPIs ----------
total = len(slice_df)
sent = slice_df["sentiment"].value_counts()
neg_ct = int(sent.get("negative", 0))
pos_ct = int(sent.get("positive", 0))
neu_ct = int(sent.get("neutral", 0))
mix_ct = int(sent.get("mixed", 0))

by_type = slice_df.groupby("thread_type").size().to_dict()
pre_ct = int(by_type.get("pregame", 0))
live_ct = int(by_type.get("live_game", 0))
post_ct = int(by_type.get("postgame", 0))

hs = heat_score(slice_df)
top_theme = safe_top_label(theme_counts_for_df(slice_df), default="None detected")
top_mention = safe_top_label(player_counts_for_df(slice_df), default="None detected")

st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
st.markdown(
    f"""
<div class="kpi"><div class="kpi-label">Risk Pulse (Heat)</div><div class="kpi-value">{hs}</div><div class="kpi-sub">Negativity + volume</div></div>
<div class="kpi"><div class="kpi-label">Negative %</div><div class="kpi-value">{pct(neg_ct, max(total,1))}%</div><div class="kpi-sub">Heuristic sentiment</div></div>
<div class="kpi"><div class="kpi-label">Positive %</div><div class="kpi-value">{pct(pos_ct, max(total,1))}%</div><div class="kpi-sub">Momentum / praise</div></div>
<div class="kpi"><div class="kpi-label">Engagement</div><div class="kpi-value">{total}</div><div class="kpi-sub">Comments analyzed</div></div>
<div class="kpi"><div class="kpi-label">Context Mix</div><div class="kpi-value">{post_ct - live_ct}</div><div class="kpi-sub">Postgame minus live</div></div>
<div class="kpi"><div class="kpi-label">Top Driver</div><div class="kpi-value">{top_theme}</div><div class="kpi-sub">Top mention: {top_mention}</div></div>
""",
    unsafe_allow_html=True
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# ---------- Sentiment distribution ----------
st.subheader("Sentiment Mix")
tone_df = pd.DataFrame(
    [
        {"Sentiment": "negative", "Count": neg_ct, "Share %": pct(neg_ct, max(total, 1))},
        {"Sentiment": "mixed", "Count": mix_ct, "Share %": pct(mix_ct, max(total, 1))},
        {"Sentiment": "neutral", "Count": neu_ct, "Share %": pct(neu_ct, max(total, 1))},
        {"Sentiment": "positive", "Count": pos_ct, "Share %": pct(pos_ct, max(total, 1))},
    ]
).sort_values("Count", ascending=False)

left, right = st.columns([1.1, 1.0])
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.dataframe(tone_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Evidence: Top upvoted comments (overall)**")
    st.caption("Usernames hidden. Score = upvotes.")
    top_overall = slice_df.sort_values("score_num", ascending=False).copy()
    cols = [c for c in ["thread_type", "score_num", "sentiment", "body"] if c in top_overall.columns]
    top_overall = top_overall[cols].head(18).rename(columns={"score_num": "score (upvotes)", "thread_type": "context"})
    st.dataframe(top_overall, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# ---------- Themes + drilldown ----------
st.subheader("Narrative Drivers (Themes)")
theme_tbl = theme_kpi_table(slice_df)
if theme_tbl.empty:
    st.info("No theme hits detected with current deterministic rules. Add more patterns to THEMES.")
else:
    left, right = st.columns([1.2, 1.0])
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.caption("Hits = number of comments matching theme. Negative % = risk within that theme.")
        st.dataframe(theme_tbl, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        theme_pick = st.selectbox("Theme drill-down", options=theme_tbl["Theme"].tolist())
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top upvoted**")
            st.dataframe(top_comments_for_theme(slice_df, theme_pick), use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**Most negative (by upvotes)**")
            st.dataframe(most_negative_for_theme(slice_df, theme_pick), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# ---------- Player / figure mentions ----------
st.subheader("Narrative Concentration (Mentions)")
pc = player_counts_for_df(slice_df)
pc_df = pd.DataFrame(pc.most_common(15), columns=["Name", "Mentions"])

colA, colB = st.columns([1.1, 1.0])
with colA:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.caption("Mentions = narrative focus, not performance.")
    if pc_df.empty:
        st.info("No mentions detected. Expand PLAYERS patterns (add nicknames).")
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
    st.markdown("**Most negative comments (overall)**")
    st.caption("Sorted by upvotes. Usernames hidden.")
    neg_overall = slice_df[slice_df["sentiment"].isin(["negative", "mixed"])].copy()
    cols = [c for c in ["thread_type", "score_num", "sentiment", "body"] if c in neg_overall.columns]
    neg_overall = neg_overall.sort_values("score_num", ascending=False)[cols].head(18)
    neg_overall = neg_overall.rename(columns={"score_num": "score (upvotes)", "thread_type": "context"})
    st.dataframe(neg_overall, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Raw validation ----------
with st.expander("Raw Validation (usernames may appear here only)", expanded=False):
    safe_df = _ensure_no_duplicate_columns(slice_df.copy())
    st.dataframe(safe_df, use_container_width=True)

