# app.py
# Bulls Fan Belief Intelligence (Streamlit) - Single-page scroll
# Deterministic rules only (no API, no ML required)

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


# -----------------------------
# Page config + Bulls styling
# -----------------------------
st.set_page_config(page_title="Bulls Fan Belief Intelligence", layout="wide")

BULLS_RED = "#CE1141"
BULLS_BLACK = "#0B0B0B"
TEXT = "#111827"
MUTED = "#6B7280"
BORDER = "#E5E7EB"
BG = "#FFFFFF"
SOFT_BG = "#F9FAFB"

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
h1, h2, h3 {{ letter-spacing: -0.02em; color: {BULLS_BLACK}; }}
h1 {{ font-weight: 850; }}
h1::after {{
  content: "";
  display: block;
  height: 4px;
  width: 92px;
  margin-top: 10px;
  border-radius: 999px;
  background: {BULLS_RED};
  opacity: 0.95;
}}
.stCaption {{ color: {MUTED}; }}

section[data-testid="stSidebar"] > div {{
  background: {SOFT_BG};
  border-right: 1px solid {BORDER};
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
  border-radius: 14px;
  padding: 12px 14px;
  box-shadow: 0 10px 18px rgba(17,24,39,0.06);
}}
.kpi-label {{
  font-size: 0.88rem;
  color: {MUTED};
  display: flex;
  gap: 8px;
  align-items: center;
}}
.kpi-value {{
  font-size: 1.55rem;
  font-weight: 850;
  margin-top: 3px;
  color: {BULLS_BLACK};
}}
.kpi-sub {{
  font-size: 0.82rem;
  color: {MUTED};
  margin-top: 4px;
  line-height: 1.25;
}}

[data-testid="stDataFrame"] {{
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid {BORDER};
}}

.section-card {{
  background: {SOFT_BG};
  border: 1px solid {BORDER};
  border-radius: 14px;
  padding: 14px 14px;
}}

.hr {{
  height: 1px;
  background: {BORDER};
  margin: 16px 0;
}}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Deterministic dictionaries
# (Inject roster/coaches/announcers here)
# -----------------------------
# You should expand this list over time. Start with your roster + coach + broadcasters.
PLAYERS: Dict[str, List[str]] = {
    # Core roster (add aliases you see fans use)
    "Matas Buzelis": [r"\bmatas\b", r"\bbuzelis\b", r"\bmatas buzelis\b"],
    "Coby White": [r"\bcoby\b", r"\bcoby white\b"],
    "Nikola Vucevic": [r"\bvooch\b", r"\bvucevic\b", r"\bvuc\b"],
    "Patrick Williams": [r"\bpwill\b", r"\bpatrick williams\b", r"\bpat will\b"],
    "Josh Giddey": [r"\bgiddey\b", r"\bjosh giddey\b"],
    "Ayo Dosunmu": [r"\bayo\b", r"\bdosunmu\b", r"\bayo dosunmu\b"],
    "Jevon Carter": [r"\bjevon\b", r"\bcarter\b", r"\bjevon carter\b"],
    "Kevin Huerter": [r"\bhuerter\b", r"\bkevin huerter\b"],
    "Tre Jones": [r"\btre jones\b", r"\btre\b"],
    "Julian Phillips": [r"\bjulian phillips\b", r"\bphillips\b"],
    "Dalen Terry": [r"\bdalen\b", r"\bdalen terry\b"],
    "Jalen Smith": [r"\bjalen smith\b", r"\bjalen\b", r"\bjsmith\b"],
    "Zach Collins": [r"\bzach collins\b", r"\bcollins\b"],
    "Isaac Okoro": [r"\bokoro\b", r"\bisaac okoro\b"],
    "Noa Essengue": [r"\bessengue\b", r"\bnoa essengue\b"],
    "Yuki Kawamura": [r"\bkawamura\b", r"\byuki\b", r"\byuki kawamura\b"],

    # Coaches
    "Billy Donovan": [r"\bbilly donovan\b", r"\bdonovan\b"],

    # Broadcasters (fan feedback)
    "Stacey King": [r"\bstacey\b", r"\bstacey king\b"],
    "Adam Amin": [r"\badam amin\b", r"\badam\b", r"\bamin\b"],
}

THEMES: Dict[str, List[str]] = {
    "injury": [r"\binjur", r"\bconcussion\b", r"\bprotocol\b", r"\bout\b", r"\bquestionable\b"],
    "coaching": [r"\bcoach\b", r"\bcoaching\b", r"\blineup\b", r"\brotation\b", r"\btimeouts?\b", r"\bdonovan\b"],
    "shooting": [r"\bshoot", r"\b3s\b", r"\bthrees\b", r"\bthree\b", r"\bbrick", r"\bfg\b", r"\bpercent\b"],
    "refs": [r"\bref", r"\bwhistle\b", r"\bfoul\b", r"\bfree throw\b", r"\bft\b"],
    "front_office": [r"\bfront office\b", r"\bakme\b", r"\bkarnisovas\b", r"\btrade\b", r"\bdeadline\b"],
    "effort_identity": [r"\beffort\b", r"\bsoft\b", r"\bheart\b", r"\bidentity\b", r"\bvibes\b"],
    "tanking": [r"\btank\b", r"\blottery\b", r"\bpicks?\b", r"\btop pick\b"],
    "announcers": [r"\bstacey\b", r"\bstacey king\b", r"\badam amin\b", r"\bamin\b", r"\bcommentary\b", r"\bbroadcast\b"],
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
# expects: YYYY-MM-DD_live_game_THREADID.csv etc
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
    gd = m.group("game_date")
    tt = normalize_thread_type(m.group("thread_type"))
    tid = m.group("thread_id")
    return gd, tt, tid


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
    return round((neg_ct / total) * 100.0 + (len(df_subset) / 60.0), 1)

def comment_hits_any_patterns(text: str, pats: List[str]) -> bool:
    txt = text or ""
    return any(re.search(p, txt, flags=re.I) for p in pats)

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
        hit_mask = df_subset["body"].apply(lambda t: comment_hits_any_patterns(t, THEMES.get(theme, [])))
        hits = int(hit_mask.sum())
        if hits == 0:
            continue

        sub = df_subset[hit_mask].copy()
        neg_pct = round(100.0 * (sub["sentiment"] == "negative").mean(), 1)

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
                "pregame_hits": pre_hits,
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
    x["hits_theme"] = x["body"].apply(lambda t: comment_hits_any_patterns(t, THEMES.get(theme, [])))
    x = x[x["hits_theme"] == True].copy()

    cols = [c for c in ["game_date", "thread_type", "score_num", "sentiment", "body"] if c in x.columns]
    x = x.sort_values("score_num", ascending=False)[cols].head(limit)
    return x.rename(columns={"score_num": "score (upvotes)"})

def most_negative_for_theme(df_subset: pd.DataFrame, theme: str, limit: int = 25) -> pd.DataFrame:
    x = df_subset.copy()
    x["hits_theme"] = x["body"].apply(lambda t: comment_hits_any_patterns(t, THEMES.get(theme, [])))
    x = x[(x["hits_theme"] == True) & (x["sentiment"].isin(["negative", "mixed"]))].copy()

    cols = [c for c in ["game_date", "thread_type", "score_num", "sentiment", "body"] if c in x.columns]
    x = x.sort_values("score_num", ascending=False)[cols].head(limit)
    return x.rename(columns={"score_num": "score (upvotes)"})

def deterministic_game_narrative(g: pd.DataFrame) -> List[str]:
    bullets: List[str] = []
    total = len(g)

    by_type = g.groupby("thread_type").size().to_dict()
    pre_ct = int(by_type.get("pregame", 0))
    live_ct = int(by_type.get("live_game", 0))
    post_ct = int(by_type.get("postgame", 0))

    s_all = Counter(g["sentiment"].astype(str).tolist())
    neg = int(s_all.get("negative", 0))
    pos = int(s_all.get("positive", 0))
    neu = int(s_all.get("neutral", 0))
    mix = int(s_all.get("mixed", 0))

    bullets.append(f"Engagement: {total} comments (pregame {pre_ct}, live {live_ct}, postgame {post_ct}).")
    bullets.append(
        f"Tone (heuristic): {pct(neg, total)}% negative, {pct(mix, total)}% mixed, "
        f"{pct(neu, total)}% neutral, {pct(pos, total)}% positive."
    )

    if live_ct and post_ct:
        if post_ct > live_ct * 1.25:
            bullets.append("Volume increased after the final (postgame > live).")
        elif live_ct > post_ct * 1.25:
            bullets.append("Volume peaked during the game (live > postgame).")
        else:
            bullets.append("Volume was steady (live and postgame similar).")

    top_themes = [k for k, _ in theme_counts_for_df(g).most_common(4)]
    if top_themes:
        bullets.append("Top narratives: " + ", ".join(top_themes) + ".")

    top_players = player_counts_for_df(g).most_common(3)
    if top_players:
        bullets.append("Conversation centers on: " + ", ".join([f"{p} ({c})" for p, c in top_players]) + ".")

    hs = heat_score(g)
    if hs >= 70:
        bullets.append("Heat level: HIGH (high negativity + volume).")
    elif hs >= 45:
        bullets.append("Heat level: MODERATE (noticeable criticism).")
    else:
        bullets.append("Heat level: LOW (calmer / neutral-to-positive).")

    return bullets


def load_uploaded_csvs(files) -> pd.DataFrame:
    all_rows = []
    for f in files:
        game_date_str, thread_type_from_name, thread_id_from_name = parse_filename_meta(f.name)

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

        # Normalize thread_type from file contents
        df["thread_type"] = df["thread_type"].apply(normalize_thread_type)

        # Override with filename meta if present
        if game_date_str:
            df["game_date"] = game_date_str
        df["game_date"] = df["game_date"].astype(str).str.slice(0, 10)

        if thread_type_from_name != "unknown":
            df["thread_type"] = thread_type_from_name

        if thread_id_from_name:
            df["thread_id"] = thread_id_from_name

        df["body"] = df["body"].apply(safe_text)
        df["sentiment"] = df["body"].apply(classify_sentiment)

        # score = upvotes
        df["score_num"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)

        all_rows.append(df)

    if not all_rows:
        return pd.DataFrame()

    out = pd.concat(all_rows, ignore_index=True)
    out["game_date"] = out["game_date"].astype(str).str.slice(0, 10)
    out["thread_type"] = out["thread_type"].apply(normalize_thread_type)
    out["thread_id"] = out["thread_id"].astype(str)
    return out


# -----------------------------
# UI (single scroll)
# -----------------------------
st.title("Bulls Fan Belief Intelligence")
st.caption("Score = upvotes. Usernames are hidden everywhere except Raw Data.")

# Sidebar: upload + filters
st.sidebar.header("Upload")
uploaded = st.sidebar.file_uploader("Upload one or more thread CSVs", type=["csv"], accept_multiple_files=True)
if not uploaded:
    st.info("Upload your `comments_by_thread/*.csv` files to start.")
    st.stop()

df = load_uploaded_csvs(uploaded)
if df.empty:
    st.error("No rows found. Make sure your CSVs contain comment rows and include a 'body' column.")
    st.stop()

# Coverage (always available, solves the “why only one thread?” problem)
st.subheader("Coverage")
cov = (
    df.groupby(["game_date", "thread_type", "thread_id"])
      .size()
      .reset_index(name="comments")
      .sort_values(["game_date", "thread_type", "comments"], ascending=[True, True, False])
)
st.dataframe(cov, use_container_width=True, hide_index=True)
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# Filters
st.sidebar.header("Filters")

all_dates = sorted([d for d in df["game_date"].dropna().unique().tolist() if re.match(r"^\d{4}-\d{2}-\d{2}$", str(d))])
if not all_dates:
    st.error("No valid game_date values detected. Use YYYY-MM-DD in filenames or include game_date in the CSV.")
    st.stop()

game_date = st.sidebar.selectbox("Game date", options=all_dates, index=len(all_dates) - 1)

thread_types = ["pregame", "live_game", "postgame"]
type_filter = st.sidebar.multiselect("Thread types", options=thread_types, default=thread_types)

available_thread_ids = (
    df[(df["game_date"] == game_date) & (df["thread_type"].isin(type_filter))]["thread_id"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)
available_thread_ids = sorted([t for t in available_thread_ids if t and t != "None"])
thread_id_filter = st.sidebar.multiselect("Thread IDs", options=available_thread_ids, default=available_thread_ids)

q = st.sidebar.text_input("Search comment text (contains)", value="").strip().lower()

# Apply filters
f = df[(df["game_date"] == game_date) & (df["thread_type"].isin(type_filter))].copy()
if thread_id_filter:
    f = f[f["thread_id"].astype(str).isin(thread_id_filter)].copy()
if q:
    f = f[f["body"].str.lower().str.contains(re.escape(q), na=False)].copy()

if f.empty:
    st.warning("No comments match your filters.")
    st.stop()

# KPI strip
total_comments = len(f)
unique_commenters = int(f["author"].nunique(dropna=True)) if "author" in f.columns else 0
pre_ct = int((f["thread_type"] == "pregame").sum())
live_ct = int((f["thread_type"] == "live_game").sum())
post_ct = int((f["thread_type"] == "postgame").sum())

sent_counts = f["sentiment"].value_counts()
neg_ct = int(sent_counts.get("negative", 0))
hs = heat_score(f)

st.subheader(f"Game View: {game_date}")
st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
st.markdown(
    f"""
<div class="kpi-card">
  <div class="kpi-label">Heat score</div>
  <div class="kpi-value">{hs}</div>
  <div class="kpi-sub">Higher = more volatility (negativity + volume)</div>
</div>
<div class="kpi-card">
  <div class="kpi-label">Comments</div>
  <div class="kpi-value">{total_comments}</div>
  <div class="kpi-sub">Filtered rows</div>
</div>
<div class="kpi-card">
  <div class="kpi-label">Unique commenters</div>
  <div class="kpi-value">{unique_commenters}</div>
  <div class="kpi-sub">Usernames hidden (Raw Data only)</div>
</div>
<div class="kpi-card">
  <div class="kpi-label">Neg %</div>
  <div class="kpi-value">{pct(neg_ct, max(total_comments, 1))}%</div>
  <div class="kpi-sub">Heuristic keyword-based</div>
</div>
<div class="kpi-card">
  <div class="kpi-label">Live game</div>
  <div class="kpi-value">{live_ct}</div>
  <div class="kpi-sub">Comments in live threads</div>
</div>
<div class="kpi-card">
  <div class="kpi-label">Postgame</div>
  <div class="kpi-value">{post_ct}</div>
  <div class="kpi-sub">Comments in postgame threads</div>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# Narrative bullets
st.subheader("Narrative Summary (deterministic)")
bullets = deterministic_game_narrative(f.copy())
st.markdown('<div class="section-card">', unsafe_allow_html=True)
for b in bullets:
    st.write(f"- {b}")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# Sentiment table
st.subheader("Tone mix (filtered)")
tone_df = sent_counts.rename_axis("tone").reset_index(name="count")
tone_df["share_%"] = tone_df.apply(lambda r: pct(int(r["count"]), max(total_comments, 1)), axis=1)
st.dataframe(tone_df, use_container_width=True, hide_index=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# Player leaderboard + drilldown
st.subheader("Player & Figure Mentions (filtered)")
pc = player_counts_for_df(f)
pc_df = pd.DataFrame(pc.most_common(30), columns=["name", "mentions"])
if pc_df.empty:
    st.info("No mentions detected with the current dictionary.")
else:
    st.dataframe(pc_df, use_container_width=True, hide_index=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# Themes + drilldown
st.subheader("Themes (filtered)")
theme_table = theme_kpi_table(f)
if theme_table.empty:
    st.info("No deterministic theme hits found with current rules.")
else:
    st.dataframe(theme_table, use_container_width=True, hide_index=True)

    st.markdown("### Theme drill-down (usernames hidden)")
    theme_pick = st.selectbox("Select a theme", options=theme_table["theme"].tolist())

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Top upvoted comments (score = upvotes)")
        st.dataframe(top_comments_for_theme(f, theme_pick, limit=25), use_container_width=True, hide_index=True)
    with c2:
        st.markdown("#### Most negative comments (by upvotes)")
        st.dataframe(most_negative_for_theme(f, theme_pick, limit=25), use_container_width=True, hide_index=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# Weekly summary (simple + robust)
st.subheader("Weekly Summary (date window)")

date_objs: List[date] = []
for d in all_dates:
    try:
        date_objs.append(datetime.strptime(d, "%Y-%m-%d").date())
    except Exception:
        pass

if date_objs:
    min_d, max_d = min(date_objs), max(date_objs)

    c1, c2 = st.columns(2)
    with c1:
        start_d = st.date_input("Start date", value=min_d, min_value=min_d, max_value=max_d, key="wk_start")
    with c2:
        end_d = st.date_input("End date", value=max_d, min_value=min_d, max_value=max_d, key="wk_end")

    if start_d > end_d:
        st.warning("Start date is after end date. Swapping them.")
        start_d, end_d = end_d, start_d

    weekly = df.copy()
    weekly["game_date_obj"] = pd.to_datetime(weekly["game_date"], errors="coerce").dt.date
    weekly = weekly[(weekly["game_date_obj"] >= start_d) & (weekly["game_date_obj"] <= end_d)].copy()

    games_included = sorted([d for d in weekly["game_date"].dropna().unique().tolist() if re.match(r"^\d{4}-\d{2}-\d{2}$", str(d))])
    st.write(f"- Games included: **{len(games_included)}**")
    st.write(f"- Total comments: **{len(weekly)}**")

    sent_all = weekly["sentiment"].value_counts()
    sent_all_df = sent_all.rename_axis("tone").reset_index(name="count")
    sent_all_df["share_%"] = sent_all_df.apply(lambda r: pct(int(r["count"]), max(len(weekly), 1)), axis=1)
    st.markdown("### Overall tone (heuristic)")
    st.dataframe(sent_all_df, use_container_width=True, hide_index=True)

    st.markdown("### Top themes (overall)")
    tcw = theme_counts_for_df(weekly)
    st.dataframe(pd.DataFrame(tcw.most_common(20), columns=["theme", "hits"]), use_container_width=True, hide_index=True)

    st.markdown("### Top mentions (overall)")
    pcw = player_counts_for_df(weekly)
    st.dataframe(pd.DataFrame(pcw.most_common(25), columns=["name", "mentions"]), use_container_width=True, hide_index=True)

    st.markdown("### Game summaries (bullets)")
    for gd in games_included:
        st.markdown(f"#### {gd}")
        out_bullets = deterministic_game_narrative(weekly[weekly["game_date"] == gd].copy())
        for b in out_bullets:
            st.write(f"- {b}")
else:
    st.info("Weekly summary unavailable (could not parse game_date values).")

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# Raw Data (collapsed, only place usernames can appear)
with st.expander("Raw Data (validation only, usernames may appear here)", expanded=False):
    safe_f = _ensure_no_duplicate_columns(f.copy())
    st.dataframe(safe_f, use_container_width=True)
