# app.py
# Bulls Fan Belief Intelligence (Deterministic rules only)

import re
from collections import Counter
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Optional Altair (nice charts). App still works without it.
try:
    import altair as alt
    ALTAIR_OK = True
except Exception:
    alt = None
    ALTAIR_OK = False


# =============================
# Page config + Bulls theme
# =============================
st.set_page_config(page_title="Bulls Fan Belief Intelligence", layout="wide")

BULLS_RED = "#CE1141"
BULLS_BLACK = "#0B0B0B"
BULLS_DARK = "#111214"
BULLS_BORDER = "rgba(255,255,255,0.10)"
BULLS_TEXT = "rgba(255,255,255,0.92)"
BULLS_MUTED = "rgba(255,255,255,0.70)"

st.markdown(
    f"""
<style>
.stApp {{
  background: radial-gradient(1200px 800px at 18% 0%, rgba(206,17,65,0.14), transparent 60%),
              radial-gradient(900px 600px at 100% 22%, rgba(206,17,65,0.10), transparent 55%),
              {BULLS_BLACK};
  color: {BULLS_TEXT};
}}
.block-container {{ padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1400px; }}
h1, h2, h3 {{ letter-spacing: -0.02em; }}
h1 {{ font-weight: 800; }}
h1::after {{
  content: ""; display: block; height: 3px; width: 78px; margin-top: 10px;
  border-radius: 999px; background: {BULLS_RED}; opacity: 0.95;
}}
.stCaption {{ color: {BULLS_MUTED}; }}

section[data-testid="stSidebar"] > div {{
  background: linear-gradient(180deg, {BULLS_DARK}, {BULLS_BLACK});
  border-right: 1px solid {BULLS_BORDER};
}}

.kpi-card {{
  background: rgba(255,255,255,0.03);
  border: 1px solid {BULLS_BORDER};
  border-radius: 14px;
  padding: 12px 14px;
  box-shadow: 0 10px 24px rgba(0,0,0,0.18);
}}
.kpi-label {{ font-size: 0.92rem; color: {BULLS_MUTED}; display: flex; gap: 8px; align-items: center; }}
.kpi-value {{ font-size: 1.65rem; font-weight: 750; margin-top: 2px; }}
.kpi-sub {{ font-size: 0.82rem; color: {BULLS_MUTED}; margin-top: 4px; line-height: 1.25; }}
.pill {{
  display: inline-block; padding: 2px 8px; border-radius: 999px;
  border: 1px solid rgba(206,17,65,0.35); background: rgba(206,17,65,0.12);
  font-size: 0.78rem; color: rgba(255,255,255,0.88);
}}

[data-testid="stDataFrame"] {{ border-radius: 14px; overflow: hidden; border: 1px solid {BULLS_BORDER}; }}
button[data-baseweb="tab"] {{ color: {BULLS_MUTED}; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color: {BULLS_TEXT}; border-bottom: 2px solid {BULLS_RED} !important; }}
a {{ color: {BULLS_RED} !important; }}
</style>
""",
    unsafe_allow_html=True,
)


# =============================
# KPI definitions (tooltips + Definitions tab)
# =============================
KPI_DEFINITIONS = {
    "Dominant belief theme": {
        "definition": "Most frequently detected outcome-independent fan narrative in the selected slice.",
        "how_calculated": "Count comments matching each belief theme’s patterns; select the highest.",
        "why_it_matters": "Shows what fans repeat and believe, not just how they feel.",
        "notes": "Deterministic keyword rules in BELIEF_THEMES.",
    },
    "Emotional anchor player": {
        "definition": "Player acting as the strongest positive emotional focal point of conversation.",
        "how_calculated": "Among players with meaningful mention volume, compute a weighted score from sentiment split + mentions; highest wins.",
        "why_it_matters": "Anchors stabilize the narrative and give comms something to build around.",
        "notes": "Deterministic heuristic. Tune weights if needed.",
    },
    "Win/Loss decoupling": {
        "definition": "Share of comments reframing losing as acceptable or desirable (tanking, development over wins).",
        "how_calculated": "Percent of comments matching WIN_LOSS_DECOUPLING patterns.",
        "why_it_matters": "Signals fans redefining success criteria.",
        "notes": "",
    },
    "Brand loyalty state": {
        "definition": "Plain-English classification of investment vs distrust vs disengagement risk.",
        "how_calculated": "Uses organizational distrust share plus presence of an emotional anchor player.",
        "why_it_matters": "Summarizes volatility and retention risk in one line.",
        "notes": "",
    },
    "Heat score": {
        "definition": "Directional indicator of reputational heat combining negativity and volume.",
        "how_calculated": "Heat = (negative/total*100) + (total/60).",
        "why_it_matters": "Quick pulse check for brand risk across games.",
        "notes": "",
    },
    "Comments": {
        "definition": "Total number of comments in the selected slice.",
        "how_calculated": "Row count after filters.",
        "why_it_matters": "Volume = narrative importance.",
        "notes": "",
    },
    "Unique commenters": {
        "definition": "Count of distinct commenters in the selected slice.",
        "how_calculated": "Unique count of author values. Author is not shown on the dashboard.",
        "why_it_matters": "Measures breadth of participation.",
        "notes": "Usernames are only visible in Raw Data.",
    },
    "Score (upvotes)": {
        "definition": "The number of upvotes a comment received (as scraped).",
        "how_calculated": "Loaded from the CSV score column; coerced to int.",
        "why_it_matters": "Used to rank representative comments by agreement.",
        "notes": "Displayed as score (upvotes).",
    },
}


def _tooltip_text(term: str) -> str:
    meta = KPI_DEFINITIONS.get(term, {})
    if not meta:
        return ""
    parts = [
        meta.get("definition", "").strip(),
        f"How: {meta.get('how_calculated','').strip()}".strip(),
        f"Why: {meta.get('why_it_matters','').strip()}".strip(),
    ]
    notes = meta.get("notes", "").strip()
    if notes:
        parts.append(f"Notes: {notes}")
    return "\n".join([p for p in parts if p])


def _render_kpi_card(label: str, value: str, sub: str = "", help_term: Optional[str] = None, show_hover: bool = True):
    help_text = _tooltip_text(help_term or label) if show_hover else ""
    title_attr = help_text.replace('"', "'")
    hover = f' title="{title_attr}"' if help_text else ""
    st.markdown(
        f"""
<div class="kpi-card"{hover}>
  <div class="kpi-label">{label} <span class="pill">hover</span></div>
  <div class="kpi-value">{value}</div>
  <div class="kpi-sub">{sub}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def kpi_definitions_df() -> pd.DataFrame:
    rows = []
    for term, meta in KPI_DEFINITIONS.items():
        rows.append(
            {
                "term": term,
                "definition": meta.get("definition", ""),
                "how_calculated": meta.get("how_calculated", ""),
                "why_it_matters": meta.get("why_it_matters", ""),
                "notes": meta.get("notes", ""),
            }
        )
    return pd.DataFrame(rows)


# =============================
# Deterministic dictionaries
# =============================
PLAYERS: Dict[str, List[str]] = {
    "Nikola Vucevic": [r"\bvooch\b", r"\bvucevic\b", r"\bvuc\b"],
    "Patrick Williams": [r"\bpwill\b", r"\bpatrick williams\b", r"\bpat\b"],
    "Coby White": [r"\bcoby\b", r"\bcoby white\b"],
    "Josh Giddey": [r"\bgiddey\b", r"\bjosh giddey\b"],
    "Lonzo Ball": [r"\blonzo\b", r"\blonzo ball\b"],
    "Ayo Dosunmu": [r"\bayo\b", r"\bdosunmu\b", r"\bayo dosunmu\b"],
    "Matas Buzelis": [r"\bmatas\b", r"\bbuzelis\b", r"\bmatas buzelis\b"],
    "Billy Donovan": [r"\bbilly donovan\b", r"\bdonovan\b"],
}

BELIEF_THEMES: Dict[str, Dict] = {
    "Build around Matas / youth": {
        "patterns": [r"\bbuild around\b", r"\byouth\b", r"\bdevelopment\b", r"\brebuild\b", r"\bplay the kids\b", r"\bmatas\b", r"\bbuzelis\b", r"\bfuture\b", r"\bcore\b"],
        "outcome_independent": True,
    },
    "Trade / move on from vets": {
        "patterns": [r"\btrade\b", r"\bdeadline\b", r"\bmove on\b", r"\bblow it up\b", r"\btear it down\b", r"\bvets\b", r"\bget rid of\b", r"\bsell\b", r"\bvuc\b", r"\bvucevic\b"],
        "outcome_independent": True,
    },
    "Team identity problem": {
        "patterns": [r"\bidentity\b", r"\bwho are we\b", r"\bno identity\b", r"\bdirection\b", r"\bplan\b"],
        "outcome_independent": True,
    },
}

WIN_LOSS_DECOUPLING: List[str] = [
    r"\bethical tank\b", r"\btank win\b", r"\blottery odds\b", r"\btop pick\b", r"\btank\b", r"\blosing is fine\b"
]

ORG_DISTRUST: List[str] = [
    r"\bfront office\b", r"\bakme\b", r"\bkarnisovas\b", r"\bownership\b", r"\breinsdorf\b", r"\bsell the team\b", r"\bno plan\b"
]

NEG_WORDS = [r"\btrash\b", r"\bawful\b", r"\bworst\b", r"\bpathetic\b", r"\bgarbage\b", r"\bsucks?\b"]
POS_WORDS = [r"\bgreat\b", r"\bamazing\b", r"\blove\b", r"\bsolid\b", r"\bproud\b", r"\bclutch\b"]


# =============================
# Filename parsing
# =============================
FILENAME_RE = re.compile(
    r"(?P<game_date>\d{4}-\d{2}-\d{2})_(?P<thread_type>pregame|live_game|postgame|game)_(?P<thread_id>[a-z0-9]+)\.csv$",
    re.I,
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


# =============================
# Helpers
# =============================
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


def comment_hits_any_patterns(text: str, pats: List[str]) -> bool:
    txt = text or ""
    return any(re.search(p, txt, flags=re.I) for p in pats)


def pct(part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(100.0 * part / total, 1)


@st.cache_data(show_spinner=False)
def load_uploaded_csvs(file_bytes_and_names: List[Tuple[bytes, str]]) -> pd.DataFrame:
    all_rows = []
    for content, name in file_bytes_and_names:
        game_date_str, thread_type_from_name, thread_id_from_name = parse_filename_meta(name)

        # Robust read_csv
        try:
            df = pd.read_csv(pd.io.common.BytesIO(content))
        except Exception:
            df = pd.read_csv(pd.io.common.BytesIO(content), encoding_errors="ignore")

        df = _ensure_no_duplicate_columns(df)

        # Ensure standard columns exist
        for col in ["body", "author", "score", "created_utc", "comment_id", "thread_id", "thread_type", "game_date"]:
            if col not in df.columns:
                df[col] = None

        # Normalize thread_type
        df["thread_type"] = df["thread_type"].apply(normalize_thread_type)

        # Prefer filename meta
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
    out["thread_type"] = out["thread_type"].apply(normalize_thread_type)
    return out


def belief_counts(df_subset: pd.DataFrame) -> Counter:
    c = Counter()
    for body in df_subset["body"].astype(str).tolist():
        for belief, meta in BELIEF_THEMES.items():
            if comment_hits_any_patterns(body, meta["patterns"]):
                c[belief] += 1
    return c


def player_mention_counts_unique(df_subset: pd.DataFrame) -> Dict[str, int]:
    out = {}
    for player, pats in PLAYERS.items():
        mask = df_subset["body"].apply(lambda t: comment_hits_any_patterns(t, pats))
        out[player] = int(mask.sum())
    return out


def heat_score(df_subset: pd.DataFrame) -> float:
    total = max(len(df_subset), 1)
    neg_ct = int((df_subset["sentiment"] == "negative").sum())
    return round((neg_ct / total) * 100.0 + (len(df_subset) / 60.0), 1)


def win_loss_decoupling_pct(df_subset: pd.DataFrame) -> float:
    mask = df_subset["body"].apply(lambda t: comment_hits_any_patterns(t, WIN_LOSS_DECOUPLING))
    return round(mask.mean() * 100.0, 1) if len(df_subset) else 0.0


def top_comments_no_author(df_subset: pd.DataFrame, mask: pd.Series, limit: int = 12) -> pd.DataFrame:
    x = df_subset[mask].copy().sort_values("score_num", ascending=False)
    cols = [c for c in ["game_date", "thread_type", "score_num", "sentiment", "body"] if c in x.columns]
    return x[cols].head(limit).rename(columns={"score_num": "score (upvotes)"})


def render_player_bar(conc: pd.DataFrame):
    if conc.empty:
        st.info("No player mentions detected with current dictionaries.")
        return

    if ALTAIR_OK:
        chart = (
            alt.Chart(conc)
            .mark_bar(color=BULLS_RED)
            .encode(
                x=alt.X("comments_mentioning:Q", title="Comments mentioning"),
                y=alt.Y("player:N", sort="-x", title=""),
                tooltip=["player", "comments_mentioning", "share_%"],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("Altair not available. Using Streamlit default chart. (Add altair to requirements.txt to enable Bulls-red bars.)")
        st.bar_chart(conc.set_index("player")[["comments_mentioning"]], height=320)


# =============================
# UI
# =============================
st.title("Bulls Fan Belief Intelligence")
st.caption("Deterministic rules only. Usernames are hidden everywhere except Raw Data. Score = upvotes.")

st.sidebar.header("Data pull (objective)")
with st.sidebar.expander("Checklist", expanded=True):
    st.markdown(
        """
- Upload one CSV per thread (pregame, live_game, postgame).
- Best filename format: `YYYY-MM-DD_postgame_THREADID.csv` (date + thread type + id)
- Required column: `body`
- Recommended: `score` (upvotes), `thread_type`
- Optional: `author` (only used for unique commenter count; shown only in Raw Data)
"""
    )

show_hover = st.sidebar.toggle("Enable hover tooltips", value=True)
debug = st.sidebar.toggle("Debug mode", value=False)

uploaded = st.sidebar.file_uploader("Upload CSVs", type=["csv"], accept_multiple_files=True)
if not uploaded:
    st.info("Upload your thread CSV files to start.")
    st.stop()

file_bytes_and_names = [(u.getvalue(), u.name) for u in uploaded]
df = load_uploaded_csvs(file_bytes_and_names)
if df.empty:
    st.error("No rows found. Confirm your CSVs contain rows and include a 'body' column.")
    st.stop()

# Debug block (shows what Streamlit sees)
if debug:
    st.sidebar.markdown("### Debug")
    st.sidebar.write("Altair OK:", ALTAIR_OK)
    st.sidebar.write("Columns:", list(df.columns))
    st.sidebar.write("Sample game_date values:", sorted(df["game_date"].dropna().unique().tolist())[:10])
    st.sidebar.write("Sample thread_type values:", sorted(df["thread_type"].dropna().unique().tolist()))

st.sidebar.header("Filters")
all_dates = sorted([d for d in df["game_date"].dropna().unique().tolist() if d and d != "None" and re.match(r"^\d{4}-\d{2}-\d{2}$", d)])
if not all_dates:
    st.error("No valid game_date values found. Make sure filenames include YYYY-MM-DD, or add a game_date column in CSV.")
    st.stop()

selected_game_date = st.sidebar.selectbox("Game", options=all_dates, index=len(all_dates) - 1)
thread_types = ["pregame", "live_game", "postgame"]
selected_types = st.sidebar.multiselect("Thread types", options=thread_types, default=thread_types)
q = st.sidebar.text_input("Search (optional)", value="").strip().lower()

f = df[(df["game_date"] == selected_game_date) & (df["thread_type"].isin(selected_types))].copy()
if q:
    f = f[f["body"].str.lower().str.contains(re.escape(q), na=False)].copy()

if f.empty:
    st.warning("No comments match the current filters.")
    st.stop()

tabs = st.tabs(["Game Snapshot", "Definitions", "Raw Data"])

with tabs[0]:
    total_comments = len(f)
    unique_commenters = int(f["author"].nunique(dropna=True)) if "author" in f.columns else 0
    neg_ct = int((f["sentiment"] == "negative").sum())

    belief_cts = belief_counts(f)
    dominant_belief = belief_cts.most_common(1)[0][0] if belief_cts else "None detected"
    decouple = win_loss_decoupling_pct(f)
    heat = heat_score(f)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _render_kpi_card("Dominant belief theme", dominant_belief, help_term="Dominant belief theme", show_hover=show_hover)
    with c2:
        _render_kpi_card("Heat score", str(heat), help_term="Heat score", show_hover=show_hover)
    with c3:
        _render_kpi_card("Comments", str(total_comments), help_term="Comments", show_hover=show_hover)
    with c4:
        _render_kpi_card("Win/Loss decoupling", f"{decouple}%", help_term="Win/Loss decoupling", show_hover=show_hover)

    st.markdown("---")
    st.subheader("Narrative concentration")
    st.caption("Counts comments mentioning each player (comment-level presence).")

    conc_rows = []
    mentions = player_mention_counts_unique(f)
    for player, ct in mentions.items():
        if ct:
            conc_rows.append({"player": player, "comments_mentioning": ct, "share_%": pct(ct, max(len(f), 1))})
    conc = pd.DataFrame(conc_rows).sort_values("comments_mentioning", ascending=False).head(10)

    if conc.empty:
        st.info("No player mentions detected with current dictionaries.")
    else:
        left, right = st.columns([1.25, 0.75])
        with left:
            render_player_bar(conc)
        with right:
            st.dataframe(conc, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Representative comments (dominant belief)")
    if dominant_belief != "None detected":
        pats = BELIEF_THEMES[dominant_belief]["patterns"]
        mask = f["body"].apply(lambda t: comment_hits_any_patterns(t, pats))
        st.caption("Score = upvotes. Usernames are hidden.")
        st.dataframe(top_comments_no_author(f, mask, limit=12), use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("Definitions")
    defs = kpi_definitions_df()
    query = st.text_input("Search definitions", value="", placeholder="Try: heat, score, decoupling").strip().lower()
    if query:
        m = (
            defs["term"].str.lower().str.contains(query, na=False)
            | defs["definition"].str.lower().str.contains(query, na=False)
            | defs["how_calculated"].str.lower().str.contains(query, na=False)
            | defs["why_it_matters"].str.lower().str.contains(query, na=False)
            | defs["notes"].str.lower().str.contains(query, na=False)
        )
        defs_view = defs[m].copy()
    else:
        defs_view = defs.copy()
    st.dataframe(defs_view, use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("Raw Data")
    st.caption("Raw rows for validation. Usernames may appear here only.")
    # Show everything here (including author), but nowhere else.
    st.dataframe(_ensure_no_duplicate_columns(f.copy()), use_container_width=True)
