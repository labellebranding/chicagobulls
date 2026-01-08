# app.py
# Bulls Fan Belief Intelligence (Deterministic rules only)
# UI/UX: insight-first, executive readable, no usernames shown anywhere except Raw Data.
# NOTE: "score" = number of upvotes (as scraped).

import re
from collections import Counter
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


# =============================
# Page config + styling
# =============================
st.set_page_config(page_title="Bulls Fan Belief Intelligence", layout="wide")

st.markdown(
    """
<style>
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1400px; }
h1, h2, h3 { letter-spacing: -0.02em; }
.stCaption { opacity: 0.85; }

/* KPI cards */
.kpi-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
.kpi-card {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.09);
  border-radius: 14px;
  padding: 12px 14px;
}
.kpi-label { font-size: 0.9rem; opacity: 0.9; display: flex; gap: 8px; align-items: center; }
.kpi-value { font-size: 1.6rem; font-weight: 650; margin-top: 2px; }
.kpi-sub { font-size: 0.82rem; opacity: 0.75; margin-top: 4px; line-height: 1.25; }

/* table wrapper */
[data-testid="stDataFrame"] {
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.09);
}

/* pills */
.pill {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
  font-size: 0.78rem;
  opacity: 0.95;
}
</style>
""",
    unsafe_allow_html=True,
)


# =============================
# KPI definitions (single source of truth)
# Used for hover tooltips + Definitions tab
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
        "notes": "Often higher during rebuild cycles.",
    },
    "Brand loyalty state": {
        "definition": "Plain-English classification of investment vs distrust vs disengagement risk.",
        "how_calculated": "Uses organizational distrust share plus presence of an emotional anchor player.",
        "why_it_matters": "Summarizes volatility and retention risk in one line.",
        "notes": "Thresholds are deterministic and tunable.",
    },
    "Heat score": {
        "definition": "Directional indicator of reputational heat combining negativity and volume.",
        "how_calculated": "Heat = (negative/total*100) + (total/60).",
        "why_it_matters": "Quick pulse check for brand risk across games.",
        "notes": "Intentionally simple so it stays comparable over time.",
    },
    "Comments": {
        "definition": "Total number of comments in the selected slice.",
        "how_calculated": "Row count after filters.",
        "why_it_matters": "Higher volume usually means higher narrative importance.",
        "notes": "",
    },
    "Unique commenters": {
        "definition": "Count of distinct commenters in the selected slice.",
        "how_calculated": "Unique count of author values.",
        "why_it_matters": "Measures breadth of participation, not just loudness.",
        "notes": "Usernames are not shown on the dashboard. Only used for this count.",
    },
    "Neg %": {
        "definition": "Percent of comments classified as negative (heuristic).",
        "how_calculated": "negative/total*100.",
        "why_it_matters": "Adds tone context, but belief themes are the main layer.",
        "notes": "Keyword heuristic from NEG_WORDS and POS_WORDS.",
    },
    "Live": {
        "definition": "Comments from live_game threads in the selected slice.",
        "how_calculated": "Count where thread_type == live_game.",
        "why_it_matters": "Captures real-time reactions (often more volatile).",
        "notes": "",
    },
    "Post": {
        "definition": "Comments from postgame threads in the selected slice.",
        "how_calculated": "Count where thread_type == postgame.",
        "why_it_matters": "Captures stabilized narratives after the outcome.",
        "notes": "",
    },
    "Narrative concentration": {
        "definition": "Distribution of comments mentioning each player (comment-level presence).",
        "how_calculated": "For each player, count comments whose text matches that player’s regex patterns.",
        "why_it_matters": "Shows who the conversation centers on, regardless of performance quality.",
        "notes": "Counts comments, not raw word hits.",
    },
    "Belief themes": {
        "definition": "Outcome-independent narrative categories expressed by fans.",
        "how_calculated": "Deterministic matching using BELIEF_THEMES patterns.",
        "why_it_matters": "This is the most actionable layer for comms and positioning.",
        "notes": "A comment can match multiple belief themes.",
    },
    "Momentum": {
        "definition": "Directional movement of a belief theme compared to a baseline period.",
        "how_calculated": "↑ if >= +20%, ↓ if <= -20%, else → based on mention counts.",
        "why_it_matters": "Shows whether a belief is spreading, stable, or fading.",
        "notes": "Baseline is deterministic in Narrative Trends.",
    },
    "Top upvoted comments": {
        "definition": "Most highly scored comments within the selected filter.",
        "how_calculated": "Sort by score (upvotes) descending; take top N.",
        "why_it_matters": "Represents what the community agreed with most.",
        "notes": "Dashboard never shows usernames. Raw Data can.",
    },
    "Score (upvotes)": {
        "definition": "The number of upvotes a comment received (as scraped).",
        "how_calculated": "Loaded from the CSV score column, coerced to an integer.",
        "why_it_matters": "Used to rank representative comments by community agreement.",
        "notes": "Displayed as score in drill-down tables.",
    },
}


# =============================
# Deterministic dictionaries (edit freely)
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
        "patterns": [
            r"\bbuild around\b", r"\byouth\b", r"\bdevelopment\b", r"\brebuild\b",
            r"\bplay the kids\b", r"\bplay the young\b", r"\bmatas\b", r"\bbuzelis\b",
            r"\bfuture\b", r"\bcore\b",
        ],
        "outcome_independent": True,
    },
    "Trade / move on from vets": {
        "patterns": [
            r"\btrade\b", r"\bdeadline\b", r"\bmove on\b", r"\bblow it up\b", r"\btear it down\b",
            r"\bvets\b", r"\bget rid of\b", r"\bship\b", r"\bsell\b",
            r"\bvuc\b", r"\bvucevic\b", r"\bcoby\b", r"\bpwill\b", r"\bpatrick williams\b",
        ],
        "outcome_independent": True,
    },
    "Rebounding / physicality issues": {
        "patterns": [
            r"\brebound", r"\bboards?\b", r"\bbox out\b", r"\bphysical\b", r"\bsoft\b",
            r"\bsize\b", r"\bpaint\b", r"\bbig\b",
        ],
        "outcome_independent": True,
    },
    "Injury mismanagement / rushing players": {
        "patterns": [
            r"\binjur", r"\bconcussion\b", r"\bprotocol\b", r"\brushed\b", r"\btoo soon\b",
            r"\bmedical\b", r"\btraining staff\b", r"\bmismanag", r"\bshouldn't be playing\b",
        ],
        "outcome_independent": True,
    },
    "Team identity problem": {
        "patterns": [
            r"\bidentity\b", r"\bwho are we\b", r"\bno identity\b", r"\bwhat is this team\b",
            r"\bdirection\b", r"\bplan\b", r"\bphilosophy\b",
        ],
        "outcome_independent": True,
    },
    "Shot selection": {
        "patterns": [
            r"\bshot selection\b", r"\bbad shots?\b", r"\bchucking\b", r"\bhero ball\b",
            r"\bsettling\b", r"\bthree\b", r"\bthrees\b", r"\bbrick\b",
        ],
        "outcome_independent": True,
    },
}

WIN_LOSS_DECOUPLING: List[str] = [
    r"\bethical tank\b", r"\bperfect tank win\b", r"\btank win\b", r"\btrade the l\b",
    r"\bI'll take the l\b", r"\blosing is fine\b", r"\bkeep losing\b", r"\bloss is fine\b",
    r"\blottery odds\b", r"\btop pick\b", r"\btank\b",
]

ORG_DISTRUST: List[str] = [
    r"\bfront office\b", r"\bakme\b", r"\bkarnisovas\b", r"\bownership\b", r"\breinsdorf\b",
    r"\bthis organization\b", r"\bno plan\b", r"\bnever change\b", r"\bsell the team\b",
    r"\bfire\b",
]

NEG_WORDS = [
    r"\btrash\b", r"\bembarrass", r"\bawful\b", r"\bworst\b", r"\bpathetic\b",
    r"\bpissed\b", r"\bfuck\b", r"\bgarbage\b", r"\bchoke\b", r"\bsucks?\b",
]
POS_WORDS = [
    r"\bgreat\b", r"\bamazing\b", r"\blove\b", r"\bwin\b", r"\bsolid\b",
    r"\bproud\b", r"\bnice\b", r"\bclutch\b", r"\bballing\b",
]


# =============================
# Filename parsing
# =============================
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


def load_uploaded_csvs(files) -> pd.DataFrame:
    all_rows = []

    for f in files:
        game_date_str, thread_type_from_name, thread_id_from_name = parse_filename_meta(f.name)

        df = pd.read_csv(f)
        df = _ensure_no_duplicate_columns(df)

        # Required-ish fields (author kept for unique commenter counts and Raw Data only)
        for col in ["body", "author", "score", "created_utc", "comment_id", "thread_id", "thread_type"]:
            if col not in df.columns:
                df[col] = None

        df["thread_type"] = df["thread_type"].apply(normalize_thread_type)

        # Prefer filename meta when present
        if game_date_str:
            df["game_date"] = game_date_str
        elif "game_date" not in df.columns:
            df["game_date"] = None

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
    return out


def belief_counts(df_subset: pd.DataFrame) -> Counter:
    c = Counter()
    bodies = df_subset["body"].astype(str).tolist()
    for body in bodies:
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


def player_sentiment_split(df_subset: pd.DataFrame, player: str) -> Counter:
    pats = PLAYERS.get(player, [])
    x = df_subset[df_subset["body"].apply(lambda t: comment_hits_any_patterns(t, pats))].copy()
    return Counter(x["sentiment"].astype(str).tolist())


def heat_score(df_subset: pd.DataFrame) -> float:
    total = max(len(df_subset), 1)
    neg_ct = int((df_subset["sentiment"] == "negative").sum())
    return round((neg_ct / total) * 100.0 + (len(df_subset) / 60.0), 1)


def win_loss_decoupling_pct(df_subset: pd.DataFrame) -> float:
    mask = df_subset["body"].apply(lambda t: comment_hits_any_patterns(t, WIN_LOSS_DECOUPLING))
    return round(mask.mean() * 100.0, 1) if len(df_subset) else 0.0


def find_emotional_anchor(df_subset: pd.DataFrame) -> Optional[str]:
    total = max(len(df_subset), 1)
    mentions = player_mention_counts_unique(df_subset)

    candidates = []
    for player, m in mentions.items():
        if m < max(3, int(0.06 * total)):
            continue
        split = player_sentiment_split(df_subset, player)
        pos = split.get("positive", 0)
        neg = split.get("negative", 0)
        mixed = split.get("mixed", 0)
        score = (pos * 2.0) + (mixed * 0.4) - (neg * 2.2) + (m * 0.15)
        candidates.append((score, player))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def brand_loyalty_state(df_subset: pd.DataFrame, anchor_player: Optional[str]) -> str:
    if len(df_subset) == 0:
        return "No data"

    distrust_mask = df_subset["body"].apply(lambda t: comment_hits_any_patterns(t, ORG_DISTRUST))
    distrust_pct = distrust_mask.mean() * 100.0

    invested = False
    if anchor_player:
        split = player_sentiment_split(df_subset, anchor_player)
        mentions = player_mention_counts_unique(df_subset).get(anchor_player, 0)
        pos = split.get("positive", 0)
        neg = split.get("negative", 0)
        invested = (mentions >= max(3, int(0.08 * len(df_subset)))) and (pos >= max(2, neg))

    if invested and distrust_pct >= 6:
        return "Emotionally invested but distrustful"
    if invested and distrust_pct < 6:
        return "Emotionally invested"
    if (not invested) and distrust_pct >= 6:
        return "Low player attachment, high organizational distrust"
    return "Low emotional investment (watch for disengagement)"


def momentum_indicator(current: int, prior: int) -> str:
    if prior == 0 and current > 0:
        return "↑"
    if current == 0 and prior > 0:
        return "↓"
    if prior == 0 and current == 0:
        return "→"
    ratio = current / max(prior, 1)
    if ratio >= 1.20:
        return "↑"
    if ratio <= 0.80:
        return "↓"
    return "→"


def belief_table_with_momentum(df_subset: pd.DataFrame, compare_subset: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    total = max(len(df_subset), 1)
    cur = belief_counts(df_subset)
    base = belief_counts(compare_subset) if compare_subset is not None else Counter()

    rows = []
    for belief, meta in BELIEF_THEMES.items():
        hits = int(cur.get(belief, 0))
        if hits == 0:
            continue
        rows.append(
            {
                "belief_theme": belief,
                "mentions": hits,
                "share_%": round(100.0 * hits / total, 1),
                "outcome_independent": "Yes" if meta.get("outcome_independent") else "No",
                "momentum": (momentum_indicator(hits, int(base.get(belief, 0))) if compare_subset is not None else ""),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["mentions", "share_%"], ascending=[False, False])


def narrative_concentration_df(df_subset: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    total = max(len(df_subset), 1)
    mentions = player_mention_counts_unique(df_subset)
    rows = []
    for player, ct in mentions.items():
        if ct == 0:
            continue
        rows.append({"player": player, "comments_mentioning": ct, "share_%": pct(ct, total)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["comments_mentioning"], ascending=False).head(top_n)


def top_comments_no_author(df_subset: pd.DataFrame, mask: pd.Series, limit: int = 12) -> pd.DataFrame:
    """
    Returns top comments by score (upvotes) WITHOUT usernames.
    Dashboard privacy rule: no identifying data such as author shown outside Raw Data.
    """
    x = df_subset[mask].copy()
    x = x.sort_values("score_num", ascending=False)

    cols = [c for c in ["game_date", "thread_type", "score_num", "sentiment", "body"] if c in x.columns]
    x = x[cols].head(limit).rename(columns={"score_num": "score (upvotes)"})
    return x


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


def tooltip_text(term: str) -> str:
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


def render_kpi_card(label: str, value: str, sub: str = "", help_term: Optional[str] = None, show_hover: bool = True):
    help_text = tooltip_text(help_term or label) if show_hover else ""
    title_attr = help_text.replace('"', "'")  # keep HTML safe-ish
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


# =============================
# Header
# =============================
st.title("Bulls Fan Belief Intelligence")
st.caption(
    "Deterministic rules only. The dashboard is designed to surface dominant fan narratives, emotional anchors, and belief movement. "
    "Usernames are never shown anywhere on the dashboard, only in Raw Data."
)


# =============================
# Sidebar: objective data pull guidance + privacy rules
# =============================
st.sidebar.header("Data pull")

with st.sidebar.expander("How to pull and format the data (objective checklist)", expanded=True):
    st.markdown(
        """
**Input files**
- Upload one CSV per thread (pregame, live_game, postgame).
- Recommended filename format:
  - `YYYY-MM-DD_live_game_THREADID.csv`
  - `YYYY-MM-DD_postgame_THREADID.csv`
  - `YYYY-MM-DD_pregame_THREADID.csv`

**Required columns**
- `body` (comment text)

**Recommended columns**
- `score` (upvotes)
- `thread_type` (pregame, live_game, postgame) or use filename to infer
- `created_utc` (optional)
- `author` (optional, only used for the Unique commenters count, and only visible in Raw Data)

**Score meaning**
- `score` is treated as the comment’s **upvote count** and is displayed as **score (upvotes)**.

**Privacy rule**
- No identifying data such as usernames is shown anywhere except the Raw Data tab.
"""
    )

show_hover = st.sidebar.toggle("Enable hover tooltips", value=True)

uploaded = st.sidebar.file_uploader("Upload one or more CSVs", type=["csv"], accept_multiple_files=True)
if not uploaded:
    st.info("Upload your thread CSV files to start.")
    st.stop()

df = load_uploaded_csvs(uploaded)
if df.empty:
    st.error("No rows found. Confirm the CSVs contain comment rows with a body column.")
    st.stop()

st.sidebar.header("Filters")
all_dates = sorted([d for d in df["game_date"].dropna().unique().tolist() if d and d != "None"])
if not all_dates:
    st.error("No game_date values detected. Use the recommended filename format.")
    st.stop()

selected_game_date = st.sidebar.selectbox("Game", options=all_dates, index=len(all_dates) - 1)

thread_types = ["pregame", "live_game", "postgame"]
selected_types = st.sidebar.multiselect("Thread types", options=thread_types, default=thread_types)

q = st.sidebar.text_input("Search (optional)", value="").strip().lower()

f = df[(df["game_date"] == selected_game_date) & (df["thread_type"].isin(selected_types))].copy()
if q:
    f = f[f["body"].str.lower().str.contains(re.escape(q), na=False)].copy()


# =============================
# Tabs
# =============================
tabs = st.tabs(["Game Snapshot", "Player Beliefs", "Fan Psychology", "Narrative Trends", "Definitions", "Raw Data"])


# =============================
# Tab 1: Game Snapshot
# =============================
with tabs[0]:
    total_comments = len(f)
    unique_commenters = int(f["author"].nunique(dropna=True)) if "author" in f.columns else 0
    live_ct = int((f["thread_type"] == "live_game").sum())
    post_ct = int((f["thread_type"] == "postgame").sum())

    neg_ct = int((f["sentiment"] == "negative").sum())
    heat = heat_score(f)

    anchor = find_emotional_anchor(f)
    belief_cts = belief_counts(f)
    dominant_belief = belief_cts.most_common(1)[0][0] if belief_cts else "None detected"
    decouple = win_loss_decoupling_pct(f)
    loyalty = brand_loyalty_state(f, anchor)

    # Top KPI row: 4 cards
    st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card("Dominant belief theme", dominant_belief, help_term="Dominant belief theme", show_hover=show_hover)
    with c2:
        render_kpi_card("Emotional anchor player", anchor or "None detected", help_term="Emotional anchor player", show_hover=show_hover)
    with c3:
        render_kpi_card("Win/Loss decoupling", f"{decouple}%", help_term="Win/Loss decoupling", show_hover=show_hover)
    with c4:
        render_kpi_card("Brand loyalty state", loyalty, help_term="Brand loyalty state", show_hover=show_hover)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")

    # Secondary strip (still useful, but not the hero)
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    with s1:
        render_kpi_card("Heat score", str(heat), help_term="Heat score", show_hover=show_hover)
    with s2:
        render_kpi_card("Comments", str(total_comments), help_term="Comments", show_hover=show_hover)
    with s3:
        render_kpi_card("Unique commenters", str(unique_commenters), help_term="Unique commenters", show_hover=show_hover)
    with s4:
        render_kpi_card("Neg %", f"{pct(neg_ct, max(total_comments,1))}%", help_term="Neg %", show_hover=show_hover)
    with s5:
        render_kpi_card("Live", str(live_ct), help_term="Live", show_hover=show_hover)
    with s6:
        render_kpi_card("Post", str(post_ct), help_term="Post", show_hover=show_hover)

    st.markdown("---")

    # Narrative concentration
    st.subheader("Narrative concentration")
    st.caption("Counts comments mentioning each player. This shows who the conversation centered on.")
    conc = narrative_concentration_df(f, top_n=10)
    if conc.empty:
        st.info("No player mentions detected with current player dictionaries.")
    else:
        left, right = st.columns([1.2, 0.8])
        with left:
            st.bar_chart(conc.set_index("player")[["comments_mentioning"]], height=300)
            top_player = conc.iloc[0]["player"]
            top_share = conc.iloc[0]["share_%"]
            st.markdown(f"**Interpretation:** The conversation centered most on **{top_player}** (about **{top_share}%** of comments).")
        with right:
            st.dataframe(conc, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Belief themes
    st.subheader("Belief themes")
    st.caption("Outcome-independent narratives. These are usually more stable than game results.")
    belief_tbl = belief_table_with_momentum(f)
    if belief_tbl.empty:
        st.info("No belief themes detected with the current belief dictionaries.")
    else:
        st.dataframe(belief_tbl, use_container_width=True, hide_index=True)

        st.markdown("#### Drill down: representative comments (no usernames)")
        pick = st.selectbox("Select a belief theme", options=belief_tbl["belief_theme"].tolist())
        pats = BELIEF_THEMES[pick]["patterns"]
        mask = f["body"].apply(lambda t: comment_hits_any_patterns(t, pats))
        st.dataframe(top_comments_no_author(f, mask, limit=12), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Tone mix (simple, minimal)
    st.subheader("Tone mix")
    st.caption("Heuristic tone. Use it as context, not as the main takeaway.")
    sent_counts = f["sentiment"].value_counts()
    tone_df = pd.DataFrame(
        [
            {"tone": "negative", "count": int(sent_counts.get("negative", 0)), "share_%": pct(int(sent_counts.get("negative", 0)), max(total_comments, 1))},
            {"tone": "mixed", "count": int(sent_counts.get("mixed", 0)), "share_%": pct(int(sent_counts.get("mixed", 0)), max(total_comments, 1))},
            {"tone": "neutral", "count": int(sent_counts.get("neutral", 0)), "share_%": pct(int(sent_counts.get("neutral", 0)), max(total_comments, 1))},
            {"tone": "positive", "count": int(sent_counts.get("positive", 0)), "share_%": pct(int(sent_counts.get("positive", 0)), max(total_comments, 1))},
        ]
    ).sort_values("count", ascending=False)
    st.dataframe(tone_df, use_container_width=True, hide_index=True)


# =============================
# Tab 2: Player Beliefs
# =============================
with tabs[1]:
    st.subheader("Player belief profiles")
    st.caption("Expandable cards. Summary first. Drill-down shows comments with score (upvotes), never usernames.")

    total = max(len(f), 1)
    player_mentions = player_mention_counts_unique(f)
    players_sorted = sorted(player_mentions.items(), key=lambda x: x[1], reverse=True)

    if not players_sorted or players_sorted[0][1] == 0:
        st.info("No player mentions detected with current player dictionaries.")
    else:
        top_n = st.slider("Players to show", min_value=4, max_value=min(18, len(players_sorted)), value=min(8, len(players_sorted)))
        for player, mention_ct in players_sorted[:top_n]:
            if mention_ct == 0:
                continue

            split = player_sentiment_split(f, player)
            pos = int(split.get("positive", 0))
            mixed = int(split.get("mixed", 0))
            neg = int(split.get("negative", 0))
            neu = int(split.get("neutral", 0))

            with st.expander(f"{player}  |  {mention_ct} comments mentioning  |  {pct(mention_ct, total)}%"):
                a, b, c, d = st.columns(4)
                a.metric("Positive", pos)
                b.metric("Mixed", mixed)
                c.metric("Negative", neg)
                d.metric("Neutral", neu)

                # Simple, clear interpretation
                signal = "Mixed or low-signal conversation."
                if mention_ct >= int(0.25 * total):
                    signal = "High conversation focus."
                if neg == 0 and pos >= 3:
                    signal = "Strong positive signal. No negative detected in this slice."
                if neg >= max(6, pos * 2):
                    signal = "Negative outweighs positive. Risk signal for this player narrative."

                st.markdown(f"**Interpretation:** {signal}")

                pats = PLAYERS.get(player, [])
                pmask = f["body"].apply(lambda t: comment_hits_any_patterns(t, pats))

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Top upvoted comments (score = upvotes)**")
                    st.dataframe(top_comments_no_author(f, pmask, limit=10), use_container_width=True, hide_index=True)
                with c2:
                    st.markdown("**Top negative or mixed (by upvotes)**")
                    x = f[pmask].copy()
                    x = x[x["sentiment"].isin(["negative", "mixed"])].copy()
                    if x.empty:
                        st.info("No negative or mixed comments detected for this player in this slice.")
                    else:
                        mask2 = pd.Series([True] * len(x), index=x.index)
                        st.dataframe(top_comments_no_author(x, mask2, limit=10), use_container_width=True, hide_index=True)


# =============================
# Tab 3: Fan Psychology
# =============================
with tabs[2]:
    st.subheader("Fan psychology indicators")
    st.caption("Captures reframing, trust, and reputational risk in simple terms.")

    anchor = find_emotional_anchor(f)
    loyalty = brand_loyalty_state(f, anchor)
    decouple = win_loss_decoupling_pct(f)
    heat = heat_score(f)

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        render_kpi_card("Heat score", str(heat), help_term="Heat score", show_hover=show_hover)
    with r2:
        render_kpi_card("Win/Loss decoupling", f"{decouple}%", help_term="Win/Loss decoupling", show_hover=show_hover)
    with r3:
        render_kpi_card("Emotional anchor player", anchor or "None detected", help_term="Emotional anchor player", show_hover=show_hover)
    with r4:
        render_kpi_card("Brand loyalty state", loyalty, help_term="Brand loyalty state", show_hover=show_hover)

    st.markdown("---")

    st.subheader("Win/Loss decoupling signals")
    st.caption("Comments that explicitly say losing is acceptable or desirable.")
    dmask = f["body"].apply(lambda t: comment_hits_any_patterns(t, WIN_LOSS_DECOUPLING))
    dquotes = top_comments_no_author(f, dmask, limit=12)
    if dquotes.empty:
        st.info("No win/loss decoupling signals detected with current patterns.")
    else:
        st.dataframe(dquotes, use_container_width=True, hide_index=True)

    st.markdown("---")

    st.subheader("Organizational distrust signals")
    st.caption("Directional keyword-based read. Useful as a risk flag.")
    omask = f["body"].apply(lambda t: comment_hits_any_patterns(t, ORG_DISTRUST))
    distrust_pct = round(omask.mean() * 100.0, 1) if len(f) else 0.0

    c1, c2 = st.columns([0.6, 1.4])
    with c1:
        render_kpi_card("Distrust share", f"{distrust_pct}%", sub="Share of comments with distrust keywords.", show_hover=False)
        if distrust_pct >= 10:
            st.markdown("**Interpretation:** High distrust presence.")
        elif distrust_pct >= 5:
            st.markdown("**Interpretation:** Meaningful distrust presence.")
        else:
            st.markdown("**Interpretation:** Low-to-moderate distrust presence.")
    with c2:
        oquotes = top_comments_no_author(f, omask, limit=12)
        if oquotes.empty:
            st.info("No distrust comments detected with current patterns.")
        else:
            st.dataframe(oquotes, use_container_width=True, hide_index=True)

    st.markdown("---")

    st.subheader("Narrative risk vs opportunity")
    st.caption("Deterministic summary to support decision-making.")

    risks = []
    opps = []

    if distrust_pct >= 6:
        risks.append("Organizational distrust is present at meaningful volume.")
    if anchor is None:
        risks.append("No clear emotional anchor detected in this slice (possible disengagement risk).")

    bc = belief_counts(f)
    if bc.get("Build around Matas / youth", 0) > 0:
        opps.append("Fans respond well to development-forward messaging when growth is visible.")
    if decouple >= 8:
        opps.append("Loss tolerance is higher when the narrative is about the future, not the win.")

    if not risks:
        risks.append("No major deterministic risk flags triggered in this slice.")
    if not opps:
        opps.append("No major deterministic opportunity flags triggered in this slice.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Risks**")
        for r in risks:
            st.write(f"- {r}")
    with right:
        st.markdown("**Opportunities**")
        for o in opps:
            st.write(f"- {o}")


# =============================
# Tab 4: Narrative Trends
# =============================
with tabs[3]:
    st.subheader("Narrative trends")
    st.caption("Compares belief movement across a selected date range. Momentum is deterministic ↑ → ↓.")

    date_objs: List[date] = []
    for d in all_dates:
        try:
            date_objs.append(datetime.strptime(d, "%Y-%m-%d").date())
        except Exception:
            pass

    if not date_objs:
        st.warning("Could not parse game dates. Use the recommended filename format.")
        st.stop()

    min_d, max_d = min(date_objs), max(date_objs)

    picked = st.date_input(
        "Select date range",
        value=(min_d, max_d),
        min_value=min_d,
        max_value=max_d,
        key="trend_range",
    )

    if isinstance(picked, (tuple, list)) and len(picked) == 2:
        start_d, end_d = picked
    else:
        start_d, end_d = min_d, max_d

    weekly = df.copy()
    weekly["game_date_obj"] = pd.to_datetime(weekly["game_date"], errors="coerce").dt.date
    weekly = weekly[(weekly["game_date_obj"] >= start_d) & (weekly["game_date_obj"] <= end_d)].copy()

    games_included = sorted([d for d in weekly["game_date"].dropna().unique().tolist() if d and d != "None"])

    a, b, c = st.columns(3)
    a.metric("Games included", len(games_included))
    b.metric("Total comments", len(weekly))
    c.metric("Heat score (range)", heat_score(weekly))

    if weekly.empty:
        st.info("No data in this date range.")
        st.stop()

    st.markdown("---")

    games_sorted = sorted(games_included)
    if len(games_sorted) >= 2:
        mid = max(1, len(games_sorted) // 2)
        prior_games = set(games_sorted[:mid])
        current_games = set(games_sorted[mid:])
        prior = weekly[weekly["game_date"].isin(prior_games)].copy()
        current = weekly[weekly["game_date"].isin(current_games)].copy()

        st.markdown("### Belief momentum (second half vs first half)")
        bt = belief_table_with_momentum(current, compare_subset=prior)
        if bt.empty:
            st.info("No belief themes detected in the selected range with current dictionaries.")
        else:
            st.dataframe(bt, use_container_width=True, hide_index=True)
            st.caption("Momentum compares the second half of the selected range to the first half.")

        st.markdown("---")

    st.markdown("### Game-by-game: dominant belief + anchor")
    rows = []
    for gd in games_sorted:
        g = weekly[weekly["game_date"] == gd].copy()
        anchor = find_emotional_anchor(g)
        bc = belief_counts(g)
        dom = bc.most_common(1)[0][0] if bc else "None"
        rows.append(
            {
                "game_date": gd,
                "dominant_belief": dom,
                "emotional_anchor": anchor or "None",
                "win_loss_decouple_%": win_loss_decoupling_pct(g),
                "heat": heat_score(g),
                "comments": len(g),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    st.markdown("### Drill down: belief theme across the range (no usernames)")
    belief_options = list(BELIEF_THEMES.keys())
    pick_belief = st.selectbox("Belief theme", options=belief_options, key="trend_belief_pick")
    pats = BELIEF_THEMES[pick_belief]["patterns"]
    mask = weekly["body"].apply(lambda t: comment_hits_any_patterns(t, pats))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top upvoted comments (score = upvotes)**")
        st.dataframe(top_comments_no_author(weekly, mask, limit=15), use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Top negative or mixed (by upvotes)**")
        x = weekly[mask].copy()
        x = x[x["sentiment"].isin(["negative", "mixed"])].copy()
        if x.empty:
            st.info("No negative or mixed comments detected for this belief theme in this range.")
        else:
            mask2 = pd.Series([True] * len(x), index=x.index)
            st.dataframe(top_comments_no_author(x, mask2, limit=15), use_container_width=True, hide_index=True)


# =============================
# Tab 5: Definitions (searchable)
# =============================
with tabs[4]:
    st.subheader("Definitions")
    st.caption("Search any KPI term. These definitions match the exact deterministic logic used in the app.")

    defs = kpi_definitions_df()
    query = st.text_input(
        "Search definitions",
        value="",
        placeholder="Try: heat, anchor, decoupling, momentum, score",
    ).strip().lower()

    if query:
        mask = (
            defs["term"].str.lower().str.contains(query, na=False)
            | defs["definition"].str.lower().str.contains(query, na=False)
            | defs["how_calculated"].str.lower().str.contains(query, na=False)
            | defs["why_it_matters"].str.lower().str.contains(query, na=False)
            | defs["notes"].str.lower().str.contains(query, na=False)
        )
        defs_view = defs[mask].copy()
    else:
        defs_view = defs.copy()

    st.dataframe(defs_view, use_container_width=True, hide_index=True)

    st.markdown("### Quick view")
    pick = st.selectbox(
        "Pick a term",
        options=(defs_view["term"].tolist() if not defs_view.empty else defs["term"].tolist()),
    )
    row = defs[defs["term"] == pick].iloc[0]
    st.markdown(f"**{row['term']}**")
    st.write(row["definition"])
    st.markdown("**How calculated**")
    st.write(row["how_calculated"])
    st.markdown("**Why it matters**")
    st.write(row["why_it_matters"])
    if str(row["notes"]).strip():
        st.markdown("**Notes**")
        st.write(row["notes"])


# =============================
# Tab 6: Raw Data (author allowed here only)
# =============================
with tabs[5]:
    st.subheader("Raw Data")
    st.caption("Raw rows for validation. Usernames may appear here only.")
    safe_f = _ensure_no_duplicate_columns(f.copy())
    st.dataframe(safe_f, use_container_width=True)
