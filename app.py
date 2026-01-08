# app.py
# Streamlit dashboard for Bulls subreddit game threads
# Deterministic rules only, UI/UX refactor to be "belief intelligence" first

import re
from collections import Counter
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


# -----------------------------
# Page config + light styling
# -----------------------------
st.set_page_config(page_title="Bulls Fan Belief Intelligence", layout="wide")

st.markdown(
    """
<style>
/* tighten default spacing */
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: -0.02em; }
small, .stCaption { opacity: 0.8; }

/* make metric cards feel more "dashboard" */
[data-testid="stMetric"] {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.08);
  padding: 14px 14px 10px 14px;
  border-radius: 14px;
}

/* tables */
[data-testid="stDataFrame"] {
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.08);
}
</style>
""",
    unsafe_allow_html=True,
)


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

# Legacy themes (kept for compatibility)
THEMES: Dict[str, List[str]] = {
    "injury": [r"\binjur", r"\bconcussion\b", r"\bprotocol\b", r"\bout\b", r"\bquestionable\b"],
    "coaching": [r"\bcoach\b", r"\bcoaching\b", r"\blineup\b", r"\brotation\b", r"\btimeouts?\b", r"\bdonovan\b"],
    "shooting": [r"\bshoot", r"\b3s\b", r"\bthrees\b", r"\bthree\b", r"\bbrick", r"\bfg\b", r"\bpercent\b"],
    "refs": [r"\bref", r"\bwhistle\b", r"\bfoul\b", r"\bfree throw\b", r"\bft\b"],
    "front_office": [r"\bfront office\b", r"\bakme\b", r"\bkarnisovas\b", r"\btrade\b", r"\bdeadline\b"],
    "effort_identity": [r"\beffort\b", r"\bsoft\b", r"\bheart\b", r"\bidentity\b", r"\bvibes\b"],
    "tanking": [r"\btank\b", r"\blottery\b", r"\bpicks?\b", r"\btop pick\b"],
}

# Belief themes (what leadership should care about)
# These are designed to be outcome-independent narratives.
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

# Win/loss decoupling signal phrases (fans reframing losses as acceptable/desirable)
WIN_LOSS_DECOUPLING: List[str] = [
    r"\bethical tank\b", r"\bperfect tank win\b", r"\btank win\b", r"\btrade the l\b",
    r"\bI'll take the l\b", r"\blosing is fine\b", r"\bkeep losing\b", r"\bloss is fine\b",
    r"\blottery odds\b", r"\btop pick\b", r"\btank\b",
]

# Organizational distrust signals
ORG_DISTRUST: List[str] = [
    r"\bfront office\b", r"\bakme\b", r"\bkarnisovas\b", r"\bownership\b", r"\breinsdorf\b",
    r"\bgarpax\b", r"\bthis organization\b", r"\bno plan\b", r"\bnever change\b",
    r"\bfire\b", r"\bsell the team\b",
]

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


def pct(part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(100.0 * part / total, 1)


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

        # numeric score
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
    """
    Counts # of comments mentioning each player (not raw mention hits).
    This matches the "Narrative Concentration" concept better.
    """
    out = {}
    for player, pats in PLAYERS.items():
        mask = df_subset["body"].apply(lambda t: comment_hits_any_patterns(t, pats))
        out[player] = int(mask.sum())
    return out


def player_sentiment_split(df_subset: pd.DataFrame, player: str) -> Counter:
    pats = PLAYERS.get(player, [])
    x = df_subset[df_subset["body"].apply(lambda t: comment_hits_any_patterns(t, pats))].copy()
    return Counter(x["sentiment"].astype(str).tolist())


def top_comments(df_subset: pd.DataFrame, mask: pd.Series, limit: int = 12) -> pd.DataFrame:
    x = df_subset[mask].copy()
    x = x.sort_values("score_num", ascending=False)
    cols = [c for c in ["game_date", "thread_type", "author", "score_num", "sentiment", "body"] if c in x.columns]
    x = x[cols].head(limit).rename(columns={"score_num": "score"})
    return x


def brand_loyalty_state(df_subset: pd.DataFrame, anchor_player: Optional[str]) -> str:
    """
    Binary-style framing: emotionally invested vs distrustful vs disengaging.
    Deterministic heuristic.
    """
    total = max(len(df_subset), 1)

    distrust_mask = df_subset["body"].apply(lambda t: comment_hits_any_patterns(t, ORG_DISTRUST))
    distrust_pct = (distrust_mask.mean() * 100.0) if total else 0.0

    # Emotional investment proxy: anchor player positivity AND mention volume
    invested = False
    if anchor_player:
        split = player_sentiment_split(df_subset, anchor_player)
        mentions = player_mention_counts_unique(df_subset).get(anchor_player, 0)
        pos = split.get("positive", 0)
        neg = split.get("negative", 0)
        invested = (mentions >= max(3, int(0.08 * total))) and (pos >= max(2, neg))

    if invested and distrust_pct >= 6:
        return "Emotionally invested but distrustful"
    if invested and distrust_pct < 6:
        return "Emotionally invested"
    if (not invested) and distrust_pct >= 6:
        return "Low player attachment, high organizational distrust"
    return "Low emotional investment (watch for disengagement)"


def find_emotional_anchor(df_subset: pd.DataFrame) -> Optional[str]:
    """
    Choose the player with meaningful conversation share and the strongest positive skew.
    """
    total = max(len(df_subset), 1)
    mentions = player_mention_counts_unique(df_subset)

    candidates = []
    for player, m in mentions.items():
        if m < max(3, int(0.06 * total)):  # must be present enough to matter
            continue
        split = player_sentiment_split(df_subset, player)
        pos = split.get("positive", 0)
        neg = split.get("negative", 0)
        mixed = split.get("mixed", 0)

        # score: prefer strong positive, penalize negative, allow some mixed
        score = (pos * 2.0) + (mixed * 0.4) - (neg * 2.2) + (m * 0.15)
        candidates.append((score, player))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def win_loss_decoupling_pct(df_subset: pd.DataFrame) -> float:
    mask = df_subset["body"].apply(lambda t: comment_hits_any_patterns(t, WIN_LOSS_DECOUPLING))
    return round(mask.mean() * 100.0, 1)


def heat_score(df_subset: pd.DataFrame) -> float:
    total = max(len(df_subset), 1)
    neg_ct = int((df_subset["sentiment"] == "negative").sum())
    # deterministic blend of negativity + volume
    return round((neg_ct / total) * 100.0 + (len(df_subset) / 60.0), 1)


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
    """
    If compare_subset is provided, show ↑ → ↓ vs that baseline.
    Otherwise momentum column is blank.
    """
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
    out = out.sort_values(["mentions", "share_%"], ascending=[False, False])
    return out


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
    out = out.sort_values(["comments_mentioning"], ascending=False).head(top_n)
    return out


def safe_table(df_in: pd.DataFrame) -> pd.DataFrame:
    x = df_in.copy()
    x = _ensure_no_duplicate_columns(x)
    return x


# -----------------------------
# Header
# -----------------------------
st.title("Bulls Fan Belief Intelligence")
st.caption("Deterministic rules only. This dashboard is designed to surface dominant fan narratives, emotional anchors, and belief movement.")


# -----------------------------
# Data upload
# -----------------------------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader(
    "Upload one or more CSVs",
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

# -----------------------------
# Global filters (minimal)
# -----------------------------
st.sidebar.header("Filters")

all_dates = sorted([d for d in df["game_date"].dropna().unique().tolist() if d and d != "None"])
if not all_dates:
    st.error("No game_date values detected. Filenames should look like YYYY-MM-DD_live_game_THREADID.csv")
    st.stop()

default_idx = len(all_dates) - 1
selected_game_date = st.sidebar.selectbox("Game", options=all_dates, index=default_idx)

thread_types = ["pregame", "live_game", "postgame"]
selected_types = st.sidebar.multiselect("Thread types", options=thread_types, default=thread_types)

q = st.sidebar.text_input("Search (optional)", value="").strip().lower()

f = df[(df["game_date"] == selected_game_date) & (df["thread_type"].isin(selected_types))].copy()
if q:
    f = f[f["body"].str.lower().str.contains(re.escape(q), na=False)].copy()


# -----------------------------
# Tabs: use navigation-first structure
# -----------------------------
tabs = st.tabs(
    [
        "Game Snapshot",
        "Player Beliefs",
        "Fan Psychology",
        "Narrative Trends",
        "Raw Data",
    ]
)


# -----------------------------
# Tab 1: Game Snapshot
# -----------------------------
with tabs[0]:
    total_comments = len(f)
    unique_authors = int(f["author"].nunique(dropna=True))
    live_ct = int((f["thread_type"] == "live_game").sum())
    post_ct = int((f["thread_type"] == "postgame").sum())
    pre_ct = int((f["thread_type"] == "pregame").sum())

    neg_ct = int((f["sentiment"] == "negative").sum())
    pos_ct = int((f["sentiment"] == "positive").sum())
    mix_ct = int((f["sentiment"] == "mixed").sum())
    neu_ct = int((f["sentiment"] == "neutral").sum())

    anchor = find_emotional_anchor(f)

    belief_cts = belief_counts(f)
    dominant_belief = belief_cts.most_common(1)[0][0] if belief_cts else "None detected"
    decouple = win_loss_decoupling_pct(f)
    heat = heat_score(f)
    loyalty = brand_loyalty_state(f, anchor)

    # Global Snapshot row: 4 cards, insight-first
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dominant belief theme", dominant_belief)
    c2.metric("Emotional anchor player", anchor or "None detected")
    c3.metric("Win/Loss decoupling", f"{decouple}%")
    c4.metric("Brand loyalty state", loyalty)

    st.markdown("")

    # Secondary KPI strip (still useful, but not the hero)
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Heat score", heat)
    k2.metric("Comments", total_comments)
    k3.metric("Unique authors", unique_authors)
    k4.metric("Neg %", f"{pct(neg_ct, total_comments)}%")
    k5.metric("Live", live_ct)
    k6.metric("Post", post_ct)

    st.markdown("---")

    # Narrative concentration: who the story is about
    st.subheader("Narrative concentration")
    st.caption("Counts comments mentioning each player (not raw word hits). This shows who the conversation centered on.")

    conc = narrative_concentration_df(f, top_n=10)
    if conc.empty:
        st.info("No player mentions found with current deterministic rules.")
    else:
        left, right = st.columns([1.2, 0.8])
        with left:
            # Simple chart: Streamlit bar chart needs index
            chart_df = conc.set_index("player")[["comments_mentioning"]]
            st.bar_chart(chart_df, height=300)
            # Always-visible insight
            top_player = conc.iloc[0]["player"]
            top_share = conc.iloc[0]["share_%"]
            st.markdown(f"**Key insight:** Conversation concentrated around **{top_player}** (about **{top_share}%** of comments).")
        with right:
            st.dataframe(conc, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Belief themes: most important layer
    st.subheader("Belief themes")
    st.caption("Outcome-independent narratives. These usually matter more than the final score.")

    belief_tbl = belief_table_with_momentum(f)
    if belief_tbl.empty:
        st.info("No belief themes detected with the current belief dictionaries.")
    else:
        st.dataframe(belief_tbl, use_container_width=True, hide_index=True)

        # Drill down: representative comments
        st.markdown("#### Drill down: representative comments")
        pick = st.selectbox("Select a belief theme", options=belief_tbl["belief_theme"].tolist())
        patterns = BELIEF_THEMES[pick]["patterns"]
        mask = f["body"].apply(lambda t: comment_hits_any_patterns(t, patterns))
        st.dataframe(top_comments(f, mask, limit=12), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Tone summary (kept short)
    st.subheader("Tone mix")
    st.caption("Heuristic sentiment. Use it as directional context, not as the story.")
    tone_df = pd.DataFrame(
        [
            {"tone": "negative", "count": neg_ct, "share_%": pct(neg_ct, total_comments)},
            {"tone": "mixed", "count": mix_ct, "share_%": pct(mix_ct, total_comments)},
            {"tone": "neutral", "count": neu_ct, "share_%": pct(neu_ct, total_comments)},
            {"tone": "positive", "count": pos_ct, "share_%": pct(pos_ct, total_comments)},
        ]
    ).sort_values("count", ascending=False)
    st.dataframe(tone_df, use_container_width=True, hide_index=True)


# -----------------------------
# Tab 2: Player Beliefs (expandable player cards)
# -----------------------------
with tabs[1]:
    st.subheader("Player belief profiles")
    st.caption("Expandable cards. Summary first, raw fan voice only when you want it.")

    total = max(len(f), 1)
    player_mentions = player_mention_counts_unique(f)
    # sort by comment-mention count desc
    players_sorted = sorted(player_mentions.items(), key=lambda x: x[1], reverse=True)

    if not players_sorted or players_sorted[0][1] == 0:
        st.info("No player mentions found with current deterministic rules.")
    else:
        # Show top players first
        top_n = st.slider("How many players to show", min_value=4, max_value=min(18, len(players_sorted)), value=min(8, len(players_sorted)))
        for player, mention_ct in players_sorted[:top_n]:
            if mention_ct == 0:
                continue

            split = player_sentiment_split(f, player)
            pos = int(split.get("positive", 0))
            mixed = int(split.get("mixed", 0))
            neg = int(split.get("negative", 0))
            neu = int(split.get("neutral", 0))

            with st.expander(f"{player}  |  {mention_ct} comments mentioning  |  {pct(mention_ct, total)}% of thread"):
                # Minimal, visual summary without noisy charts
                a, b, c, d = st.columns(4)
                a.metric("Positive", pos)
                b.metric("Mixed", mixed)
                c.metric("Negative", neg)
                d.metric("Neutral", neu)

                # Deterministic signal line
                signal_parts = []
                if mention_ct >= int(0.25 * total):
                    signal_parts.append("High narrative concentration")
                if neg == 0 and pos >= 3:
                    signal_parts.append("Zero negative detected (rare)")
                if neg >= max(6, pos * 2):
                    signal_parts.append("Negative outweighs positive")
                if not signal_parts:
                    signal_parts.append("Mixed or low-signal conversation")

                st.markdown(f"**Signal:** {'. '.join(signal_parts)}.")

                # Representative comments
                pats = PLAYERS.get(player, [])
                pmask = f["body"].apply(lambda t: comment_hits_any_patterns(t, pats))

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Top upvoted comments**")
                    st.dataframe(top_comments(f, pmask, limit=10), use_container_width=True, hide_index=True)
                with c2:
                    st.markdown("**Top negative or mixed (by score)**")
                    pm = f[pmask].copy()
                    pm = pm[pm["sentiment"].isin(["negative", "mixed"])].copy()
                    if pm.empty:
                        st.info("No negative or mixed comments detected for this player (with current heuristics).")
                    else:
                        cols = [c for c in ["game_date", "thread_type", "author", "score_num", "sentiment", "body"] if c in pm.columns]
                        pm = pm.sort_values("score_num", ascending=False)[cols].head(10).rename(columns={"score_num": "score"})
                        st.dataframe(pm, use_container_width=True, hide_index=True)


# -----------------------------
# Tab 3: Fan Psychology
# -----------------------------
with tabs[2]:
    st.subheader("Fan psychology indicators")
    st.caption("This is where you capture reframing, trust, and reputational risk.")

    total_comments = len(f)
    heat = heat_score(f)
    decouple = win_loss_decoupling_pct(f)
    anchor = find_emotional_anchor(f)
    loyalty = brand_loyalty_state(f, anchor)

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Heat score", heat)
    r2.metric("Win/Loss decoupling", f"{decouple}%")
    r3.metric("Emotional anchor", anchor or "None detected")
    r4.metric("Brand loyalty state", loyalty)

    st.markdown("---")

    # Win/Loss decoupling quote carousel (simple: show top comments matching patterns)
    st.subheader("Win/Loss decoupling signals")
    st.caption("Comments explicitly reframing losing as acceptable or desirable.")

    dmask = f["body"].apply(lambda t: comment_hits_any_patterns(t, WIN_LOSS_DECOUPLING))
    dquotes = top_comments(f, dmask, limit=12)
    if dquotes.empty:
        st.info("No win/loss decoupling comments detected with current patterns.")
    else:
        st.dataframe(dquotes, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Organizational distrust
    st.subheader("Organizational distrust signals")
    st.caption("Deterministic keyword-based read. Useful as a directional flag.")

    omask = f["body"].apply(lambda t: comment_hits_any_patterns(t, ORG_DISTRUST))
    distrust_pct = round(omask.mean() * 100.0, 1) if len(f) else 0.0

    c1, c2 = st.columns([0.6, 1.4])
    with c1:
        st.metric("Distrust share", f"{distrust_pct}%")
        if distrust_pct >= 10:
            st.markdown("**Interpretation:** elevated distrust risk.")
        elif distrust_pct >= 5:
            st.markdown("**Interpretation:** meaningful distrust presence.")
        else:
            st.markdown("**Interpretation:** low-to-moderate distrust presence.")
    with c2:
        oquotes = top_comments(f, omask, limit=12)
        if oquotes.empty:
            st.info("No organizational distrust comments detected with current patterns.")
        else:
            st.dataframe(oquotes, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Risk vs opportunity framing (executive language, deterministic)
    st.subheader("Narrative risk vs opportunity")
    risks = []
    opps = []

    # Simple, deterministic triggers
    if distrust_pct >= 6:
        risks.append("Organizational distrust is present at meaningful volume.")
    if anchor is None:
        risks.append("No clear emotional anchor player detected in this slice (risk of disengagement).")

    # Belief-driven opportunities
    bc = belief_counts(f)
    if bc.get("Build around Matas / youth", 0) > 0:
        opps.append("Fans are open to development-forward messaging when growth is visible.")
    if decouple >= 8:
        opps.append("Loss tolerance is higher when the narrative is about future value, not wins.")

    # fallback
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


# -----------------------------
# Tab 4: Narrative Trends (multi-game)
# -----------------------------
with tabs[3]:
    st.subheader("Narrative trends")
    st.caption("Compare belief movement across a selected date range. Momentum is deterministic ↑ → ↓.")

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
    b.metric("Unique authors", int(weekly["author"].nunique(dropna=True)))
    c.metric("Heat score (range)", heat_score(weekly))

    if weekly.empty:
        st.info("No data in this date range.")
        st.stop()

    st.markdown("---")

    # Baseline for momentum: first half vs second half of selected range (by date order)
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

    st.markdown("### Drill down: pick a belief theme across the range")
    belief_options = list(BELIEF_THEMES.keys())
    pick_belief = st.selectbox("Belief theme", options=belief_options, key="trend_belief_pick")
    pats = BELIEF_THEMES[pick_belief]["patterns"]
    mask = weekly["body"].apply(lambda t: comment_hits_any_patterns(t, pats))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top upvoted comments (range)**")
        st.dataframe(top_comments(weekly, mask, limit=15), use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Most negative or mixed (range, by score)**")
        x = weekly[mask].copy()
        x = x[x["sentiment"].isin(["negative", "mixed"])].copy()
        if x.empty:
            st.info("No negative or mixed comments detected for this belief theme (with current heuristics).")
        else:
            cols = [c for c in ["game_date", "thread_type", "author", "score_num", "sentiment", "body"] if c in x.columns]
            x = x.sort_values("score_num", ascending=False)[cols].head(15).rename(columns={"score_num": "score"})
            st.dataframe(x, use_container_width=True, hide_index=True)


# -----------------------------
# Tab 5: Raw Data
# -----------------------------
with tabs[4]:
    st.subheader("Raw data (filtered)")
    st.caption("Use this for debugging dictionaries and validating edge cases.")
    st.dataframe(safe_table(f), use_container_width=True)

