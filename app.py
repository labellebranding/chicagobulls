# app.py
# Bulls Fan Belief Intelligence (Streamlit)
# Deterministic rules + optional NLP discovered themes
#
# NON-NEGOTIABLE UI/PRIVACY RULES
# - White background, Bulls red/black accents (easy on the eyes)
# - "score" = number of upvotes (always labeled as such)
# - No identifying data (no usernames/author) anywhere except Raw Data tab
# - Deterministic themes remain available + drill-down into top comments
# - Optional NLP discovered themes (embeddings + clustering) + drill-down
# - Searchable Definitions tab for all KPI terms used
# - Hover tooltips (toggle in sidebar) for KPI cards / key UI elements

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Optional charting (app works without it)
try:
    import altair as alt  # type: ignore
    ALTAIR_OK = True
except Exception:
    alt = None
    ALTAIR_OK = False

# Optional NLP (requires nlp.py + requirements)
try:
    from nlp import build_discovered_themes  # type: ignore
    NLP_OK = True
    NLP_IMPORT_ERROR = ""
except Exception as e:
    build_discovered_themes = None
    NLP_OK = False
    NLP_IMPORT_ERROR = str(e)


# -----------------------------
# Page config + Bulls light theme
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

button[data-baseweb="tab"] {{ color: {MUTED}; }}
button[data-baseweb="tab"][aria-selected="true"] {{
  color: {BULLS_BLACK};
  border-bottom: 3px solid {BULLS_RED} !important;
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
.pill {{
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  border: 1px solid rgba(206,17,65,0.25);
  background: rgba(206,17,65,0.08);
  font-size: 0.76rem;
  color: {BULLS_BLACK};
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
# KPI Definitions (searchable tab + hover tooltips)
# -----------------------------
KPI_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "Heat score": {
        "definition": "Directional pulse combining negativity and volume.",
        "how_calculated": "Heat = (negative/total*100) + (total/60).",
        "why_it_matters": "Quickly flags volatility / reputational risk in the conversation.",
        "notes": "Simple by design so it stays comparable across games.",
    },
    "Comments": {
        "definition": "Total number of comments in the selected slice.",
        "how_calculated": "Row count after filters.",
        "why_it_matters": "More volume usually means the narrative matters more.",
        "notes": "",
    },
    "Unique commenters": {
        "definition": "Count of distinct commenters in the selected slice.",
        "how_calculated": "Unique count of author values.",
        "why_it_matters": "Shows breadth (not just a few loud voices).",
        "notes": "Usernames are hidden everywhere except Raw Data.",
    },
    "Neg %": {
        "definition": "Percent of comments classified as negative (heuristic).",
        "how_calculated": "negative / total * 100.",
        "why_it_matters": "Adds tone context (themes are the main layer).",
        "notes": "Keyword heuristic via NEG_WORDS/POS_WORDS.",
    },
    "Live game": {
        "definition": "Number of comments from live game threads in the selected slice.",
        "how_calculated": "Count where thread_type == live_game.",
        "why_it_matters": "Separates in-game reactions from postgame reflection.",
        "notes": "",
    },
    "Postgame": {
        "definition": "Number of comments from postgame threads in the selected slice.",
        "how_calculated": "Count where thread_type == postgame.",
        "why_it_matters": "Postgame often contains the clearest narratives and blame assignment.",
        "notes": "",
    },
    "Score (upvotes)": {
        "definition": "Number of upvotes a comment received (as scraped).",
        "how_calculated": "CSV column score coerced to integer.",
        "why_it_matters": "Used to rank comments by agreement/visibility.",
        "notes": "Displayed as score (upvotes).",
    },
    "Player mentions": {
        "definition": "How often players/coaches/broadcast are referenced in comment text.",
        "how_calculated": "Regex alias matching; counts hit occurrences.",
        "why_it_matters": "Shows who the conversation centers on (not performance quality).",
        "notes": "A comment can mention multiple people.",
    },
    "Theme hits": {
        "definition": "How many comments match a theme’s patterns.",
        "how_calculated": "Regex pattern match per theme (comment-level).",
        "why_it_matters": "Shows which narratives dominate.",
        "notes": "A comment can match multiple themes.",
    },
    "Theme negative %": {
        "definition": "Percent of theme-matching comments that are negative.",
        "how_calculated": "Within theme hits, negative/hits*100.",
        "why_it_matters": "Separates high-volume themes from high-risk themes.",
        "notes": "",
    },
    "Discovered themes (NLP)": {
        "definition": "Themes automatically discovered from comment meaning (embeddings + clustering).",
        "how_calculated": "Embeddings -> clustering -> short labels from TF-IDF terms.",
        "why_it_matters": "Catches new storylines without hand-writing rules.",
        "notes": "Separate from deterministic themes. Requires sentence-transformers + scikit-learn + nlp.py.",
    },
}


def _tooltip_text(term: str) -> str:
    meta = KPI_DEFINITIONS.get(term, {})
    if not meta:
        return ""
    parts = []
    if meta.get("definition"):
        parts.append(meta["definition"].strip())
    if meta.get("how_calculated"):
        parts.append(f"How: {meta['how_calculated'].strip()}")
    if meta.get("why_it_matters"):
        parts.append(f"Why: {meta['why_it_matters'].strip()}")
    if meta.get("notes"):
        parts.append(f"Notes: {meta['notes'].strip()}")
    return "\n".join([p for p in parts if p])


def _render_kpi_card(label: str, value: str, sub: str = "", help_term: Optional[str] = None, show_hover: bool = True):
    help_text = _tooltip_text(help_term or label) if show_hover else ""
    title_attr = help_text.replace('"', "'")
    hover = f' title="{title_attr}"' if help_text else ""
    st.markdown(
        f"""
<div class="kpi-card"{hover}>
  <div class="kpi-label">{label}</div>
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


# -----------------------------
# People dictionary (Roster + coach + broadcast)
# Deterministic. Edit freely.
# -----------------------------
PEOPLE: Dict[str, List[str]] = {
    # Core Bulls
    "Matas Buzelis": [r"\bmatas\b", r"\bbuzelis\b", r"\bmatas buzelis\b"],
    "Coby White": [r"\bcoby\b", r"\bcoby white\b"],
    "Nikola Vucevic": [r"\bvooch\b", r"\bvucevic\b", r"\bvuc\b"],
    "Patrick Williams": [r"\bpwill\b", r"\bpatrick williams\b", r"\bpat williams\b"],
    "Ayo Dosunmu": [r"\bayo\b", r"\bdosunmu\b", r"\bayo dosunmu\b"],
    "Josh Giddey": [r"\bgiddey\b", r"\bjosh giddey\b"],
    "Lonzo Ball": [r"\blonzo\b", r"\blonzo ball\b"],
    "Jevon Carter": [r"\bjevon\b", r"\bjevon carter\b"],
    "Zach Collins": [r"\bzach collins\b", r"\bcollins\b"],
    "Kevin Huerter": [r"\bhuerter\b", r"\bkevin huerter\b"],
    "Tre Jones": [r"\btre jones\b", r"\bjones\b"],
    "Julian Phillips": [r"\bjulian phillips\b", r"\bphillips\b"],
    "Dalen Terry": [r"\bdalen\b", r"\bdalen terry\b"],
    "Jalen Smith": [r"\bjalen smith\b", r"\bsmith\b"],
    "Isaac Okoro": [r"\bokoro\b", r"\bisaac okoro\b"],
    "Noa Essengue": [r"\bessengue\b", r"\bnoa essengue\b"],
    "Trentyn Flowers": [r"\bflowers\b", r"\btrentyn flowers\b"],
    "Yuki Kawamura": [r"\bkawamura\b", r"\byuki\b", r"\byuki kawamura\b"],
    "Emanuel Miller": [r"\bemanu(?:el)? miller\b", r"\bemanu(?:el)?\b"],
    "Lachlan Olbrich": [r"\bolbrich\b", r"\blachlan olbrich\b"],

    # Coach
    "Billy Donovan": [r"\bbilly donovan\b", r"\bdonovan\b"],

    # Front office / org (optional but useful)
    "AKME": [r"\bakme\b", r"\bak\b", r"\bkarnisovas\b", r"\beversley\b", r"\bartūras\b"],

    # Broadcast / media
    "Stacey King": [r"\bstacey king\b", r"\bstacey\b", r"\bking\b"],
    "Adam Amin": [r"\badam amin\b", r"\badam\b", r"\bamin\b"],
}

# -----------------------------
# Deterministic themes (edit freely)
# -----------------------------
THEMES: Dict[str, List[str]] = {
    "injury": [r"\binjur", r"\bconcussion\b", r"\bprotocol\b", r"\bout\b", r"\bquestionable\b"],
    "coaching": [r"\bcoach\b", r"\bcoaching\b", r"\blineup\b", r"\brotation\b", r"\btimeouts?\b", r"\bdonovan\b"],
    "shooting": [r"\bshoot", r"\b3s\b", r"\bthrees\b", r"\bthree\b", r"\bbrick", r"\bfg\b", r"\bpercent\b"],
    "refs": [r"\bref", r"\bwhistle\b", r"\bfoul\b", r"\bfree throw\b", r"\bft\b"],
    "front_office": [r"\bfront office\b", r"\bakme\b", r"\bkarnisovas\b", r"\btrade\b", r"\bdeadline\b"],
    "effort_identity": [r"\beffort\b", r"\bsoft\b", r"\bheart\b", r"\bidentity\b", r"\bvibes\b"],
    "tanking": [r"\btank\b", r"\blottery\b", r"\bpicks?\b", r"\btop pick\b"],
    "broadcast": [r"\bstacey king\b", r"\badam amin\b", r"\bannounc", r"\bcommentator", r"\bbroadcast"],
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


def comment_hits_person(body: str, person: str) -> bool:
    pats = PEOPLE.get(person, [])
    return comment_hits_any_patterns(body, pats)


def pct(part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(100.0 * part / total, 1)


def heat_score(df_subset: pd.DataFrame) -> float:
    total = max(len(df_subset), 1)
    neg_ct = int((df_subset["sentiment"] == "negative").sum())
    return round((neg_ct / total) * 100.0 + (len(df_subset) / 60.0), 1)


def people_counts_for_df(df_subset: pd.DataFrame) -> Counter:
    c = Counter()
    bodies = df_subset["body"].astype(str).tolist()
    for body in bodies:
        for person, pats in PEOPLE.items():
            hits = 0
            for p in pats:
                hits += len(re.findall(p, body, flags=re.I))
            if hits:
                c[person] += hits
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


def _safe_comment_view_cols(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Privacy: NO author/usernames anywhere except Raw Data tab.
    """
    cols = ["game_date", "thread_type", "score_num", "sentiment", "body"]
    cols = [c for c in cols if c in df_in.columns]
    out = df_in[cols].copy()
    if "score_num" in out.columns:
        out = out.rename(columns={"score_num": "score (upvotes)"})
    return out


def top_comments_for_theme(df_subset: pd.DataFrame, theme: str, limit: int = 25) -> pd.DataFrame:
    x = df_subset.copy()
    x["hits_theme"] = x["body"].apply(lambda t: comment_hits_theme(t, theme))
    x = x[x["hits_theme"] == True].copy()
    x = x.sort_values("score_num", ascending=False).head(limit)
    return _safe_comment_view_cols(x)


def most_negative_for_theme(df_subset: pd.DataFrame, theme: str, limit: int = 25) -> pd.DataFrame:
    x = df_subset.copy()
    x["hits_theme"] = x["body"].apply(lambda t: comment_hits_theme(t, theme))
    x = x[x["hits_theme"] == True].copy()
    x = x[x["sentiment"].isin(["negative", "mixed"])].copy()
    x = x.sort_values("score_num", ascending=False).head(limit)
    return _safe_comment_view_cols(x)


def top_comments_for_person(df_subset: pd.DataFrame, person: str, limit: int = 25) -> pd.DataFrame:
    x = df_subset.copy()
    x["hits_person"] = x["body"].apply(lambda t: comment_hits_person(t, person))
    x = x[x["hits_person"] == True].copy()
    x = x.sort_values("score_num", ascending=False).head(limit)
    return _safe_comment_view_cols(x)


def most_negative_for_person(df_subset: pd.DataFrame, person: str, limit: int = 25) -> pd.DataFrame:
    x = df_subset.copy()
    x["hits_person"] = x["body"].apply(lambda t: comment_hits_person(t, person))
    x = x[x["hits_person"] == True].copy()
    x = x[x["sentiment"].isin(["negative", "mixed"])].copy()
    x = x.sort_values("score_num", ascending=False).head(limit)
    return _safe_comment_view_cols(x)


def deterministic_game_narrative(g: pd.DataFrame) -> Dict[str, object]:
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

    bullets.append(f"Engagement: {total_comments} total comments (live {live_ct}, post {post_ct}, pre {pre_ct}).")
    bullets.append(
        f"Tone (heuristic): {pct(neg, total_comments)}% negative, {pct(mix, total_comments)}% mixed, "
        f"{pct(neu, total_comments)}% neutral, {pct(pos, total_comments)}% positive."
    )

    if live_ct > 0 and post_ct > 0:
        if post_ct > live_ct * 1.25:
            bullets.append("Volume moved upward after the final (postgame higher than live).")
        elif live_ct > post_ct * 1.25:
            bullets.append("Volume peaked during the game (live higher than postgame).")
        else:
            bullets.append("Volume was steady (live and postgame similar).")

    themes_all = theme_counts_for_df(g)
    top_themes = [k for k, _ in themes_all.most_common(4)]
    if top_themes:
        bullets.append("Top narratives: " + ", ".join(top_themes) + ".")

    ppl_all = people_counts_for_df(g)
    top_people = ppl_all.most_common(6)
    if top_people:
        bullets.append("Most discussed: " + ", ".join([f"{p} ({c})" for p, c in top_people[:3]]) + ".")

    hs = heat_score(g)
    if hs >= 70:
        bullets.append("Heat level: HIGH (high negativity + volume).")
    elif hs >= 45:
        bullets.append("Heat level: MODERATE (noticeable criticism).")
    else:
        bullets.append("Heat level: LOW (calmer / neutral-to-positive).")

    # Fan quotes (top upvoted), privacy-safe
    fan_quotes = g.copy().sort_values("score_num", ascending=False).head(15)
    fan_quotes = _safe_comment_view_cols(fan_quotes)

    return {
        "bullets": bullets,
        "heat_score": hs,
        "themes_all": themes_all,
        "people_all": ppl_all,
        "sent_counts": s_all,
        "fan_quotes": fan_quotes,
    }


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

        # Clean + enrich
        df["body"] = df["body"].apply(safe_text)
        df["sentiment"] = df["body"].apply(classify_sentiment)

        # score = upvotes (always numeric)
        df["score_num"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)

        all_rows.append(df)

    if not all_rows:
        return pd.DataFrame()

    out = pd.concat(all_rows, ignore_index=True)
    out["game_date"] = out["game_date"].astype(str).str.slice(0, 10)
    out["thread_type"] = out["thread_type"].apply(normalize_thread_type)
    return out


# -----------------------------
# NLP wrapper (cached)
# -----------------------------
@st.cache_data(show_spinner=False)
def run_nlp_discovery(df_slice: pd.DataFrame, model_name: str, k: int) -> Dict[str, object]:
    if not NLP_OK or build_discovered_themes is None:
        return {
            "themes_table": pd.DataFrame(),
            "assignments": pd.DataFrame(),
            "reps": {},
            "note": f"NLP not available. Import error: {NLP_IMPORT_ERROR}",
        }

    cols = [c for c in ["body", "score_num", "sentiment", "thread_type", "game_date"] if c in df_slice.columns]
    x = df_slice[cols].copy()
    return build_discovered_themes(
        x,
        text_col="body",
        score_col="score_num",
        model_name=model_name,
        k=int(k),
        min_comments=30,
    )


# -----------------------------
# UI
# -----------------------------
st.title("Bulls Fan Belief Intelligence")
st.caption("Score = upvotes. Usernames are hidden everywhere except Raw Data.")

# Sidebar: objective data pull guidance
st.sidebar.header("Data pull instructions (objective)")
with st.sidebar.expander("How to format your CSVs", expanded=True):
    st.markdown(
        """
**Best practice**
- Upload one CSV per thread (pregame, live_game, postgame)

**Best filename format**
- `YYYY-MM-DD_live_game_THREADID.csv`
- `YYYY-MM-DD_postgame_THREADID.csv`
- `YYYY-MM-DD_pregame_THREADID.csv`

**Required column**
- `body` (comment text)

**Recommended**
- `score` (upvotes)
- `thread_type` (or rely on filename)
- `game_date` (or rely on filename)
"""
    )

show_hover = st.sidebar.toggle("Enable hover tooltips", value=True)
debug_mode = st.sidebar.toggle("Debug mode", value=False)

st.sidebar.header("Upload")
uploaded = st.sidebar.file_uploader("Upload one or more thread CSVs", type=["csv"], accept_multiple_files=True)
if not uploaded:
    st.info("Upload your thread CSV files to start.")
    st.stop()

df = load_uploaded_csvs(uploaded)
if df.empty:
    st.error("No rows found. Confirm your CSVs contain comment rows and include a 'body' column.")
    st.stop()

# Sidebar: Filters
st.sidebar.header("Filters")

all_dates = sorted(
    [d for d in df["game_date"].dropna().unique().tolist() if re.match(r"^\d{4}-\d{2}-\d{2}$", str(d))]
)
if not all_dates:
    st.error("No valid game_date values detected. Use YYYY-MM-DD in filenames or include game_date in the CSV.")
    st.stop()

game_date = st.sidebar.selectbox("Game date", options=all_dates, index=len(all_dates) - 1)

thread_types = ["pregame", "live_game", "postgame"]
type_filter = st.sidebar.multiselect("Thread types", options=thread_types, default=thread_types)

st.sidebar.markdown("### Search")
q = st.sidebar.text_input("Search comment text (contains)", value="").strip().lower()

# Sidebar: NLP controls (only if NLP is installed)
st.sidebar.header("NLP (optional)")
use_nlp = st.sidebar.toggle("Enable discovered themes (NLP)", value=False, disabled=not NLP_OK)
nlp_model = st.sidebar.selectbox(
    "Embedding model",
    options=["all-MiniLM-L6-v2"],
    index=0,
    disabled=(not NLP_OK or not use_nlp),
    help="Small + fast model that works well on Streamlit Cloud.",
)
k_override = st.sidebar.slider(
    "Number of discovered themes",
    min_value=3,
    max_value=14,
    value=8,
    disabled=(not NLP_OK or not use_nlp),
    help="Higher = more granular. Lower = broader.",
)

if debug_mode:
    st.sidebar.markdown("### Debug")
    st.sidebar.write("Altair OK:", ALTAIR_OK)
    st.sidebar.write("NLP OK:", NLP_OK)
    if not NLP_OK:
        st.sidebar.write("NLP import error:", NLP_IMPORT_ERROR)
    st.sidebar.write("Columns:", list(df.columns))
    st.sidebar.write("thread_type values:", sorted(df["thread_type"].dropna().unique().tolist()))
    st.sidebar.write("game_date values:", all_dates[:10])

# Apply filters
f = df[(df["game_date"] == game_date) & (df["thread_type"].isin(type_filter))].copy()
if q:
    f = f[f["body"].str.lower().str.contains(re.escape(q), na=False)].copy()

if f.empty:
    st.warning("No comments match your filters.")
    st.stop()

tabs = st.tabs(["Dashboard", "Game-by-Game Report", "Weekly Report", "Definitions", "Raw Data"])


# -----------------------------
# Dashboard tab
# -----------------------------
with tabs[0]:
    st.subheader("Dashboard")

    total_comments = len(f)
    unique_commenters = int(f["author"].nunique(dropna=True)) if "author" in f.columns else 0
    live_ct = int((f["thread_type"] == "live_game").sum())
    post_ct = int((f["thread_type"] == "postgame").sum())

    sent_counts = f["sentiment"].value_counts()
    neg_ct = int(sent_counts.get("negative", 0))
    hs = heat_score(f)

    st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
    _render_kpi_card("Heat score", str(hs), "Higher = more risk/volatility.", help_term="Heat score", show_hover=show_hover)
    _render_kpi_card("Comments", str(total_comments), "", help_term="Comments", show_hover=show_hover)
    _render_kpi_card("Unique commenters", str(unique_commenters), "Usernames hidden (Raw Data only).", help_term="Unique commenters", show_hover=show_hover)
    _render_kpi_card("Neg %", f"{pct(neg_ct, max(total_comments, 1))}%", "", help_term="Neg %", show_hover=show_hover)
    _render_kpi_card("Live game", str(live_ct), "", help_term="Live game", show_hover=show_hover)
    _render_kpi_card("Postgame", str(post_ct), "", help_term="Postgame", show_hover=show_hover)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # Tone mix
    st.markdown("### Tone mix (filtered)")
    tone_df = sent_counts.rename_axis("tone").reset_index(name="count")
    tone_df["share_%"] = tone_df.apply(lambda r: pct(int(r["count"]), max(total_comments, 1)), axis=1)
    st.dataframe(tone_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # People leaderboard
    st.markdown("### People mention leaderboard (players/coaches/broadcast, filtered)")
    pc = people_counts_for_df(f)
    pc_df = pd.DataFrame(pc.most_common(30), columns=["person", "mentions"])
    if pc_df.empty:
        st.info("No people mentions detected with the current dictionary.")
    else:
        if ALTAIR_OK:
            top10 = pc_df.head(10)
            chart = (
                alt.Chart(top10)
                .mark_bar(color=BULLS_RED)
                .encode(
                    x=alt.X("mentions:Q", title="Mentions"),
                    y=alt.Y("person:N", sort="-x", title=""),
                    tooltip=["person", "mentions"],
                )
                .properties(height=320)
            )
            st.altair_chart(chart, use_container_width=True)
        st.dataframe(pc_df, use_container_width=True, hide_index=True)

    st.markdown("### People drill-down (privacy-safe)")
    if not pc_df.empty:
        person_pick = st.selectbox("Select a person", options=pc_df["person"].tolist(), key="person_pick")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Top upvoted comments (score = upvotes)")
            st.dataframe(top_comments_for_person(f, person_pick, limit=25), use_container_width=True, hide_index=True)
        with c2:
            st.markdown("#### Most negative comments (by upvotes)")
            st.dataframe(most_negative_for_person(f, person_pick, limit=25), use_container_width=True, hide_index=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # Deterministic themes
    st.markdown("### Theme tracking (deterministic, filtered)")
    theme_table = theme_kpi_table(f)
    if theme_table.empty:
        st.info("No deterministic theme hits found with current rules.")
    else:
        st.dataframe(theme_table, use_container_width=True, hide_index=True)

        st.markdown("### Theme drill-down (deterministic, usernames hidden)")
        theme_pick = st.selectbox("Select a theme", options=theme_table["theme"].tolist(), key="det_theme_pick")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Top upvoted comments (score = upvotes)")
            st.dataframe(top_comments_for_theme(f, theme_pick, limit=25), use_container_width=True, hide_index=True)
        with c2:
            st.markdown("#### Most negative comments (by upvotes)")
            st.dataframe(most_negative_for_theme(f, theme_pick, limit=25), use_container_width=True, hide_index=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # NLP themes
    st.subheader("Discovered themes (NLP)")
    st.caption("Automatically discovered from comment meaning. Separate from deterministic themes. Usernames are hidden.")

    if not use_nlp:
        if not NLP_OK:
            st.info("NLP is not available in this deployment.")
            if debug_mode:
                st.code(NLP_IMPORT_ERROR)
        else:
            st.info("Turn on 'Enable discovered themes (NLP)' in the sidebar to generate themes.")
    else:
        max_for_nlp = 2500
        if len(f) > max_for_nlp:
            st.warning(f"NLP runs on the first {max_for_nlp} comments for speed.")
            f_nlp = f.head(max_for_nlp).copy()
        else:
            f_nlp = f.copy()

        nlp_out = run_nlp_discovery(f_nlp, model_name=nlp_model, k=int(k_override))
        note = nlp_out.get("note", "")
        if note:
            st.warning(note)

        themes_tbl = nlp_out.get("themes_table", pd.DataFrame())
        assignments = nlp_out.get("assignments", pd.DataFrame())
        reps = nlp_out.get("reps", {})

        if themes_tbl is None or themes_tbl.empty:
            st.info("No NLP themes available for this slice.")
        else:
            st.dataframe(themes_tbl, use_container_width=True, hide_index=True)

            pick = st.selectbox(
                "Select a discovered theme",
                options=themes_tbl["theme_label"].tolist(),
                key="nlp_theme_pick",
            )

            sub = assignments[assignments["cluster_label"] == pick].copy()
            sub = sub.sort_values("score_num", ascending=False)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Top upvoted comments (score = upvotes)")
                safe_sub = _safe_comment_view_cols(sub)
                st.dataframe(safe_sub.head(25), use_container_width=True, hide_index=True)

            with c2:
                st.markdown("#### Representative comments (closest to theme center)")
                cid_series = assignments.loc[assignments["cluster_label"] == pick, "cluster_id"]
                cid = int(cid_series.iloc[0]) if not cid_series.empty else None
                if cid is None or cid not in reps:
                    st.info("No representative comments available.")
                else:
                    rep_rows = assignments.iloc[reps[cid]].copy()
                    rep_rows = rep_rows.sort_values("score_num", ascending=False)
                    safe_rep = _safe_comment_view_cols(rep_rows)
                    st.dataframe(safe_rep.head(25), use_container_width=True, hide_index=True)


# -----------------------------
# Game-by-Game Report tab
# -----------------------------
with tabs[1]:
    st.subheader(f"Game-by-Game Report: {game_date}")

    g = df[df["game_date"] == game_date].copy()
    out = deterministic_game_narrative(g)

    st.markdown("### Narrative summary (deterministic)")
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    for b in out["bullets"]:
        st.write(f"- {b}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")

    st.markdown("### Heat score")
    st.write(f"**{out['heat_score']}** (higher = more risk/volatility)")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.markdown("### Top themes (game)")
    tc = out["themes_all"]
    st.dataframe(pd.DataFrame(tc.most_common(20), columns=["theme", "hits"]), use_container_width=True, hide_index=True)

    st.markdown("### Top people mentions (game)")
    pc2 = out["people_all"]
    st.dataframe(pd.DataFrame(pc2.most_common(25), columns=["person", "mentions"]), use_container_width=True, hide_index=True)

    st.markdown("### Fan quotes (top upvoted, usernames hidden)")
    st.dataframe(out["fan_quotes"], use_container_width=True, hide_index=True)


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
        st.warning("Could not parse game_date values. Use YYYY-MM-DD in filenames or include game_date in the CSV.")
        st.stop()

    min_d, max_d = min(date_objs), max(date_objs)

    # Avoid Streamlit tuple/range differences across versions:
    cA, cB = st.columns(2)
    with cA:
        start_d = st.date_input("Start date", value=min_d, min_value=min_d, max_value=max_d, key="wk_start")
    with cB:
        end_d = st.date_input("End date", value=max_d, min_value=min_d, max_value=max_d, key="wk_end")

    if start_d > end_d:
        st.warning("Start date is after end date. Swapping them.")
        start_d, end_d = end_d, start_d

    weekly = df.copy()
    weekly["game_date_obj"] = pd.to_datetime(weekly["game_date"], errors="coerce").dt.date
    weekly = weekly[(weekly["game_date_obj"] >= start_d) & (weekly["game_date_obj"] <= end_d)].copy()

    games_included = sorted(
        [d for d in weekly["game_date"].dropna().unique().tolist() if re.match(r"^\d{4}-\d{2}-\d{2}$", str(d))]
    )

    st.write(f"- Games included: **{len(games_included)}**")
    st.write(f"- Total comments: **{len(weekly)}**")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.markdown("### Overall tone (heuristic)")
    sent_all = weekly["sentiment"].value_counts()
    sent_all_df = sent_all.rename_axis("tone").reset_index(name="count")
    sent_all_df["share_%"] = sent_all_df.apply(lambda r: pct(int(r["count"]), max(len(weekly), 1)), axis=1)
    st.dataframe(sent_all_df, use_container_width=True, hide_index=True)

    st.markdown("### Top themes (overall)")
    tcw = theme_counts_for_df(weekly)
    st.dataframe(pd.DataFrame(tcw.most_common(20), columns=["theme", "hits"]), use_container_width=True, hide_index=True)

    st.markdown("### Top people mentions (overall)")
    pcw = people_counts_for_df(weekly)
    st.dataframe(pd.DataFrame(pcw.most_common(25), columns=["person", "mentions"]), use_container_width=True, hide_index=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.markdown("### Theme drill-down (weekly, usernames hidden)")
    theme_table_w = theme_kpi_table(weekly)
    if theme_table_w.empty:
        st.info("No theme hits found in this date range.")
    else:
        st.dataframe(theme_table_w, use_container_width=True, hide_index=True)
        theme_pick_w = st.selectbox(
            "Select a theme (weekly)",
            options=theme_table_w["theme"].tolist(),
            key="theme_weekly_pick",
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Top upvoted comments (weekly, score = upvotes)")
            st.dataframe(top_comments_for_theme(weekly, theme_pick_w, limit=25), use_container_width=True, hide_index=True)
        with c2:
            st.markdown("#### Most negative comments (weekly, by upvotes)")
            st.dataframe(most_negative_for_theme(weekly, theme_pick_w, limit=25), use_container_width=True, hide_index=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.markdown("### People drill-down (weekly, usernames hidden)")
    pcw_df = pd.DataFrame(pcw.most_common(40), columns=["person", "mentions"])
    if pcw_df.empty:
        st.info("No people mentions detected in this date range.")
    else:
        person_pick_w = st.selectbox("Select a person (weekly)", options=pcw_df["person"].tolist(), key="person_weekly_pick")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Top upvoted comments (weekly)")
            st.dataframe(top_comments_for_person(weekly, person_pick_w, limit=25), use_container_width=True, hide_index=True)
        with c2:
            st.markdown("#### Most negative comments (weekly)")
            st.dataframe(most_negative_for_person(weekly, person_pick_w, limit=25), use_container_width=True, hide_index=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.markdown("### Game summaries (deterministic bullets)")
    for gd in games_included:
        st.markdown(f"#### {gd}")
        out_g = deterministic_game_narrative(weekly[weekly["game_date"] == gd].copy())
        for b in out_g["bullets"]:
            st.write(f"- {b}")


# -----------------------------
# Definitions tab (searchable)
# -----------------------------
with tabs[3]:
    st.subheader("Definitions")
    st.caption("Search KPI terms used in this dashboard.")

    defs = kpi_definitions_df()
    query = st.text_input("Search definitions", value="", placeholder="Try: heat, score, themes, NLP").strip().lower()

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


# -----------------------------
# Raw Data tab (author allowed only here)
# -----------------------------
with tabs[4]:
    st.subheader("Raw Data")
    st.caption("Raw rows for validation. Usernames may appear here only.")

    safe_f = _ensure_no_duplicate_columns(f.copy())
    st.dataframe(safe_f, use_container_width=True)
