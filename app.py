# app.py
# Bulls Fan Sentiment Intelligence (Streamlit) — Demo-ready, deterministic rules + optional model sentiment
# - One-page executive scroll (no tabs)
# - Game switcher (All games + any game_date)
# - Exec KPIs: heat, sentiment, engagement, postgame-vs-live shifts
# - Drilldowns: select a theme/player and see top upvoted + most negative comments
# - Privacy: NO usernames anywhere except optional Raw Data expander (off by default)
# - Definitions: searchable KPI definitions section
# - Hover tooltips: optional hover help on KPIs + key labels
#
# Expected input: per-thread CSVs from scraper:
#   data/comments_by_thread/YYYY-MM-DD_{pregame|live_game|postgame}_THREADID.csv
#
# Required column: body
# Recommended: score, created_utc, thread_type, thread_id
#
# NOTE:
# - Deterministic mode uses keyword rules (fast, stable, not sarcasm-aware).
# - Model mode uses a social-text sentiment model + confidence and adds a review queue.

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
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

# Optional NLP sentiment (Transformers). App runs without it.
MODEL_SENT_OK = False
MODEL_SENT_ERR = ""
try:
    from transformers import pipeline  # type: ignore
    MODEL_SENT_OK = True
except Exception as e:
    MODEL_SENT_OK = False
    MODEL_SENT_ERR = str(e)


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

.small-note {{
  color: {MUTED};
  font-size: 0.86rem;
}}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# KPI dictionary (definitions)
# -----------------------------
KPI_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "Heat score": {
        "definition": "Directional pulse combining negativity and volume.",
        "how": "Heat = (negative/total * 100) + (total/60). Uses selected sentiment mode.",
        "why": "Flags volatile or reputation-risk conversations fast.",
        "notes": "Simple on purpose so it stays comparable across games.",
    },
    "Net sentiment": {
        "definition": "Average sentiment score from -100 to +100 (model mode only).",
        "how": "Net sentiment = mean((p_pos - p_neg) * 100).",
        "why": "More stable than raw negative percent when tone is mixed.",
        "notes": "Only shown when model sentiment is enabled and available.",
    },
    "Negative % (confident)": {
        "definition": "Percent of comments classified as negative with high confidence (model mode).",
        "how": "Counts only comments where confidence >= threshold.",
        "why": "Avoids over-claiming when sarcasm or ambiguity is likely.",
        "notes": "Threshold is shown in Definitions and used in Needs Review.",
    },
    "Negative % (rules)": {
        "definition": "Percent of comments classified as negative using keyword rules.",
        "how": "Matches NEG_WORDS and POS_WORDS. Mixed if both match.",
        "why": "Fast, stable baseline that runs anywhere.",
        "notes": "Not sarcasm-aware.",
    },
    "Uncertain %": {
        "definition": "Percent of comments with low model confidence, or sarcasm risk flagged.",
        "how": "Uncertain if confidence < threshold OR sarcasm_risk is true.",
        "why": "Creates a human review queue for high-visibility comments.",
        "notes": "Only shown in model mode.",
    },
    "Sarcasm risk %": {
        "definition": "Percent of comments flagged as possible sarcasm by simple patterns.",
        "how": "Looks for markers like 'yeah right', 'lol', /s, caps, repeated punctuation.",
        "why": "Sarcasm is a common failure mode for sentiment classifiers.",
        "notes": "Heuristic only, not definitive.",
    },
    "Comments analyzed": {
        "definition": "Total number of comments in the current filter slice.",
        "how": "Row count after filters are applied.",
        "why": "Higher volume usually means the narrative matters more.",
        "notes": "",
    },
    "Live vs Post volume": {
        "definition": "Comment volume split between live game and postgame threads.",
        "how": "Counts of thread_type == live_game vs postgame.",
        "why": "Postgame often contains clearer narratives and blame assignment.",
        "notes": "",
    },
    "score (upvotes)": {
        "definition": "Number of upvotes a comment received (as scraped).",
        "how": "CSV column 'score' coerced to integer.",
        "why": "Used to rank visibility and agreement.",
        "notes": "Displayed as score (upvotes) everywhere.",
    },
    "Theme hits": {
        "definition": "How many comments match a theme’s patterns.",
        "how": "Regex match per theme (comment-level).",
        "why": "Shows which narratives dominate attention.",
        "notes": "A comment can match multiple themes.",
    },
    "Mentions": {
        "definition": "How often a player/staff entity is referenced in comment text.",
        "how": "Regex alias matching; counts hit occurrences.",
        "why": "Shows who the conversation centers on (not performance quality).",
        "notes": "One comment can mention multiple players.",
    },
    "Needs review": {
        "definition": "High visibility comments where sentiment is uncertain or sarcasm risk is flagged.",
        "how": "needs_review = (conf < threshold) OR sarcasm_risk.",
        "why": "Makes the system safer and more useful in real workflows.",
        "notes": "Only shown in model mode.",
    },
}

MODEL_CONF_THRESHOLD = 0.60
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def defs_df() -> pd.DataFrame:
    rows = []
    for term, meta in KPI_DEFINITIONS.items():
        rows.append(
            {
                "term": term,
                "definition": meta.get("definition", ""),
                "how_calculated": meta.get("how", ""),
                "why_it_matters": meta.get("why", ""),
                "notes": meta.get("notes", ""),
            }
        )
    return pd.DataFrame(rows)


def tooltip(term: str) -> str:
    meta = KPI_DEFINITIONS.get(term, {})
    if not meta:
        return ""
    parts = []
    if meta.get("definition"):
        parts.append(meta["definition"])
    if meta.get("how"):
        parts.append(f"How: {meta['how']}")
    if meta.get("why"):
        parts.append(f"Why: {meta['why']}")
    if meta.get("notes"):
        parts.append(f"Notes: {meta['notes']}")
    return "\n".join([p for p in parts if p])


def kpi_card(label: str, value: str, sub: str, help_term: str, enable_hover: bool = True):
    t = tooltip(help_term) if enable_hover else ""
    t = t.replace('"', "'")
    title_attr = f' title="{t}"' if t else ""
    st.markdown(
        f"""
<div class="kpi-card"{title_attr}>
  <div class="kpi-label">{label}</div>
  <div class="kpi-value">{value}</div>
  <div class="kpi-sub">{sub}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def label_with_hover(text: str, help_term: str, enable_hover: bool = True) -> str:
    t = tooltip(help_term) if enable_hover else ""
    t = t.replace('"', "'")
    if t:
        return f"<span title='{t}'>{text}</span>"
    return text


# -----------------------------
# Dictionaries (EDIT THESE)
# -----------------------------
PLAYERS: Dict[str, List[str]] = {
    "Matas Buzelis": [r"\bmatas\b", r"\bbuzelis\b", r"\bmatas buzelis\b"],
    "Coby White": [r"\bcoby\b", r"\bcoby white\b"],
    "Nikola Vucevic": [r"\bvooch\b", r"\bvucevic\b", r"\bvuc\b", r"\bvuce\b"],
    "Patrick Williams": [r"\bpatrick williams\b", r"\bpwill\b", r"\bp will\b", r"\bpat will\b"],
    "Ayo Dosunmu": [r"\bayo\b", r"\bdosunmu\b", r"\bayo dosunmu\b"],
    "Josh Giddey": [r"\bgiddey\b", r"\bjosh giddey\b"],
    "Lonzo Ball": [r"\blonzo\b", r"\blonzo ball\b"],
    "Kevin Huerter": [r"\bhuerter\b", r"\bkevin huerter\b"],
    "Tre Jones": [r"\btre jones\b"],
    "Jevon Carter": [r"\bjevon\b", r"\bjevon carter\b"],
    "Julian Phillips": [r"\bjulian phillips\b", r"\bphillips\b"],
    "Dalen Terry": [r"\bdalen\b", r"\bdalen terry\b"],
    "Jalen Smith": [r"\bjalen smith\b", r"\bj\s*smith\b"],
    "Zach Collins": [r"\bzach collins\b"],
    "Isaac Okoro": [r"\bisaac okoro\b", r"\bokoro\b"],
    "Noa Essengue": [r"\bessengue\b", r"\bnoa essengue\b"],
    "Yuki Kawamura": [r"\bkawamura\b", r"\byuki kawamura\b"],
    "Billy Donovan": [r"\bbilly donovan\b", r"\bcoach donovan\b", r"\bdonovan\b"],
    "Stacey King": [r"\bstacey king\b", r"\bstacey\b"],
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

# Optional deterministic context (no API)
GAME_CONTEXT: Dict[str, Dict[str, str]] = {}


# -----------------------------
# Sarcasm risk heuristic (simple, useful for review queue)
# -----------------------------
SARCASM_HINTS = [
    r"\byeah right\b", r"\bthanks for nothing\b", r"\bwhat a joke\b", r"\bclassic bulls\b",
    r"\bof course\b", r"\bsure\b", r"\bgenius\b", r"\bbrilliant\b",
    r"\blol\b", r"\blmao\b", r"\brofl\b", r"\b/s\b",
    r"\bgreat job\b", r"\bnice job\b",
]


def sarcasm_risk(text: str) -> bool:
    t = (text or "")
    tl = t.lower()
    caps = sum(1 for ch in t if ch.isupper())
    if caps >= 20:
        return True
    if "??" in tl or "!!" in tl:
        return True
    return any(re.search(p, tl, flags=re.I) for p in SARCASM_HINTS)


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


def classify_sentiment_rules(text: str) -> str:
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


def get_active_sentiment_col(use_model: bool, df_subset: pd.DataFrame) -> str:
    if use_model and "sent_model_label" in df_subset.columns:
        return "sent_model_label"
    return "sent_rules"


def heat_score(df_subset: pd.DataFrame, use_model: bool) -> float:
    total = max(len(df_subset), 1)
    col = get_active_sentiment_col(use_model, df_subset)
    neg_ct = int((df_subset[col] == "negative").sum())
    return round((neg_ct / total) * 100.0 + (len(df_subset) / 60.0), 1)


def theme_kpi_table(df_subset: pd.DataFrame, use_model: bool) -> pd.DataFrame:
    total = max(len(df_subset), 1)
    sent_col = get_active_sentiment_col(use_model, df_subset)

    rows = []
    for theme in THEMES.keys():
        hit_mask = df_subset["body"].apply(lambda t: comment_hits_any(t, THEMES.get(theme, [])))
        hits = int(hit_mask.sum())
        if hits == 0:
            continue

        sub = df_subset[hit_mask].copy()
        neg_pct = round(100.0 * (sub[sent_col] == "negative").mean(), 1)
        mix_pct = round(100.0 * (sub[sent_col] == "mixed").mean(), 1) if "mixed" in sub[sent_col].unique().tolist() else 0.0

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
    out = pd.DataFrame(pc.most_common(30), columns=["player_or_staff", "mentions"])
    return out


def top_comments(df_subset: pd.DataFrame, use_model: bool, limit: int = 20) -> pd.DataFrame:
    x = df_subset.copy()
    sent_col = get_active_sentiment_col(use_model, x)
    cols = [c for c in ["game_date", "thread_type", "score_num", sent_col, "body"] if c in x.columns]
    x = x.sort_values("score_num", ascending=False)[cols].head(limit)
    x = x.rename(columns={"score_num": "score (upvotes)", sent_col: "sentiment"})
    return x


def top_comments_for_theme(df_subset: pd.DataFrame, use_model: bool, theme: str, limit: int = 25) -> pd.DataFrame:
    pats = THEMES.get(theme, [])
    x = df_subset.copy()
    x = x[x["body"].apply(lambda t: comment_hits_any(t, pats))].copy()
    sent_col = get_active_sentiment_col(use_model, x)
    cols = [c for c in ["game_date", "thread_type", "score_num", sent_col, "body"] if c in x.columns]
    x = x.sort_values("score_num", ascending=False)[cols].head(limit)
    x = x.rename(columns={"score_num": "score (upvotes)", sent_col: "sentiment"})
    return x


def most_negative_for_theme(df_subset: pd.DataFrame, use_model: bool, theme: str, limit: int = 25) -> pd.DataFrame:
    pats = THEMES.get(theme, [])
    x = df_subset.copy()
    x = x[x["body"].apply(lambda t: comment_hits_any(t, pats))].copy()
    sent_col = get_active_sentiment_col(use_model, x)
    x = x[x[sent_col].isin(["negative", "mixed"])].copy()
    cols = [c for c in ["game_date", "thread_type", "score_num", sent_col, "body"] if c in x.columns]
    x = x.sort_values("score_num", ascending=False)[cols].head(limit)
    x = x.rename(columns={"score_num": "score (upvotes)", sent_col: "sentiment"})
    return x


def top_comments_for_player(df_subset: pd.DataFrame, use_model: bool, player: str, limit: int = 25) -> pd.DataFrame:
    pats = PLAYERS.get(player, [])
    x = df_subset.copy()
    x = x[x["body"].apply(lambda t: comment_hits_any(t, pats))].copy()
    sent_col = get_active_sentiment_col(use_model, x)
    cols = [c for c in ["game_date", "thread_type", "score_num", sent_col, "body"] if c in x.columns]
    x = x.sort_values("score_num", ascending=False)[cols].head(limit)
    x = x.rename(columns={"score_num": "score (upvotes)", sent_col: "sentiment"})
    return x


def narrative_bullets(df_slice: pd.DataFrame, use_model: bool) -> List[str]:
    bullets: List[str] = []
    total = len(df_slice)

    by_type = df_slice.groupby("thread_type").size().to_dict()
    pre_ct = int(by_type.get("pregame", 0))
    live_ct = int(by_type.get("live_game", 0))
    post_ct = int(by_type.get("postgame", 0))

    sent_col = get_active_sentiment_col(use_model, df_slice)
    sent = df_slice[sent_col].value_counts()
    neg = int(sent.get("negative", 0))
    pos = int(sent.get("positive", 0))
    neu = int(sent.get("neutral", 0))
    mix = int(sent.get("mixed", 0))

    mode_label = "Model" if (use_model and "sent_model_label" in df_slice.columns) else "Rules"
    bullets.append(f"Engagement: {total} comments (Pregame {pre_ct}, Live {live_ct}, Post {post_ct}).")
    bullets.append(
        f"Tone ({mode_label}): {pct(neg, total)}% negative, {pct(mix, total)}% mixed, "
        f"{pct(neu, total)}% neutral, {pct(pos, total)}% positive."
    )

    if live_ct > 0 and post_ct > 0:
        if post_ct > live_ct * 1.25:
            bullets.append("Postgame volume rose vs live, suggesting stronger reflection after the final.")
        elif live_ct > post_ct * 1.25:
            bullets.append("Live thread dominated volume, suggesting in-game reactions were the primary driver.")
        else:
            bullets.append("Live and postgame volume were similar, suggesting steady engagement before and after the final.")

    tc = theme_counts_for_df(df_slice)
    if tc:
        top = [k for k, _ in tc.most_common(4)]
        bullets.append("Top narratives: " + ", ".join(top) + ".")

    pc = player_counts_for_df(df_slice)
    if pc:
        top_p = pc.most_common(3)
        bullets.append("Most discussed: " + ", ".join([f"{p} ({c})" for p, c in top_p]) + ".")

    hs = heat_score(df_slice, use_model)
    if hs >= 70:
        bullets.append("Heat: HIGH. Elevated frustration and reputational risk signal.")
    elif hs >= 45:
        bullets.append("Heat: MODERATE. Noticeable criticism, not a full meltdown.")
    else:
        bullets.append("Heat: LOW. Conversation relatively calm or neutral-to-positive.")

    return bullets


# -----------------------------
# Model sentiment
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_sentiment_pipe():
    if not MODEL_SENT_OK:
        raise RuntimeError(MODEL_SENT_ERR or "Transformers not available.")
    return pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        return_all_scores=True,
        truncation=True,
    )


@st.cache_data(show_spinner=False)
def apply_model_sentiment(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      sent_model_label: negative/neutral/positive
      sent_model_conf: top probability
      sent_model_score: p_pos - p_neg (-1..+1)
      sarcasm_risk: bool
      needs_review: bool (low confidence or sarcasm risk)
    """
    if not MODEL_SENT_OK or df_in.empty:
        return df_in

    x = df_in.copy()
    pipe = get_sentiment_pipe()
    texts = x["body"].astype(str).tolist()

    results: List[List[Dict]] = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        preds = pipe(chunk)  # list[list[{label, score}]]
        results.extend(preds)

    labels = []
    confs = []
    scores = []
    p_negs = []
    p_neus = []
    p_poss = []

    for row in results:
        probs = {d["label"].lower(): float(d["score"]) for d in row}
        p_neg = probs.get("negative", 0.0)
        p_neu = probs.get("neutral", 0.0)
        p_pos = probs.get("positive", 0.0)
        top_label, top_conf = max([("negative", p_neg), ("neutral", p_neu), ("positive", p_pos)], key=lambda z: z[1])
        labels.append(top_label)
        confs.append(float(top_conf))
        p_negs.append(float(p_neg))
        p_neus.append(float(p_neu))
        p_poss.append(float(p_pos))
        scores.append(float(p_pos - p_neg))

    x["sent_model_label"] = labels
    x["sent_model_conf"] = confs
    x["sent_model_score"] = scores
    x["p_neg"] = p_negs
    x["p_neu"] = p_neus
    x["p_pos"] = p_poss

    x["sarcasm_risk"] = x["body"].apply(sarcasm_risk)
    x["needs_review"] = (x["sent_model_conf"] < MODEL_CONF_THRESHOLD) | (x["sarcasm_risk"])
    return x


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

        # score = upvotes
        df["score_num"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)

        # Deterministic sentiment baseline
        df["sent_rules"] = df["body"].apply(classify_sentiment_rules)

        all_rows.append(df)

    if not all_rows:
        return pd.DataFrame()

    out = pd.concat(all_rows, ignore_index=True)
    out["game_date"] = out["game_date"].astype(str).str.slice(0, 10)
    out["thread_type"] = out["thread_type"].apply(normalize_thread_type)
    out = out[out["thread_type"].isin(THREAD_TYPES)].copy()
    return out


def safe_df_for_display(df_in: pd.DataFrame, allow_author: bool) -> pd.DataFrame:
    df_out = df_in.copy()

    drop_cols = []
    if not allow_author and "author" in df_out.columns:
        drop_cols.append("author")

    for c in ["thread_id", "comment_id", "permalink"]:
        if c in df_out.columns:
            drop_cols.append(c)

    df_out = df_out.drop(columns=[c for c in drop_cols if c in df_out.columns], errors="ignore")
    return df_out


# -----------------------------
# Header
# -----------------------------
st.title("Bulls Fan Sentiment Intelligence")
st.caption("Narrative drivers from Reddit threads. Score = upvotes. Usernames are hidden on the dashboard.")


# -----------------------------
# Sidebar controls
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

enable_hover = st.sidebar.toggle(
    "Enable hover help",
    value=True,
    help="Adds hover tooltips to KPI cards and key labels.",
    key="enable_hover",
)

show_author_raw = st.sidebar.toggle(
    "Show usernames (Raw Data only)",
    value=False,
    help="Leave off for demos. Only enable if you need to validate raw rows.",
    key="show_author_raw",
)

st.sidebar.markdown("---")
st.sidebar.header("3) Sentiment mode")

use_model_sent = st.sidebar.toggle(
    "Use model sentiment (better for slang and sarcasm)",
    value=False,
    disabled=not MODEL_SENT_OK,
    help="If enabled, uses a social-text sentiment model and adds confidence + review queue.",
    key="use_model_sent",
)
if not MODEL_SENT_OK:
    st.sidebar.caption(f"Model sentiment unavailable: {MODEL_SENT_ERR}")

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

# Apply model sentiment if enabled
if use_model_sent and MODEL_SENT_OK:
    try:
        slice_df = apply_model_sentiment(slice_df)
    except Exception as e:
        st.warning("Model sentiment failed to load. Falling back to rules sentiment.")
        st.caption(str(e))
        use_model_sent = False


# Context line
def game_context_line(game_date: str) -> str:
    meta = GAME_CONTEXT.get(game_date, {})
    opp = meta.get("opponent", "Unknown")
    venue = meta.get("venue", "Unknown")
    return f"Opponent: {opp} • Venue: {venue}"


# -----------------------------
# “What you’re viewing” strip
# -----------------------------
view_label = "Weekly view (all games)" if selected_game == GAME_ALL else f"Game: {selected_game}"
ctx_label = ", ".join(contexts) if contexts else "none"
sent_label = "Model" if (use_model_sent and "sent_model_label" in slice_df.columns) else "Rules"
st.markdown(
    f"<div class='small-note'><b>Viewing:</b> {view_label} • <b>Contexts:</b> {ctx_label} • <b>Sentiment:</b> {sent_label} • <b>Score</b> = upvotes</div>",
    unsafe_allow_html=True,
)
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)


# -----------------------------
# Executive summary
# -----------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)

if selected_game == GAME_ALL:
    st.markdown("### Executive Summary <span class='badge'>Weekly view</span>", unsafe_allow_html=True)
    st.write("This dashboard summarizes fan sentiment and narrative drivers across all uploaded games.")
else:
    st.markdown(f"### Executive Summary <span class='badge'>{selected_game}</span>", unsafe_allow_html=True)
    st.caption(game_context_line(selected_game))

bullets = narrative_bullets(slice_df, use_model_sent)
for b in bullets:
    st.write(f"- {b}")

st.markdown("</div>", unsafe_allow_html=True)
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)


# -----------------------------
# KPI row
# -----------------------------
total_comments = len(slice_df)
by_type = slice_df.groupby("thread_type").size().to_dict()
pre_ct = int(by_type.get("pregame", 0))
live_ct = int(by_type.get("live_game", 0))
post_ct = int(by_type.get("postgame", 0))

hs = heat_score(slice_df, use_model_sent)

st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)

kpi_card("Heat score", f"{hs}", "Higher = more volatility (negativity + volume).", "Heat score", enable_hover)

if use_model_sent and "sent_model_label" in slice_df.columns:
    net_sent = round(float(slice_df["sent_model_score"].mean()) * 100.0, 1)

    confident = slice_df[slice_df["sent_model_conf"] >= MODEL_CONF_THRESHOLD].copy()
    neg_ct_conf = int((confident["sent_model_label"] == "negative").sum())
    neg_pct_conf = pct(neg_ct_conf, max(len(confident), 1))

    uncertain_ct = int((slice_df["sent_model_conf"] < MODEL_CONF_THRESHOLD).sum())
    uncertain_pct = pct(uncertain_ct, max(total_comments, 1))

    sarcasm_pct = pct(int(slice_df["sarcasm_risk"].sum()), max(total_comments, 1))

    kpi_card("Net sentiment", f"{net_sent}", "Range: -100 (neg) to +100 (pos).", "Net sentiment", enable_hover)
    kpi_card("Negative %", f"{neg_pct_conf}%", f"Confident only (conf ≥ {MODEL_CONF_THRESHOLD}).", "Negative % (confident)", enable_hover)
    kpi_card("Uncertain %", f"{uncertain_pct}%", "Low confidence and needs review.", "Uncertain %", enable_hover)
    kpi_card("Sarcasm risk %", f"{sarcasm_pct}%", "Heuristic flag for possible sarcasm.", "Sarcasm risk %", enable_hover)
else:
    sent_counts = slice_df["sent_rules"].value_counts()
    neg_ct = int(sent_counts.get("negative", 0))
    neg_pct = pct(neg_ct, max(total_comments, 1))
    kpi_card("Negative %", f"{neg_pct}%", "Keyword rules baseline. Not sarcasm-aware.", "Negative % (rules)", enable_hover)
    kpi_card("Pregame volume", f"{pre_ct}", "Pregame expectations and mood.", "Comments analyzed", enable_hover)
    kpi_card("Live vs Post volume", f"{live_ct} / {post_ct}", "Live vs postgame comments.", "Live vs Post volume", enable_hover)
    kpi_card("Comments analyzed", f"{total_comments}", "Based on uploaded CSVs.", "Comments analyzed", enable_hover)

# Always include these two to keep the row stable
kpi_card("Comments analyzed", f"{total_comments}", "Based on uploaded CSVs.", "Comments analyzed", enable_hover)
kpi_card("Live vs Post volume", f"{live_ct} / {post_ct}", "Live vs postgame comments.", "Live vs Post volume", enable_hover)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)


# -----------------------------
# Trend context
# -----------------------------
st.subheader("Trend Context")

trend = df[df["thread_type"].isin(contexts)].copy()

# Trend always uses rules (so it stays fast across all games), but you can upgrade later if needed.
trend["neg"] = (trend["sent_rules"] == "negative").astype(int)
trend["pos"] = (trend["sent_rules"] == "positive").astype(int)

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
    st.markdown("### Weekly baseline")
    baseline_neg = float(trend_by_game["negative_pct"].mean()) if len(trend_by_game) else 0.0
    baseline_comments = float(trend_by_game["comments"].mean()) if len(trend_by_game) else 0.0
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.write(f"- Avg negative % (rules): **{baseline_neg:.1f}%**")
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
            y=alt.Y("negative_pct:Q", title="Negative % (rules)"),
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

theme_table = theme_kpi_table(slice_df, use_model_sent)
if theme_table.empty:
    st.info("No theme hits detected with the current theme rules.")
else:
    c1, c2 = st.columns([1.05, 0.95])

    with c1:
        header = label_with_hover("### Theme KPI table", "Theme hits", enable_hover)
        st.markdown(header, unsafe_allow_html=True)
        st.dataframe(theme_table, use_container_width=True, hide_index=True)

    with c2:
        st.markdown("### Theme drilldown (evidence)")
        theme_pick = st.selectbox(
            "Select a theme to view supporting comments",
            options=theme_table["theme"].tolist(),
            key="theme_pick",
        )
        st.caption("Ranked by score (upvotes). Usernames are hidden.")

        a, b = st.columns(2)
        with a:
            st.markdown("**Top upvoted (theme)**")
            st.dataframe(
                safe_df_for_display(top_comments_for_theme(slice_df, use_model_sent, theme_pick, 25), allow_author=False),
                use_container_width=True,
                hide_index=True,
            )
        with b:
            st.markdown("**Most negative (theme)**")
            st.dataframe(
                safe_df_for_display(most_negative_for_theme(slice_df, use_model_sent, theme_pick, 25), allow_author=False),
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
        header = label_with_hover("### Mentions leaderboard", "Mentions", enable_hover)
        st.markdown(header, unsafe_allow_html=True)
        st.dataframe(player_table, use_container_width=True, hide_index=True)

    with c2:
        st.markdown("### Player drilldown (evidence)")
        pick = st.selectbox(
            "Select a player/staff to view top comments",
            options=player_table["player_or_staff"].tolist(),
            key="player_pick",
        )
        st.caption("Ranked by score (upvotes). Usernames are hidden.")
        st.dataframe(
            safe_df_for_display(top_comments_for_player(slice_df, use_model_sent, pick, 25), allow_author=False),
            use_container_width=True,
            hide_index=True,
        )

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)


# -----------------------------
# Top fan quotes
# -----------------------------
st.subheader("Top Fan Quotes (Highest Upvoted)")
st.caption("Most visible or agreed-with comments in the selected slice. Score = upvotes. Usernames are hidden.")
st.dataframe(
    safe_df_for_display(top_comments(slice_df, use_model_sent, limit=25), allow_author=False),
    use_container_width=True,
    hide_index=True,
)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)


# -----------------------------
# Needs review (model mode only)
# -----------------------------
if use_model_sent and "needs_review" in slice_df.columns:
    st.subheader("Needs Review (High visibility + uncertain or sarcasm risk)")
    st.caption("High-upvote comments where model confidence is low or sarcasm risk is flagged. This is the human review queue.")

    review = slice_df[slice_df["needs_review"] == True].copy()
    review = review.sort_values("score_num", ascending=False)

    cols = ["game_date", "thread_type", "score_num", "sent_model_label", "sent_model_conf", "sarcasm_risk", "body"]
    cols = [c for c in cols if c in review.columns]
    review = review[cols].head(25).rename(columns={"score_num": "score (upvotes)"})

    st.dataframe(
        safe_df_for_display(review, allow_author=False),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)


# -----------------------------
# Definitions (searchable)
# -----------------------------
st.subheader("Definitions (Searchable)")
st.caption("Search terms used in this dashboard (what they mean and how to interpret them).")

defs = defs_df()
query = st.text_input("Search definitions", value="", placeholder="Try: heat, net, sarcasm, score, mentions").strip().lower()

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

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)


# -----------------------------
# Optional raw data (hidden behind expander + toggle)
# -----------------------------
with st.expander("Raw Data (for validation only)", expanded=False):
    st.caption("Usernames appear only if you enable the sidebar toggle. Score = upvotes.")
    raw_view = safe_df_for_display(slice_df, allow_author=show_author_raw)
    st.dataframe(raw_view, use_container_width=True)
