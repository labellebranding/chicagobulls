import pandas as pd
import streamlit as st
from collections import Counter

st.set_page_config(page_title="Bulls Fan Revenue Intelligence", layout="wide")
st.title("Bulls Fan Revenue Intelligence")
st.caption("Upload your CSVs from data/comments_by_thread to generate game + weekly summaries.")

st.sidebar.header("Upload")
uploaded = st.sidebar.file_uploader(
    "Upload one or more comments CSV files",
    type=["csv"],
    accept_multiple_files=True
)

if not uploaded:
    st.info("Upload your comments CSV files in the left sidebar to see the dashboard.")
    st.stop()

@st.cache_data
def load_files(file_objs):
    dfs = []
    for f in file_objs:
        df = pd.read_csv(f)
        df["source_file"] = getattr(f, "name", "uploaded.csv")
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)

    # Clean/standardize expected columns
    for col in ["game_date", "thread_type", "thread_id", "author", "body", "created_utc"]:
        if col in out.columns:
            out[col] = out[col].fillna("").astype(str)

    if "score" in out.columns:
        out["score"] = pd.to_numeric(out["score"], errors="coerce").fillna(0).astype(int)

    return out

df = load_files(uploaded)

# Sidebar filters
st.sidebar.header("Filters")

if "game_date" in df.columns:
    dates = sorted([d for d in df["game_date"].unique() if d])
    selected_dates = st.sidebar.multiselect("Game dates", dates, default=dates)
else:
    selected_dates = []

if "thread_type" in df.columns:
    types = sorted([t for t in df["thread_type"].unique() if t])
    selected_types = st.sidebar.multiselect("Thread types", types, default=types)
else:
    selected_types = []

f = df.copy()
if selected_dates and "game_date" in f.columns:
    f = f[f["game_date"].isin(selected_dates)]
if selected_types and "thread_type" in f.columns:
    f = f[f["thread_type"].isin(selected_types)]

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Games", f["game_date"].nunique() if "game_date" in f.columns else "—")
c2.metric("Threads", f["thread_id"].nunique() if "thread_id" in f.columns else "—")
c3.metric("Total comments", f"{len(f):,}")
c4.metric("Unique commenters", f["author"].nunique() if "author" in f.columns else "—")

st.divider()

# Volume by game + thread type
if "game_date" in f.columns and "thread_type" in f.columns:
    st.subheader("Comment volume by game and thread type")
    vol = (
        f.groupby(["game_date", "thread_type"])
        .size()
        .reset_index(name="comments")
        .sort_values(["game_date", "thread_type"])
    )
    st.dataframe(vol, use_container_width=True)
else:
    st.warning("Missing game_date or thread_type columns. Make sure you’re uploading the per-thread comments CSVs.")

st.divider()

# Quick narrative proxy: top recurring words
st.subheader("Top recurring words (quick narrative proxy)")
if "body" in f.columns:
    text = " ".join(f["body"].astype(str).tolist()).lower()
    stop = {
        "the","and","to","of","a","in","is","it","for","on","that","we","this","with","was","are","as",
        "they","be","but","have","not","at","you","i","our","so","if","just","im","its","from","like"
    }
    tokens = [t.strip(".,!?()[]{}:;\"'") for t in text.split()]
    tokens = [t for t in tokens if len(t) >= 4 and t not in stop]
    top = Counter(tokens).most_common(40)
    st.table(pd.DataFrame(top, columns=["word", "count"]))
else:
    st.warning("No 'body' column found in uploaded CSVs.")

st.divider()

# Raw preview
st.subheader("Raw comments preview")
preview_cols = [c for c in ["game_date","thread_type","thread_id","author","score","created_utc","body","source_file"] if c in f.columns]
st.dataframe(f[preview_cols].head(300), use_container_width=True)
