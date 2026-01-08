# nlp.py
# Lightweight NLP theme discovery for Reddit comments
# - Embeddings via sentence-transformers
# - Clustering via KMeans
# - Cluster labels via top TF-IDF terms
# - Representative comments via distance-to-centroid
#
# Designed for Streamlit use with caching in app.py.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


DEFAULT_MODEL = "all-MiniLM-L6-v2"


def _clean_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    return " ".join(s.split()).strip()


def embed_texts(
    texts: List[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Returns embeddings shape (n, d).
    """
    from sentence_transformers import SentenceTransformer  # local import to avoid import errors at module load

    model = SentenceTransformer(model_name)
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embs


def choose_k(n: int) -> int:
    """
    Deterministic K choice:
    - Too small -> everything merges
    - Too large -> noisy themes
    """
    if n < 40:
        return 4
    if n < 100:
        return 6
    if n < 250:
        return 8
    if n < 600:
        return 10
    return 12


def cluster_embeddings(embs: np.ndarray, k: int, random_state: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (labels, centroids)
    """
    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    labels = km.fit_predict(embs)
    centroids = km.cluster_centers_
    return labels, centroids


def label_clusters_tfidf(
    texts: List[str],
    labels: np.ndarray,
    top_n: int = 5,
    stop_words: str = "english",
    max_features: int = 5000,
) -> Dict[int, str]:
    """
    Builds short human-readable labels per cluster using TF-IDF top terms.
    """
    vec = TfidfVectorizer(stop_words=stop_words, max_features=max_features)
    X = vec.fit_transform(texts)
    terms = np.array(vec.get_feature_names_out())

    out: Dict[int, str] = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            out[int(c)] = f"Theme {int(c)+1}"
            continue

        # Average TF-IDF weights within cluster
        sub = X[idx].mean(axis=0)
        sub = np.asarray(sub).ravel()

        top_idx = sub.argsort()[::-1][:top_n]
        top_terms = [t for t in terms[top_idx] if t]
        if not top_terms:
            out[int(c)] = f"Theme {int(c)+1}"
        else:
            out[int(c)] = " / ".join(top_terms[:3])
    return out


def representative_indices(
    embs: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    top_n: int = 8,
) -> Dict[int, List[int]]:
    """
    Returns indices of comments closest to centroid (representative examples).
    """
    rep: Dict[int, List[int]] = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            rep[int(c)] = []
            continue
        cvec = centroids[int(c)]
        # cosine distance since embeddings normalized
        d = 1.0 - np.dot(embs[idx], cvec)
        order = np.argsort(d)[:top_n]
        rep[int(c)] = idx[order].tolist()
    return rep


def build_discovered_themes(
    df: pd.DataFrame,
    text_col: str = "body",
    score_col: str = "score_num",
    model_name: str = DEFAULT_MODEL,
    k: Optional[int] = None,
    min_comments: int = 30,
) -> Dict[str, object]:
    """
    Returns a dict with:
      - themes_table: DataFrame (cluster_id, label, comments, share_%)
      - assignments: df copy with cluster_id + cluster_label
      - reps: dict cluster_id -> representative row indices
    """
    x = df.copy()
    x[text_col] = x[text_col].astype(str).map(_clean_text)
    x = x[x[text_col].str.len() > 0].copy()

    n = len(x)
    if n < min_comments:
        return {
            "themes_table": pd.DataFrame(),
            "assignments": x.assign(cluster_id=None, cluster_label=None),
            "reps": {},
            "note": f"Not enough comments for NLP discovery (need {min_comments}+; found {n}).",
        }

    texts = x[text_col].tolist()
    embs = embed_texts(texts, model_name=model_name)

    k_use = k if k is not None else choose_k(n)
    k_use = max(3, min(k_use, 14))
    labels, centroids = cluster_embeddings(embs, k=k_use)

    cluster_labels = label_clusters_tfidf(texts, labels)
    reps = representative_indices(embs, labels, centroids, top_n=8)

    x["cluster_id"] = labels.astype(int)
    x["cluster_label"] = x["cluster_id"].map(cluster_labels)

    # Summary table
    counts = x["cluster_id"].value_counts().sort_values(ascending=False)
    total = max(len(x), 1)
    rows = []
    for cid, ct in counts.items():
        rows.append(
            {
                "cluster_id": int(cid),
                "theme_label": cluster_labels.get(int(cid), f"Theme {int(cid)+1}"),
                "comments": int(ct),
                "share_%": round(100.0 * int(ct) / total, 1),
            }
        )
    themes_table = pd.DataFrame(rows)

    # Safety: ensure score exists
    if score_col not in x.columns:
        x[score_col] = 0

    return {
        "themes_table": themes_table,
        "assignments": x,
        "reps": reps,
        "note": "",
    }
