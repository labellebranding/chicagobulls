import csv
import os
import re
import time
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import requests

SUBREDDIT_NEW_URL = "https://www.reddit.com/r/chicagobulls/new.json?limit=50"
COMMENTS_URL_TEMPLATE = "https://www.reddit.com/comments/{thread_id}.json"

# Detect thread types from titles
THREAD_PATTERNS = [
    ("postgame", re.compile(r"(post game thread|post-game thread)", re.I)),
    ("game", re.compile(r"\b(game thread)\b", re.I)),
    ("pregame", re.compile(r"(pre game thread|pre-game thread)", re.I)),
]

THREAD_FIELDS = [
    "thread_id",
    "thread_type",
    "title",
    "url",
    "created_utc",
    "author",
    "score",
    "num_comments",
    "flair",
]

COMMENT_FIELDS = [
    "comment_id",
    "thread_id",
    "thread_type",
    "created_utc",
    "author",
    "score",
    "depth",
    "parent_id",
    "permalink",
    "body",
]


def user_agent() -> str:
    # Set REDDIT_USER_AGENT env var if you want. Default is fine for local runs.
    return os.environ.get(
        "REDDIT_USER_AGENT",
        "bulls-thread-scraper (local) by u/your_reddit_username",
    )


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": user_agent(),
            "Accept": "application/json",
        }
    )
    return s


def utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def utc_stamp() -> str:
    # filename-safe timestamp like 2026-01-07T21-15-03Z
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def detect_thread_type(title: str) -> Optional[str]:
    for ttype, pat in THREAD_PATTERNS:
        if pat.search(title or ""):
            return ttype
    return None


def safe_get_json(s: requests.Session, url: str) -> dict:
    # raw_json=1 prevents some escaping issues
    if "raw_json=1" not in url:
        joiner = "&" if "?" in url else "?"
        url = f"{url}{joiner}raw_json=1"

    resp = s.get(url, timeout=30)
    if resp.status_code == 429:
        raise RuntimeError("Rate limited (429). Try again later or slow down.")
    resp.raise_for_status()
    return resp.json()


def fetch_new_threads(s: requests.Session) -> List[Dict]:
    data = safe_get_json(s, SUBREDDIT_NEW_URL)
    children = data.get("data", {}).get("children", [])

    threads: List[Dict] = []
    for item in children:
        post = item.get("data", {})
        title = post.get("title", "")
        ttype = detect_thread_type(title)
        if not ttype:
            continue

        thread_id = post.get("id")
        permalink = post.get("permalink", "")
        threads.append(
            {
                "thread_id": thread_id,
                "thread_type": ttype,
                "title": title,
                "url": f"https://www.reddit.com{permalink}",
                "created_utc": utc_iso(post.get("created_utc", 0)),
                "author": post.get("author"),
                "score": post.get("score"),
                "num_comments": post.get("num_comments"),
                "flair": post.get("link_flair_text") or "",
            }
        )
    return threads


def walk_comment_tree(
    children: Iterable[dict], thread_id: str, thread_type: str
) -> Iterable[Dict]:
    stack: List[Tuple[dict, int, str]] = []  # (node, depth, parent_id)
    for ch in children:
        stack.append((ch, 0, ""))

    while stack:
        node, depth, parent_id = stack.pop()
        kind = node.get("kind")
        data = node.get("data", {})

        if kind != "t1":
            continue  # skip "more" placeholders

        comment_id = data.get("id")
        body = data.get("body") or ""
        author = data.get("author")
        score = data.get("score")
        created = utc_iso(data.get("created_utc", 0))
        permalink = data.get("permalink", "")

        yield {
            "comment_id": comment_id,
            "thread_id": thread_id,
            "thread_type": thread_type,
            "created_utc": created,
            "author": author,
            "score": score,
            "depth": depth,
            "parent_id": parent_id,
            "permalink": f"https://www.reddit.com{permalink}",
            "body": body,
        }

        replies = data.get("replies")
        if replies and isinstance(replies, dict):
            reply_children = replies.get("data", {}).get("children", [])
            for rc in reply_children:
                stack.append((rc, depth + 1, comment_id or parent_id))


def fetch_thread_comments(s: requests.Session, thread_id: str, thread_type: str) -> List[Dict]:
    url = COMMENTS_URL_TEMPLATE.format(thread_id=thread_id)
    payload = safe_get_json(s, url)

    if not isinstance(payload, list) or len(payload) < 2:
        return []

    comments_listing = payload[1]
    children = comments_listing.get("data", {}).get("children", [])
    return list(walk_comment_tree(children, thread_id, thread_type))


def write_csv(path: str, rows: List[Dict], fieldnames: List[str]) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main():
    s = session()

    # Put one OR many thread IDs here:
    # Example: ["1q3ewd6"] or ["1q3ewd6", "1q56f7l"]
    TARGET_THREAD_IDS = ["1q3ewd6"]  # <-- change these IDs

    threads = fetch_new_threads(s)
    print(f"Found {len(threads)} likely game threads in newest 50 posts.")

    # Save threads list for reference (overwrites each run)
    write_csv("data/threads.csv", threads, THREAD_FIELDS)
    print("Saved: data/threads.csv")

    # Create lookup for thread_type if present in newest list
    meta_by_id: Dict[str, Dict] = {t["thread_id"]: t for t in threads}

    all_comments: List[Dict] = []
    stamp = utc_stamp()

    for idx, thread_id in enumerate(TARGET_THREAD_IDS, start=1):
        meta = meta_by_id.get(thread_id)
        thread_type = meta["thread_type"] if meta else "unknown"

        print(f"[{idx}/{len(TARGET_THREAD_IDS)}] Fetching comments for {thread_type} thread {thread_id}...")

        comments = fetch_thread_comments(s, thread_id, thread_type)
        print(f"  -> {len(comments)} comments captured")

        per_thread_path = f"data/comments_by_thread/comments_{thread_id}_{thread_type}_{stamp}.csv"
        write_csv(per_thread_path, comments, COMMENT_FIELDS)
        print(f"  -> saved {per_thread_path}")

        all_comments.extend(comments)

        time.sleep(1.2)

    master_path = f"data/comments_all_{stamp}.csv"
    write_csv(master_path, all_comments, COMMENT_FIELDS)
    print(f"Saved combined: {master_path}")


if __name__ == "__main__":
    main()
