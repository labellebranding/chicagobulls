import csv
import os
import re
import time
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import requests

SUBREDDIT_NEW_URL = "https://www.reddit.com/r/chicagobulls/new.json?limit=50"
COMMENTS_URL_TEMPLATE = "https://www.reddit.com/comments/{thread_id}.json"

# Basic title detection
THREAD_PATTERNS = [
    ("postgame", re.compile(r"(post game thread|post-game thread)", re.I)),
    ("game", re.compile(r"\b(game thread)\b", re.I)),
    ("pregame", re.compile(r"(pre game thread|pre-game thread)", re.I)),
]

def user_agent() -> str:
    # Strongly recommended by Reddit. Set your own in env if you want.
    return os.environ.get(
        "REDDIT_USER_AGENT",
        "bulls-thread-scraper (github) by u/your_reddit_username"
    )

def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": user_agent(),
        "Accept": "application/json",
    })
    return s

def utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

def detect_thread_type(title: str) -> Optional[str]:
    for ttype, pat in THREAD_PATTERNS:
        if pat.search(title or ""):
            return ttype
    return None

def safe_get_json(s: requests.Session, url: str) -> dict:
    # raw_json=1 prevents some escaping issues (especially useful if you later capture more fields)
    if "raw_json=1" not in url:
        joiner = "&" if "?" in url else "?"
        url = f"{url}{joiner}raw_json=1"

    resp = s.get(url, timeout=30)
    if resp.status_code == 429:
        raise RuntimeError("Rate limited (429). Slow down your schedule or add delays.")
    resp.raise_for_status()
    return resp.json()

def fetch_new_threads(s: requests.Session) -> List[Dict]:
    data = safe_get_json(s, SUBREDDIT_NEW_URL)
    children = data.get("data", {}).get("children", [])
    threads = []
    for item in children:
        post = item.get("data", {})
        title = post.get("title", "")
        ttype = detect_thread_type(title)
        if not ttype:
            continue

        thread_id = post.get("id")
        permalink = post.get("permalink", "")
        threads.append({
            "thread_id": thread_id,
            "thread_type": ttype,
            "title": title,
            "url": f"https://www.reddit.com{permalink}",
            "created_utc": utc_iso(post.get("created_utc", 0)),
            "author": post.get("author"),
            "score": post.get("score"),
            "num_comments": post.get("num_comments"),
            "flair": post.get("link_flair_text") or "",
        })
    return threads

def walk_comment_tree(children: Iterable[dict], thread_id: str, thread_type: str) -> Iterable[Dict]:
    """
    Flattens the nested comment tree into rows.
    Reddit JSON returns 'kind' = 't1' for comments; some 'more' nodes appear too.
    """
    stack: List[Tuple[dict, int, str]] = []  # (node, depth, parent_id)
    for ch in children:
        stack.append((ch, 0, ""))

    while stack:
        node, depth, parent_id = stack.pop()
        kind = node.get("kind")
        data = node.get("data", {})

        if kind != "t1":
            # skip "more" placeholders
            continue

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
            # push replies onto stack
            for rc in reply_children:
                stack.append((rc, depth + 1, comment_id or parent_id))

def fetch_thread_comments(s: requests.Session, thread_id: str, thread_type: str) -> List[Dict]:
    url = COMMENTS_URL_TEMPLATE.format(thread_id=thread_id)
    payload = safe_get_json(s, url)

    # payload is a list: [postListing, commentsListing]
    if not isinstance(payload, list) or len(payload) < 2:
        return []

    comments_listing = payload[1]
    children = comments_listing.get("data", {}).get("children", [])
    return list(walk_comment_tree(children, thread_id, thread_type))

def write_csv(path: str, rows: List[Dict], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

def main():
    s = session()

    threads = fetch_new_threads(s)
    print(f"Found {len(threads)} likely game threads in newest 50 posts.")

    all_comments: List[Dict] = []
    for i, t in enumerate(threads, start=1):
        thread_id = t["thread_id"]
        thread_type = t["thread_type"]
        print(f"[{i}/{len(threads)}] Fetching comments for {thread_type} thread {thread_id}...")
        comments = fetch_thread_comments(s, thread_id, thread_type)
        print(f"  -> {len(comments)} comments captured")
        all_comments.extend(comments)

        # be polite to Reddit
        time.sleep(1.2)

    # Save outputs
    write_csv(
        "data/threads.csv",
        threads,
        ["thread_id", "thread_type", "title", "url", "created_utc", "author", "score", "num_comments", "flair"]
    )
    write_csv(
        "data/comments.csv",
        all_comments,
        ["comment_id", "thread_id", "thread_type", "created_utc", "author", "score", "depth", "parent_id", "permalink", "body"]
    )

    print("Saved:")
    print(" - data/threads.csv")
    print(" - data/comments.csv")

if __name__ == "__main__":
    main()
