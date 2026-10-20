"""Step 1b: download all comments from the discussion threads you approved.

Input:
  - data/threads.csv  (created by discover.py; you removed irrelevant rows)
  - data/manual.csv   (optional; columns: text,url,thread)  e.g. Reddit comments
Output: data/posts.csv
Usage:  python collect.py
"""
import csv
import html
import io
import re
import time
from pathlib import Path

import requests

HEADERS = {"User-Agent": "voc-research-student-project/0.1"}
MIN_LEN = 80    # too short: not enough signal for analysis
MAX_LEN = 1500  # too long: usually pasted articles or rants, not opinions
# Drop obvious self-promotion and job posts
SPAM = re.compile(
    r"seeking work|we(?:'re| are) hiring|pre-?order|indiegogo|kickstarter|"
    r"book a chat|use my (?:code|link)",
    re.IGNORECASE,
)


def clean(text: str) -> str:
    """Convert raw HTML into plain, single-spaced text."""
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def read_csv_rows(path: Path):
    """Read a CSV saved by Excel or a script: detects encoding and delimiter.

    Excel on Windows often saves CSV as cp1252 and uses ';' as the delimiter,
    so we try UTF-8 first, then fall back to Windows encodings.
    """
    raw = path.read_bytes()
    text = None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    first_line = text.splitlines()[0] if text else ""
    delimiter = max([",", ";", "\t"], key=first_line.count)
    return list(csv.DictReader(io.StringIO(text, newline=""), delimiter=delimiter))


def fetch_thread(thread_id: str):
    """Yield every comment in a HN thread (the whole nested tree)."""
    r = requests.get(
        f"https://hn.algolia.com/api/v1/items/{thread_id}", headers=HEADERS, timeout=30
    )
    r.raise_for_status()
    root = r.json()
    title = root.get("title", "")
    stack = list(root.get("children", []))  # iterative walk: no recursion limit issues
    while stack:
        node = stack.pop()
        stack.extend(node.get("children", []))
        text = clean(node.get("text"))
        if text:
            # Author names are intentionally NOT stored (privacy)
            yield {
                "id": f"hn-{node['id']}",
                "source": "hackernews",
                "thread": title,
                "text": text,
                "url": f"https://news.ycombinator.com/item?id={node['id']}",
                "date": (node.get("created_at") or "")[:10],
            }


def main():
    rows, seen = [], set()

    def add(items):
        """Append records, skipping duplicates, bad length and spam."""
        for row in items:
            text = row["text"]
            if row["id"] in seen or not (MIN_LEN <= len(text) <= MAX_LEN):
                continue
            if SPAM.search(text):
                continue
            seen.add(row["id"])
            rows.append(row)

    threads_file = Path("data/threads.csv")
    if threads_file.exists():
        threads = read_csv_rows(threads_file)
        for i, t in enumerate(threads, 1):
            try:
                add(fetch_thread(t["id"]))
                print(f"[{i}/{len(threads)}] ok: {t['title'][:70]} (total {len(rows)})")
            except Exception as ex:
                print(f"[{i}/{len(threads)}] failed: {t['title'][:70]}: {ex}")
            time.sleep(1)  # be polite to the API
    else:
        print("data/threads.csv not found: run discover.py first")

    # Manual input, e.g. Reddit comments you copied yourself
    manual = Path("data/manual.csv")
    if manual.exists():
        for i, r in enumerate(read_csv_rows(manual)):
            add([{
                "id": f"manual-{i}",
                "source": "manual",
                "thread": r.get("thread", ""),
                "text": clean(r["text"]),
                "url": r.get("url", ""),
                "date": "",
            }])

    Path("data").mkdir(exist_ok=True)
    with open("data/posts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["id", "source", "thread", "text", "url", "date"]
        )
        w.writeheader()
        w.writerows(rows)
    print(f"\nDone: {len(rows)} texts -> data/posts.csv")


if __name__ == "__main__":
    main()
