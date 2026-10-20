"""Step 1a: discover Hacker News discussions about preventive health and testing.

Instead of searching for individual comments (noisy), we search for whole
discussion threads (stories) that people actively debated, then let YOU decide
which threads are relevant before any comments are downloaded.

Output: data/threads.csv  (delete the rows you do not want, then run collect.py)
Usage:  python discover.py
"""
import csv
import time
from pathlib import Path

import requests

# Topics around attitudes to preventive medicine, paid testing and longevity
TOPIC_QUERIES = [
    "blood test",
    "preventive health",
    "full body MRI",
    "longevity clinic",
    "health screening",
    "overdiagnosis",
    "Function Health",
    "Superpower health",
    "Prenuvo",
    "biomarkers",
    "direct-to-consumer testing",
    "ApoB cholesterol",
    "biohacking",
]
HEADERS = {"User-Agent": "voc-research-student-project/0.1"}
MIN_COMMENTS = 20  # skip threads with little discussion


def search_threads(query: str):
    """Return HN stories matching the query that had a real discussion."""
    r = requests.get(
        "https://hn.algolia.com/api/v1/search",
        params={
            "query": query,
            "tags": "story",
            "numericFilters": f"num_comments>={MIN_COMMENTS}",
            "hitsPerPage": 20,
        },
        headers=HEADERS,
        timeout=20,
    )
    r.raise_for_status()
    return r.json().get("hits", [])


def main():
    threads = {}
    for q in TOPIC_QUERIES:
        try:
            for h in search_threads(q):
                threads[h["objectID"]] = {
                    "id": h["objectID"],
                    "title": h.get("title", ""),
                    "points": h.get("points", 0),
                    "num_comments": h.get("num_comments", 0),
                    "date": h.get("created_at", "")[:10],
                    "url": f"https://news.ycombinator.com/item?id={h['objectID']}",
                }
            print(f"ok: {q} (threads so far: {len(threads)})")
        except Exception as ex:
            print(f"failed: {q}: {ex}")
        time.sleep(1)

    # Most discussed first
    rows = sorted(threads.values(), key=lambda t: t["num_comments"], reverse=True)
    Path("data").mkdir(exist_ok=True)
    with open("data/threads.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["id", "title", "points", "num_comments", "date", "url"]
        )
        w.writeheader()
        w.writerows(rows)

    print(f"\n{len(rows)} threads saved to data/threads.csv\n")
    for t in rows[:40]:
        print(f"{t['num_comments']:>4} comments | {t['title'][:90]}")


if __name__ == "__main__":
    main()
