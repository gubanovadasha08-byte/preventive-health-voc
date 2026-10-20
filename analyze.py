"""Step 3: aggregate the classified comments and prepare material for the report.

Reads:  data/posts.csv, data/classified.jsonl
Writes: data/relevant.csv          (relevant comments, for manual quality checks)
        data/summary.json          (counts + verified example quotes for the report)
        data/topics_by_stance.png  (chart)
Usage:  py analyze.py
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to a file, no window needed
import matplotlib.pyplot as plt
import pandas as pd

STANCES = ["skeptical", "mixed", "supportive", "neutral"]
EXAMPLES_PER_STANCE = 2  # example quotes kept per topic and stance


def norm(s: str) -> str:
    """Normalize text so quotes can be compared reliably."""
    s = str(s).replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", s).strip().lower()


def quote_is_verbatim(quote: str, text: str) -> bool:
    """True if the model's quote really appears in the original comment."""
    if not isinstance(quote, str) or len(quote.strip()) < 10:
        return False
    parts = re.split(r"\.\.\.|\u2026|\[\.\.\.\]", quote)  # allow '...' inside quotes
    text_n = norm(text)
    return all(norm(p) in text_n for p in parts if len(p.strip()) > 3)


def main():
    with open("data/classified.jsonl", encoding="utf-8") as f:
        cl = pd.DataFrame([json.loads(line) for line in f if line.strip()])
    cl = cl.drop_duplicates("id")
    posts = pd.read_csv("data/posts.csv")
    m = cl.merge(posts[["id", "source", "thread", "text", "url", "date"]],
                 on="id", how="left")
    m["relevant"] = m["relevant"].astype(bool)
    m["quote_verified"] = [quote_is_verbatim(q, t) for q, t in zip(m["quote"], m["text"])]

    rel = m[m["relevant"]].copy()

    # Optional manual cleaning: ids (one per line) of comments you judged off-topic
    n_excluded = 0
    exclude_file = Path("data/exclude_ids.txt")
    if exclude_file.exists():
        excl = {ln.strip() for ln in exclude_file.read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.strip().startswith("#")}
        before = len(rel)
        rel = rel[~rel["id"].isin(excl)].copy()
        n_excluded = before - len(rel)
        print(f"Manually excluded {n_excluded} comments listed in {exclude_file}")

    print(f"Classified: {len(m)} | relevant: {len(rel)} "
          f"({len(rel) / max(len(m), 1):.0%})")
    if len(rel) < 60:
        print("WARNING: fewer than 60 relevant comments; conclusions will be weak.")

    # Quality signal: how many quotes are really verbatim
    print(f"Verbatim quotes among relevant: {rel['quote_verified'].mean():.0%}")

    # Stance overall and by source (to spot bias between HN and manual comments)
    stance_counts = rel["stance"].value_counts().reindex(STANCES, fill_value=0)
    print("\nStance (relevant comments):")
    print(stance_counts.to_string())
    by_source = pd.crosstab(rel["source"], rel["stance"]).reindex(
        columns=STANCES, fill_value=0)
    print("\nStance by source:")
    print(by_source.to_string())

    # Topics x stance
    # explode() repeats row labels; reset them so crosstab works
    exp = rel.explode("topics").dropna(subset=["topics"]).reset_index(drop=True)
    ct = pd.crosstab(exp["topics"], exp["stance"]).reindex(columns=STANCES, fill_value=0)
    ct["total"] = ct.sum(axis=1)
    ct = ct.sort_values("total", ascending=False)
    print("\nTopics by stance:")
    print(ct.to_string())

    # Chart
    plot_df = ct.drop(columns="total").iloc[::-1]  # biggest topic on top
    ax = plot_df.plot(kind="barh", stacked=True, figsize=(9, 5.5))
    ax.set_title("What people discuss about paid preventive testing, by stance")
    ax.set_xlabel("Number of comments")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig("data/topics_by_stance.png", dpi=150)

    # Verified example quotes for each topic (used by the report step).
    # Prefer reasonably long quotes from focused comments (few topics), and never
    # reuse the same comment under two topics. Rarest topics choose first.
    examples = {}
    good = exp[exp["quote_verified"] & (exp["quote"].str.len() >= 50)].copy()
    good["n_topics"] = good.groupby("id")["topics"].transform("count")
    used_ids = set()
    for topic in ct.sort_values("total").index:
        g = good[good["topics"] == topic].sample(frac=1, random_state=1)
        g = g.sort_values("n_topics", kind="stable")
        picked = []
        for stance in STANCES:
            taken = 0
            for r in g[g["stance"] == stance].itertuples():
                if r.id in used_ids:
                    continue
                picked.append(r)
                used_ids.add(r.id)
                taken += 1
                if taken >= EXAMPLES_PER_STANCE:
                    break
        examples[topic] = [
            {"stance": r.stance, "argument": r.argument, "quote": r.quote,
             "source": r.source, "url": r.url}
            for r in picked
        ]

    summary = {
        "n_classified": int(len(m)),
        "n_relevant": int(len(rel)),
        "n_excluded": int(n_excluded),
        "stance_counts": {k: int(v) for k, v in stance_counts.items()},
        "stance_by_source": by_source.to_dict(orient="index"),
        "topic_stance": ct.drop(columns="total").to_dict(orient="index"),
        "topic_totals": {k: int(v) for k, v in ct["total"].items()},
        "examples": examples,
    }
    Path("data/summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=int), encoding="utf-8")

    rel[["id", "source", "thread", "date", "stance", "topics", "argument", "quote",
         "quote_verified", "text", "url"]].to_csv(
        "data/relevant.csv", index=False, encoding="utf-8-sig")
    print("\nSaved: data/relevant.csv, data/summary.json, data/topics_by_stance.png")


if __name__ == "__main__":
    main()
