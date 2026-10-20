"""Manual accuracy check of the classifier (about 15 minutes).

Step 1:  py validate.py make
         -> creates data/check20.csv with 20 random comments the model called "relevant".
Step 2:  open it in Excel and fill the three empty columns for every row with 1 or 0:
           relevant_ok : 1 if the comment really is about paid/elective health testing,
                         screening, scans, longevity clinics or preventive medicine
                         for healthy people; 0 if it is off-topic
           stance_ok   : 1 if the model's stance (skeptical/supportive/mixed/neutral)
                         is right; 0 if wrong (leave empty if relevant_ok is 0)
           topics_ok   : 1 if the model's topics fit the comment; 0 if not
                         (leave empty if relevant_ok is 0)
         Save as CSV.
Step 3:  py validate.py score
         -> prints a sentence you can paste into: py report.py --validation "..."
"""
import csv
import io
import sys
from pathlib import Path

import pandas as pd

CHECK = Path("data/check20.csv")
YES = {"1", "y", "yes", "\u0434\u0430", "+"}
NO = {"0", "n", "no", "\u043d\u0435\u0442", "-"}


def make():
    rel = pd.read_csv("data/relevant.csv")
    sample = rel.sample(20, random_state=7)[
        ["id", "text", "stance", "topics", "argument"]
    ].rename(columns={"stance": "model_stance", "topics": "model_topics",
                      "argument": "model_argument"})
    for col in ["relevant_ok", "stance_ok", "topics_ok"]:
        sample[col] = ""
    sample.to_csv(CHECK, index=False, encoding="utf-8-sig")
    print(f"Created {CHECK}. Fill relevant_ok / stance_ok / topics_ok with 1 or 0, "
          "save as CSV, then run: py validate.py score")


def read_marks(path: Path):
    """Read the CSV after Excel: detect encoding and delimiter."""
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    first = text.splitlines()[0]
    delim = max([",", ";", "\t"], key=first.count)
    return list(csv.DictReader(io.StringIO(text, newline=""), delimiter=delim))


def mark(value):
    v = (value or "").strip().lower()
    if v in YES:
        return True
    if v in NO:
        return False
    return None  # empty or unrecognized


def score():
    rows = read_marks(CHECK)
    rel = [mark(r.get("relevant_ok")) for r in rows]
    if any(m is None for m in rel):
        sys.exit("Every row needs 1 or 0 in relevant_ok. Fill the column and save as CSV.")
    ok_rows = [r for r, m in zip(rows, rel) if m]
    stance = [mark(r.get("stance_ok")) for r in ok_rows]
    topics = [mark(r.get("topics_ok")) for r in ok_rows]
    if any(m is None for m in stance + topics):
        sys.exit("For rows with relevant_ok = 1, fill stance_ok and topics_ok too.")

    n, n_ok = len(rows), len(ok_rows)
    text = (f"{n_ok}/{n} comments were truly relevant; among those, the stance was "
            f"correct in {sum(stance)}/{n_ok} and the topics in {sum(topics)}/{n_ok}")
    print(text)
    print("\nNow run:")
    print(f'py report.py --validation "{text}"')


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("make", "score"):
        sys.exit("Usage: py validate.py make | score")
    make() if sys.argv[1] == "make" else score()
