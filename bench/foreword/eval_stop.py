"""Before/after stop-word filtering on Paper 0 (hybrid + standalone FTS defaults).
Usage: python bench/foreword/eval_stop.py   (prints a summary; append to docs by hand/script)"""
import json, sys
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
import app.core.search as S
from app.core.embedder import create_embedder

EXTRA = [  # question-form queries, relevant ids judged by reading
 ("Кто такие Настройщики Мышления?", {"0:5.9", "0:2.3"}),
 ("What is the Paradise Trinity?", {"0:12.1", "0:12.2", "0:12.9", "0:1.14"}),
 ("Что такое Рай и где он находится?", {"0:4.11", "0:4.12", "0:4.13"}),
 ("Как возникает душа человека?", {"0:5.10"}),
 ("Who is the Universal Father?", {"0:2.12", "0:3.10", "0:5.5"}),
 ("Что такое эталон?", {"0:6.10", "0:6.11", "0:6.13"}),
]
qs = [(q["query"], {r["id"] for r in q["relevant"]}) for q in
      json.load(open(HERE / "queries.json", encoding="utf-8"))["queries"]] + EXTRA
s = S.Searcher(HERE / "paper0.db", embedder=create_embedder("bge-m3", runtime="llamacpp-vulkan", device="auto"))
orig = S._is_stop


def run(label, fn):
    r10 = first = n_first = 0
    for q, rel in qs:
        ids = [r["id"] for r in fn(q)][:10]
        r10 += len(set(ids) & rel) / len(rel)
        pos = next((i + 1 for i, x in enumerate(ids) if x in rel), None)
        if pos: first += pos; n_first += 1
    print(f"{label}: share of needed paragraphs in top-10 = {r10/len(qs):.3f}; "
          f"avg position of first good = {first/max(n_first,1):.2f} (found for {n_first}/{len(qs)})")


for stop_on in (False, True):
    S._is_stop = orig if stop_on else (lambda w, l=None: False)
    tag = "stop-words ON " if stop_on else "stop-words OFF"
    run(f"{tag} | standalone FTS", lambda q: s.fts(q, k=10))
    run(f"{tag} | hybrid       ", lambda q: s.hybrid(q, k=10))
S._is_stop = orig
