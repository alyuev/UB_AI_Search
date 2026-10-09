import json, sys
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
import app.core.search as S
from app.core.embedder import create_embedder
exec(open(HERE/"eval_stop.py").read().split("s = S.Searcher")[0])
s = S.Searcher(HERE/"paper0.db", embedder=create_embedder("bge-m3", runtime="llamacpp-vulkan", device="auto"))
for q, rel in qs:
    s.fts_stopwords=False; a={r["id"] for r in s.hybrid(q,k=10)}
    s.fts_stopwords=True; b={r["id"] for r in s.hybrid(q,k=10)}
    lost=(a-b)&rel
    if lost:
        dropped=[w for w in S._WORD_RE.findall(q) if S._is_stop(w, S._get_morph().parse(w)[0].normal_form)]
        print(q, "| lost", sorted(lost), "| gained", sorted((b-a)&rel), "| stop dropped", dropped)
