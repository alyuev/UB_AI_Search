"""Compare float32 vs int8 vector storage on a bench DB (no re-embedding needed).

Uses stored paragraph vectors as queries:
  1) cross-lingual pair retrieval: RU vector -> EN matrix (and EN -> RU), rank of the paired paragraph;
  2) top-10 overlap for scheme D (max over EN/RU) between float32 and int8 matrices.
"""
import sqlite3
import sys
import time

import numpy as np

db = sys.argv[1] if len(sys.argv) > 1 else "bench/bge-m3.db"
con = sqlite3.connect(db)
rows = con.execute("select para_id, lang, vector from embeddings order by para_id").fetchall()
vecs = {"en": {}, "ru": {}}
for pid, lang, blob in rows:
    vecs[lang][pid] = np.frombuffer(blob, dtype=np.float32)
ids = sorted(vecs["en"])
EN = np.stack([vecs["en"][i] for i in ids])
RU = np.stack([vecs["ru"][i] for i in ids])
n, dim = EN.shape


def quant(m):
    """Symmetric per-vector int8 quantization; returns int8 matrix and per-row scales."""
    scale = np.abs(m).max(axis=1, keepdims=True) / 127.0
    q = np.clip(np.round(m / scale), -127, 127).astype(np.int8)
    return q, scale.astype(np.float32)


EN_q, EN_s = quant(EN)
RU_q, RU_s = quant(RU)


def scores_f32(q, M):
    return M @ q


def scores_i8(q, Mq, Ms):
    # dequantize on the fly: (Mq * s) @ q  == (Mq @ q) * s
    return (Mq.astype(np.float32) @ q) * Ms[:, 0]


def pair_eval(Q, M, Mq, Ms):
    r1_f = r1_q = rr_f = rr_q = 0.0
    for i in range(n):
        sf = scores_f32(Q[i], M)
        sq = scores_i8(Q[i], Mq, Ms)
        rank_f = int((sf > sf[i]).sum()) + 1
        rank_q = int((sq > sq[i]).sum()) + 1
        r1_f += rank_f == 1
        r1_q += rank_q == 1
        rr_f += 1 / rank_f
        rr_q += 1 / rank_q
    return r1_f / n, rr_f / n, r1_q / n, rr_q / n


print(f"db={db} n={n} dim={dim}")
print(f"storage float32: {EN.nbytes + RU.nbytes:,} B; int8: {EN_q.nbytes + RU_q.nbytes + EN_s.nbytes + RU_s.nbytes:,} B")
recon = np.sum(EN * (EN_q * EN_s), axis=1)
print(f"cosine(original, int8-reconstructed): mean={recon.mean():.5f} min={recon.min():.5f}")

for name, Q, M, Mq, Ms in [("RU->EN", RU, EN, EN_q, EN_s), ("EN->RU", EN, RU, RU_q, RU_s)]:
    r1f, mf, r1q, mq = pair_eval(Q, M, Mq, Ms)
    print(f"{name}: float32 R@1={r1f:.4f} MRR={mf:.4f} | int8 R@1={r1q:.4f} MRR={mq:.4f}")

rng = np.random.default_rng(0)
sample = rng.choice(n, 300, replace=False)
ov10, same1 = [], 0
for i in sample:
    q = RU[i]  # RU paragraph as query, scheme D
    sf = np.maximum(scores_f32(q, EN), scores_f32(q, RU))
    sq = np.maximum(scores_i8(q, EN_q, EN_s), scores_i8(q, RU_q, RU_s))
    tf = np.argsort(-sf)[1:11]  # skip itself
    tq = np.argsort(-sq)[1:11]
    ov10.append(len(set(tf) & set(tq)) / 10)
    same1 += tf[0] == tq[0]
print(f"scheme D top-10 overlap float32 vs int8: mean={np.mean(ov10):.4f} min={np.min(ov10):.2f}; same top-1: {same1/len(sample):.4f}")

# latency on full-corpus-sized matrix (29170 x dim)
big = np.tile(EN, (int(np.ceil(29170 / n)), 1))[:29170]
bq, bs = quant(big)
q = RU[0]
t = time.perf_counter(); [big @ q for _ in range(50)]; tf = (time.perf_counter() - t) / 50 * 1000
t = time.perf_counter(); [(bq.astype(np.float32) @ q) * bs[:, 0] for _ in range(50)]; tq = (time.perf_counter() - t) / 50 * 1000
print(f"latency 29170x{dim}: float32 {tf:.2f} ms; int8 (dequant each query) {tq:.2f} ms")
