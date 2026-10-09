# Stage 3 Benchmark — Search Quality & Embedding Speed

Benchmarked on a **20-paper / 1,572-paragraph subset** of the corpus (not the
full 14,585 paragraphs — the user changed plans mid-benchmark to keep
iteration fast; full-corpus timings below are extrapolated from measured
paras/sec). Subset papers: `0,1,2,3,4,5,6,7,19,20,39,40,43,63,66,69,75,101,108,113`
(Foreword/Trinity/Thought Adjuster material, Melchizedek, Lucifer rebellion,
Michael of Nebadon, Andon & Fonta's origin-of-man narrative, marriage/family,
guardian angels, personality survival). 29 EN/RU/cross-lingual queries in
`bench/queries.json`, all `relevant` IDs verified to fall inside the subset.

Machine: Ryzen AI 9 HX 370 (12 physical / 24 logical cores), Radeon 890M
iGPU, 48GB RAM, Windows 11, Python 3.14.

## 1. Embedding speed: CPU vs DirectML vs llama.cpp-Vulkan

Measured on 300 real paragraphs (`bench/subset.json[:300]`), best batch size
per config shown (full data: `bench/profile_results.json`,
`bench/profile_vulkan_results.json`).

| Model | CPU (12 threads) | CPU (10 threads, cores−2) | DirectML (GPU) | llama.cpp Vulkan (GPU) |
|---|---|---|---|---|
| multilingual-e5-small | 110.6/s | **113.1/s** | 52.9/s | **319.4/s** |
| multilingual-e5-base | 43.6/s | **44.2/s** | 20.2/s | **128.9/s** |
| bge-m3 | **15.2/s** | 14.9/s | 6.3/s | **46.1/s** |
| qwen3-embedding-0.6b | ~13/s* | ~12.7/s* | 3.9/s | **30.8/s** |

\* qwen3 CPU numbers were re-measured after an earlier run was found to be
contended by a concurrently-loaded LM Studio LLM (contended: ~7.5-8.7/s).

**llama.cpp's Vulkan backend is 2.8x-4x faster than the best CPU config for
every model tried**, and DirectML (onnxruntime-directml) is consistently
*slower* than CPU on this iGPU for these small/short-context models — DML's
per-dispatch overhead dominates at this scale.

### Why DirectML was dropped in favor of Vulkan (UI lag)

The user observed that ONNX Runtime's DirectML execution provider made the
whole Windows desktop feel laggy while embedding. Root cause, confirmed with
`Get-Counter '\GPU Engine(*)\Utilization Percentage'` sampled once per
second during a load:

- **onnxruntime-directml**: creates a D3D12 **DIRECT** (3D/graphics) command
  queue on the shared Radeon 890M — the same queue class the desktop
  compositor uses — so heavy embedding work directly competes with UI
  rendering.
- **llama.cpp (Vulkan backend)**: dispatches through a **Compute** queue.
  Confirmed directly: during a bge-m3 llama-server load, `Get-Counter`
  showed 75-105% utilization on `engtype_Compute 0` for that process's PID,
  and the process **never appeared at all** under `engtype_3D`.

Because of this, **`device="auto"` in `app/core/embedder.py` now means CPU
only** — DirectML is never auto-selected, only available as an explicit
`device="gpu"` opt-in with a UI-facing warning
(`app.core.embedder.DEVICE_INFO["gpu"]`). The recommended fast GPU path is
`runtime="llamacpp-vulkan"` instead (see below).

## 2. NPU (AMD XDNA2) — findings, no SDK installed

- `onnxruntime-directml`'s `DmlExecutionProvider` was tried with
  `provider_options={"device_filter": "npu"}` and rejected outright at
  session-creation with `"Invalid Filter provided for DirectML EP device
  selection"` — this build's compiled-in filter enum only accepts `"gpu"`
  (`"npu"`, `"any"`, and case variants were all rejected). DirectML on this
  system cannot reach the NPU.
- No Vitis AI / Ryzen AI EP is installed (`ort.get_available_providers()` is
  still just `['DmlExecutionProvider', 'CPUExecutionProvider']`), and no
  Ryzen AI SW installation was found on the system. **Not installed**, per
  instruction (it's a large separate SDK requiring model-specific Quark
  INT8/BF16 quantization and static shapes).
- Searched Hugging Face's `amd/` org and general search for ready NPU
  embedding models: found only vision models (`amd/RyzenAI-clip-vit-base-...
  -NPU`, `ryzenai-hrnet-bg-seg`, `ryzenai-esrgan`, etc.) — **no ready-made
  AMD-NPU text embedding model exists** that we could find.
- **Conclusion: NPU is not feasible for this project** without installing
  the Ryzen AI/Vitis AI SDK and doing custom model quantization —
  out of scope. Dropped from further consideration.

## 3. GGUF quality gate: llama.cpp vs ONNX (cosine similarity)

Before trusting llama.cpp-Vulkan's (much faster) output, every GGUF
candidate's embeddings were compared against the corresponding ONNX model's
embeddings on the same sample texts. Bar: cosine ≥ 0.98.

| Model | GGUF source | Cosine vs ONNX | Verdict |
|---|---|---|---|
| **bge-m3** | gpustack/bge-m3-GGUF Q8_0 | **0.985–0.990** | **PASS** |
| **multilingual-e5-base** | dinab/multilingual-e5-base-Q8_0-GGUF | **0.983–0.988** | **PASS** |
| multilingual-e5-small | keisuke-miyako/...-gguf-q8_0 | 0.77–0.82 (weak query/passage discrimination: 0.915 vs 0.899) | FAIL |
| multilingual-e5-small | milimyname/...-Q8_0-GGUF (2nd source) | crashed at load (`unsupported ggml op types`) | FAIL (broken) |
| qwen3-embedding-0.6b | Qwen/Qwen3-Embedding-0.6B-GGUF Q8_0 (official) | 0.78–0.90 | FAIL |

**multilingual-e5-small and qwen3-embedding-0.6b were dropped from the
benchmark** — the community e5-small GGUF conversions are unreliable across
two different sources tried, and the qwen3 GGUF disagrees with our ONNX
implementation by more than the quality bar allows (root cause not fully
isolated — bge-m3 passing proves the overall comparison harness is sound,
so this could be a subtlety in our custom ONNX decoder-forward workaround
for Qwen3's KV-cache-requiring export, not necessarily llama.cpp being
wrong; not chased further given time spent). Both are kept in
`app.core.embedder.LLAMACPP_REGISTRY` with `quality_ok=False` and a
`not_recommended_reason` string, in case someone wants to use them via the
(always-available) ONNX/CPU path.

**Only bge-m3 and multilingual-e5-base were benchmarked for search quality**,
both run through the verified-fast `runtime="llamacpp-vulkan"` path.

## 4. Subset index build time (llama.cpp-Vulkan)

| Model | Build time (1,572 paragraphs) | paras/sec | DB size | Extrapolated full corpus (14,585) |
|---|---|---|---|---|
| bge-m3 | 90.7s | 17.3 | 24.9 MB | ~842s (~14 min) |
| multilingual-e5-base | 43.8s | 35.9 | 23.4 MB | ~407s (~7 min) |

(Includes parsing, paragraph insert, RU lemmatization, FTS5 build, and
embedding both EN and RU columns — not just the embedding step.)

## 5. Search quality (queries.json-based) — ⚠️ BIASED, see §6 for the trustworthy numbers

Recall@10 / MRR@10 / avg latency, averaged over the 29 queries in
`bench/queries.json` (full JSON: `bench/eval_results.json`). **These
numbers looked like they favored multilingual-e5-base and made semantic
search look worse than FTS — both turned out to be measurement artifacts,
not real quality differences. See §6 for why, and §8 for which numbers to
actually trust.**

### bge-m3

| Method | Recall@10 | MRR@10 | Latency (ms) |
|---|---|---|---|
| FTS | 0.292 | 0.345 | 1.9 |
| Semantic (scheme D: max(en,ru)) | 0.197 | 0.266 | 156 |
| Scheme A (RU-only) | 0.208 | 0.299 | 155 |
| Scheme B (EN-only) | 0.189 | 0.247 | 154 |
| **Hybrid RRF k=60, cand=50** | **0.284** | **0.425** | 155 |
| Hybrid RRF k=30, cand=30 | 0.275 | 0.422 | 157 |

### multilingual-e5-base — **winner**

| Method | Recall@10 | MRR@10 | Latency (ms) |
|---|---|---|---|
| FTS | 0.292 | 0.345 | 0.5 |
| Semantic (scheme D: max(en,ru)) | 0.223 | 0.352 | 151 |
| Scheme A (RU-only) | 0.137 | 0.226 | 151 |
| Scheme B (EN-only) | 0.165 | 0.267 | 152 |
| Scheme C (EN+RU concatenated) | 0.224 | 0.313 | 150 (+43s one-off embed pass) |
| **Hybrid RRF k=60, cand=50** | **0.290** | **0.457** | 154 |
| Hybrid RRF k=30, cand=30 | 0.290 | 0.458 | 152 |

(FTS numbers are identical across models — expected, since FTS5/lemma
matching doesn't depend on the embedding model.)

### Reading the schemes

- **Scheme D (separate EN/RU vectors, take max cosine) beats A (RU-only) and
  B (EN-only)** for e5-base, confirming that keeping both languages queryable
  independently — rather than picking one — genuinely helps cross-lingual
  recall. For bge-m3, A (RU-only) narrowly beats D, likely because bge-m3's
  training already handles cross-lingual matching well from either side, so
  splitting adds noise; the margin is small (0.208 vs 0.197).
- **Scheme C (EN+RU concatenated into one vector) essentially ties scheme D**
  for e5-base (0.224 vs 0.223 recall, but D's MRR 0.352 > C's 0.313), while
  costing extra storage (one vector instead of two per paragraph) and hitting
  a real practical snag: **e5-base's native context is only 512 tokens**, so
  a few of our paragraphs' EN+RU concatenation genuinely exceeds it and had
  to be character-truncated for this test. **Scheme D is simpler, already
  implemented, and at least as good — kept as the shipped default.**
- **Hybrid (FTS + semantic via Reciprocal Rank Fusion) clearly beats either
  method alone** for both models — e.g. e5-base hybrid MRR@10 0.457 vs 0.352
  (semantic alone) vs 0.345 (FTS alone). RRF k=60/cand=50 and k=30/cand=30
  are statistically indistinguishable here (29 queries is a small sample);
  k=60/cand=50 kept as the default for slightly more stable ranking with a
  larger candidate pool.
- Reading this table in isolation, multilingual-e5-base looks like the
  winner on semantic recall (0.223 vs 0.197) and hybrid MRR (0.457 vs
  0.425). **§6 explains why this conclusion is not reliable, and §7 gives
  the objective test that overturns it.**

## 6. Why the queries.json numbers are biased

`bench/queries.json`'s "relevant" IDs were found by **grepping the corpus
for the query's own exact terms** (e.g. searching for "Melchizedek" to find
paragraphs containing "Melchizedek"). This has a structural bias: it
rewards systems that match on the same lexical terms — i.e. FTS — and
systematically **undercounts** semantic/concept-level matches that use
different words for the same idea, which is exactly what embedding search
is *for*. Two pieces of evidence confirmed this after the fact:

1. **Pool bias check.** Pooling each system's top-10 across all 29 queries
   and checking what fraction is *not* in the existing qrels:

   | System | Fraction of top-10 NOT in qrels ("unjudged") |
   |---|---|
   | FTS | 0.448 |
   | Semantic (bge-m3) | **0.852** |
   | Semantic (e5-base) | **0.834** |

   FTS's results overlap the qrels far more (unsurprising — the qrels were
   built by finding FTS-matchable terms). Both semantic systems return
   ~83-85% of their top-10 *outside* the judged set. Those results are not
   necessarily wrong — most are plausibly good concept-level matches that
   simply weren't captured by grepping for exact terms — but recall@10
   computed against an incomplete qrel set unfairly penalizes semantic
   search for every one of them.
2. **e5-base "won" on a benchmark that structurally favors its weaker
   cross-lingual generalization** (see §7) — a red flag that something in
   the measurement, not the model, was doing the work.

**Conclusion: §5's recall/MRR numbers should not be used to choose between
bge-m3 and multilingual-e5-base.** They remain useful for one thing only:
confirming that hybrid search beats either FTS or semantic alone (that
comparison isn't sensitive to the qrel bias in the same way, since hybrid's
gain comes from combining ranks, not from a recall level).

## 7. Objective, qrel-free evaluation (this is what the model choice is actually based on)

Two tests that don't depend on manually/grep-curated relevance judgments,
run on the same 1,572-paragraph subset DBs.

### 7a. Cross-lingual pair retrieval (parallel-corpus ground truth)

For 200 randomly sampled paragraphs: embed the **RU** text and search only
the **EN** vector column for the paragraph's own paired EN text (same `id`)
— and the reverse direction. No qrels needed: the parallel corpus itself
*is* the ground truth. Tested with both the full paragraph text (easy — long
text is very distinctive) and just the **first sentence** (harder, and a
much better proxy for how a short real search query behaves).

| Direction | bge-m3 R@1 / R@10 / MRR | e5-base R@1 / R@10 / MRR |
|---|---|---|
| RU full paragraph → EN vectors | 0.995 / 1.000 / 0.998 | 0.990 / 1.000 / 0.994 |
| **RU first sentence → EN vectors** | **0.775 / 0.830 / 0.798** | 0.655 / 0.810 / 0.706 |
| EN full paragraph → RU vectors | 0.995 / 1.000 / 0.998 | 0.995 / 1.000 / 0.996 |
| **EN first sentence → RU vectors** | **0.735 / 0.835 / 0.774** | 0.595 / 0.770 / 0.650 |

Full-paragraph retrieval is close to saturated for both models. The
first-sentence condition is where they actually separate, and **bge-m3
wins clearly** (e.g. RU→EN MRR 0.798 vs 0.706; EN→RU MRR 0.774 vs 0.650) —
consistent with bge-m3's training being explicitly optimized for
cross-lingual retrieval, whereas e5's multilingual training is comparatively
lighter-weight for this specific short-query cross-lingual case.

### 7b. Paraphrase/concept queries — skipped

Generating good short paraphrase queries cheaply (different wording, same
meaning as a paragraph) isn't feasible without an LLM to write them; noted
here rather than faked with a naive keyword-strip, which would not produce
a meaningfully different query from the original text.

### 7c. Query latency breakdown (embed time vs matrix search, persistent connection)

The ~150ms per-query latency reported in §5 was mostly artifact: each call
was opening a fresh `httpx.Client` (new TCP handshake) per request. Fixed in
`app/core/embedder.py` — `LlamaCppEmbedder` now keeps one persistent
keep-alive client for its whole lifetime. Real cost with a warm connection:

| Model | Embed (ms) | Matrix search (ms) |
|---|---|---|
| bge-m3 | 30.4 | 0.88 |
| multilingual-e5-base | 19.7 | 0.75 |

e5-base is faster per query (~10ms less), but both are well under any
noticeable threshold for interactive search; this difference does not
change the model recommendation.

### 7d. Full-corpus context-length risk (a practical tiebreaker)

While building the production `library/urantia/book.db`, **the
multilingual-e5-base build crashed** partway through embedding: llama-server
returned a context-overflow error on a paragraph in the full 14,585-paragraph
corpus that exceeds e5-base's native 512-token window (not present in the
1,572-paragraph benchmark subset, which happened not to include a paragraph
that long). bge-m3's native 8,192-token context has no such risk for this
corpus. A truncate-and-retry safeguard (`LlamaCppEmbedder._embed_batch`,
bisects an oversized batch and truncates via the model's own tokenizer)
was added afterward so this can no longer crash a build regardless of model,
but it remains a real quality risk for e5-base: any paragraph over 512
tokens gets truncated before embedding, silently losing content from the
vector.

## 8. Which numbers to trust, and the final decision

**Use §7 (objective, qrel-free), not §5 (queries.json-based, biased toward
FTS-matchable exact terms).** §7a shows bge-m3 generalizes better to the
harder, more realistic short-query cross-lingual case, and §7d shows
e5-base's 512-token context is a genuine full-corpus risk that bge-m3 (8192
tokens) doesn't share. The ~10ms latency difference (§7c) does not
outweigh either finding.

**Decision: bge-m3**, despite "losing" the biased §5 comparison.

## 9. Chosen defaults

- **Model: `bge-m3`**
- **Runtime: `llamacpp-vulkan`** (falls back automatically to ONNX/CPU if
  the Vulkan binary or GPU is unavailable — see `create_embedder()` in
  `app/core/embedder.py`)
- **Scheme: D** (separate EN/RU vectors, max cosine)
- **Search: hybrid**, RRF k=60, candidate pool=50 per side

Set in `app/config.py` (`DEFAULT_SETTINGS["embedding_model"]`,
`DEFAULT_SETTINGS["embedding_runtime"]`) and used to build the production
`library/urantia/book.db` (see below).

## 10. Correctness / safety notes

- `books.embedding_model` and `books.embedding_runtime` are stored in every
  `book.db`. `app.core.db.check_embedder_match()` compares a `Searcher`'s
  embedder against these at load time and logs a clear warning if they
  don't match — a query embedded with the wrong model/runtime lands in a
  different vector space and would otherwise silently return meaningless
  similarity scores.
- `LlamaCppEmbedder` registers an `atexit` cleanup and also terminates its
  `llama-server` subprocess if startup fails, so a crashed or unused server
  is never left running.
- `create_embedder(model, runtime="llamacpp-vulkan")` automatically falls
  back to the ONNX/CPU path if the Vulkan server can't start for any reason
  (missing binary, no GPU, port conflict, etc.) — logged as a warning, not a
  hard failure.
- `LlamaCppEmbedder._embed_batch` catches llama-server's 400/500 responses
  for oversized requests (either one text too long, or a batch's cumulative
  token count exceeding `-ub`), recursively bisects the batch to isolate the
  offending item(s), and truncates any single item still too long using the
  model's own tokenizer (`models/<name>/tokenizer.json`) before retrying —
  so one unusually long paragraph can no longer fail an entire indexing run
  (this is what crashed the first e5-base full-corpus build attempt; see
  §7d). `llama-server` is started with `-c 8192` (bge-m3's native context)
  so it never artificially caps below what the model actually supports.
