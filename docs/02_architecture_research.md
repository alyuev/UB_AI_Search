# Stage 2 — Architecture Research (fully local Windows app)

Date: 2026-09-26. Machine used for measurements (MEASURED via PowerShell `Get-CimInstance`):
- CPU: AMD Ryzen AI 9 HX 370 (12 cores / 24 logical threads)
- GPU: AMD Radeon 890M iGPU (integrated, ~4GB shared VRAM reported), no discrete NVIDIA — confirms AMD path is required (DirectML/Vulkan, not CUDA).
- RAM: 48 GB (51,129,462,784 bytes)
- Python installed: 3.14.3 (default `python`), 3.13 also present via `py -3.13`, 3.8 also present. **No 3.12 installed.**
- LM Studio: reachable at `http://localhost:1234`, has an embedding model text-embedding-multilingual-e5-large/small, qwen3-embedding-0.6b, nomic-embed already downloaded, plus several chat/VLM models (all `state: not-loaded` — left untouched per instructions, no load/unload performed).

All numbers below are labeled **MEASURED** (verified hands-on in the scratchpad venv) or **ESTIMATE** (from vendor docs/model cards/web research, not run locally).

---

## 1. Embedding model choice

| Model | Params | Dim | Max tokens | License | ONNX | Notes |
|---|---|---|---|---|---|---|
| multilingual-e5-small | 118M | 384 | 512 | MIT | yes | good RU/EN, fast, small |
| multilingual-e5-base | 278M | 768 | 512 | MIT | yes | better recall, 2x cost |
| multilingual-e5-large | 560M | 1024 | 512 | MIT | yes | best of the E5 family, ~1.1GB fp32 / ~300MB int8 |
| paraphrase-multilingual-MiniLM-L12-v2 | 118M | 384 | 512 (128 typical) | Apache-2.0 | yes | older but solid RU/EN paraphrase model, MEASURED below |
| BGE-M3 | 568M | 1024 (+sparse+colbert) | 8192 | MIT | yes (dense-only export) | strongest multilingual hybrid retrieval, heavier; long-context is overkill for paragraph-level search |
| LaBSE | 471M | 768 | 512 (256 typical) | Apache-2.0 | yes | strong bitext/cross-lingual alignment, older, less MTEB-tuned |
| Qwen3-Embedding-0.6B | 600M | 1024 (MRL down to 32) | 32k | Apache-2.0 | yes (already in LM Studio as GGUF) | #1-ish on multilingual MTEB among small models, but needs instruction-prefixed queries and a heavier runtime (transformer decoder, not BERT-tiny) |
| jina-embeddings-v3 | 572M | 1024 (Matryoshka) | 8192 | CC-BY-NC 4.0 | yes | license blocks a shipped commercial-ish redistributable; fine for personal use only |
| EmbeddingGemma-300m | 300M | 768 (MRL) | 2048 | Gemma license | yes | good multilingual, decent size, newer (2026) |
| USER-bge-m3 | 359M | 1024 | 8192 | MIT | via BGE-M3 export | Russian-tuned distilled BGE-M3, strong ruMTEB, smaller than full BGE-M3 |

**Recommendation:** ship **multilingual-e5-small** (or -base if disk/RAM budget allows) as the bundled default, quantized ONNX (int8), run via `fastembed`/`onnxruntime`. Rationale:
- Book is short paragraphs (not long documents) → 512-token context is enough; no need for BGE-M3's 8k window.
- MIT license, small download (~130MB int8 e5-small vs 300MB+ for larger models), CPU-friendly.
- E5/USER-bge-m3 score well on ruMTEB specifically (per arXiv:2408.12503, "Russian-focused embedders" survey) — plain E5 multilingual is a known solid RU baseline; USER-bge-m3 is worth a second look if RU recall quality turns out inadequate in Stage 3 testing.
- Keep BGE-M3 or Qwen3-Embedding-0.6B (both already present as GGUF in LM Studio) as an optional "high quality" mode toggle for power users, loaded via llama.cpp/GGUF rather than duplicating in ONNX.
- Avoid jina-v3 (non-commercial license) and avoid depending on LM Studio for embeddings (per requirement) — LM Studio embedding models are for *interactive* use in the app's chat features, not for the offline bulk embedding pass.

## 2. Runtime choice

**MEASURED** in scratchpad (`venv314`, Python 3.14.3 embedded-equivalent):
- `pip install onnxruntime fastembed` — installs cleanly, no build-from-source, no Rust/C++ toolchain needed. Wheels exist for Python 3.14 on win_amd64 already.
- `onnxruntime.get_available_providers()` → `['AzureExecutionProvider','CPUExecutionProvider']` by default.
- `pip install onnxruntime-directml` → adds `DmlExecutionProvider`, confirmed available for AMD iGPU (Radeon 890M) on this Python 3.14 build. (Only provider *availability* was verified; a full embedding pass through DML was not run — GPU speed is an ESTIMATE below.)
- Ran `fastembed.TextEmbedding("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")` (auto-downloads quantized ONNX, ~110MB): embedding 300 mixed RU/EN short sentences (batch_size=32) took **0.98s → 305.6 sentences/sec on CPU** (single process, default thread count on the 24-thread Ryzen). Load time 13.9s dominated by first-time HF download, not inference.
- Extrapolated: 15–20k paragraphs would embed in **~50–65s on CPU alone** — fast enough that GPU/DirectML acceleration is a nice-to-have, not a requirement, for the one-time indexing pass.

**Recommendation:** `onnxruntime` + `fastembed` (or a thin custom wrapper around `onnxruntime` directly, skipping fastembed's HF-Hub auto-download in favor of bundling the exact ONNX+tokenizer files in the zip). This is the smallest, most portable option:
- No PyTorch dependency (sentence-transformers pulls in torch, which is 500MB–2GB+ install and much slower cold-start) — reject sentence-transformers/torch for the shipped app; fine only as a dev-time tool for exporting/quantizing models.
- No llama.cpp/GGUF requirement for embeddings (keeps the embedding path independent of the LM Studio/llama.cpp stack, satisfying "must not depend on LM Studio").
- DirectML gives an optional speed path for AMD GPUs without needing ROCm (which barely supports Windows/consumer AMD anyway) or Vulkan.

## 3. Vector storage

**MEASURED:** numpy brute-force cosine search over a synthetic 40,000 × 384 float32 matrix (matching the book's paragraph×dim scale): mean **1.04 ms**, max 1.77 ms per query (`mat @ query` + `argpartition` top-10), on the Ryzen AI 9 HX 370 CPU, single-threaded numpy call. At 15–20k real paragraphs this will be well under 1ms.

**Recommendation:** Do **not** add FAISS/Chroma/LanceDB/Qdrant — massive overkill and extra install size (FAISS wheel alone is 10s of MB plus MKL/OpenBLAS; Chroma/LanceDB pull in additional Rust binaries and dependency trees) for a dataset this small. Store vectors as either:
- a `.npy`/`.npz` file loaded once into RAM at startup (fastest, simplest), with paragraph IDs as a parallel index, or
- BLOB columns in the same SQLite DB (keeps everything in one portable file), reconstructed into a numpy matrix at startup.
Either way, brute-force numpy matmul is the answer; a proper ANN index (even sqlite-vec) is unnecessary at 15–40k vectors.

**sqlite-vec (MEASURED):** `pip install sqlite-vec` installed cleanly and **loaded successfully** via `conn.enable_load_extension(True); sqlite_vec.load(conn)` on the stock Python 3.14.3 sqlite3 module — `vec_version()` returned `v0.1.9`. So it is a viable option if the team prefers SQL-native vector queries (`vec0` virtual tables) over manual numpy management, at negligible cost (small extension DLL, no extra Python-level vector-math code). Given the brute-force numbers above, this is a matter of code-ergonomics preference, not a performance requirement.

**SQLite/FTS5 (MEASURED):** Python 3.14.3's bundled `sqlite3` module reports SQLite **3.50.4** and successfully creates an `fts5` virtual table — FTS5 is compiled in, no extra extension needed.
- `tokenize='unicode61 remove_diacritics 2'` was tested and correctly tokenizes/matches Cyrillic text case-insensitively (confirmed a Russian-language MATCH query returned the expected row; console mangled the Cyrillic *display* due to a terminal codepage issue, but the match itself succeeded, proving tokenization works).
- SQLite has no built-in Snowball/Russian stemmer. Options: (a) rely on `unicode61` for exact/prefix Cyrillic matching (recommended baseline — the semantic/hybrid layer covers fuzzy recall), (b) add the `fts5 trigram` tokenizer (also confirmed to instantiate) for substring/fuzzy matching, or (c) pre-stem Russian text at index time with `pymorphy3`/`pymorphy2` (pure-Python, no native deps) before feeding FTS5, which most closely emulates what Snowball gives other languages. Recommend (a)+(c): store both the raw and a pymorphy3-lemmatized column, FTS5-index both, so search can match on lemma forms.

## 4. LM Studio API

**MEASURED** (server was already running on `http://localhost:1234`, no models loaded/unloaded per instructions):
- `GET /v1/models` — OpenAI-compatible endpoint, returns `{data:[{id, object:"model", owned_by}], object:"list"}`, minimal fields only. 25 models registered (LLMs, VLMs, and embedding models), all reported implicitly as installed (this endpoint doesn't include load state).
- `GET /api/v0/models` — LM Studio's native REST API, richer: adds `type` (llm/vlm/embeddings), `publisher`, `arch`, `compatibility_type` (gguf), `quantization`, **`state`** (`not-loaded`/`loaded`), `max_context_length`, and `capabilities` (e.g. `tool_use`). This is the endpoint to use for a Settings/model-picker UI, since it shows context length and load state per model without needing to load anything.
- Confirmed embedding models already present in this LM Studio instance: `text-embedding-multilingual-e5-large`, `text-embedding-multilingual-e5-small`, `text-embedding-qwen3-embedding-0.6b`, `text-embedding-nomic-embed-text-v1.5` (two quants) — useful as an *optional* alternate embedding backend the user can pick in Settings, but the app's default/bundled embedding pipeline must remain the local ONNX one (per requirement) so first-run search works even if LM Studio isn't running.
- Did **not** perform a chat completion, since LM Studio JIT-loads a model on first request when "just-in-time model loading" is enabled — that would count as loading a model, which the task explicitly said not to do. Documenting API shape instead from LM Studio's published OpenAI-compatible spec: `POST /v1/chat/completions` with `{model, messages, max_tokens, temperature, stream}`; `stream:true` gives SSE chunks in the OpenAI format; no built-in request timeout is enforced server-side — the client must set its own HTTP timeout (recommend a generous one, e.g. 120s+, since local CPU/iGPU token generation is much slower than a cloud API, especially on longer contexts with an iGPU). `POST /v1/embeddings` follows the same OpenAI-compatible shape (`{model, input}` → `{data:[{embedding}]}`).
- Practical implication for RAG/translation-QA features: use `/api/v0/models` at app startup/settings-open to populate a model dropdown filtered by `type` and to show `max_context_length`; use `/v1/chat/completions` (OpenAI-compatible) for actual inference calls so the code is portable to any OpenAI-compatible local server, not LM Studio-specific.

## 5. Windows packaging

- **Embedded Python distribution** (`python-3.x-embed-amd64.zip` from python.org) — recommended. ~15–25MB base (ESTIMATE, per python.org embeddable package sizes), plus `pip`-installed deps (`onnxruntime` ~15MB wheel, `fastembed`/tokenizers ~10MB, `numpy` ~15MB, model files ~50–300MB depending on chosen embedding model) → **total portable zip likely 150–400MB** (ESTIMATE), dominated by the embedding model weights and onnxruntime binaries, not Python itself. Works well with a `start.bat` that calls the embedded `python.exe` directly; no installation, no registry writes, no admin rights — matches the "portable zip" requirement exactly.
- **PyInstaller** (bundle app + interpreter into one exe/folder) — rejected as primary approach: harder to debug, slower startup (unpacking to temp or larger single-file overhead), historically flaky with onnxruntime's native DLLs and dynamic model loading via `huggingface_hub`/`fastembed`'s runtime file access patterns, and produces a similarly-sized or larger artifact (ESTIMATE) with less transparency for the "local, inspectable" portable-app goal. An embedded-Python + plain `.py` app + `start.bat` is simpler, more debuggable, and just as portable.
- **Python version:** stick with **3.14.3** (already installed and default on this machine) rather than downgrading to 3.12. All target packages (`onnxruntime`, `onnxruntime-directml`, `fastembed`, `sqlite-vec`, `numpy`) were MEASURED to install and import successfully via prebuilt wheels on 3.14 — no wheel-availability problem was found in practice, so there's no benefit to the extra complexity of sourcing/bundling a 3.12 embeddable distribution. (Python 3.13 is also present locally as a fallback if any future dependency lags on 3.14 wheel support.)

## Summary of concrete recommendations

1. Embeddings: `multilingual-e5-small` (upgrade path to -base or USER-bge-m3) as quantized ONNX, run via `onnxruntime`(+`onnxruntime-directml` optional)/`fastembed`, fully independent of LM Studio.
2. Storage: SQLite single-file DB — FTS5 (`unicode61` + optional pymorphy3 lemma column, trigram as fallback) for lexical search, BLOB-stored or `.npy`-cached vectors + brute-force numpy cosine (measured ~1ms/query at this scale) for semantic search, combined via RRF for hybrid. sqlite-vec is a viable optional convenience layer, not a necessity.
3. LM Studio integration: use `/api/v0/models` for capability discovery, OpenAI-compatible `/v1/chat/completions` and `/v1/embeddings` for actual calls (portable to any compatible server), generous client-side timeouts, streaming supported.
4. Packaging: embedded Python 3.14 distribution + `start.bat`, plain-`.py` app (not PyInstaller), estimated 150–400MB portable zip.
