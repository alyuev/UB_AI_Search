# Stage 1 Data Analysis — Urantia Bilingual RAG Corpus

Source files (D:\Develop\KU\SmartSearch):
- `urantia_bilingual_rag.json` (~21.9 MB)
- `urantia_bilingual_rag.md` (~19.5 MB, human-readable mirror)

Analysis performed with standalone Python scripts (Python 3.14.3, tiktoken 0.14.0, `cl100k_base` encoding) reading the files directly — never loaded into the assistant's context.

## 1. Structure

JSON top level is a flat **array of 14,585 objects**. Each record has these fields:

| field | type | notes |
|---|---|---|
| `id` | str | e.g. `"0:0.1"` → `Paper:Section.Paragraph` |
| `paper` | int | matches the `id` string in every record |
| `section` | int | matches the `id` string in every record |
| `paragraph` | int | matches the `id` string in every record |
| `en` | str | English paragraph text |
| `ru` | str | Russian paragraph text |
| `meta` | dict | `en_words`, `en_chars`, `ru_words`, `ru_chars` — precomputed counts, present on every record and consistent with actual string lengths |

No other/extra top-level fields exist beyond `meta` (no titles, no section/paper names, no source citations).

The `.md` file is a plain-text mirror: a one-line header (`Paragraphs: 14585`), then repeating blocks of `ID:`, `EN:`, `RU:`, separated by `---`.

## 2. Counts

- Total records (paragraphs): **14,585**
- Unique papers: **197** (paper numbers 0–196; paper 0 is the Foreword)
- Unique (paper, section) pairs: **1,626**
- Unique (paper, section, paragraph) triples: **14,585** (equals record count — no collisions)

## 3. Data quality

- **Missing fields:** none. Every record has all 6 top-level fields.
- **Null/empty text:** 0 null or empty `en`, 0 null or empty `ru` values across all 14,585 records.
- **Malformed IDs:** 0. Every `id` matches `^\d+:\d+\.\d+$` and is internally consistent with the `paper`/`section`/`paragraph` integer fields (0 mismatches).
- **Duplicate IDs:** 0 duplicate `id` strings.
- **Ordering:** records are in strictly increasing (paper, section, paragraph) order throughout the file (0 out-of-order transitions); within every section the paragraph numbers also appear in ascending order.
- **Paragraph-number gaps:** 3 sections have non-contiguous paragraph numbers (i.e., min..max range doesn't match the count present), meaning a paragraph number was skipped in the source numbering (not necessarily a data error — Urantia Book paragraph numbering itself has known skips):
  - Paper 0, Section 12: paragraphs 1–14 present but **10** missing (13 of 14 present)
  - Paper 134, Section 6: paragraphs 1–17 present but **14** missing (16 of 17 present)
  - Paper 144, Section 5: paragraphs 1–108 present but **11, 25, 38, 54, 73, 87** missing (102 of 108 present)

  These are very likely gaps in the original source numbering (e.g., merged/renumbered paragraphs in some print editions) rather than parser bugs, but worth spot-checking against a reference edition before relying on paragraph continuity.

## 4. Character counts

| | total chars | min | median | p95 | max |
|---|---|---|---|---|---|
| EN | 6,546,966 | 10 | 418 | 950 | 2,879 |
| RU | 6,479,573 | 10 | 416 | 938 | 2,474 |

## 5. Token estimates

Computed with **tiktoken `cl100k_base`** (installed via pip; exact byte-pair-encoding count, not a heuristic).

| | total tokens | min | median | p95 | max |
|---|---|---|---|---|---|
| EN | 1,317,560 | 3 | 83 | 194 | 620 |
| RU | 3,017,649 | 6 | 193 | 436 | 1,197 |
| EN+RU combined | 4,335,209 | 10 | 277 | 625 | 1,817 |

Note the large EN/RU token asymmetry: cl100k_base was trained mostly on English/Latin-script text, so Cyrillic text tokenizes far less efficiently (roughly 2.3x more tokens than English for similar content). If the embedding model has its own tokenizer (e.g. a multilingual BERT/XLM-R-style model), real token counts for Russian will likely be lower than this cl100k-based estimate — treat the RU numbers here as an upper-bound sanity check, not the ground truth for a specific embedding model.

### Records exceeding 512 tokens (embedding-model limit check)

- EN text alone > 512 tokens: **3 records**
- RU text alone > 512 tokens: **351 records**
- Either EN or RU > 512 tokens: **351 records** (2.4% of all paragraphs)
- Combined EN+RU (if concatenated into one input) > 512 tokens: **1,645 records** (11.3% of all paragraphs)

Implication: if paragraphs are embedded separately per language, ~2.4% will need truncation or chunking for a 512-token model, dominated by the Russian side. If EN+RU are ever concatenated into a single embedding input, over 11% would exceed 512 tokens — chunking/splitting strategy should be planned for Stage 2.

## 6. JSON vs. MD consistency

- MD file header declares `Paragraphs: 14585`.
- Actual count of `^ID: ` lines in the MD file: **14,585**.
- Matches the JSON record count exactly. No discrepancy found.

## 7. Vector index size estimates

Based on 14,585 paragraphs, computed for common embedding dimensions/dtypes and 1 vs. 2 vectors stored per paragraph (e.g., separate EN/RU vectors, or vector + reranking vector):

| dim | dtype | vectors/paragraph | raw vector data size |
|---|---|---|---|
| 1024 | float32 | 1 | 59.74 MB |
| 1024 | float32 | 2 | 119.48 MB |
| 1024 | int8 | 1 | 14.94 MB |
| 1024 | int8 | 2 | 29.87 MB |
| 768 | float32 | 1 | 44.81 MB |
| 768 | float32 | 2 | 89.61 MB |
| 768 | int8 | 1 | 11.20 MB |
| 768 | int8 | 2 | 22.40 MB |

These are raw embedding payload sizes only (no index overhead such as HNSW graph links, IDs, metadata, or on-disk storage/replication factors — actual index footprint, e.g. FAISS HNSW or a vector DB, will typically run 1.3–2x higher due to graph/index structures). In all cases the corpus is small enough that even the largest configuration (1024-dim float32, 2 vectors/paragraph, ~120 MB) comfortably fits in memory on any modern machine.

## Summary of key numbers

- 14,585 paragraphs, 197 papers, 1,626 sections — fully well-formed (no nulls, no duplicate/malformed IDs, JSON/MD counts match exactly).
- Only 3 sections have paragraph-number gaps (6 total missing paragraph numbers) likely from source-edition numbering, not corruption.
- EN: 6.55M chars / 1.32M tokens (cl100k). RU: 6.48M chars / 3.02M tokens (cl100k) — RU tokenizes ~2.3x heavier under this English-oriented tokenizer.
- 351 records (2.4%) exceed 512 tokens on at least one language side (almost all driven by RU); 1,645 (11.3%) would exceed 512 tokens if EN+RU were concatenated into one input.
- Vector index payload is tiny regardless of config: 11–120 MB across all tested dim/dtype/vector-count combinations — storage is a non-issue at this corpus size.
