# FTS tuning on Paper 0 (Foreword)

> **Caveat: the relevance judgments (`bench/foreword/queries.json`) are a DRAFT, status "draft - needs user review".**
> They were judged by one reader (an AI agent) on 20 queries over 171 paragraphs, so differences of a few
> hundredths are noise. Treat results as a smoke test of the mechanism, not as a tuning conclusion for the whole book.

Setup: scratch DB `bench/foreword/paper0.db` (Paper 0 only, bge-m3 via llama.cpp Vulkan). Weights (bm25, positional): `{"en": 1.0, "ru": 1.0, "ru_lemma": 0.5, "title_en": 2.0, "title_ru": 2.0}`.
"Titles" = the new `title_en`/`title_ru` FTS columns (paper + section titles) both indexed and matched.
Reproduce: `.venv/Scripts/python bench/foreword/eval.py`.

"AND" (current default) requires every query word to match, which is why long natural-language queries score ~0 in FTS;
"OR-mode" (`fts_mode_*` settings) matches any word and lets bm25 rank. Column weights and titles
barely move the numbers on Paper 0 (short titles, few title-driven queries); the big effect is AND vs OR.

## Overall (mean over 20 queries)

| System | Recall@5 | Recall@10 | MRR@10 |
|---|---|---|---|
| FTS old (unweighted, no titles) | 0.079 | 0.092 | 0.150 |
| FTS weighted, no titles | 0.079 | 0.092 | 0.150 |
| FTS unweighted + titles | 0.087 | 0.100 | 0.167 |
| FTS weighted + titles | 0.087 | 0.100 | 0.167 |
| FTS OR-mode, unweighted, no titles | 0.522 | 0.685 | 0.872 |
| FTS OR-mode, weighted + titles | 0.499 | 0.695 | 0.854 |
| Semantic (bge-m3) | 0.698 | 0.784 | 0.967 |
| Hybrid RRF (old FTS) | 0.698 | 0.797 | 0.942 |
| Hybrid RRF (weighted + titles) | 0.698 | 0.797 | 0.942 |
| Hybrid RRF (OR-mode, weighted + titles) | 0.667 | 0.856 | 0.975 |
| NEW DEFAULT standalone FTS (and_fallback_or, weighted+titles) | 0.440 | 0.587 | 0.814 |
| NEW DEFAULT hybrid (OR FTS, weighted+titles) | 0.667 | 0.856 | 0.975 |

## By query language (Recall@10 / MRR@10)

| System | RU Recall@10 | EN Recall@10 | RU MRR | EN MRR |
|---|---|---|---|---|
| FTS old (unweighted, no titles) | 0.093 | 0.091 | 0.222 | 0.091 |
| FTS weighted, no titles | 0.093 | 0.091 | 0.222 | 0.091 |
| FTS unweighted + titles | 0.111 | 0.091 | 0.259 | 0.091 |
| FTS weighted + titles | 0.111 | 0.091 | 0.259 | 0.091 |
| FTS OR-mode, unweighted, no titles | 0.680 | 0.690 | 0.903 | 0.848 |
| FTS OR-mode, weighted + titles | 0.728 | 0.669 | 0.870 | 0.841 |
| Semantic (bge-m3) | 0.783 | 0.785 | 0.926 | 1.000 |
| Hybrid RRF (old FTS) | 0.783 | 0.808 | 0.926 | 0.955 |
| Hybrid RRF (weighted + titles) | 0.783 | 0.808 | 0.926 | 0.955 |
| Hybrid RRF (OR-mode, weighted + titles) | 0.824 | 0.883 | 1.000 | 0.955 |
| NEW DEFAULT standalone FTS (and_fallback_or, weighted+titles) | 0.543 | 0.623 | 0.796 | 0.828 |
| NEW DEFAULT hybrid (OR FTS, weighted+titles) | 0.824 | 0.883 | 1.000 | 0.955 |

## Per-query MRR@10

| Query | FTS old | FTS weighted, no titles | FTS unweighted + titles | FTS weighted + titles | FTS OR-mode, unweighted, no titles | FTS OR-mode, weighted + titles | Semantic | Hybrid RRF | Hybrid RRF | Hybrid RRF | NEW DEFAULT standalone FTS | NEW DEFAULT hybrid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| q01 What is the difference between Deity and | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | 0.11 | 1.00 | 1.00 | 1.00 | 1.00 | 0.11 | 1.00 |
| q02 Что такое божественность? | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | 0.33 | 1.00 | 1.00 | 1.00 | 1.00 | 0.33 | 1.00 |
| q03 the seven levels on which Deity function | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| q04 Как следует понимать слово «Бог» в этих  | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| q05 Paradise Trinity | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| q06 союз Отца, Сына и Духа как единое божест | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.33 | 0.33 | 0.33 | 1.00 | 1.00 | 1.00 |
| q07 Настройщик Мышления, божественный дух в  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| q08 how does the immortal human soul come in | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| q09 Чем личность отличается от тела, разума  | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| q10 Who is the First Source and Center and h | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| q11 everything that exists traces back to on | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| q12 Why is the central Isle motionless and w | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| q13 Три Абсолюта: Божество-Абсолют, Безуслов | 0.00 | 0.00 | 0.33 | 0.33 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.33 | 1.00 |
| q14 difference between force, energy and pow | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| q15 Что такое эталон и чем он отличается от  | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 |
| q16 finite, absonite and absolute levels of  | 0.00 | 0.00 | 0.00 | 0.00 | 0.20 | 0.14 | 1.00 | 0.50 | 0.50 | 0.50 | 0.00 | 0.50 |
| q17 Верховное Существо - эволюционирующий Бо | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| q18 how mortal creatures approach God in sev | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| q19 Понятие Я ЕСТЬ как философская уступка к | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| q20 purpose of this Foreword and why English | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

Notes: queries have 2-8 relevant paragraphs (list-style paragraphs inflate counts); Foreword section titles are
short, so title weight mainly helps queries naming a section ("The Trinities", "God the Sevenfold").

## Update: new defaults (approved)

Settings: `fts_mode_hybrid` = "or" (hybrid FTS leg is OR, bm25-weighted, titles on), `fts_mode_standalone` =
"and_fallback_or" (AND; OR only if AND returns nothing). Quoted phrases stay exact in all modes.
`last_timings["fts_mode_used"]` reports "and"/"or". The two rows "NEW DEFAULT ..." in the table above are these defaults:
hybrid R@10 0.856 / MRR 0.975 (vs 0.797 / 0.942 before); standalone FTS R@10 0.587 / MRR 0.814 (vs 0.100 / 0.167).
Standalone stays below pure OR because AND returns a few weak hits for some queries, so the fallback never fires.

## Update: stop-word filtering (unquoted terms, AND and OR)

Compact RU (~150, incl. lemma forms via pymorphy) and EN stop lists; quoted phrases untouched; if every term is a
stop word the original terms are used; stop words are not matched, so they are never highlighted.
Test: 20 eval queries + 6 question-form queries (`bench/foreword/eval_stop.py`), Paper 0 only.

In plain words (share of needed paragraphs found in the top 10 / average position of the first good one):
- Standalone FTS: before 73% / 1.81, after 71% / 1.46. The first good hit moves up, total coverage dips slightly.
- Hybrid: before 88% / 1.08, after 82% / 1.08. First hit unchanged, coverage down about 6 points.
The small loss on Paper 0 comes from enumerated-list paragraphs that matched via common words; with 171 paragraphs and
a vocabulary dominated by "God/Deity", such incidental hits happen to be relevant. On the full book, where common words
match thousands of paragraphs, the effect is expected to be the opposite (the reported "кто такие агондонтеры" case
now matches only the content word). This is not verifiable on Paper 0 (no such term), and qrels are still a draft.

## Update: `fts_stopwords` setting and list audit

Setting `fts_stopwords` (default true) switches filtering on/off (`Searcher(fts_stopwords=...)` overrides).
Audit of hybrid queries that lost needed paragraphs: 4 of 26 queries lost 1-2 paragraphs each. Most losses were incidental
(a paragraph that matched only on "это", "в", "what", "does"). One was a real content-word problem: the key term
"Я ЕСТЬ" was removed as pronoun + "to be". Removed "я", "есть", "быть", "суть" from the RU list. After the fix hybrid coverage is
84% (was 88% without filtering, 82% before the fix); standalone 72% (73%), first good hit still earlier (1.46 vs 1.81).
