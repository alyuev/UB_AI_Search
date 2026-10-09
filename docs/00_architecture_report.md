# Архитектурный отчёт (п. 49 ТЗ)

Основано на: `01_data_analysis.md`, `02_architecture_research.md`. [И] = измерено, [О] = оценка.

## Исходные данные
14 585 абзацев, 197 Paper, 1 626 разделов; 0 битых/пустых записей [И]. ~6,5 млн символов на язык.
Медиана абзаца: 83 ток. EN / 193 ток. RU (cl100k) [И]. RU >512 ток.: 351 абзац; EN+RU вместе >512: 1 645 [И].

## 1. Embedding-модель
Кандидаты на финальный benchmark (Этап 6/14, реальные RU↔EN запросы):
| Модель | dim | контекст | размер ONNX | роль |
|---|---|---|---|---|
| multilingual-e5-small | 384 | 512 | ~120 MB (int8) | быстрый вариант по умолчанию |
| multilingual-e5-base | 768 | 512 | ~280 MB (int8) | компромисс |
| BGE-M3 | 1024 | 8192 | ~570 MB (int8) | максимум качества, без обрезки длинных абзацев |

Рекомендация: **выбирать по benchmark**; ожидаемый фаворит по качеству — BGE-M3 (сильный cross-lingual, нет проблемы 512 токенов), по скорости/размеру — e5-small. Все MIT, все работают через ONNX.
jina-v3 исключена (некоммерческая лицензия).

## 2. Runtime
**onnxruntime** (CPU) + опционально **onnxruntime-directml** (AMD Radeon 890M) [И: устанавливается, DmlExecutionProvider доступен]. Без torch (экономия ~2 GB). Токенизатор — `tokenizers` (HF). LM Studio для embeddings не нужен; его embedding-модели — лишь опциональный альтернативный backend.

## 3–4. БД и хранение векторов
Один файл **`book.db` (SQLite)**: таблицы `books`, `paragraphs`, `chunks`, `embeddings` (BLOB float32/int8, привязка к модели), FTS5.
Поиск: при старте векторы грузятся в numpy-матрицу, brute-force cosine — **~1 мс на запрос** для 40k×384 [И]. ANN не нужен.
FAISS/Chroma/LanceDB/Qdrant — избыточны (лишние зависимости/сервер/файлы). sqlite-vec загружается [И], держим как опцию.

## 5. FTS5
Встроен в SQLite 3.50.4 [И]. Tokenizer `unicode61 remove_diacritics 2` [И: кириллица ок]. Две колонки `en`, `ru` → поиск по языку или обоим, фразы, префиксы, bm25.
RU-морфология: дополнительная колонка `ru_lemma` (pymorphy3) — запрос тоже лемматизируется. Опционально trigram для подстрок.

## 6. Hybrid search
**RRF** (k=60) по спискам FTS (top 50) и semantic (top 50); параметры подтвердит benchmark (20–30 запросов, метрики Recall@10 / MRR) против FTS и semantic отдельно.

## 7. Chunking
**1 абзац = 1 chunk** (v1). Соседние абзацы берутся по сквозному номеру `seq`. Вариант «окно из 3 соседних» — проверить в benchmark; включать, только если даст прирост.
Схема индексации текста: эксперимент A/B/C/D; ожидаемо лучшая — **D (отдельные EN и RU векторы, score = max)**; C отпадает для 512-токенных моделей (11% обрезки).

## 8. LM Studio
OpenAI-совместимый `/v1/models`, `/v1/chat/completions` (stream SSE) + нативный `/api/v0/models` (state, max_context_length) [И: оба отвечают, 25 моделей]. URL, модель, temperature, max_tokens, timeout — в настройках; модель не зашивается. Размер батча анализа перевода вычисляется от контекста модели.

## 9. Backend
Python 3.14 + **FastAPI/uvicorn**, модули: `parser`, `db`, `indexer`, `embedder`, `search` (fts/semantic/hybrid), `llm` (LM Studio клиент), `rag`, `translation`, `jobs` (фоновые потоки, прогресс в памяти + SQLite), логи `logs/app|index|analysis.log`.
API: `/api/search`, `/api/paragraph/{id}` (+context), `/api/jobs/index|rag|translation-analysis`, `/api/jobs/{id}` (+SSE), `/api/settings/*`.
Анализ перевода: парсер ссылок («Paper 147 Section 2») → прямая выборка из БД, не semantic.

## 10. Frontend
Статический HTML + vanilla JS (без сборки): вкладки Поиск / Вопрос / Анализ перевода / Подготовка базы / Настройки (LLM, Embeddings). Реальный прогресс через SSE/поллинг jobs.

## 11. Packaging
Embeddable Python + предустановленные пакеты + `start.bat` (открывает браузер). Режимы: User (app + book.db + модель) и Builder (+ исходники JSON). Структура `library/<book_id>/book.db`.

## 12. Размер дистрибутива [О]
Python+пакеты ~120 MB; модель 120–570 MB; book.db ~40 MB текст+FTS + векторы 22–120 MB. **Итого ~250–850 MB** в зависимости от модели (zip меньше).

## 13. Время индексации
e5-small/CPU: ~300 предл./с [И на MiniLM] → ~1–2 мин на 29k текстов (EN+RU) [О]. BGE-M3/CPU: ~10–20 мин [О]; с DirectML быстрее.

## 14–15. RAM, CPU/GPU
Поиск: 300 MB–1,5 GB RAM (зависит от модели). CPU достаточно; GPU — только ускорение индексации. LLM — отдельно в LM Studio (для gemma-4-26b ~16+ GB).

## 16. Offline
Полностью: после установки нет обращений в сеть (модели лежат локально, HF offline mode). Без LM Studio работают FTS/semantic/hybrid; RAG и анализ перевода сообщают о недоступности LLM.

## Открытые вопросы к согласованию
1. Модель по умолчанию: решить по benchmark (e5-small vs e5-base vs BGE-M3) — ок?
2. FastAPI + vanilla JS — ок?
3. Добавлять ли pymorphy3 (лемматизация RU в FTS, +~15 MB)?
