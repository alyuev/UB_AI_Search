"""RAG pipeline: query preparation (LLM search plan) -> retrieval (FTS on keywords + semantic on
the plan sentence and on the original question, merged by RRF) -> context building ->
generation. Each stage is reported through an optional progress callback.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
import time
from typing import Callable, Optional

from app import options as opts_mod
from app import prompts as prompts_mod
from app.llm.lmstudio import LMStudioClient, LMStudioError, llm_title
from app.log import get_logger
from app.version import __version__

logger_rag = get_logger("app")
_LIST_ITEM_RE = re.compile(r"\s*(\d{1,3})[.)]\s")   # "1. Onagar…", "12) …"

# A question that wants a full enumeration (backup for the plan's answer_type).
_LIST_QUESTION_RE = re.compile(
    r"перечисл|перечен|список|назови\s+вс|кто\s+вход|кто\s+был[иа]?\s+член|кто\s+состо|"
    r"из\s+(?:чего|кого)\s+состо|какие\s+(?:бывают|есть|существуют)|все\s+виды|"
    r"\blist\b|\benumerate|name\s+all|who\s+(?:are|were)\s+the\s+members|what\s+are\s+all",
    re.IGNORECASE)

# Words never counted when matching a question against section titles.
_TITLE_STOP = {
    "и", "в", "во", "на", "о", "об", "по", "с", "со", "к", "от", "для", "из", "у", "за", "до", "the", "of",
    "and", "in", "on", "to", "a", "an", "for", "with", "by", "at", "from", "his", "their",
}


def is_list_question(question: str, plan: Optional[dict]) -> bool:
    return bool((plan and plan.get("answer_type") == "list") or _LIST_QUESTION_RE.search(question or ""))


def _stems(text: str) -> set[str]:
    """Crude stems for title matching: lower case, ё->е, first 5 letters (enough for Russian endings)."""
    out = set()
    for w in re.findall(r"[0-9A-Za-zА-Яа-яЁё]+", (text or "").lower().replace("ё", "е")):
        if w in _TITLE_STOP or len(w) < 3:
            continue
        out.add(w[:5])
    return out


def match_section_titles(conn, question: str, plan: Optional[dict], limit: int = 2) -> list[dict]:
    """Sections whose title (RU or EN) is covered by the question and the plan's keywords.
    A match needs ALL content words of the title and at least two of them (a one-word title like «Адам»
    would match far too often). Best (longest) titles first."""
    words = _stems(question)
    if plan:
        for t in plan.get("keywords_ru", []) + plan.get("keywords_en", []):
            words |= _stems(t)
    if not words or conn is None:
        return []
    found = []
    try:
        rows = conn.execute("SELECT paper, section, en, ru FROM sections WHERE section > 0").fetchall()
    except Exception:  # noqa: BLE001 - a bonus, never fatal
        return []
    for paper, section, en, ru in rows:
        best = 0
        for title in (ru, en):
            st = _stems(title)
            if len(st) >= 2 and st <= words:
                best = max(best, len(st))
        if best:
            found.append({"paper": paper, "section": section, "title": ru or en, "words": best})
    found.sort(key=lambda s: -s["words"])
    return found[:limit]

SYSTEM_PROMPT = (
    "You are a careful research assistant answering questions about a bilingual (EN/RU) book "
    "STRICTLY from the numbered excerpts provided below. Ты отвечаешь на вопрос СТРОГО по "
    "приведённым ниже абзацам книги (EN/RU), не используя никаких других знаний.\n\n"
    "RULES / ПРАВИЛА:\n"
    "1. Every excerpt has an id in square brackets, like [0:1.2]. After EVERY sentence that states a "
    "fact, put the id(s) of the excerpt(s) it comes from, each in its own brackets: [0:1.2][0:2.10]. "
    "A sentence you cannot support with an id must not be written. "
    "После КАЖДОГО предложения с утверждением ставь номера абзацев-источников, каждый в своих "
    "скобках: [0:1.2][0:2.10]. Предложение без источника писать нельзя.\n"
    "2. Cite ONLY ids that appear in the excerpts, copied exactly. Never invent an id, a quote or a "
    "fact. Цитируй только номера из списка, точно как написано. Не выдумывай номера, цитаты, факты.\n"
    "3. Use only what the excerpts say. No outside knowledge, no guessing, no general introductions "
    "such as 'according to the sources'. Только то, что сказано в абзацах; без домыслов и вводных "
    "фраз вроде «на основании источников».\n"
    "4. If the excerpts answer only PART of the question, answer that part and say plainly what is "
    "missing (e.g. 'В найденных абзацах нет сведений о ...'). If they do not contain the answer at all, "
    "reply only: «В найденных абзацах нет достаточных данных для ответа на этот вопрос.» (in the "
    "question's language) and cite nothing. Если данных мало или нет — скажи об этом прямо.\n"
    "5. Answer in the language of the question (Russian question -> Russian answer, English -> "
    "English). Be concise and well structured (short paragraphs or a list). Отвечай на языке вопроса, "
    "кратко и по делу.\n"
)

QUERY_REWRITE_PROMPT = (
    "You turn a user's question about a book into a search plan. The book exists in two languages "
    "(English and Russian) and is searched in both. Ты превращаешь вопрос пользователя в план "
    "поиска по книге (текст есть на английском и русском).\n"
    "Output ONLY a JSON object, no prose: "
    '{"keywords_ru": [...], "keywords_en": [...], "semantic_query": "...", "answer_type": "..."}\n'
    "- keywords_ru: 1-{max_terms} key terms or names in Russian, dictionary (nominative, singular) form, "
    "WITHOUT question words (что, кто, как, почему, зачем, в чём, различие...) and without filler.\n"
    "- keywords_en: the English equivalents of the same key terms as they would appear in the English "
    "text of the book (translate terms, transliterate names).\n"
    "- The user message may end with «Слова из вопроса и их соответствия в книге: …». These hints were "
    "checked against the real text of the book: PREFER exactly those spellings (Russian and English) for the "
    "words they cover. Подсказки проверены по тексту книги — бери написание из них.\n"
    "- NEVER replace a word of the question with another word and never translate it into a DIFFERENT real "
    "word (a name like «эдем» is not «отёк»/edema). If a word is unknown or unclear, keep it AS THE USER "
    "WROTE IT (in English: transliterate it letter by letter). Never invent spellings, forms or related "
    "words. Не заменяй слова пользователя другими, не выдумывай написания: непонятное слово оставь как есть "
    "(для английского — транслитерация).\n"
    "- Use ONLY the concepts the question itself names. Never add related, broader, narrower or "
    "example terms and no synonym lists: 'Что такое божество?' -> only «божество»/deity. "
    "Только понятия из самого вопроса, без связанных и «примерных» слов.\n"
    "- EXCEPTION for questions about a quantity, duration, date or order (how long / how many / how old / "
    "when / which came first): ALSO add the few directly related entities and events that probably hold "
    "the needed numbers, e.g. for «Сколько прожила Ева?» -> «Ева», «смерть Евы», «Адам» (its partner, "
    "the death, the age). Still no unrelated concepts and at most {max_terms} terms in total. "
    "Для вопросов «сколько / когда / как долго» добавь 1–3 прямо связанных понятия, где вероятны цифры.\n"
    "- semantic_query: ONE short declarative phrase in the language of the question that describes "
    "the passage that would answer it (not a question).\n"
    '- answer_type: "list" if the question asks to enumerate or name ALL members, kinds, steps, parts '
    "etc. of something (перечисли, назови всех, кто входит, какие бывают, из чего состоит, list, name all); "
    'otherwise "other". Вопрос-перечень -> "list".\n'
    "Example: question «В чём разница между разумом и духом?» -> "
    '{"keywords_ru":["разум","дух"],"keywords_en":["mind","spirit"],'
    '"semantic_query":"различие между разумом и духом","answer_type":"other"}'
)

HINTS_LABEL = "Слова из вопроса и их соответствия в книге: "

INFERENCE_ADDENDUM = (
    "\nLABELED INFERENCES ARE ALLOWED / РАЗРЕШЕНЫ ПОМЕЧЕННЫЕ ВЫВОДЫ (this extends rules 1 and 4):\n"
    "6. If no excerpt states the answer directly, but it FOLLOWS from several excerpts by simple "
    "reasoning or arithmetic (subtraction, addition, comparison, ordering of events), do NOT reply "
    "«no data». First write everything the excerpts state directly (with ids), then ONE separate final "
    "block that starts with the word «Вывод:». A missing LINK that is a natural, plainly stated assumption "
    "(e.g. that two people who began life together were of the same age, or that both statements count "
    "years from the same moment) does NOT forbid an inference: make the result conditional on that "
    "assumption («Допущение: …») and call it approximate. But if a needed fact or number is simply absent "
    "from the excerpts and no such natural assumption exists, do NOT infer: say what is missing (rule 4). "
    "Если ответ прямо не сказан, но следует из "
    "нескольких абзацев простым рассуждением или арифметикой — не отвечай «данных нет»: сначала прямые "
    "факты, затем отдельный блок «Вывод:».\n"
    "«Вывод:» is ONLY for a result that is not written in any excerpt and had to be derived (a computed "
    "number, a date, an order of events). NEVER use it for a summary, overview or restatement of what the "
    "excerpts say, and never when the question is answered directly: then simply answer, with citations. "
    "«Вывод:» — только для результата, которого нет ни в одном абзаце и который пришлось вычислить; "
    "не используй его для краткого пересказа или резюме.\n"
    "7. Inside «Вывод:»: (a) show the steps one by one; (b) cite EVERY fact you use, each with its id in "
    "brackets; (c) write every assumption on its own line starting with «Допущение: …» (for example that two "
    "statements are about the same person or that years are counted the same way); (d) say that this is a "
    "conclusion drawn from the sources, not a statement of the book; (e) use no outside knowledge. Never "
    "present an inference as a stated fact and never put an inference outside the «Вывод:» block. "
    "В блоке «Вывод:»: шаги, ссылки на КАЖДЫЙ использованный факт, допущения отдельными строками "
    "«Допущение: …», пометка, что это вывод, а не слова книги.\n"
    "WORKED EXAMPLE (invented names, follow this pattern). Question: «Сколько лет прожила Алма?» Excerpts: "
    "[1:1.1] «Борн дожил до 80 лет; Алма умерла на пять лет раньше.» Good answer: «Прямые факты: Борн прожил "
    "80 лет [1:1.1]; Алма умерла на пять лет раньше Борна [1:1.1]. Точной цифры для Алмы в источниках нет.\n"
    "Вывод: 1) Борн умер в 80 лет [1:1.1]. 2) Алма умерла на 5 лет раньше [1:1.1]. 3) Если Алма и Борн были "
    "ровесниками, она прожила около 80 − 5 = 75 лет. Допущение: Алма и Борн ровесники (в источниках это прямо "
    "не сказано). Это вывод из источников, а не прямое утверждение книги.» A reply that only says «данных "
    "недостаточно» in such a case is WRONG; so is stating «75 лет» as a plain fact.\n"
)

# Appended to the user message (recency matters for small models): before answering "no data", check
# whether a labeled inference is possible.
INFERENCE_REMINDER = (
    "\n\n(Если ответ написан в абзацах прямо — просто ответь со ссылками, без блока «Вывод:». "
    "Перед ответом «данных нет» проверь: нельзя ли получить ответ простым рассуждением или расчётом "
    "из найденных абзацев — например, сложить или вычесть числа из РАЗНЫХ абзацев? Если можно — сначала "
    "напиши прямые факты, затем блок «Вывод:» с шагами, ссылками на каждый факт и допущениями. "
    "Недостающее очевидное звено можно принять как явное «Допущение: …» (например, что двое, начавшие жизнь "
    "вместе, ровесники) — тогда вывод условный и приблизительный. "
    "Если же нужного числа или факта нет вовсе и разумного допущения тоже нет — не выводи, а скажи, чего не хватает.)"
)

# Appended to the user message when the question asks for a full enumeration.
LIST_REMINDER = (
    "\n\n(Это вопрос-перечень: перечисли ВСЕ подходящие пункты из источников, ничего не пропуская, "
    "каждый со ссылкой на его абзац. Если часть перечня в источниках не названа — скажи, какая.)"
)

# Appended LAST to the system prompt: how the answer should look (rendered as Markdown in the UI).
FORMATTING_PROMPT = (
    "\nFORMATTING / ОФОРМЛЕНИЕ ОТВЕТА: use light Markdown. Short paragraphs; **bold** for key terms and "
    "definitions; *italic* for assumptions; bullet lists (- ) or numbered lists (1. ) for enumerations; "
    "citation ids in square brackets right after each claim, e.g. [0:1.2]; headings no higher than ### "
    "(preferably none); no tables, no emojis, no horizontal rules, no code blocks. If a separate conclusion "
    "block is required by the rules above, keep its label and layout exactly as specified there.\n"
    "Используй лёгкую разметку Markdown: короткие абзацы; **жирным** — ключевые термины и определения; "
    "*курсивом* — допущения; маркированные (- ) и нумерованные (1. ) списки для перечислений; номера абзацев "
    "в квадратных скобках сразу после утверждения; заголовки не выше ### (лучше без них); без таблиц, эмодзи, "
    "горизонтальных линий и блоков кода.\n"
)

# Added to the system prompt only when the setting «Язык ответа» is «всегда по-русски».
ANSWER_LANG_RU_PROMPT = (
    "\nANSWER LANGUAGE / ЯЗЫК ОТВЕТА: ALWAYS answer in Russian, whatever language the question is written "
    "in; keep proper names and quoted terms as in the book. Всегда отвечай по-русски, на каком бы языке "
    "ни был задан вопрос.\n"
)

# How the question and the numbered excerpts are put into the user message.
# Placeholders: {sources} (required), {question} (required), {inference_reminder} (optional).
USER_TEMPLATE = "Sources / Источники:\n{sources}\n\nQuestion / Вопрос: {question}{inference_reminder}"

STRICTNESS_MODES = ("facts_only", "allow_inferences")


def system_prompt(strictness: str = "allow_inferences") -> str:
    """facts_only = the strict prompt as before; allow_inferences adds the labeled-inference rules."""
    return SYSTEM_PROMPT + (INFERENCE_ADDENDUM if strictness == "allow_inferences" else "")


_INFERENCE_RE = re.compile(r"(?:^|\n)[ \t>*_#-]*Вывод\s*:")


def has_inference(text: str) -> bool:
    return bool(_INFERENCE_RE.search(text or ""))


DEFAULT_STAGES = [
    "analysis", "fts", "semantic", "hybrid", "context", "generation",
]

NO_ANSWER_MARKER = "нет достаточных данных"


# UI setting values ("авто"/"вкл"/"выкл") -> the reasoning_effort argument passed to
# LMStudioClient.chat. "auto" resolves to our data-backed recommendation per mode
# (see RagPipeline/TranslationAnalyzer docstrings and docs/04_benchmark_translation.md).
def resolve_reasoning_effort(setting: Optional[str], auto_default: str = "none") -> Optional[str]:
    setting = (setting or "auto").lower()
    if setting == "auto":
        return auto_default
    if setting in ("off", "выкл"):
        return "none"
    if setting in ("on", "вкл"):
        return None  # omit the field entirely -> model's own default (reasoning on)
    return setting  # pass through e.g. "low"/"medium"/"high" verbatim


def estimate_tokens(text: str) -> int:
    """Rough token estimate: chars/3 (per architecture doc), works ok for mixed EN/RU."""
    return max(1, len(text) // 3)


def fit_context(paragraphs: list[dict], max_context_tokens: int, reserved_tokens: int = 800,
                lang: str = "both") -> list[dict]:
    """Trim paragraph list to fit the model's context window (token estimate chars/3).
    `lang` ("both"/"ru"/"en") must match what build_context_block will really send."""
    budget = max(0, max_context_tokens - reserved_tokens)
    out = []
    used = 0
    for p in paragraphs:
        cost = estimate_tokens(build_context_block([p], lang))
        if used + cost > budget and out:
            break
        out.append(p)
        used += cost
    return out


def build_context_block(paragraphs: list[dict], lang: str = "both") -> str:
    lines = []
    for p in paragraphs:
        lines.append(f"[{p['id']}] (Paper {p.get('paper')}, Section {p.get('section')}, Para {p.get('paragraph')})")
        if p.get("en") and lang in ("both", "en"):
            lines.append(f"EN: {p['en']}")
        if p.get("ru") and lang in ("both", "ru"):
            lines.append(f"RU: {p['ru']}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- citations ---

_ID_RE = re.compile(r"\d+:\d+\.\d+")


def extract_citations(text: str) -> list[str]:
    """Paragraph ids cited in the answer, unique, in order of first appearance.
    Understands [0:1.2], [0:1.2][0:2.10] and grouped [0:1.2, 0:2.10]."""
    seen: list[str] = []
    for m in re.finditer(r"\[([^\[\]]+)\]", text or ""):
        for part in re.split(r"\s*[,;]\s*", m.group(1).strip()):
            part = part.strip()
            if _ID_RE.fullmatch(part) and part not in seen:
                seen.append(part)
    return seen


# ------------------------------------------------------------ query rewriting ---

def _clean_terms(items, limit: int = 8) -> list[str]:
    out: list[str] = []
    if not isinstance(items, list):
        return out
    for it in items:
        if not isinstance(it, str):
            continue
        t = re.sub(r"[\"“”«»]", "", it).strip()
        if t and t.lower() not in [o.lower() for o in out]:
            out.append(t[:60])
        if len(out) >= limit:
            break
    return out


def parse_search_plan(text: str, max_terms: int = 8) -> Optional[dict]:
    """Parse the model's JSON search plan (at most `max_terms` terms per language); None if unusable."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    ru = _clean_terms(obj.get("keywords_ru"), max_terms)
    en = _clean_terms(obj.get("keywords_en"), max_terms)
    sq = obj.get("semantic_query")
    sq = sq.strip() if isinstance(sq, str) else ""
    if not ru and not en:
        return None
    at = obj.get("answer_type")
    return {"keywords_ru": ru, "keywords_en": en, "semantic_query": sq,
            "answer_type": "list" if isinstance(at, str) and at.strip().lower() == "list" else "other"}


def rewrite_query(llm, model: str, question: str, timeout: float = 60.0, prompt: Optional[str] = None,
                  max_terms: int = 6, hints: str = "") -> Optional[dict]:
    """One short, deterministic LLM call (reasoning off). Returns the plan or None on ANY failure
    (the caller then falls back to plain retrieval on the question). `prompt` is the already-rendered
    system prompt (default: the built-in one with {max_terms} filled in). `hints` = verified words of the
    question (from the book's vocabulary / glossary), appended to the user message."""
    if prompt is None:
        prompt = QUERY_REWRITE_PROMPT.replace("{max_terms}", str(max_terms))
    user = question + (f"\n\n{HINTS_LABEL}{hints}" if hints else "")
    try:
        raw = llm.chat(
            [{"role": "system", "content": prompt},
             {"role": "user", "content": user}],
            model=model, temperature=0.0, max_tokens=200, timeout=timeout, reasoning_effort="none",
        )
    except Exception:  # noqa: BLE001 - any failure => fallback, never break the question
        return None
    return parse_search_plan(raw if isinstance(raw, str) else "", max_terms)


def fts_query_from_plan(plan: dict) -> str:
    """Search string for Searcher.fts(): multi-word terms become quoted phrases, single words stay
    bare (the searcher OR-combines them and matches RU lemmas)."""
    parts: list[str] = []
    for term in plan["keywords_ru"] + plan["keywords_en"]:
        term = term.strip()
        if not term:
            continue
        parts.append(f'"{term}"' if " " in term else term)
    return " ".join(parts)


def plan_summary(plan: dict) -> str:
    """Plain-Russian one-liner for the status panel, e.g. 'Ищу: агондонтеры / agondonters'."""
    ru = ", ".join(plan["keywords_ru"])
    en = ", ".join(plan["keywords_en"])
    both = " / ".join(x for x in (ru, en) if x)
    return f"Ищу: {both}"


def rrf_merge(lists: list[list[dict]], k: int, rrf_k: int = 60) -> list[dict]:
    """Reciprocal-rank fusion of several ranked paragraph lists. The first list is the FTS list,
    the others are semantic lists (used only to label fts_rank / sem_rank)."""
    score: dict[str, float] = {}
    first: dict[str, dict] = {}
    fts_rank: dict[str, int] = {}
    sem_rank: dict[str, int] = {}
    for li, lst in enumerate(lists):
        for rank, item in enumerate(lst, start=1):
            pid = item["id"]
            score[pid] = score.get(pid, 0.0) + 1.0 / (rrf_k + rank)
            first.setdefault(pid, item)
            if li == 0:
                fts_rank.setdefault(pid, rank)
            else:
                sem_rank[pid] = min(sem_rank.get(pid, rank), rank)
    ranked = sorted(score, key=lambda pid: -score[pid])[:k]
    out = []
    for pid in ranked:
        item = dict(first[pid])
        item["score"] = score[pid]
        item["fts_rank"] = fts_rank.get(pid)
        item["sem_rank"] = sem_rank.get(pid)
        out.append(item)
    return out


def compose_system_prompt(prompts, strictness: str, answer_lang: str) -> str:
    """base instruction [+ inference rules] [+ 'answer in Russian' rule] + formatting instruction."""
    parts = [prompts.text("rag_system")]
    if strictness == "allow_inferences":
        parts.append(prompts.text("rag_inference_rules"))
    if answer_lang == "ru":
        parts.append(prompts.text("rag_answer_lang_ru"))
    parts.append(prompts.text("rag_formatting"))
    return "".join(parts)


def _terms_summary(terms: list[dict]) -> str:
    """'Ищу: эдем, эдемия / eden, edentia' - what is really searched (verified words only)."""
    ru = [t.get("display") or t["text"] for t in terms if t.get("ok") and t["lang"] == "ru"]
    en = [t.get("display") or t["text"] for t in terms if t.get("ok") and t["lang"] == "en"]
    return "Ищу: " + " / ".join(x for x in (", ".join(ru), ", ".join(en)) if x)


def ensure_rare_terms(retrieved: list[dict], merged: Optional[dict], searcher, k: int,
                      max_hits: int, slots: int) -> tuple[list[dict], list[dict]]:
    """Rare-term guarantee: paragraphs holding a verified, rare (<= max_hits paragraphs in the book) word of the
    user/glossary always reach the model, even if fusion pushed them below the top k. They take up to `slots`
    places (best bm25 on the rare words only) and the rest is filled by the fused rank.
    Returns (new list, [{"id", "terms"}] of the guaranteed paragraphs)."""
    if merged is None or slots <= 0 or max_hits <= 0:
        return retrieved, []
    from app.core.terms import SearchTerms
    rare = [t for t in merged["terms"] if t.get("ok") and not t.get("too_common") and t.get("src") != "model"
            and 0 < (t.get("hits") or 0) <= max_hits]
    if not rare:
        return retrieved, []
    st = SearchTerms()
    for t in rare:
        add = st.add_ru if t["lang"] == "ru" else st.add_en
        if t.get("phrase"):
            add(phrase=t["key"])
        else:
            add(exact=t["key"], prefix=t.get("prefix") or "")
    try:
        found = searcher.fts_terms(st, k=slots)
    except Exception:  # noqa: BLE001 - a bonus, never fatal
        logger_rag.exception("rare-term lookup failed")
        return retrieved, []
    found = found[:slots]
    if not found:
        return retrieved, []
    g_ids = {r["id"] for r in found}
    by_id = {r["id"]: r for r in retrieved}
    guaranteed = [by_id.get(r["id"], r) for r in found]
    rest = [r for r in retrieved if r["id"] not in g_ids]
    info = [{"id": r["id"], "terms": [t.get("display") or t["text"] for t in rare],
             "in_top_k_before": r["id"] in {x["id"] for x in retrieved[:k]}} for r in found]
    return guaranteed + rest[:max(0, k - len(guaranteed))], info


class RagPipeline:
    def __init__(self, searcher, llm_client: LMStudioClient, model: str,
                 temperature: float = 0.2, max_tokens: int = 1024,
                 timeout: float = 120, max_context_length: int = 8192,
                 reasoning_effort: Optional[str] = "none", query_rewrite: Optional[bool] = None,
                 strictness: Optional[str] = None, options: Optional[dict] = None,
                 prompts=None, conn=None, meta: Optional[dict] = None,
                 rewrite_llm=None, rewrite_model: Optional[str] = None):
        self.searcher = searcher
        self.rewrite_llm, self.rewrite_model = rewrite_llm, rewrite_model  # optional separate provider for query preparation
        self.llm = llm_client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_context_length = max_context_length
        # Empirically (see docs/04_benchmark_translation.md and the RAG smoke test),
        # this Gemma build's reasoning ("thinking") mode eats most of max_tokens on
        # reasoning_content and returns a truncated/empty visible answer, and is
        # ~3x slower. Default reasoning OFF for RAG; still overridable per request.
        self.reasoning_effort = reasoning_effort
        o = dict(opts_mod.DEFAULTS)
        o.update(options or {})
        if query_rewrite is not None:
            o["rag_query_rewrite"] = bool(query_rewrite)
        if strictness in STRICTNESS_MODES:
            o["rag_strictness"] = strictness
        self.o = o
        self.query_rewrite = bool(o["rag_query_rewrite"])
        self.strictness = o["rag_strictness"] if o["rag_strictness"] in STRICTNESS_MODES else "allow_inferences"
        self.prompts = prompts or prompts_mod.ResolvedPrompts()
        self.conn = conn          # needed only for "neighbouring paragraphs"
        self.meta = meta or {}    # extra facts for the experiment snapshot (preset, reasoning label, search settings)
        self.system_prompt = compose_system_prompt(self.prompts, self.strictness, o["rag_answer_lang"])

    # ---------------------------------------------------------------- retrieval ---
    def _retrieve(self, question: str, k: int, plan: Optional[dict], report,
                  merged: Optional[dict] = None) -> tuple[list[dict], dict]:
        o = self.o
        cand_fts = o["rag_candidates_fts"] or max(k * 3, 30)
        cand_sem = o["rag_candidates_semantic"] or max(k * 3, 30)
        rrf_k = o["rag_rrf_k"]
        if merged is not None and not merged["search"].empty():
            # verified words: the user's own + the model's that exist in the book + glossary translations
            shown = merged["summary"]
            report("fts", 0, 1, f"поиск по словам · {shown}")
            fts_list = self.searcher.fts_terms(merged["search"], k=cand_fts)
            report("fts", 1, 1, f"найдено абзацев: {len(fts_list)} · {shown}")
            report("semantic", 0, 1, f"поиск по смыслу · {shown}")
            sem_lists = []
            if plan and plan.get("semantic_query"):
                sem_lists.append(self.searcher.semantic(plan["semantic_query"], k=cand_sem))
            sem_lists.append(self.searcher.semantic(question, k=cand_sem))
            n_sem = len({r["id"] for lst in sem_lists for r in lst})
            report("semantic", 1, 1, f"найдено абзацев: {n_sem} · {shown}")
            report("hybrid", 0, 1, f"объединение результатов · {shown}")
            res = rrf_merge([fts_list] + sem_lists, k, rrf_k=rrf_k)
            report("hybrid", 1, 1, f"выбрано абзацев: {len(res)}")
            return res, {"fts": len(fts_list), "semantic": n_sem}
        if plan is None:
            # fallback = the plain hybrid search on the question, as before
            report("fts", 0, 1, "поиск по словам")
            report("fts", 1, 1, "по тексту вопроса")
            report("semantic", 0, 1, "поиск по смыслу")
            report("semantic", 1, 1, "по тексту вопроса")
            report("hybrid", 0, 1, "объединение результатов")
            res = self.searcher.hybrid(question, k=k, rrf_k=rrf_k,
                                       cand=max(o["rag_candidates_fts"], o["rag_candidates_semantic"]) or 50)
            report("hybrid", 1, 1, f"выбрано абзацев: {len(res)}")
            return res, {"fts": sum(1 for r in res if r.get("fts_rank")),
                         "semantic": sum(1 for r in res if r.get("sem_rank"))}
        fts_q = fts_query_from_plan(plan)
        mode = getattr(self.searcher, "fts_mode_hybrid", None) or "or"
        shown = plan_summary(plan)  # keep the keywords visible in the status panel while searching
        report("fts", 0, 1, f"поиск по словам · {shown}")
        fts_list = self.searcher.fts(fts_q, k=cand_fts, lang="both", mode=mode)
        report("fts", 1, 1, f"найдено абзацев: {len(fts_list)} · {shown}")
        report("semantic", 0, 1, f"поиск по смыслу · {shown}")
        sem_lists = []
        if plan.get("semantic_query"):
            sem_lists.append(self.searcher.semantic(plan["semantic_query"], k=cand_sem))
        sem_lists.append(self.searcher.semantic(question, k=cand_sem))
        n_sem = len({r["id"] for lst in sem_lists for r in lst})
        report("semantic", 1, 1, f"найдено абзацев: {n_sem} · {shown}")
        report("hybrid", 0, 1, f"объединение результатов · {shown}")
        res = rrf_merge([fts_list] + sem_lists, k, rrf_k=rrf_k)
        report("hybrid", 1, 1, f"выбрано абзацев: {len(res)}")
        return res, {"fts": len(fts_list), "semantic": n_sem}

    def _with_neighbors(self, items: list[dict], n: int) -> list[dict]:
        """Add n paragraphs before/after every source (reading order, deduplicated). Neighbours are
        flagged with neighbor_of and may be cited like any source."""
        if n <= 0 or self.conn is None or not items:
            return items
        from app.core import db as core_db

        main_ids = {it["id"] for it in items}
        seen: set[str] = set()
        out: list[dict] = []
        for it in items:
            try:
                ctx = core_db.get_context(self.conn, it["id"], n, n)
            except Exception:  # noqa: BLE001 - neighbours are a bonus, never fatal
                ctx = []
            if not ctx:
                ctx = [it]
            for p in ctx:
                if p["id"] == it["id"]:
                    if p["id"] not in seen:
                        seen.add(p["id"])
                        out.append(it)
                elif p["id"] not in main_ids and p["id"] not in seen:
                    seen.add(p["id"])
                    out.append({**p, "neighbor_of": it["id"]})
        return out

    def _with_lists(self, items: list[dict], max_items: int) -> list[dict]:
        """Pull in whole numbered lists. A source that introduces a list (ends with ':') gets the
        numbered paragraphs that follow it; a source that is itself a list item ('7. ...') gets the
        intro and every other item of its list. Added paragraphs are flagged with neighbor_of and
        placed right after their source."""
        if max_items <= 0 or self.conn is None or not items:
            return items
        from app.core import db as core_db

        def num(p):
            for t in (p.get("ru"), p.get("en")):
                m = _LIST_ITEM_RE.match(t or "")
                if m:
                    return int(m.group(1))
            return None

        def is_intro(p):
            return any((t or "").rstrip().endswith(":") for t in (p.get("ru"), p.get("en")))

        main_ids = {it["id"] for it in items}
        seen: set[str] = set()
        out: list[dict] = []
        for it in items:
            if it["id"] not in seen:
                seen.add(it["id"])
                out.append(it)
            n0 = num(it)
            if n0 is None and not is_intro(it):
                continue
            try:
                ctx = core_db.get_context(self.conn, it["id"], max_items + 1, max_items + 1)
            except Exception:  # noqa: BLE001 - a bonus, never fatal
                continue
            ctx = [p for p in ctx if p["paper"] == it["paper"] and p["section"] == it["section"]]
            idx = next((i for i, p in enumerate(ctx) if p["id"] == it["id"]), None)
            if idx is None:
                continue
            block: list[dict] = []
            if n0 is not None:                      # walk back to the start of the list (and its intro)
                i, expect = idx - 1, n0 - 1
                while i >= 0 and expect >= 1 and num(ctx[i]) == expect:
                    block.insert(0, ctx[i]); i -= 1; expect -= 1
                if i >= 0 and expect == 0 and is_intro(ctx[i]):
                    block.insert(0, ctx[i])
                expect = n0 + 1
            else:                                   # intro: the list must follow immediately
                nxt = num(ctx[idx + 1]) if idx + 1 < len(ctx) else None
                if nxt is None:
                    continue
                expect = nxt
            i = idx + 1
            while i < len(ctx) and num(ctx[i]) == expect:
                block.append(ctx[i]); i += 1; expect += 1
            for p in block[:max_items]:
                if p["id"] not in main_ids and p["id"] not in seen:
                    seen.add(p["id"])
                    out.append({**p, "neighbor_of": it["id"]})
        return out

    def _expand_sections(self, items: list[dict], sections: list[dict], budget: int, lang: str) -> tuple[list[dict], list[dict]]:
        """Put whole sections in front of the other sources (fit_context drops from the end). A section
        that does not fit into ~3/4 of the budget is cut: from the first found paragraph of it onwards
        (or from its start), as much as fits. Returns (new items, what was added for the report)."""
        if not sections or self.conn is None or budget <= 0:
            return items, []
        from app.core import db as core_db

        by_id = {it["id"]: it for it in items}
        left = int(budget * 0.75)
        front: list[dict] = []
        taken: set[str] = set()
        info = []
        for s in sections:
            try:
                paras = core_db.get_range(self.conn, s["paper"], s["section"])
            except Exception:  # noqa: BLE001 - a bonus, never fatal
                continue
            paras = [p for p in paras if p["id"] not in taken]
            if not paras:
                continue
            costs = [estimate_tokens(build_context_block([p], lang)) for p in paras]
            whole = sum(costs) <= left
            start = 0
            if not whole:
                hit = next((i for i, p in enumerate(paras) if p["id"] in by_id), None)
                start = max(0, hit - 1) if hit is not None else 0
            label = f"{s['paper']}:{s['section']}"
            n = 0
            for p, c in zip(paras[start:], costs[start:]):
                if c > left:
                    break
                left -= c
                taken.add(p["id"])
                front.append(by_id.get(p["id"]) or {**p, "neighbor_of": label, "expanded_section": label})
                n += 1
            if n:
                info.append({**s, "id": label, "added": n, "total": len(paras), "whole": n == len(paras)})
            if left <= 0:
                break
        rest = [it for it in items if it["id"] not in taken]
        return front + rest, info

    # ------------------------------------------------------------------ answer ---
    def answer(self, question: str, k: Optional[int] = None,
               progress: Optional[Callable[[str, int, int, Optional[int], str], None]] = None) -> dict:
        def report(stage, current, total, message=""):
            if progress:
                progress(stage, current, total, None, message)

        o = self.o
        k = int(k or o["rag_top_k"])
        t_total = time.monotonic()
        timings: dict = {}

        # --- 1. query preparation (LLM search plan) ------------------------------------------
        plan = None
        rewrite_prompt = None
        rw_llm = self.rewrite_llm or self.llm
        rw_model = self.rewrite_model or self.model
        title = llm_title(self.llm)
        tt = getattr(self.searcher, "terms", None) if self.query_rewrite else None
        if tt is not None and not (hasattr(tt, "available") and hasattr(self.searcher, "fts_terms") and tt.available()):
            tt = None
        user_terms: list[dict] = []
        hints = ""
        if tt is not None:
            try:
                user_terms = tt.analyze_user(question)
                hints = tt.hints_text(user_terms)
            except Exception:  # noqa: BLE001 - the vocabulary is a bonus, never fatal
                logger_rag.exception("term analysis failed")
                tt, user_terms, hints = None, [], ""
        report("analysis", 0, 1, f"Подготовка запроса ({llm_title(rw_llm)} · {rw_model})…"
               + (f"\nСлова из вопроса: {hints}" if hints else ""))
        t0 = time.monotonic()
        if self.query_rewrite:
            rewrite_prompt = prompts_mod.render(self.prompts.text("query_rewrite"),
                                                max_terms=o["rag_rewrite_max_terms"])
            plan = rewrite_query(rw_llm, rw_model, question, timeout=min(self.timeout, 90),
                                 prompt=rewrite_prompt, max_terms=o["rag_rewrite_max_terms"], hints=hints)
        timings["plan_s"] = round(time.monotonic() - t0, 2)
        merged = None
        if tt is not None:
            try:
                merged = tt.merge(user_terms, plan)
                merged["summary"] = _terms_summary(merged["terms"])
                merged["hints"] = hints
            except Exception:  # noqa: BLE001
                logger_rag.exception("term merge failed")
                merged = None
        if merged is not None:
            msg = merged["summary"] + "".join("\n" + ln for ln in merged["lines"])
            if merged["model_all_missing"]:
                msg += "\nВсе слова модели не найдены в книге — ищу по словам из вопроса и по словарю."
            report("analysis", 1, 1, msg)
        else:
            report("analysis", 1, 1, plan_summary(plan) if plan else "Ищу по тексту вопроса")

        # --- 2. retrieval ----------------------------------------------------------------------
        t0 = time.monotonic()
        retrieved, candidates = self._retrieve(question, k, plan, report, merged)
        retrieved, rare_info = ensure_rare_terms(retrieved, merged, self.searcher, k,
                                                 int(o["rag_rare_term_max_hits"]), int(o["rag_rare_term_slots"]))
        n_retrieved = len(retrieved)
        rel = float(o["rag_min_relevance"])
        if rel > 0 and retrieved:
            top = max((r.get("score") or 0.0) for r in retrieved)
            g = {x["id"] for x in rare_info}
            kept = [r for r in retrieved if r["id"] in g or (r.get("score") or 0.0) >= top * rel]
            retrieved = kept or retrieved[:1]
        n_after_filter = len(retrieved)
        auto_expand = bool(o["rag_auto_expand"])
        list_q = auto_expand and is_list_question(question, plan)
        expand_secs: list[dict] = []
        if auto_expand:
            expand_secs = match_section_titles(self.conn, question, plan)
            if list_q:   # a «перечисли…» question: the sections of the best sources as well
                for r in retrieved[:2]:
                    expand_secs.append({"paper": r["paper"], "section": r["section"],
                                        "title": (r.get("section_title") or {}).get("ru") or ""})
            seen_s, uniq = set(), []
            for s in expand_secs:
                if (s["paper"], s["section"]) not in seen_s:
                    seen_s.add((s["paper"], s["section"]))
                    uniq.append(s)
            expand_secs = uniq[:3]
            retrieved = self._with_lists(retrieved, 60)
        retrieved = self._with_neighbors(retrieved, int(o["rag_neighbors"]))
        timings["retrieval_s"] = round(time.monotonic() - t0, 2)

        # --- 3. context ---------------------------------------------------------------------------
        t0 = time.monotonic()
        report("context", 0, 1, "формирование контекста")
        lang = o["rag_sources_lang"]
        inference_on = self.strictness == "allow_inferences"
        reminder = self.prompts.text("rag_inference_reminder") if inference_on else ""
        user_tmpl = self.prompts.text("rag_user_template")
        overhead = estimate_tokens(self.system_prompt) + estimate_tokens(user_tmpl) + estimate_tokens(reminder)
        avail = max(0, self.max_context_length - (self.max_tokens + overhead + 200))
        budget = int(avail * float(o["rag_context_share"]))
        retrieved, expanded = self._expand_sections(retrieved, expand_secs, budget, lang)
        fitted = fit_context(retrieved, budget, reserved_tokens=0, lang=lang)
        context_block = build_context_block(fitted, lang)
        msg = f"использовано источников: {len(fitted)}"
        if list_q:
            msg += "\nВопрос-перечень: передаю модели разделы целиком"
        for e in expanded:
            how = "целиком" if e["whole"] else f"{e['added']} из {e['total']} абзацев (не влез целиком)"
            msg += f"\nРаздел {e['id']}" + (f" «{e['title']}»" if e.get("title") else "") + f": {how}"
        report("context", 1, 1, msg)
        user_message = prompts_mod.render(user_tmpl, sources=context_block, question=question,
                                          inference_reminder=reminder)
        if list_q:
            user_message += LIST_REMINDER
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        timings["context_s"] = round(time.monotonic() - t0, 2)

        report("generation", 0, 1, f"отправка в {title} · {self.model}")

        gen_t0 = time.monotonic()
        gen_chars = 0
        last_report = 0.0

        def on_delta(chunk_text: str):
            # Live progress while tokens stream in. No per-chunk token count is available
            # from the API, so tok/s here is a rough live estimate (chars/4); the accurate
            # figure from usage.completion_tokens is reported once generation finishes.
            nonlocal gen_chars, last_report
            gen_chars += len(chunk_text)
            now = time.monotonic()
            if now - last_report < 0.7:  # throttle: one progress/log line per ~0.7 s, not per token
                return
            last_report = now
            elapsed = now - gen_t0
            est_tps = round((gen_chars / 4) / elapsed, 1) if elapsed > 0.2 else None
            msg = f"получение ответа… (~{est_tps} ток/с)" if est_tps else "получение ответа…"
            report("generation", 0, 1, msg)

        plan_out = None
        if plan:
            plan_out = {**plan, "summary": plan_summary(plan)}
        if merged is not None:
            base = plan_out or {"keywords_ru": [], "keywords_en": [], "semantic_query": ""}
            plan_out = {**base, "summary": merged["summary"] if not plan else plan_out["summary"],
                        "terms": merged["terms"], "terms_lines": merged["lines"], "hints": merged["hints"],
                        "model_failed": plan is None, "dropped": merged["dropped"],
                        "search_terms": merged["search"].to_dict()}

        used_ids = ["rag_system", "rag_formatting", "rag_user_template"]
        if inference_on:
            used_ids += ["rag_inference_rules", "rag_inference_reminder"]
        if o["rag_answer_lang"] == "ru":
            used_ids.append("rag_answer_lang_ru")
        if self.query_rewrite:
            used_ids.append("query_rewrite")

        def experiment(stats_dict=None):
            timings["generation_s"] = round(time.monotonic() - gen_t0, 2)
            timings["total_s"] = round(time.monotonic() - t_total, 2)
            return {
                "kind": "rag",
                "app": {"version": __version__, "date": _dt.date.today().isoformat()},
                "model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
                "reasoning": {"setting": self.meta.get("reasoning_setting"), "effort": self.reasoning_effort},
                "context_length": self.max_context_length,
                "preset": self.prompts.preset,
                "prompt_versions": self.prompts.versions(used_ids),
                "prompt_texts": self.prompts.texts(used_ids),
                "prompts": {"system": self.system_prompt, "user": user_message,
                            "rewrite_system": rewrite_prompt, "rewrite_user": question if rewrite_prompt else None},
                "options": dict(o),
                "search_settings": {k_: self.meta.get(k_) for k_ in ("fts_weights", "fts_mode_hybrid", "fts_mode_standalone",
                                                                   "fts_stopwords")},
                "retrieval": {
                    "k": k, "fallback_plain_hybrid": plan is None and merged is None,
                    "candidates_fts_asked": o["rag_candidates_fts"] or max(k * 3, 30),
                    "candidates_semantic_asked": o["rag_candidates_semantic"] or max(k * 3, 30),
                    "candidates_found": candidates, "rrf_k": o["rag_rrf_k"],
                    "retrieved": n_retrieved, "after_relevance_filter": n_after_filter,
                    "with_neighbors": len(retrieved), "sources_sent": len(fitted),
                    "sources_lang": lang, "context_share": o["rag_context_share"],
                    "search_terms": merged["search"].to_dict() if merged is not None else None,
                    "rare_term_guaranteed": rare_info,
                    "list_question": list_q, "expanded_sections": expanded,
                    "terms_used": merged is not None and not merged["search"].empty(),
                },
                "plan": plan_out,
                "timings": dict(timings),
                "stats": stats_dict,
            }

        gen_kwargs = {}
        if hasattr(self.llm, "profile"):   # real provider client: backup-model switches are shown in the status
            gen_kwargs["on_status"] = lambda text: report("generation", 0, 1, text)
        try:
            answer_text, stats = self.llm.chat_with_stats(
                messages, model=self.model, temperature=self.temperature,
                max_tokens=self.max_tokens, timeout=self.timeout,
                reasoning_effort=self.reasoning_effort, on_delta=on_delta, **gen_kwargs,
            )
        except LMStudioError as exc:
            partial = getattr(exc, "partial", None)
            pstats = getattr(exc, "partial_stats", None)
            report("generation", 1, 1, f"ошибка ({title} · {self.model}): {exc}")
            out = {
                "answer": partial or None,
                "error": str(exc),
                "sources": fitted,
                "plan": plan_out,
                "candidates": candidates,
                "prompt_warnings": self.prompts.warnings,
                "warnings": list(self.prompts.warnings),
                "experiment": experiment(pstats.to_dict() if pstats else None),
            }
            if partial:
                cited = extract_citations(partial)
                known = {p["id"] for p in fitted}
                out.update({
                    "truncated": True,
                    "warnings": list(self.prompts.warnings) + ["Ответ оборван: провайдер прервал передачу. "
                                                               "Показано то, что успело прийти."],
                    "cited_ids": [c for c in cited if c in known],
                    "unknown_citations": [c for c in cited if c not in known],
                    "no_answer": False, "has_inference": has_inference(partial), "strictness": self.strictness,
                    "stats": pstats.to_dict() if pstats else None,
                    "stats_summary": pstats.summary_line() if pstats else None,
                })
            return out
        report("generation", 1, 1, stats.summary_line())

        cited = extract_citations(answer_text)
        known = {p["id"] for p in fitted}
        warns = list(self.prompts.warnings)
        return {
            "answer": answer_text,
            "error": None,
            "sources": fitted,
            "plan": plan_out,
            "cited_ids": [c for c in cited if c in known],
            "unknown_citations": [c for c in cited if c not in known],
            "no_answer": NO_ANSWER_MARKER in (answer_text or "").lower(),
            "has_inference": has_inference(answer_text),
            "strictness": self.strictness,
            "candidates": candidates,
            "stats": stats.to_dict(),
            "stats_summary": stats.summary_line(),
            "answered_model": getattr(stats, "model", None),
            "fallback_note": getattr(stats, "fallback_note", None),
            "prompt_warnings": self.prompts.warnings,
            "warnings": warns,
            "experiment": experiment(stats.to_dict()),
        }
