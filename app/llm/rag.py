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
    '{"keywords_ru": [...], "keywords_en": [...], "semantic_query": "...", "answer_type": "...", '
    '"main_ru": "...", "main_en": "..."}\n'
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
    "- main_ru / main_en: the ONE subject (a person, group, name or thing) that EVERY passage answering the "
    "question must mention, in dictionary form (Russian) and as written in the English text. Not an abstract "
    "word the book may not use for it (conflict, relation, meaning...) and not a subject present in almost "
    "every passage (Jesus, God). «Какие были споры Иисуса с фарисеями?» -> «фарисей» / «Pharisees». "
    'Empty strings "" if the question has no such single subject (e.g. it compares two concepts). '
    "Главное слово — то, без чего абзац точно не ответ; если такого нет — пустая строка.\n"
    "Example: question «В чём разница между разумом и духом?» -> "
    '{"keywords_ru":["разум","дух"],"keywords_en":["mind","spirit"],'
    '"semantic_query":"различие между разумом и духом","answer_type":"other","main_ru":"","main_en":""}'
)

HINTS_LABEL = "Слова из вопроса и их соответствия в книге: "

INFERENCE_ADDENDUM = (
    "\nLABELED INFERENCES ARE ALLOWED / РАЗРЕШЕНЫ ПОМЕЧЕННЫЕ ВЫВОДЫ (this extends rules 1 and 4):\n"
    "6. If no excerpt states the answer directly, but it FOLLOWS from several excerpts by simple "
    "reasoning or arithmetic (subtraction, addition, comparison, ordering of events), do NOT reply "
    "«no data». Start the answer with ONE short line that begins with «Коротко:» and gives the derived "
    "result itself (one number, a narrow range or a short statement) marked as an estimate drawn from the "
    "sources, e.g. «Коротко: примерно 6–8 тысяч человек (оценка по найденным числам, не слова книги).» "
    "Then write everything the excerpts state directly (with ids), then ONE separate final "
    "block that starts with the word «Вывод:». Do NOT open the answer with «точной цифры нет» / «прямого "
    "ответа нет»: say it later, among the direct facts. A missing LINK that is a natural, plainly stated assumption "
    "(e.g. that two people who began life together were of the same age, or that both statements count "
    "years from the same moment) does NOT forbid an inference: make the result conditional on that "
    "assumption («Допущение: …») and call it approximate. But if a needed fact or number is simply absent "
    "from the excerpts and no such natural assumption exists, do NOT infer: say what is missing (rule 4). "
    "When the user asks for an approximate count or estimate («приблизительно», «оцени», «прикинь», "
    "«сколько всего»), adding up the numbers of different excerpts with stated assumptions (overlaps, "
    "vague words like «несколько тысяч» taken as a round figure) IS such a simple inference: give the "
    "rough total, never refuse because the figures are imprecise or may overlap. "
    "Если ответ прямо не сказан, но следует из "
    "нескольких абзацев простым рассуждением или арифметикой — не отвечай «данных нет»: первая строка "
    "«Коротко: …» с самим результатом (пометь, что это оценка), затем прямые факты, затем отдельный блок "
    "«Вывод:». Просьбу «приблизительно / оцени / сколько всего» выполняй: сложи числа с явными допущениями.\n"
    "«Вывод:» is ONLY for a result that is not written in any excerpt and had to be derived (a computed "
    "number, a date, an order of events). NEVER use it for a summary, overview or restatement of what the "
    "excerpts say, and never when the question is answered directly: then simply answer, with citations. "
    "«Вывод:» — только для результата, которого нет ни в одном абзаце и который пришлось вычислить; "
    "не используй его для краткого пересказа или резюме.\n"
    "7. Inside «Вывод:»: (a) show the steps one by one; (b) cite EVERY fact you use, each with its id in "
    "brackets; (c) write every assumption on its own line starting with «Допущение: …» (for example that two "
    "statements are about the same person or that years are counted the same way); (d) say that this is a "
    "conclusion drawn from the sources, not a statement of the book; (e) use no outside knowledge. Never "
    "present an inference as a stated fact and never put an inference outside the «Вывод:» block, except "
    "the opening «Коротко:» line, which repeats only the final result and is marked as an estimate. "
    "В блоке «Вывод:»: шаги, ссылки на КАЖДЫЙ использованный факт, допущения отдельными строками "
    "«Допущение: …», пометка, что это вывод, а не слова книги.\n"
    "WORKED EXAMPLE (invented names, follow this pattern). Question: «Сколько лет прожила Алма?» Excerpts: "
    "[1:1.1] «Борн дожил до 80 лет; Алма умерла на пять лет раньше.» Good answer: «Коротко: около `75 лет` "
    "(вывод с допущением, не слова книги).\nПрямые факты: Борн прожил `80 лет` [1:1.1]; Алма умерла "
    "`на пять лет раньше` Борна [1:1.1]. Возраст Алмы в источниках прямо не назван.\n"
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
    "начни с одной строки «Коротко: …» с итогом (помеченным как оценка), потом прямые факты, затем блок "
    "«Вывод:» с шагами, ссылками на каждый факт и допущениями. Если просят приблизительно оценить или "
    "подсчитать — сложи числа из абзацев с явными допущениями и дай итог, а не только перечень. "
    "Недостающее очевидное звено можно принять как явное «Допущение: …» (например, что двое, начавшие жизнь "
    "вместе, ровесники) — тогда вывод условный и приблизительный. "
    "Если же нужного числа или факта нет вовсе и разумного допущения тоже нет — не выводи, а скажи, чего не хватает.)"
)

# «Дозапрос недостающих фактов»: when the first answer says a fact is missing, the model names the facts that
# would let the answer be derived, they are searched for and the question is answered again.
FOLLOWUP_PROMPT = (
    "A first attempt to answer a question about a book from found excerpts did not fully succeed: the draft "
    "answer below says that some fact is missing. Первая попытка ответить по найденным абзацам не удалась: "
    "в черновике сказано, что каких-то сведений нет. You also get the titles of the book sections where the "
    "excerpts were found: they often name the event a fact is tied to. Даны и названия разделов, где нашлись абзацы.\n"
    "Your task: find the FEW facts that, if found ELSEWHERE in the book, would let the answer be computed by "
    "simple arithmetic or reasoning. Most often these are DATES: the year (or date) of the event where a known "
    "number is given and the year of the event the question is about. Example of reasoning: age at death = age "
    "at a known moment + (year of death - year of that moment). A person's death or another event may coincide "
    "with a well-known event mentioned in the draft or in the section titles (a trial, a crucifixion, a "
    "journey...): then ask for the date of THAT well-known event. Подумай, из каких фактов (чаще всего дат) ответ "
    "можно вычислить, и назови недостающие; если событие совпадает с известным событием — спроси дату известного.\n"
    "NEVER repeat the question itself or ask for the very thing the draft says is not stated (e.g. «возраст Иуды "
    "при смерти» for the question «Сколько лет было Иуде, когда он умер?» is useless). Do not ask for facts the "
    "draft already has. Не повторяй сам вопрос — ищи то, из чего ответ можно вычислить.\n"
    "Output ONLY a JSON object, no prose: {\"how\": \"...\", \"queries\": [\"...\", \"...\"]}. \"how\": one short "
    "sentence in Russian: how the answer can be computed. \"queries\": 1 to 3 short search phrases in Russian, "
    "each about ONE dated event or ONE number (a declarative phrase that names the concrete people and events, "
    "e.g. «год …», «дата …»). If no other fact from the book could help, output {\"how\": \"\", \"queries\": []}.\n"
    "Example. Question: «Сколько лет было Алме, когда она умерла?» Draft: «Алме было двадцать лет, когда она "
    "вышла замуж за Борна [1:1.1]. Она умерла в день великого наводнения [1:2.3]. Возраст при смерти не указан.» "
    "Sections: «Документ 1. Семья Борна · Наводнение». Output: {\"how\": \"20 лет + (год наводнения − год свадьбы)\", "
    "\"queries\": [\"год свадьбы Алмы и Борна\", \"в каком году было великое наводнение\"]}"
)

FOLLOWUP_PER_QUERY = 3   # new paragraphs taken per missing-fact search

# Appended to the user message of the second attempt.
FOLLOWUP_NOTE = (
    "\n\n(Второй проход: к источникам добавлены абзацы, найденные по недостающим фактам ({facts}). Теперь "
    "ОБЯЗАТЕЛЬНО попробуй вычислить ответ по фактам из РАЗНЫХ абзацев в блоке «Вывод:» — со ссылками на каждый "
    "факт. Если для расчёта не хватает одного звена, но есть естественное допущение (например, что человек умер "
    "в том же году, что и описанные вместе с его смертью события, или что годы считаются одинаково), прими его "
    "явно строкой «Допущение: …» и дай приблизительный ответ. Ответить «данных нет», когда приблизительный "
    "расчёт с явным допущением возможен, — НЕВЕРНО. Только если нужных чисел нет совсем — скажи, чего именно "
    "не хватает.)"
)

_GAP_RE = re.compile(
    r"нет\s+(?:достаточных\s+|прямых\s+|точных\s+)?(?:данных|сведений|информации|указани)"
    r"|не\s+(?:указан|сказан|упомина|приведен|приведён|сообщ)"
    r"|(?:no|not enough|insufficient)\s+(?:data|information)|not\s+(?:stated|mentioned|given|specified)",
    re.IGNORECASE)


_DATE_WORD_RE = re.compile(r"\b(?:год|года|году|дата|дату|даты|когда|век|year|date|when)\b", re.IGNORECASE)
_DATE_RE = re.compile(r"\b\d{1,4}\s+(?:год|года|году)\s+(?:до\s+)?н\.\s*э\.|\bA\.\s*D\.\s*\d+|\b\d+\s+B\.\s*C\.",
                      re.IGNORECASE)


def has_date(p: dict) -> bool:
    """The paragraph names a calendar year («30 года н. э.», «A.D. 30»)."""
    return bool(_DATE_RE.search(p.get("ru") or "") or _DATE_RE.search(p.get("en") or ""))


def answer_has_gap(text: str) -> bool:
    """The answer says that something needed is missing (fully or partly)."""
    return bool(_GAP_RE.search(text or ""))


def parse_followup(text: str, max_queries: int = 3) -> list[str]:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return []
    items = data.get("queries") if isinstance(data, dict) else None
    out: list[str] = []
    for q in items if isinstance(items, list) else []:
        q = re.sub(r"\s+", " ", q).strip() if isinstance(q, str) else ""
        if q and q.lower() not in (x.lower() for x in out):
            out.append(q[:200])
    return out[:max_queries]


def section_titles(paragraphs: list[dict], limit: int = 15) -> list[str]:
    """«Документ 186. Перед распятием · Конец Иуды Искариота» for the sources, unique, in order."""
    out: list[str] = []
    for p in paragraphs:
        pt, st = (p.get("paper_title") or {}).get("ru"), (p.get("section_title") or {}).get("ru")
        t = f"Документ {p.get('paper')}" + (f". {pt}" if pt else "") + (f" · {st}" if st else "")
        if (pt or st) and t not in out:
            out.append(t)
    return out[:limit]


def plan_followup(llm, model: str, question: str, draft: str, timeout: float = 60.0,
                  prompt: Optional[str] = None, sections: Optional[list[str]] = None) -> list[str]:
    """One short LLM call: search phrases for the facts the draft answer lacks. [] on any failure."""
    user = f"Вопрос: {question}\n\nЧерновик ответа:\n{draft}"
    if sections:
        user += "\n\nРазделы, где нашлись абзацы:\n" + "\n".join(f"- {t}" for t in sections)
    try:
        raw = llm.chat(
            [{"role": "system", "content": prompt or FOLLOWUP_PROMPT},
             {"role": "user", "content": user}],
            model=model, temperature=0.0, max_tokens=400, timeout=timeout, reasoning_effort="none",
        )
    except Exception:  # noqa: BLE001 - the second attempt is a bonus, never fatal
        return []
    return parse_followup(raw if isinstance(raw, str) else "")


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
    "(preferably none); no tables, no emojis, no horizontal rules, no code blocks. Wrap each concrete fact "
    "value in backticks so it is shown as a highlighted chip: numbers with their unit (`несколько тысяч`, "
    "`от 500 до 1500 человек`, `80 лет`), dates and years (`28 г. н. э.`, `15 марта`), sums and results "
    "(`≈ 7100`). Only the short value itself — never a whole sentence, never a citation id. If a separate "
    "conclusion block is required by the rules above, keep its label and layout exactly as specified there.\n"
    "Используй лёгкую разметку Markdown: короткие абзацы; **жирным** — ключевые термины и определения; "
    "*курсивом* — допущения; маркированные (- ) и нумерованные (1. ) списки для перечислений; номера абзацев "
    "в квадратных скобках сразу после утверждения; заголовки не выше ### (лучше без них); без таблиц, эмодзи, "
    "горизонтальных линий и блоков кода. Конкретные значения — числа с единицами, даты, годы, итоги расчёта — "
    "бери в обратные кавычки: `80 лет`, `28 г. н. э.`, `≈ 7100` (только само значение, не фразу и не номер абзаца).\n"
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

# ---- «Обзор по всей книге»: every section with the main word goes to the model in parts ----
OVERVIEW_MIN_SECTIONS = 15      # auto mode: a list question whose main word is spread over at least this many sections
OVERVIEW_MAX_SECTIONS = 150     # more would take too long: the best-matching ones are kept
OVERVIEW_PART_TOKENS = 9000     # one part (a few sections) of book text per request
OVERVIEW_THINKING_TOKENS = 6000 # retry budget for a model that always reasons before answering
OVERVIEW_ANSWER_TOKENS = 12000  # the final merged answer: notes from the whole book make it long

OVERVIEW_MAP_PROMPT = (
    "Ты помогаешь составить полный обзор по книге. Тебе даны вопрос и часть книги — абзацы с номерами в квадратных "
    "скобках. Выпиши из ЭТИХ абзацев всё, что относится к вопросу: каждый эпизод, случай или факт — отдельной "
    "короткой строкой: дефис, своими словами что произошло или что сказано, и номер абзаца в квадратных скобках "
    "(можно несколько). Пример строки: «- Фарисеи обвинили учеников в нарушении субботы за сбор колосьев [147:6.4]». "
    "Пропускай абзацы, где нет ничего по вопросу. Пиши на языке вопроса. "
    "Только то, что есть в абзацах, ничего от себя. Помни, что о том же может говориться другими словами "
    "(например, «шпионы» вместо «фарисеи»). Если в этой части нет ничего по вопросу — ответь одним словом НЕТ."
)
OVERVIEW_REDUCE_TEMPLATE = (
    "Вопрос: {question}\n\n"
    "Ниже выписки, сделанные по ВСЕЙ книге (по порядку книги; в скобках — номера абзацев):\n\n{notes}\n\n"
    "Составь полный ответ на вопрос по этим выпискам: сгруппируй по темам или по порядку событий, объедини повторы, "
    "не теряй ни одного эпизода и сохрани ссылки на абзацы в квадратных скобках."
)


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

    def main(key: str) -> str:
        v = obj.get(key)
        return re.sub(r"[\"“”«»]", "", v).strip()[:60] if isinstance(v, str) else ""
    return {"keywords_ru": ru, "keywords_en": en, "semantic_query": sq,
            "answer_type": "list" if isinstance(at, str) and at.strip().lower() == "list" else "other",
            "main_ru": main("main_ru"), "main_en": main("main_en")}


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
    # ------------------------------------------------------------------ overview ---
    def _overview_sections(self, merged: Optional[dict]) -> list[list[dict]]:
        """All sections holding the main word, each as its paragraphs with the word + one neighbour on each side
        (a story is often told around the word: «шпионы» next to «фарисеи»). Book order. [] without a main word."""
        req = getattr(merged["search"], "required", None) if merged is not None else None
        if req is None or req.empty() or self.conn is None or not hasattr(self.searcher, "fts_terms"):
            return []
        from app.core import db as core_db
        try:
            hits = self.searcher.fts_terms(req, k=5000)
        except Exception:  # noqa: BLE001 - the overview is a bonus, never fatal
            logger_rag.exception("overview lookup failed")
            return []
        best: dict = {}
        for rank, h in enumerate(hits):
            best.setdefault((h["paper"], h["section"]), rank)
        keep = sorted(best, key=lambda s: best[s])[:OVERVIEW_MAX_SECTIONS]
        keep_set = set(keep)
        by_sec: dict = {}
        seen: set = set()
        for h in hits:
            sec = (h["paper"], h["section"])
            if sec not in keep_set:
                continue
            try:
                ctx = core_db.get_context(self.conn, h["id"], 1, 1)
            except Exception:  # noqa: BLE001
                ctx = [h]
            for p in ctx or [h]:
                if (p["paper"], p["section"]) == sec and p["id"] not in seen:
                    seen.add(p["id"])
                    by_sec.setdefault(sec, []).append(p)

        def order(p):
            return (p["paper"], p["section"], p.get("paragraph") or 0)
        return [sorted(by_sec[s], key=order) for s in sorted(by_sec)]

    def _overview(self, question: str, plan: Optional[dict], merged: dict, sections: list[list[dict]], report,
                  t_total: float, timings: dict, rewrite_prompt: Optional[str]) -> dict:
        """Map: the sections in parts, the model writes out what each part says on the question.
        Reduce: one answer from all the notes."""
        o = self.o
        lang = "en" if not re.search(r"[а-яё]", question.lower()) else "ru"   # one language: half the text
        title = llm_title(self.llm)
        part_budget = max(2000, min(OVERVIEW_PART_TOKENS, (self.max_context_length - 1500) // 2))
        parts: list[list[dict]] = [[]]
        used = 0
        for sec in sections:
            cost = sum(estimate_tokens(build_context_block([p], lang)) for p in sec)
            if parts[-1] and used + cost > part_budget:
                parts.append([])
                used = 0
            parts[-1].extend(sec)
            used += cost
        parts = [p for p in parts if p]
        n_par = sum(len(p) for p in parts)
        report("context", 1, 1, f"Обзор по всей книге: разделов {len(sections)}, абзацев {n_par}, "
                                f"частей {len(parts)} (главное слово: {merged.get('required', {}).get('display', '')})")
        t0 = time.monotonic()
        notes: list[str] = []
        failed = 0
        thinking_only = 0   # parts where the model spent every token on reasoning and wrote nothing
        stopped = ""
        for i, part in enumerate(parts, 1):
            first, last = part[0]["id"], part[-1]["id"]
            report("generation", 0, 1, f"Обзор: часть {i} из {len(parts)} ({first} … {last}) · {title} · {self.model}")
            msgs = [{"role": "system", "content": OVERVIEW_MAP_PROMPT},
                    {"role": "user", "content": f"Вопрос: {question}\n\nАбзацы книги:\n{build_context_block(part, lang)}"}]
            txt = ""
            # a model that always thinks first (it ignores «reasoning off») may use up a small budget on thinking
            for budget in (900, OVERVIEW_THINKING_TOKENS):
                try:
                    txt, st = self.llm.chat_with_stats(msgs, model=self.model, temperature=0.1, max_tokens=budget,
                                                       timeout=self.timeout, reasoning_effort="none")
                except Exception as exc:  # noqa: BLE001 - one bad part must not lose the others
                    logger_rag.warning("overview part %d failed: %s", i, exc)
                    txt, st = None, None
                    break
                txt = (txt or "").strip() if isinstance(txt, str) else ""
                if txt or getattr(st, "stop_reason", None) != "length":
                    break
                report("generation", 0, 1, f"Обзор: часть {i} из {len(parts)} — модель израсходовала {budget} токенов на "
                                           f"рассуждения и ничего не написала; повтор с запасом {OVERVIEW_THINKING_TOKENS}…")
            if txt is None:
                failed += 1
                continue
            if not txt and getattr(st, "stop_reason", None) == "length":
                failed += 1
                thinking_only += 1
                if not notes and thinking_only >= 2:   # it will not get better on the next parts: stop early
                    stopped = (f"Модель «{self.model}» тратит все токены на рассуждения и не выдаёт ответ "
                               f"(даже при запасе {OVERVIEW_THINKING_TOKENS} токенов). Для обзора выберите модель, "
                               f"которая умеет отвечать без обязательного рассуждения.")
                    break
                continue
            if txt and not re.fullmatch(r"\W*нет\W*", txt.lower()):
                notes.append(txt)
        timings["overview_parts_s"] = round(time.monotonic() - t0, 2)
        if stopped:
            failed = len(parts)

        plan_out = {**(plan or {"keywords_ru": [], "keywords_en": [], "semantic_query": ""}),
                    "summary": merged["summary"] if merged else (plan_summary(plan) if plan else ""),
                    "terms": merged["terms"], "terms_lines": merged["lines"], "hints": merged.get("hints"),
                    "model_failed": plan is None, "dropped": merged["dropped"],
                    "search_terms": merged["search"].to_dict()}
        all_paras = [p for part in parts for p in part]
        known = {p["id"] for p in all_paras}
        ov_info = {"sections": len(sections), "paragraphs": n_par, "parts": len(parts), "failed_parts": failed,
                   "parts_with_notes": len(notes), "required": merged.get("required", {}).get("display")}
        base = {"sources": all_paras, "plan": plan_out, "candidates": {"fts": n_par, "semantic": 0},
                "overview": ov_info, "prompt_warnings": self.prompts.warnings}
        if not notes:
            msg = (stopped or "Не удалось получить ответ ни по одной части книги." if failed == len(parts)
                   else "В найденных разделах модель не нашла ничего по вопросу. " + NO_ANSWER_MARKER.capitalize() + ".")
            return {**base, "answer": msg if failed < len(parts) else None,
                    "error": msg if failed == len(parts) else None, "cited_ids": [], "unknown_citations": [],
                    "no_answer": True, "has_inference": False, "strictness": self.strictness,
                    "warnings": list(self.prompts.warnings)}

        user_message = OVERVIEW_REDUCE_TEMPLATE.format(question=question, notes="\n".join(notes))
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user_message}]
        # what is left of the model's window after the notes (a small window must not make the request fail)
        room = max(1000, self.max_context_length - estimate_tokens(self.system_prompt + user_message) - 300)
        report("generation", 0, 1, f"Обзор: сводный ответ по {len(notes)} частям · {title} · {self.model}")
        gen_t0 = time.monotonic()
        last = [0.0]

        def on_delta(_chunk: str):
            now = time.monotonic()
            if now - last[0] >= 1.0:
                last[0] = now
                report("generation", 0, 1, "Обзор: сводный ответ… получение текста")
        warns = list(self.prompts.warnings)
        if failed:
            warns.append(f"Обзор неполный: {failed} из {len(parts)} частей книги модель не обработала (ошибка).")
        try:
            answer_text, stats = self.llm.chat_with_stats(
                messages, model=self.model, temperature=self.temperature,
                max_tokens=max(self.max_tokens, min(OVERVIEW_ANSWER_TOKENS, room)), timeout=self.timeout,
                reasoning_effort=self.reasoning_effort, on_delta=on_delta)
        except LMStudioError as exc:
            partial = getattr(exc, "partial", None)
            report("generation", 1, 1, f"ошибка ({title} · {self.model}): {exc}")
            cited = extract_citations(partial or "")
            return {**base, "answer": partial or None, "error": str(exc), "truncated": bool(partial),
                    "cited_ids": [c for c in cited if c in known], "unknown_citations": [c for c in cited if c not in known],
                    "no_answer": False, "has_inference": False, "strictness": self.strictness, "warnings": warns}
        timings["generation_s"] = round(time.monotonic() - gen_t0, 2)
        timings["total_s"] = round(time.monotonic() - t_total, 2)
        report("generation", 1, 1, stats.summary_line())
        cited = extract_citations(answer_text)
        return {**base,
                "answer": answer_text, "error": None,
                "cited_ids": [c for c in cited if c in known],
                "unknown_citations": [c for c in cited if c not in known],
                "no_answer": NO_ANSWER_MARKER in (answer_text or "").lower(),
                "has_inference": has_inference(answer_text), "strictness": self.strictness,
                "stats": stats.to_dict(),
                "stats_summary": f"Обзор: {len(parts)} частей, {timings.get('total_s', 0):.0f} с всего · " + stats.summary_line(),
                "answered_model": getattr(stats, "model", None),
                "fallback_note": getattr(stats, "fallback_note", None),
                "warnings": warns,
                "experiment": {
                    "kind": "rag", "mode": "overview",
                    "app": {"version": __version__, "date": _dt.date.today().isoformat()},
                    "model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
                    "reasoning": {"setting": self.meta.get("reasoning_setting"), "effort": self.reasoning_effort},
                    "context_length": self.max_context_length, "preset": self.prompts.preset,
                    "prompts": {"system": self.system_prompt, "user": user_message, "map_system": OVERVIEW_MAP_PROMPT,
                                "rewrite_system": rewrite_prompt, "rewrite_user": question if rewrite_prompt else None},
                    "options": dict(o), "plan": plan_out, "overview": ov_info,
                    "timings": dict(timings), "stats": stats.to_dict(),
                }}

    def _followup(self, question: str, draft: str, fitted: list[dict], budget: int, lang: str,
                  make_message: Callable[[list[dict]], str], report, rw_llm, rw_model: str) -> dict:
        """«Дозапрос недостающих фактов»: the model names the facts the first answer lacks, they are searched for
        and the question is answered again with them added. Returns the info for the result/snapshot; on success
        also answer/stats/sources/user_message of the second attempt (the caller takes them out)."""
        report("followup", 0, 1, "В найденных абзацах ответа нет — ищу недостающие факты…")
        queries = plan_followup(rw_llm, rw_model, question, draft, timeout=min(self.timeout, 90),
                                prompt=self.prompts.text("rag_followup"), sections=section_titles(fitted))
        info: dict = {"queries": queries, "added": [], "first_answer": draft}
        if not queries:
            report("followup", 1, 1, "Модель не назвала недостающих фактов — остаётся первый ответ")
            return info
        known = {p["id"] for p in fitted}
        added: list[dict] = []
        for q in queries:
            try:
                found = self.searcher.hybrid(q, k=FOLLOWUP_PER_QUERY * 10)
            except Exception:  # noqa: BLE001
                logger_rag.exception("follow-up search failed: %s", q)
                continue
            if _DATE_WORD_RE.search(q):   # «год / дата …»: the paragraphs where a date is really written go first
                found = sorted(found, key=lambda r: not has_date(r))
            fresh = [r for r in found if r["id"] not in known][:FOLLOWUP_PER_QUERY]
            known.update(r["id"] for r in fresh)
            added += fresh
        if not added:
            report("followup", 1, 1, "Ищу: " + "; ".join(queries) + "\nНовых абзацев не нашлось — остаётся первый ответ")
            return info
        extra = fit_context(added, budget, reserved_tokens=0, lang=lang)
        rest = fit_context(fitted, budget - estimate_tokens(build_context_block(extra, lang)), reserved_tokens=0, lang=lang)
        sources = rest + extra
        info["added"] = [p["id"] for p in extra]
        message = make_message(sources) + FOLLOWUP_NOTE.replace("{facts}", "; ".join(queries))
        report("followup", 0, 1, "Ищу: " + "; ".join(queries)
               + f"\nДобавлено абзацев: {len(extra)} ({', '.join(info['added'])}) · отвечаю заново ({self.model})…")
        try:
            text, stats = self.llm.chat_with_stats(
                [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": message}],
                model=self.model, temperature=self.temperature, max_tokens=self.max_tokens, timeout=self.timeout,
                reasoning_effort=self.reasoning_effort)
        except LMStudioError as exc:
            info["error"] = str(exc)
            report("followup", 1, 1, f"ошибка второго прохода: {exc} — остаётся первый ответ")
            return info
        report("followup", 1, 1, stats.summary_line())
        info.update(answer=text, stats=stats, sources=sources, user_message=message)
        return info

    def answer(self, question: str, k: Optional[int] = None,
               progress: Optional[Callable[[str, int, int, Optional[int], str], None]] = None,
               overview: Optional[bool] = None) -> dict:
        """overview: True - «обзор по всей книге», False - never, None - decide (list question + spread main word)."""
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

        # --- «обзор по всей книге» -------------------------------------------------------------
        ov_sections = self._overview_sections(merged)
        if overview is None:
            # not when the question names a section of the book: then the whole list is in one place
            overview = bool(o["rag_overview_auto"]) and is_list_question(question, plan) \
                and len(ov_sections) >= OVERVIEW_MIN_SECTIONS and not match_section_titles(self.conn, question, plan)
        if overview:
            return self._overview(question, plan, merged, ov_sections, report, t_total, timings, rewrite_prompt)

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
        msg = f"использовано источников: {len(fitted)}"
        if list_q:
            msg += "\nВопрос-перечень: передаю модели разделы целиком"
        for e in expanded:
            how = "целиком" if e["whole"] else f"{e['added']} из {e['total']} абзацев (не влез целиком)"
            msg += f"\nРаздел {e['id']}" + (f" «{e['title']}»" if e.get("title") else "") + f": {how}"
        report("context", 1, 1, msg)
        def make_message(srcs: list[dict]) -> str:
            m = prompts_mod.render(user_tmpl, sources=build_context_block(srcs, lang), question=question,
                                   inference_reminder=reminder)
            return m + LIST_REMINDER if list_q else m

        user_message = make_message(fitted)
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
        followup = None   # «дозапрос недостающих фактов», set after the first answer

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
                "followup": followup,
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

        if o["rag_followup"] and inference_on and answer_has_gap(answer_text):
            t0 = time.monotonic()
            followup = self._followup(question, answer_text, fitted, budget, lang, make_message, report, rw_llm, rw_model)
            timings["followup_s"] = round(time.monotonic() - t0, 2)
            if followup.get("answer"):
                answer_text, stats = followup.pop("answer"), followup.pop("stats")
                fitted, user_message = followup.pop("sources"), followup.pop("user_message")
                used_ids.append("rag_followup")

        cited = extract_citations(answer_text)
        known = {p["id"] for p in fitted}
        warns = list(self.prompts.warnings)
        if followup and followup.get("error"):
            warns.append(f"Второй проход (дозапрос фактов) не удался: {followup['error']}. Показан первый ответ.")
        return {
            "followup": followup,
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
