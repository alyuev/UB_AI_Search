"use strict";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

// ------------------------------------------------------------- theme -----
(function initTheme() {
  const saved = localStorage.getItem("theme");
  if (saved) document.documentElement.setAttribute("data-theme", saved);
  $("#theme-toggle").addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", cur);
    localStorage.setItem("theme", cur);
  });
})();

// -------------------------------------------------------------- tabs -----
$$(".tab-btn[data-tab]").forEach((btn) => {
  btn.addEventListener("click", () => {
    $$(".tab-btn[data-tab]").forEach((b) => b.classList.remove("active"));
    $$(".tab-pane").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $(`#tab-${btn.dataset.tab}`).classList.add("active");
    if (btn.dataset.tab === "prepare") loadPrepareTab();
    if (btn.dataset.tab === "settings") loadSettingsTab();
  });
});

async function api(path, opts) {
  let res;
  try {
    res = await fetch(path, opts);
  } catch (e) {   // the browser got no answer at all ("Failed to fetch"): the program is not running
    throw new Error("нет связи с программой — похоже, она закрыта или перезапускается. Запустите start.bat и обновите страницу.");
  }
  if (!res.ok) {
    let msg = res.statusText;
    try { const j = await res.json(); msg = j.detail || msg; } catch (e) {}
    throw new Error(msg);
  }
  return res.json();
}

function escapeHtml(s) {
  return (s || "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function highlight(text, q) {
  if (!q) return escapeHtml(text);
  const esc = escapeHtml(text);
  try {
    const re = new RegExp("(" + q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "ig");
    return esc.replace(re, "<mark>$1</mark>");
  } catch (e) { return esc; }
}

// ------------------------------------------------------ status panel -----
const STAGE_LABELS = {
  analysis: "Подготовка запроса", fts: "Поиск по словам", semantic: "Поиск по смыслу",
  hybrid: "Объединение", context: "Контекст", generation: "Генерация ответа",
  followup: "Дозапрос фактов",
  "Сравнение EN/RU": "Сравнение EN/RU",
  "Анализ смысловых расхождений": "Анализ расхождений",
  "Формирование отчёта": "Формирование отчёта",
  // real app.core.indexer stage names
  parsing: "Разбор книги", database: "Запись абзацев", lemmatize: "Лемматизация RU",
  embedding: "Эмбеддинги", finalize: "Финализация",
};

function renderStages(panel, job) {
  const stagesEl = panel.querySelector(".stages");
  stagesEl.innerHTML = (job.stages || []).map((s) => {
    const dot = s.state === "done" ? "✓" : s.state === "active" ? "●" : "○";
    return `<span class="stage-item ${s.state}"><span class="dot">${dot}</span>${STAGE_LABELS[s.name] || s.name}</span>`;
  }).join("");
  const bar = panel.querySelector(".progress-bar-inner");
  if (bar) bar.style.width = `${Math.round((job.progress || 0) * 100)}%`;
  const spinner = panel.querySelector(".spinner");
  if (spinner) spinner.classList.toggle("hidden", job.status !== "running");
  // For the embedding stage specifically, the core indexer's `message` says
  // "embedding en" / "embedding ru" -- surface that as an explicit EN/RU pass badge.
  let passBadge = "";
  const msgLower = (job.message || "").toLowerCase();
  if (job.stage === "embedding") {
    if (msgLower.includes("embedding en")) passBadge = " · Проход: EN";
    else if (msgLower.includes("embedding ru")) passBadge = " · Проход: RU";
  }
  panel.querySelector(".msg").textContent = (job.message || "") + passBadge;
  const counters = panel.querySelector(".counters");
  if (counters) {
    const eta = job.eta != null ? `ETA: ${Math.round(job.eta)}с` : "";
    const rate = job.rate != null ? `${job.rate} текст./с` : "";
    counters.textContent = `${job.current || 0}/${job.total || 0}  Прошло: ${Math.round(job.elapsed || 0)}с  ${rate}  ${eta}`;
  }
  const logEl = panel.querySelector(".job-log");
  if (logEl) logEl.textContent = (job.log || []).join("\n");
}

function pollJob(jobId, panel, onDone) {
  panel.classList.remove("hidden");
  const timer = setInterval(async () => {
    try {
      const job = await api(`/api/jobs/${jobId}`);
      renderStages(panel, job);
      if (["done", "error", "cancelled"].includes(job.status)) {
        clearInterval(timer);
        onDone(job);
      }
    } catch (e) {
      clearInterval(timer);
      onDone({ status: "error", error: String(e) });
    }
  }, 500);
  return timer;
}

// ------------------------------------------------------------ search -----
let lastQuery = "";

$("#query-btn").addEventListener("click", () => runQuery());
$("#query-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); runQuery(); }
});

// The query box grows in height with a long query (up to 40% of the window, then it scrolls).
function autosizeQuery() {
  const box = $("#query-input");
  box.style.height = "auto";
  const max = window.innerHeight * 0.4;
  box.style.height = Math.min(box.scrollHeight + 2, max) + "px";
  box.style.overflowY = box.scrollHeight + 2 > max ? "auto" : "hidden";
}
$("#query-input").addEventListener("input", autosizeQuery);
window.addEventListener("resize", autosizeQuery);

function currentQMode() {
  return $("#qmode-select").value;
}

function syncSubmodeVisibility() {
  $("#search-submode").style.display = currentQMode() === "search" ? "flex" : "none";
}
$("#qmode-select").addEventListener("change", syncSubmodeVisibility);
syncSubmodeVisibility();

// replayId: id of a history entry -> re-run with EXACTLY its saved prompts/parameters (rag / translation only).
async function runQuery(replayId = null) {
  const q = $("#query-input").value.trim();
  if (!q) return;
  setLastQuery(q);
  lastView = null;
  currentHistoryEntry = null;
  updateHistoryNote();
  const replay = replayId != null ? { replay: replayId } : {};
  const mode = currentQMode();
  const results = $("#results");
  results.innerHTML = "";
  const panel = $("#status-panel");
  panel.classList.remove("hidden");
  renderStages(panel, { stages: [], progress: 0, message: "Выполняется…", status: "running" });

  if (mode === "search") {
    const smode = $$('input[name="smode"]').find((r) => r.checked).value;
    try {
      renderStages(panel, { stages: [{ name: "fts", state: "active" }], progress: 0.5, message: "Поиск…", status: "running" });
      const books = selectedBooks();
      const data = await api(`/api/search?q=${encodeURIComponent(q)}&mode=${smode}&k=10${books.length ? "&book_ids=" + encodeURIComponent(books.join(",")) : ""}`);
      renderStages(panel, { stages: [{ name: "fts", state: "done" }], progress: 1, message: `Найдено: ${data.results.length}`, status: "done" });
      const warn = (data.warnings || []).map((w) => `<div class="msg-box info">${escapeHtml(bookNameOf(w.book_id))}: ${escapeHtml(w.message)}</div>`).join("");
      renderResults(data.results, warn, (data.book_ids || []).length > 1);
      afterQuery();
    } catch (e) {
      renderStages(panel, { stages: [], progress: 1, message: "Ошибка: " + e.message, status: "error" });
    }
  } else if (mode === "rag" || mode === "overview") {
    try {
      const { job_id } = await api("/api/jobs/rag", {
        method: "POST", headers: { "Content-Type": "application/json" },
        // no `k`: the server uses the «Сколько абзацев передавать модели» setting;
        // «Обзор по всей книге» forces the overview, otherwise the server decides
        body: JSON.stringify({ question: q, book_ids: selectedBooks(), ...replay, ...(mode === "overview" ? { overview: true } : {}) }),
      });
      pollJob(job_id, panel, (job) => {
        afterQuery();
        if (job.status === "done") renderRagAnswer(job.result);
        else if (job.result && job.result.answer) renderRagAnswer(job.result, job.error);   // broken off: show what arrived
        else $("#results").innerHTML = `<div class="card issue-card critical"><strong>Ошибка:</strong> ${escapeHtml(job.error || "неизвестна")}</div>`;
      });
    } catch (e) {
      renderStages(panel, { stages: [], progress: 1, message: "Ошибка: " + e.message, status: "error" });
    }
  } else if (mode === "translation") {
    try {
      const { job_id } = await api("/api/jobs/translation-analysis", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reference: q, ...replay }),
      });
      pollJob(job_id, panel, (job) => {
        afterQuery();
        if (job.status === "done") renderTranslationReport(job.result, job.job_id);
        else results.innerHTML = `<div class="card">Ошибка: ${escapeHtml(job.error || "неизвестна")}</div>`;
      });
    } catch (e) {
      renderStages(panel, { stages: [], progress: 1, message: "Ошибка: " + e.message, status: "error" });
    }
  }
}

function titleLine(p) {
  const parts = [];
  if (p.paper_title) parts.push(`«${escapeHtml(p.paper_title.ru || p.paper_title.en || "")}»`);
  if (p.section_title) parts.push(`— ${escapeHtml(p.section_title.ru || p.section_title.en || "")}`);
  return parts.length ? `<div class="title-line">${parts.join(" ")}</div>` : "";
}

// Card header: "Paper title · Section title" in the language of the query.
// Titles link to the reading page (read.html) with the query, so the found words stay highlighted there.
function crumbs(p) {
  const pick = (t) => (t ? escapeHtml((queryIsRu() ? t.ru || t.en : t.en || t.ru) || "") : "");
  const paper = pick(p.paper_title), section = pick(p.section_title);
  if (!paper && !section) return "";
  const pa = paper ? `<a class="read-link" href="${readHref({ paper: p.paper, book: p.book_id })}" title="Читать весь документ"><b>${paper}</b></a>` : "";
  const se = section ? `<a class="read-link" href="${readHref({ paper: p.paper, section: p.section, book: p.book_id })}" title="Читать с начала раздела">${section}</a>` : "";
  return `<span class="crumbs">${pa}${paper && section ? " · " : ""}${se}</span>`;
}

function readHref({ paper, section = null, para = null, book = "" }) {
  const h = new URLSearchParams();
  h.set("p", paper);
  if (section != null) h.set("s", section);
  if (para) h.set("para", para);
  if (lastQuery) h.set("q", lastQuery);
  if (book) h.set("b", book);
  return `read.html#${h.toString()}`;
}

// Position of the paragraph in the word (FTS) or meaning (semantic) ranking; top-3 highlighted.
function rankBadge(label, n) {
  if (n == null) return "";
  const what = label === "слова" ? "поиске по словам" : "поиске по смыслу";
  return `<span class="rank${n <= 3 ? " top" : ""}" title="Место в ${what}">${label} №${n}</span>`;
}

const queryIsRu = () => /[а-яё]/i.test(lastQuery);
// The language of the query goes first in the EN/RU columns (see .ru-first in style.css).
function setLastQuery(q) {
  lastQuery = q || "";
  document.body.classList.toggle("ru-first", queryIsRu());
}

// One language column of a result card. `snippet` is server-side HTML-escaped text that
// contains only <mark> tags (safe for innerHTML); when absent (semantic-only hits, no FTS
// match in this language) fall back to the full text with client-side highlighting.
// `fullHtml` is the book's inline formatting (italics etc.); the snippet always stays plain text.
function langBlock(cls, snippet, full, fullHtml) {
  const fullShown = RichText.render(full, fullHtml, (t) => highlight(t, lastQuery));
  if (snippet) {
    return `<div class="${cls}"><span class="snip">${snippet}</span><span class="full">${fullShown}</span></div>`;
  }
  return `<div class="${cls}">${fullShown}</div>`;
}

// Show the book name on a hit only when hits of several books can meet (set by renderResults).
let badgeBooks = false;

// The EN/RU columns of one paragraph; a one-language book (one of them empty) gets a single wide column.
function langsInner(p, withSnippets) {
  const hasEn = !!(p.en || "").trim(), hasRu = !!(p.ru || "").trim();
  const col = (cls, snip, full, html) => (withSnippets ? langBlock(cls, snip, full, html)
    : `<div class="${cls}">${RichText.render(full, html, escapeHtml)}</div>`);
  const en = hasEn || !hasRu ? col("en", p.snippet_en, p.en, p.en_html) : "";
  const ru = hasRu || !hasEn ? col("ru", p.snippet_ru, p.ru, p.ru_html) : "";
  return { html: en + ru, mono: !(hasEn && hasRu) };
}

function cardHtml(p) {
  const hasSnip = !!(p.snippet_en || p.snippet_ru);
  const langs = langsInner(p, true);
  const badge = badgeBooks && p.book_name ? `<span class="book-badge" title="Книга: ${escapeHtml(p.book_name)}">${escapeHtml(p.book_name)}</span>` : "";
  const tint = badgeBooks && p.book_id ? ` book-tinted" style="--bk:${bookColor(p.book_id)}` : "";
  return `
    <div class="card${tint}" data-id="${p.id}" data-book="${escapeHtml(p.book_id || "")}">
      <div class="card-head">
        ${badge}
        <a class="ref read-link" href="${readHref({ paper: p.paper, para: p.id, book: p.book_id })}" title="Открыть абзац в тексте документа (документ ${p.paper}, раздел ${p.section}, абзац ${p.paragraph})">${p.id}</a>
        ${crumbs(p)}
        <span class="ranks" title="score: ${(p.score ?? 0).toFixed?.(3) ?? p.score}">${rankBadge("слова", p.fts_rank)}${rankBadge("смысл", p.sem_rank)}</span>
      </div>
      <div class="card-langs${langs.mono ? " mono" : ""}">${langs.html}</div>
      <div class="card-actions">
        <button class="secondary ctx-btn" data-id="${p.id}">Контекст</button>
        ${hasSnip ? `<button class="secondary snip-btn">Весь абзац</button>` : ""}
      </div>
      <div class="context-block" style="display:none"></div>
    </div>`;
}

function bindCardHandlers(root) {
  root.querySelectorAll(".ctx-btn").forEach((btn) => btn.addEventListener("click", () => toggleContext(btn)));
  root.querySelectorAll(".snip-btn").forEach((btn) => btn.addEventListener("click", () => {
    const card = btn.closest(".card");
    card.classList.toggle("expanded");
    btn.textContent = card.classList.contains("expanded") ? "Фрагмент" : "Весь абзац";
  }));
}

// Search results as the server ranked them; kept so that changing «Сортировка» re-sorts what is shown.
let lastSearchItems = null;

function sortedForView(items) {
  if ($("#search-sort").value !== "para") return items;
  return [...items].sort((a, b) => (a.paper - b.paper) || (a.section - b.section) || (a.paragraph - b.paragraph));
}

function renderResults(items, prefixHtml = "", badges = null) {
  const results = $("#results");
  lastView = null;
  lastSearchItems = items;
  badgeBooks = badges != null ? badges : (new Set(items.map((p) => p.book_id)).size > 1 || selectedBooks().length > 1);
  if (!items.length) { results.innerHTML = prefixHtml + `<div class="card">Ничего не найдено.</div>`; return; }
  results.innerHTML = prefixHtml + sortedForView(items).map(cardHtml).join("");
  bindCardHandlers(results);
}

$("#search-sort").addEventListener("change", () => {
  // only while the shown list is a word/meaning search result (not a RAG answer or a report)
  // (RAG answers, reports and comparisons set lastView; plain search results leave it null)
  if (lastView === null && lastSearchItems && lastSearchItems.length && $("#results .card[data-id]")) renderResults(lastSearchItems);
});

async function toggleContext(btn) {
  const card = btn.closest(".card");
  const block = card.querySelector(".context-block");
  if (block.style.display !== "none") { block.style.display = "none"; return; }
  const id = btn.dataset.id;
  const book = card.dataset.book || "";
  const data = await api(`/api/paragraph/${id}/context?before=2&after=2${book ? "&book_id=" + encodeURIComponent(book) : ""}`);
  block.innerHTML = data.context.map((p) => {
    const l = langsInner(p, false);
    return `<div class="card-langs${l.mono ? " mono" : ""}" data-id="${p.id}" style="margin-top:6px;${String(p.id) === id ? "font-weight:600" : ""}">${l.html}</div>`;
  }).join("");
  block.style.display = "block";
}

// Turn "[0:1.2]" / "[0:1.2, 0:2.10]" in the (escaped) answer into clickable citations.
function linkifyCitations(escapedAnswer) {
  return escapedAnswer.replace(/\[([\d:.,;\s]+)\]/g, (m, inner) =>
    "[" + inner.split(/\s*[,;]\s*/).map((id) => /^\d+:\d+\.\d+$/.test(id.trim())
      ? `<span class="cite" data-id="${id.trim()}" title="Показать абзац">${id.trim()}</span>` : escapeHtml(id)).join(", ") + "]");
}

// Scroll to a source card (opening the collapsed "other paragraphs" block if needed) and highlight it.
function jumpToSource(id) {
  const card = document.querySelector(`#results .card[data-id="${CSS.escape(id)}"]`);
  if (!card) return;
  const det = card.closest("details");
  if (det) det.open = true;
  card.classList.add("expanded", "flash");
  card.scrollIntoView({ behavior: "smooth", block: "center" });
  setTimeout(() => card.classList.remove("flash"), 2200);
}

// Answer text -> HTML: clickable citations, «Допущение:» lines in italics and the final «Вывод:» block
// (a deduction from several facts, not a statement of the book) in its own labelled box.
const mdOn = true;  // model answers are always rendered as Markdown (the on/off checkbox was removed)
let lastView = null;   // function that re-renders the current answer (used by the Markdown toggle)

// Concrete values the model put in `backticks` are <code> -> CSS chips; short **bold** text with a digit
// (older answers, models that ignore the backticks rule) gets the same chip look - only a value, not a
// bold heading like «Лагерь в Вифсаиде (май-октябрь 28 г.):».
function chipify(html) {
  return html.replace(/<strong>([^<:]{1,30})<\/strong>/g, (m, t) =>
    /\d/.test(t) && t.trim().split(/\s+/).length <= 4 ? `<strong class="fact-chip">${t}</strong>` : m);
}

// The opening «Коротко: …» line (the result of a derived answer) goes into its own box above the answer.
function renderAnswerHtml(text) {
  const s = /^[ \t>*_#]*Коротко\s*:\**[ \t]*([^\n]*)(?:\n|$)/.exec(text || "");
  if (!s) return chipify(renderAnswerBody(text));
  const line = s[1].replace(/[*_]+$/, "").trim();
  const rest = text.slice(s[0].length).replace(/^\s+/, "");
  return `<div class="answer-summary"><span class="summary-label">Коротко</span>` +
    `<span class="summary-text">${chipify(Md.render(line, { inline: linkifyCitations }).replace(/^<p>|<\/p>$/g, ""))}</span></div>` +
    chipify(renderAnswerBody(rest));
}

function renderAnswerBody(text) {
  const fmt = mdOn
    ? (t) => Md.render(t, { inline: linkifyCitations, assumption: true })
    : (t) => linkifyCitations(escapeHtml(t)).replace(
        /(^|\n)([ \t*_>-]*)(Допущение\s*:[^\n]*)/g, '$1$2<em class="assumption">$3</em>');
  const m = /(^|\n)[ \t>*_#-]*Вывод\s*:\**/.exec(text);
  if (!m) return fmt(text);
  const start = m.index + (m[1] ? m[1].length : 0);
  const head = text.slice(0, start).replace(/\s+$/, "");
  const body = text.slice(start + m[0].length - (m[1] ? m[1].length : 0)).replace(/^\s+/, "");
  return (head ? fmt(head) + (mdOn ? "" : "\n\n") : "") +
    `<div class="inference-box"><div class="inference-label">Вывод из источников <span>(это рассуждение по найденным фактам, а не прямая цитата книги)</span></div>${fmt(body)}</div>`;
}

// «Модель: …» above the answer: which LLM actually produced this result
function modelLineHtml(result) {
  const pr = (result && result.provider) || {};
  const m = (result && result.answered_model) || pr.model;
  if (!m) return "";
  const where = pr.is_cloud ? `☁ ${escapeHtml(pr.name || "облако")}` : `🖥 ${escapeHtml(pr.name || "локально")}`;
  const backup = result.answered_model && pr.model && result.answered_model !== pr.model
    ? ` <span class="hint-text">(запасная; основная — ${escapeHtml(pr.model)})</span>` : "";
  return `<div class="model-line${pr.is_cloud ? " cloud" : ""}">Модель: <strong>${escapeHtml(m)}</strong>${backup} · ${where}</div>`;
}

function renderRagAnswer(result, errText) {
  badgeBooks = false;   // a question is answered over one book
  const results = $("#results");
  lastView = () => renderRagAnswer(result, errText);
  const brokenText = errText || (result && result.truncated ? result.error : "");
  if (!result || (result.error && !result.answer)) {
    results.innerHTML = `<div class="card">Ошибка LLM: ${escapeHtml(result?.error || "неизвестна")}</div>`;
    return;
  }
  const sources = result.sources || [];
  const byId = new Map(sources.map((p) => [p.id, p]));
  const cited = (result.cited_ids || []).filter((id) => byId.has(id));
  const citedSet = new Set(cited);
  const others = sources.filter((p) => !citedSet.has(p.id));

  const brokenHtml = brokenText
    ? `<div class="issue-card critical"><strong>Ответ оборван.</strong> Ниже то, что модель успела передать; он может быть неполным. Причина: ${escapeHtml(brokenText)}</div>` : "";
  const fallbackHtml = result.fallback_note ? `<div class="hint-warning">${escapeHtml(result.fallback_note)}</div>` : "";
  const noteHtml = brokenHtml + fallbackHtml + (result.model_note ? `<div class="hint-warning">${escapeHtml(result.model_note)}</div>` : "") +
    (result.replay_note ? `<div class="info-note">${escapeHtml(result.replay_note)}</div>` : "") +
    (result.prompt_warnings || []).map((w) => `<div class="hint-warning">${escapeHtml(w)}</div>`).join("");
  const ov = result.overview;
  const overviewHtml = ov
    ? `<div class="plan-line">📚 Обзор по всей книге: прочитано разделов ${ov.sections} (абзацев ${ov.paragraphs}) за ${ov.parts} ${ov.parts === 1 ? "часть" : "частей"}` +
      (ov.required ? `, главное слово «${escapeHtml(ov.required)}»` : "") +
      (ov.failed_parts ? ` · не обработано частей: ${ov.failed_parts}` : "") + `</div>` : "";
  const fu = result.followup;
  const followupHtml = fu && (fu.queries || []).length
    ? `<div class="plan-line">🔁 Дозапрос недостающих фактов: ${escapeHtml(fu.queries.join("; "))}` +
      ((fu.added || []).length ? ` · добавлено абзацев: ${fu.added.length}, ответ получен заново` : " · новых абзацев не нашлось") +
      `</div>` : "";
  const planHtml = overviewHtml + followupHtml + (result.plan
    ? `<div class="plan-line">${escapeHtml(result.plan.summary || "")}` +
      (result.plan.semantic_query ? ` · по смыслу: «${escapeHtml(result.plan.semantic_query)}»` : "") + `</div>` +
      ((result.plan.terms_lines || []).length
        ? `<details class="terms-details"><summary>Как проверены слова запроса (по тексту книги)</summary><div class="terms-lines">${result.plan.terms_lines.map(escapeHtml).join("<br>")}</div></details>` : "")
    : `<div class="plan-line">Поиск по тексту вопроса (подготовка запроса не использовалась).</div>`);
  const unknownHtml = (result.unknown_citations || []).length
    ? `<div class="hint-warning">Внимание: ответ ссылается на номера, которых нет среди найденных абзацев (${escapeHtml(result.unknown_citations.join(", "))}) — проверьте их вручную.</div>`
    : "";
  const noCiteHtml = (!cited.length && !result.no_answer)
    ? `<div class="hint-warning">В ответе нет ссылок на найденные абзацы — относитесь к нему осторожно.</div>` : "";

  let html = modelLineHtml(result) + noteHtml + `<div class="rag-answer${mdOn ? " md-on" : ""}">${renderAnswerHtml(result.answer || "")}</div>` + unknownHtml + noCiteHtml +
    `<div class="card rag-meta">${planHtml}` +
    (ov ? "" : `<div class="stats-line">Найдено кандидатов: по словам ${result.candidates?.fts ?? "-"}, по смыслу ${result.candidates?.semantic ?? "-"}; передано модели абзацев: ${sources.length}</div>`) +
    (result.stats_summary ? `<div class="stats-line">${escapeHtml(result.stats_summary)}</div>` : "") + `</div>`;

  if (cited.length) {
    html += `<h3 class="sources-head">Источники, на которые ссылается ответ (${cited.length})</h3>` +
      cited.map((id) => cardHtml(byId.get(id))).join("");
  }
  if (others.length) {
    html += `<details class="other-sources" ${cited.length ? "" : "open"}><summary>Другие найденные абзацы (${others.length})</summary>` +
      others.map(cardHtml).join("") + `</details>`;
  }
  results.innerHTML = html;
  bindCardHandlers(results);
  results.querySelectorAll(".cite").forEach((el) => el.addEventListener("click", () => jumpToSource(el.dataset.id)));
}

// free text of the translation report: Markdown (safe renderer) or plain escaped text
function mdText(t) {
  return mdOn ? `<div class="md-on">${Md.render(t || "")}</div>` : `<div style="white-space:pre-wrap">${escapeHtml(t || "")}</div>`;
}

function renderTranslationReport(result, jobId, exportHref) {
  const results = $("#results");
  lastView = () => renderTranslationReport(result, jobId, exportHref);
  const issues = result.issues || [];
  const noteHtml = (result.model_note ? `<div class="hint-warning">${escapeHtml(result.model_note)}</div>` : "") +
    (result.replay_note ? `<div class="info-note">${escapeHtml(result.replay_note)}</div>` : "");
  const head = modelLineHtml(result) + noteHtml + `<div class="card"><strong>${escapeHtml(result.summary || "")}</strong>
    ${result.stats_summary ? `<div class="stats-line">${escapeHtml(result.stats_summary)}</div>` : ""}
    <div class="card-actions"><a class="secondary" href="${exportHref || `/api/jobs/${jobId}/translation-report.md`}" download>Экспорт в Markdown</a></div></div>`;
  const sevOrder = { critical: 0, major: 1, minor: 2 };
  const body = issues.slice().sort((a, b) => (sevOrder[a.severity] ?? 3) - (sevOrder[b.severity] ?? 3)).map((i) => `
    <div class="card issue-card ${i.severity}">
      <div class="card-head"><span>[${i.id}] ${i.severity.toUpperCase()} — ${escapeHtml(i.category)}</span></div>
      ${titleLine(i)}
      <div><strong>Проблема:</strong></div>${mdText(i.problem)}
      ${i.why_significant ? `<div><strong>Значимость:</strong></div>${mdText(i.why_significant)}` : ""}
      ${i.suggestion ? `<div><strong>Предложение:</strong></div>${mdText(i.suggestion)}` : ""}
      <div class="card-langs" style="margin-top:8px"><div class="en">${escapeHtml(i.en)}</div><div class="ru">${escapeHtml(i.ru)}</div></div>
    </div>`).join("");
  results.innerHTML = head + body;
}

// ------------------------------------------------------------ prepare ----
let prepareLoaded = false;
let embeddingModelsCache = [];
let runtimesCache = [];

function runtimeLabel(rt) {
  return `${rt.name}${rt.available ? "" : " (недоступно)"}`;
}

function populateRuntimeSelect(selectEl) {
  // value = rt.name (e.g. "onnx-cpu"/"onnx-dml"/"llamacpp-vulkan"), which is unique;
  // rt.runtime/rt.device (data-* attrs) are what actually gets sent to the API, since
  // "onnx-cpu" and "onnx-dml" both map to runtime="onnx" with a different device.
  selectEl.innerHTML = runtimesCache.map((rt) =>
    `<option value="${rt.name}" data-runtime="${rt.runtime}" data-device="${rt.device}" title="${escapeHtml(rt.description)}" ${rt.available ? "" : "disabled"}>${escapeHtml(runtimeLabel(rt))}</option>`
  ).join("");
}

function selectedRuntimeDevice(selectEl) {
  const opt = selectEl.selectedOptions[0];
  if (!opt) return { runtime: "auto", device: "auto" };
  return { runtime: opt.dataset.runtime, device: opt.dataset.device };
}

function selectRuntimeByValue(selectEl, runtime, device) {
  // runtime="auto" has no single matching entry in runtimesCache (which lists concrete
  // onnx-cpu/onnx-dml/llamacpp-vulkan options) -- in that case just leave the <select>'s
  // own default (first option) rather than guessing.
  if (!runtime || runtime === "auto") return;
  const match = runtimesCache.find((rt) => rt.runtime === runtime && (!device || device === "auto" || rt.device === device));
  if (match) selectEl.value = match.name;
}

function checkRuntimeWarning(modelName, runtimeSelectEl, warningEl) {
  const { runtime } = selectedRuntimeDevice(runtimeSelectEl);
  const model = embeddingModelsCache.find((m) => m.name === modelName);
  if (!model) { warningEl.style.display = "none"; return; }
  const compat = (model.runtime_compatibility || []).find((c) => c.runtime === runtime);
  if (compat && compat.quality_ok === false) {
    warningEl.textContent = `Внимание: модель «${modelName}» не прошла проверку качества для среды «${runtime}»: ${compat.reason}`;
    warningEl.style.display = "block";
  } else {
    warningEl.style.display = "none";
  }
}

async function initBuildForm() {
  try {
    const files = await api("/api/source-files");
    $("#source-file").innerHTML = files.files.length
      ? files.files.map((f) => `<option value="${escapeHtml(f)}">${escapeHtml(f)}</option>`).join("")
      : `<option value="">(в папке программы нет файлов .json/.md)</option>`;
  } catch (e) { console.error("source-files:", e); }
  try {
    const models = await api("/api/embeddings/models");
    embeddingModelsCache = models.models;
    $("#embed-model-select").innerHTML = embeddingModelsCache.map((m) =>
      `<option value="${m.name}">${m.name} (dim ${m.dim}${m.downloaded ? "" : ", не скачана"})</option>`).join("");
    const settings = await api("/api/settings");
    if (settings.embedding_model) $("#embed-model-select").value = settings.embedding_model;
  } catch (e) { console.error("embedding models:", e); }
  try {
    const runtimes = await api("/api/embeddings/runtimes");
    runtimesCache = runtimes.runtimes;
    populateRuntimeSelect($("#index-runtime"));
    const settings = await api("/api/settings");
    selectRuntimeByValue($("#index-runtime"), settings.embedding_runtime, settings.device);
    checkRuntimeWarning($("#embed-model-select").value, $("#index-runtime"), $("#runtime-warning"));
  } catch (e) { console.error("runtimes:", e); }

  $("#embed-model-select").addEventListener("change", () =>
    checkRuntimeWarning($("#embed-model-select").value, $("#index-runtime"), $("#runtime-warning")));
  $("#index-runtime").addEventListener("change", () => {
    checkRuntimeWarning($("#embed-model-select").value, $("#index-runtime"), $("#runtime-warning"));
    const { device } = selectedRuntimeDevice($("#index-runtime"));
    if (device) $("#index-device").value = device;
  });
  $("#rebuild-all-details").addEventListener("toggle", () => { detailsTouchedByUser = true; });
}

async function loadPrepareTab() {
  if (!prepareLoaded) {
    prepareLoaded = true;
    await initBuildForm();
  }
  await refreshDbStatus();
  await attachRunningJob();
}

// ---------------------------------------------------------- DB status card ---
const STATUS_ICON = { ok: "✓", warn: "⚠", error: "✗", info: "ℹ" };
const BADGE_CLASS = { ready: "ok", attention: "warn", missing: "error" };
let dbStatus = null;
let detailsTouchedByUser = false;
let currentJobId = null;

function showDbMessage(text, kind) {
  const box = $("#db-message");
  box.textContent = text;
  box.className = `msg-box ${kind || "error"}`;
  if (kind === "error") console.error("[db]", text);
}
function hideDbMessage() { $("#db-message").className = "msg-box hidden"; }

async function refreshDbStatus() {
  try {
    dbStatus = await api("/api/db/status");
    renderDbStatus(dbStatus);
  } catch (e) {
    console.error("db status:", e);
    $("#db-badge").textContent = "ошибка";
    $("#db-badge").className = "badge error";
    showDbMessage("Не удалось получить состояние базы: " + e.message, "error");
  }
}

function renderDbStatus(st) {
  const badge = $("#db-badge");
  badge.textContent = (st.state === "ready" ? "✓ " : st.state === "attention" ? "⚠ " : "✗ ") + st.state_label;
  badge.className = `badge ${BADGE_CLASS[st.state] || ""}`;
  $("#db-summary").textContent = st.summary || "";
  $("#db-nodb").classList.toggle("hidden", st.state !== "missing");
  $("#db-rows").innerHTML = st.state === "missing" ? "" : (st.rows || []).map((r) =>
    `<li class="db-row ${r.status}"><span class="ic">${STATUS_ICON[r.status] || "•"}</span><span>${escapeHtml(r.text)}</span></li>`
  ).join("");
  const actionable = (st.warnings || []).filter((w) => w.level !== "info" || w.action);
  $("#db-warnings").innerHTML = st.state === "missing" ? "" : actionable.map((w, i) =>
    `<div class="db-warn ${w.level}"><span>${escapeHtml(w.text)}</span>` +
    (w.action ? `<button type="button" class="secondary small" data-warn-action="${w.action.id}">${escapeHtml(w.action.label)}</button>` : "") +
    `</div>`).join("");
  $$("[data-warn-action]").forEach((b) => b.addEventListener("click", () => doWarnAction(b.dataset.warnAction)));

  // the full build form: expanded when there is no database, collapsed once it is ready
  const det = $("#rebuild-all-details");
  if (st.state === "missing") det.open = true;
  else if (!detailsTouchedByUser) det.open = false;
  if (st.book_id && st.state === "missing") $("#book-id").value = st.book_id;
  setBusy(!!(st.busy));
}

function doWarnAction(id) {
  if (id === "rebuild_fts") {
    $("#rebuild-fts-confirm").classList.remove("hidden");
    $("#rebuild-fts-confirm").scrollIntoView({ behavior: "smooth", block: "center" });
  } else if (id === "build_glossary") {
    $("#build-glossary-btn").scrollIntoView({ behavior: "smooth", block: "center" });
    $("#build-glossary-btn").focus();
  } else if (id === "rebuild_all") {
    detailsTouchedByUser = true;
    const det = $("#rebuild-all-details");
    det.open = true;
    det.scrollIntoView({ behavior: "smooth", block: "start" });
  } else if (id === "open_embeddings") {
    document.querySelector('.tab-btn[data-tab="settings"]').click();
  } else if (id === "select_book") {
    openBooksMenu();
  }
}

function setBusy(busy) {
  for (const sel of ["#rebuild-fts-btn", "#rebuild-fts-yes", "#start-index-btn", "#start-index-yes", "#build-glossary-btn"]) {
    $(sel).disabled = !!busy;
  }
  $("#cancel-index-btn").disabled = !busy;
}

// -------------------------------------------------------------------- jobs ---
const JOB_LABELS = { index: "Построение базы", "rebuild-fts": "Перестроение полнотекстового индекса",
                     "build-glossary": "Построение словаря терминов" };

async function startJob(url, body, label) {
  hideDbMessage();
  try {
    const { job_id } = await api(url, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
    });
    trackJob(job_id, label);
  } catch (e) {
    console.error("start job failed:", url, e);
    showDbMessage(`Не удалось запустить «${label}»: ${e.message}`, "error");
    setBusy(false);
  }
}

function trackJob(jobId, label) {
  currentJobId = jobId;
  setBusy(true);
  $("#cancel-index-btn").dataset.jobId = jobId;
  pollJob(jobId, $("#index-status"), (job) => {
    currentJobId = null;
    const msgEl = $("#index-status .msg");
    if (job.status === "done" && job.result && !job.result.cancelled) {
      const r = job.result;
      const text = r.pairs != null
        ? `${label}: готово — в словаре ${r.pairs} пар слов (надёжных, взаимных: ${r.mutual}).`
        : r.rows != null
        ? `${label}: готово — проиндексировано абзацев: ${r.rows}.`
        : `${label}: готово — ${r.db_path}` + (r.db_size_bytes ? ` (${(r.db_size_bytes / 1e6).toFixed(1)} МБ)` : "");
      msgEl.textContent = text;
      showDbMessage(text, "ok");
    } else if (job.status === "error") {
      const text = `${label}: ошибка — ${job.error || "неизвестная ошибка"}`;
      msgEl.textContent = text;
      showDbMessage(text, "error");
    } else {
      msgEl.textContent = `${label}: отменено (прежние данные не изменены).`;
      showDbMessage(`${label}: отменено, прежние данные не изменены.`, "info");
    }
    refreshDbStatus();
    loadBooks();
    refreshDbBanner();
  });
}

async function attachRunningJob() {
  if (currentJobId) return;
  try {
    const { jobs } = await api("/api/jobs");
    const running = jobs.find((j) => (j.type === "index" || j.type === "rebuild-fts" || j.type === "build-glossary") &&
                                     (j.status === "running" || j.status === "queued"));
    if (running) trackJob(running.job_id, JOB_LABELS[running.type]);
  } catch (e) { console.error("jobs:", e); }
}

// --- rebuild FTS: inline two-step confirmation (no native confirm() dialog) ---
$("#rebuild-fts-btn").addEventListener("click", () => {
  hideDbMessage();
  $("#rebuild-fts-confirm").classList.remove("hidden");
});
$("#rebuild-fts-no").addEventListener("click", () => $("#rebuild-fts-confirm").classList.add("hidden"));
$("#rebuild-fts-yes").addEventListener("click", () => {
  $("#rebuild-fts-confirm").classList.add("hidden");
  startJob("/api/jobs/rebuild-fts", null, JOB_LABELS["rebuild-fts"]);
});

// --- glossary of terms (RU <-> EN): a job with real progress ---
$("#build-glossary-btn").addEventListener("click", () => {
  hideDbMessage();
  startJob("/api/jobs/build-glossary", null, JOB_LABELS["build-glossary"]);
});

// --- full build ---
function collectIndexRequest() {
  const { runtime, device } = selectedRuntimeDevice($("#index-runtime"));
  return {
    json_path: $("#source-file").value,
    model_name: $("#embed-model-select").value,
    runtime: runtime || "auto",
    device: $("#index-device").value || device || "auto",
    vector_dtype: $("#index-vector-dtype").value || "int8",
    embed_batch_size: 32,
    scheme: $("#index-scheme").value,
    book_id: $("#book-id").value.trim(),
    book_name: $("#book-name").value.trim(),
  };
}

$("#start-index-btn").addEventListener("click", () => {
  hideDbMessage();
  const req = collectIndexRequest();
  if (!req.json_path) { showDbMessage("Выберите исходный файл (в папке программы нет файлов .json/.md).", "error"); return; }
  if (!req.book_id) { showDbMessage("Укажите код книги (латиницей, без пробелов).", "error"); return; }
  const replacing = dbStatus && dbStatus.exists && dbStatus.book_id === req.book_id;
  if (replacing) {
    $("#start-index-confirm-text").textContent =
      `База книги «${req.book_id}» уже есть и будет заменена после успешной сборки (около 20 минут). Продолжить?`;
    $("#start-index-confirm").classList.remove("hidden");
  } else {
    startJob("/api/jobs/index", req, JOB_LABELS.index);
  }
});
$("#start-index-no").addEventListener("click", () => $("#start-index-confirm").classList.add("hidden"));
$("#start-index-yes").addEventListener("click", () => {
  $("#start-index-confirm").classList.add("hidden");
  startJob("/api/jobs/index", collectIndexRequest(), JOB_LABELS.index);
});

$("#cancel-index-btn").addEventListener("click", async () => {
  const jobId = $("#cancel-index-btn").dataset.jobId;
  if (!jobId) return;
  try {
    await api(`/api/jobs/${jobId}/cancel`, { method: "POST" });
    $("#index-status .msg").textContent = "Отмена… дождитесь остановки задачи.";
  } catch (e) {
    console.error("cancel failed:", e);
    showDbMessage("Не удалось отменить задачу: " + e.message, "error");
  }
});

$("#db-refresh-btn").addEventListener("click", () => { refreshDbStatus(); });

window.addEventListener("error", (ev) => console.error("[ui error]", ev.message, ev.filename, ev.lineno));
window.addEventListener("unhandledrejection", (ev) => console.error("[ui promise]", ev.reason));

// ------------------------------------------------------------ settings ---
let settingsLoaded = false;
async function loadSettingsTab() {
  const settings = await api("/api/settings");
  $("#set-temperature").value = settings.temperature ?? 0.2;
  $("#set-max-tokens").value = settings.max_tokens ?? 1024;
  $("#set-timeout").value = settings.llm_timeout ?? 120;
  $("#set-device").value = settings.device || "cpu";
  const fw = settings.fts_weights || {};
  for (const k of ["en", "ru", "ru_lemma", "title_en", "title_ru"]) {
    $(`#set-w-${k}`).value = fw[k] ?? 1;
  }
  $("#set-fts-mode-hybrid").value = settings.fts_mode_hybrid || "or";
  $("#set-fts-mode-standalone").value = settings.fts_mode_standalone || "or";
  $("#set-fts-stopwords").checked = settings.fts_stopwords !== false;
  if (window.loadOptionsForm) loadOptionsForm();
  if (window.loadPromptsUi) loadPromptsUi();

  if (!settingsLoaded) {
    settingsLoaded = true;
    try {
      const embedModels = await api("/api/embeddings/models");
      embeddingModelsCache = embedModels.models;
      const opts = embeddingModelsCache.map((m) => `<option value="${m.name}">${m.name}</option>`).join("");
      $("#set-embed-model").innerHTML = opts;
      $("#set-embed-model").value = settings.embedding_model;
    } catch (e) {}
    try {
      const runtimes = await api("/api/embeddings/runtimes");
      runtimesCache = runtimes.runtimes;
      populateRuntimeSelect($("#set-embed-runtime"));
      selectRuntimeByValue($("#set-embed-runtime"), settings.embedding_runtime, settings.device);
      checkRuntimeWarning($("#set-embed-model").value, $("#set-embed-runtime"), $("#set-runtime-warning"));
    } catch (e) {}
    $("#set-embed-model").addEventListener("change", () =>
      checkRuntimeWarning($("#set-embed-model").value, $("#set-embed-runtime"), $("#set-runtime-warning")));
    $("#set-embed-runtime").addEventListener("change", () => {
      checkRuntimeWarning($("#set-embed-model").value, $("#set-embed-runtime"), $("#set-runtime-warning"));
      const { device } = selectedRuntimeDevice($("#set-embed-runtime"));
      if (device) $("#set-device").value = device;
    });
  }
  if (window.loadProvidersUi) loadProvidersUi();
  loadAppInfo();
  $$("#tab-settings select").forEach(syncSelectTitle);
}

// a long option text may not fit a closed <select>: keep the full text in the tooltip
function syncSelectTitle(sel) {
  const o = sel.selectedOptions && sel.selectedOptions[0];
  sel.title = o ? o.textContent : "";
}
document.addEventListener("change", (ev) => { if (ev.target.matches && ev.target.matches("#tab-settings select")) syncSelectTitle(ev.target); });

// settings blocks: open/closed state is remembered per browser; the side menu opens a block and scrolls to it
const SET_OPEN_KEY = "settings_open_blocks";
function saveOpenBlocks() {
  try { localStorage.setItem(SET_OPEN_KEY, JSON.stringify($$("#tab-settings .set-block").filter((d) => d.open).map((d) => d.id))); } catch (e) {}
}
(() => {
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(SET_OPEN_KEY) || "null"); } catch (e) {}
  if (Array.isArray(saved)) $$("#tab-settings .set-block").forEach((d) => { d.open = saved.includes(d.id); });
  $$("#tab-settings .set-block").forEach((d) => d.addEventListener("toggle", saveOpenBlocks));
})();
$("#set-expand-all").addEventListener("click", () => { $$("#tab-settings .set-block").forEach((d) => { d.open = true; }); });
$("#set-collapse-all").addEventListener("click", () => { $$("#tab-settings .set-block").forEach((d) => { d.open = false; }); });
$$("#settings-nav a").forEach((a) => a.addEventListener("click", (ev) => {
  ev.preventDefault();
  const det = document.getElementById(a.dataset.sec);
  if (!det) return;
  det.open = true;
  det.scrollIntoView({ behavior: "smooth", block: "start" });
  $$("#settings-nav a").forEach((x) => x.classList.toggle("active", x === a));
}));

// «Программа»: where the program runs from (folders can be copied with one click)
async function loadAppInfo() {
  const box = $("#app-info");
  if (!box) return;
  try {
    const a = (await api("/api/restart")).app || {};
    const rows = [
      ["Папка программы", a.root, true], ["Запущена файлом", a.launcher, true], ["Рабочая папка", a.cwd, true],
      ["Данные и настройки", a.data_dir, true], ["Журналы", a.logs_dir, true], ["Python", a.python, true],
      ["Версия", a.version], ["Адрес", a.port ? `http://localhost:${a.port}` : location.origin],
      ["Запущена", a.started], ["Номер процесса", a.pid],
    ].filter((r) => r[1] !== undefined && r[1] !== null && r[1] !== "");
    box.innerHTML = rows.map(([k, v, copy]) => `<div class="it-row"><div class="it-k">${escapeHtml(k)}</div>
      <div class="it-v"><code>${escapeHtml(String(v))}</code>${copy ? `<button type="button" class="link-btn" data-copy="${escapeHtml(String(v))}">копировать</button>` : ""}</div></div>`).join("");
    box.querySelectorAll("[data-copy]").forEach((b) => b.addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(b.dataset.copy); b.textContent = "скопировано ✓"; }
      catch (e) { b.textContent = "не удалось"; }
      setTimeout(() => { b.textContent = "копировать"; }, 1500);
    }));
  } catch (e) {
    box.innerHTML = `<span class="err-text">Не удалось получить сведения: ${escapeHtml(e.message)}</span>`;
  }
}

function setStatus(el, text, kind) { el.textContent = text; el.className = "save-status" + (kind ? " " + kind : ""); }

$("#save-llm-settings-btn").addEventListener("click", async () => {
  const st = $("#llm-settings-status");
  try {
    await api("/api/settings", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        temperature: parseFloat($("#set-temperature").value),
        max_tokens: parseInt($("#set-max-tokens").value, 10),
        llm_timeout: parseInt($("#set-timeout").value, 10),
      }),
    });
    setStatus(st, "✓ Сохранено", "ok");
  } catch (e) {
    setStatus(st, "✗ " + e.message, "err");
  }
});

$("#load-embed-btn").addEventListener("click", async () => {
  const model_name = $("#set-embed-model").value;
  const { runtime, device: rtDevice } = selectedRuntimeDevice($("#set-embed-runtime"));
  const device = $("#set-device").value || rtDevice || "auto";
  $("#embed-status").textContent = "Загрузка…";
  try {
    const res = await api("/api/embeddings/load", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_name, runtime: runtime || "auto", device }),
    });
    $("#embed-status").textContent = `✓ ${model_name}, runtime=${res.runtime}, dim=${res.dim}, статус=${res.status}`;
  } catch (e) {
    $("#embed-status").textContent = "✗ " + e.message;
  }
});


$("#save-search-settings-btn").addEventListener("click", async () => {
  const status = $("#search-settings-status");
  const weights = {};
  for (const k of ["en", "ru", "ru_lemma", "title_en", "title_ru"]) {
    weights[k] = parseFloat($(`#set-w-${k}`).value);
  }
  setStatus(status, "Сохранение…");
  try {
    await api("/api/settings", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        fts_weights: weights,
        fts_stopwords: $("#set-fts-stopwords").checked,
        fts_mode_hybrid: $("#set-fts-mode-hybrid").value,
        fts_mode_standalone: $("#set-fts-mode-standalone").value,
      }),
    });
    setStatus(status, "✓ Сохранено, поисковый индекс пересоздан", "ok");
  } catch (e) {
    setStatus(status, "✗ " + e.message, "err");
  }
});


// ------------------------------------------------------------ book picker ---
// The list of books with checkboxes: search covers the ticked books (saved on the server, «search_books»);
// the «основная» book is the one for reading, the table of contents and the «База данных» tab.
let booksData = { books: [], search_books: [], active_book: null };
// Stable per-book colour (by position in the library list) to tell hits of different books apart.
const BOOK_COLORS = ["#2f7dd1", "#d9822b", "#2e9e6a", "#b4489c", "#c9a227", "#16a2b8", "#d04545", "#6f5bd6"];
function bookColor(id) {
  const i = (booksData.books || []).findIndex((b) => b.book_id === id);
  if (i >= 0 && booksData.books[i].color) return booksData.books[i].color;   // chosen in «О книге»
  return BOOK_COLORS[(i < 0 ? 0 : i) % BOOK_COLORS.length];
}
const bookNameOf = (id) => (booksData.books.find((b) => b.book_id === id) || {}).book_name || id;
const selectedBooks = () => booksData.books.filter((b) => b.selected).map((b) => b.book_id);

function setSelectedBooks(ids) {
  const known = new Set(booksData.books.map((b) => b.book_id));
  const want = new Set(ids.filter((i) => known.has(i)));
  if (!want.size) return;
  booksData.books.forEach((b) => { b.selected = want.has(b.book_id); });
  renderBooksMenu();
  saveBookSelection();
}

let saveSelTimer = null;
function saveBookSelection() {
  clearTimeout(saveSelTimer);
  saveSelTimer = setTimeout(() => {
    api("/api/books/search-selection", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ book_ids: selectedBooks() }),
    }).catch((e) => console.error("save book selection:", e));
  }, 250);
}

function bookMeta(b) {
  const lang = b.bilingual ? "EN+RU" : ({ ru: "рус", en: "англ" }[b.language] || b.language || "");
  return [lang, b.paragraphs != null ? `${b.paragraphs.toLocaleString("ru-RU")} абз.` : ""].filter(Boolean).join(" · ");
}

function renderBooksSummary() {
  const bs = booksData.books, sel = bs.filter((b) => b.selected);
  let txt = "(нет баз)";
  if (bs.length) txt = sel.length === bs.length && bs.length > 1 ? `все (${bs.length})` : sel.length === 1 ? sel[0].book_name : `${sel.length} из ${bs.length}`;
  $("#books-summary").textContent = txt;
  $("#books-btn").title = sel.map((b) => b.book_name).join("\n") || "Книги, по которым идёт поиск";
}

// Order of the list: fixed when the menu opens (ticked books first, each group by name), so ticking a box
// doesn't reshuffle rows under the mouse. Books added later go to the end until the next opening.
let booksOrder = [];
function sortBooksForMenu() {
  const byName = (a, b) => a.book_name.localeCompare(b.book_name, "ru", { sensitivity: "base" });
  const bs = booksData.books;
  booksOrder = [...bs.filter((b) => b.selected).sort(byName), ...bs.filter((b) => !b.selected).sort(byName)].map((b) => b.book_id);
}
function orderedBooks() {
  const bs = booksData.books, pos = new Map(booksOrder.map((id, i) => [id, i]));
  return [...bs].sort((a, b) => (pos.has(a.book_id) ? pos.get(a.book_id) : 1e9) - (pos.has(b.book_id) ? pos.get(b.book_id) : 1e9));
}
const normFilter = (s) => (s || "").toLowerCase().replace(/ё/g, "е").trim();
function visibleBooks() {
  const words = normFilter($("#bm-filter") ? $("#bm-filter").value : "").split(/\s+/).filter(Boolean);
  return orderedBooks().filter((b) => {
    const hay = normFilter(b.book_name + " " + b.book_id);
    return words.every((w) => hay.includes(w));
  });
}

function renderBooksMenu() {
  const menu = $("#books-menu");
  renderBooksSummary();
  if (!booksData.books.length) { menu.innerHTML = `<div class="bm-foot">В библиотеке нет книг. Нажмите «+», чтобы добавить.</div>`; return; }
  if (!$("#bm-filter")) {   // skeleton once: the filter box keeps its text and focus while the list redraws
    menu.innerHTML = `
      <div class="bm-head">
        <input id="bm-filter" type="search" placeholder="Быстрый поиск книги…" autocomplete="off" spellcheck="false" />
        <div class="bm-row bm-all"><label><input type="checkbox" id="bm-all"> <span id="bm-all-text">Все книги</span></label></div>
      </div>
      <div class="bm-list" id="bm-list"></div>
      <div class="bm-foot">Поиск идёт по отмеченным книгам. «Вопрос к тексту» отвечает по одной книге (основной, если она отмечена).</div>`;
    $("#bm-filter").addEventListener("input", renderBooksList);
    $("#bm-filter").addEventListener("keydown", (ev) => { if (ev.key === "Escape" && ev.target.value) { ev.stopPropagation(); ev.target.value = ""; renderBooksList(); } });
    $("#bm-all").addEventListener("change", (ev) => {
      // with a filter typed, «все» means all books shown by the filter
      const vis = visibleBooks(), bs = booksData.books;
      vis.forEach((b) => { b.selected = ev.target.checked; });
      if (!bs.some((b) => b.selected)) (bs.find((b) => b.active) || bs[0]).selected = true;   // at least one book stays
      renderBooksSummary(); renderBooksList(); saveBookSelection();
    });
  }
  renderBooksList();
}

function renderBooksList() {
  const list = $("#bm-list");
  if (!list) return;
  const bs = booksData.books, vis = visibleBooks(), filtered = !!normFilter($("#bm-filter").value);
  const allBox = $("#bm-all");
  allBox.checked = vis.length > 0 && vis.every((b) => b.selected);
  allBox.indeterminate = !allBox.checked && vis.some((b) => b.selected);
  $("#bm-all-text").textContent = filtered ? `Все найденные (${vis.length})` : `Все книги (${bs.length})`;
  list.innerHTML = vis.length ? vis.map((b) => `<div class="bm-row" data-id="${escapeHtml(b.book_id)}">
      <label title="Искать в этой книге"><input type="checkbox" data-sel="${escapeHtml(b.book_id)}" ${b.selected ? "checked" : ""}>
        <span class="bm-dot" style="background:${bookColor(b.book_id)}"></span><span class="bm-name" title="${escapeHtml(b.book_name)}">${escapeHtml(b.book_name)}</span></label>
      <span class="bm-meta">${escapeHtml(bookMeta(b))}</span>
      <button type="button" class="bm-btn" data-info="${escapeHtml(b.book_id)}" title="О книге: файл, название, цвет">ℹ</button>
      ${b.active ? `<span class="bm-main" title="Основная книга: чтение, оглавление, вкладка «База данных»">основная</span>`
        : `<button type="button" class="bm-btn" data-main="${escapeHtml(b.book_id)}" title="Сделать основной (чтение, оглавление, «База данных»)">☆</button>
           <button type="button" class="bm-btn" data-del="${escapeHtml(b.book_id)}" title="Удалить книгу из библиотеки">🗑</button>`}
    </div>`).join("") : `<div class="bm-empty">Ничего не найдено</div>`;
  list.querySelectorAll("[data-sel]").forEach((cb) => cb.addEventListener("change", () => {
    const b = bs.find((x) => x.book_id === cb.dataset.sel);
    b.selected = cb.checked;
    if (!bs.some((x) => x.selected)) { b.selected = true; cb.checked = true; }   // at least one book stays
    renderBooksSummary(); renderBooksList(); saveBookSelection();
  }));
  list.querySelectorAll("[data-main]").forEach((btn) => btn.addEventListener("click", () => makeMainBook(btn.dataset.main)));
  list.querySelectorAll("[data-del]").forEach((btn) => btn.addEventListener("click", () => deleteBook(btn.dataset.del)));
  list.querySelectorAll("[data-info]").forEach((btn) => btn.addEventListener("click", () => openBookInfo(btn.dataset.info)));
}

// ------------------------------------------------------- «О книге» -----
const fmtBytes = (n) => n == null ? "—" : n >= 1048576 ? (n / 1048576).toFixed(1).replace(".", ",") + " МБ" : Math.max(1, Math.round(n / 1024)) + " КБ";
const LANG_RU = { ru: "русский", en: "английский" };
let bookInfoId = null;

async function openBookInfo(book_id) {
  const dlg = $("#book-info-dialog");
  bookInfoId = book_id;
  $("#bi-error").classList.add("hidden");
  $("#bi-facts").innerHTML = "Загрузка…";
  if (!dlg.open) dlg.showModal();
  try {
    const i = await api(`/api/books/${encodeURIComponent(book_id)}/info`);
    $("#bi-name").value = i.book_name;
    $("#bi-color").value = i.color || bookColor(book_id);
    $("#bi-color-auto").classList.toggle("hidden", !i.color);
    const lang = i.translation ? `${LANG_RU[i.language] || i.language} + ${LANG_RU[i.translation] || i.translation}` : (LANG_RU[i.language] || i.language || "—");
    const when = i.created_at ? new Date(i.created_at + (i.created_at.endsWith("Z") ? "" : "Z")).toLocaleString("ru-RU") : "—";
    const row = (k, v) => `<dt>${k}</dt><dd>${v}</dd>`;
    $("#bi-facts").innerHTML = `<dl>` +
      (i.author ? row("Автор", escapeHtml(i.author)) : "") +
      row("Исходный файл", i.source_file ? `<code>${escapeHtml(i.source_file)}</code> ${i.source_exists ? `(${fmtBytes(i.source_size_bytes)})` : `<span class="bi-warn">— файла больше нет</span>`}` : "—") +
      (i.title_in_file && i.title_in_file !== i.book_name ? row("Название в файле", escapeHtml(i.title_in_file)) : "") +
      row("Язык", escapeHtml(lang)) +
      row("Объём", `${(i.chapters || 0).toLocaleString("ru-RU")} глав · ${(i.paragraphs || 0).toLocaleString("ru-RU")} абзацев`) +
      row("База", `<code>${escapeHtml(i.db_path)}</code> (${fmtBytes(i.db_size_bytes)})`) +
      row("Проиндексирована", escapeHtml(when)) +
      row("Модель смысла", escapeHtml([i.embedding_model, i.embedding_runtime].filter(Boolean).join(" · ") || "—")) +
      row("ID", `<code>${escapeHtml(i.book_id)}</code>${i.active ? " · основная книга" : ""}`) +
      `</dl>`;
  } catch (e) {
    $("#bi-facts").innerHTML = "";
    $("#bi-error").textContent = "Не удалось получить сведения: " + e.message;
    $("#bi-error").classList.remove("hidden");
  }
}

async function saveBookInfo(body) {
  try {
    await api(`/api/books/${encodeURIComponent(bookInfoId)}`, {
      method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    await loadBooks();
    // repaint shown hits with the new name / colour
    if (lastView === null && lastSearchItems && $("#results .card[data-id]")) {
      lastSearchItems.forEach((p) => { if (p.book_id === bookInfoId) p.book_name = bookNameOf(bookInfoId); });
      renderResults(lastSearchItems, "", badgeBooks);
    } else {
      document.querySelectorAll(`#results .card[data-book="${CSS.escape(bookInfoId)}"]`).forEach((c) => c.style.setProperty("--bk", bookColor(bookInfoId)));
    }
    return true;
  } catch (e) {
    $("#bi-error").textContent = "Не удалось сохранить: " + e.message;
    $("#bi-error").classList.remove("hidden");
    return false;
  }
}

$("#bi-save").addEventListener("click", async () => {
  const b = booksData.books.find((x) => x.book_id === bookInfoId) || {};
  const body = {};
  const name = $("#bi-name").value.trim();
  if (name && name !== b.book_name) body.book_name = name;
  const col = $("#bi-color").value;
  if (col.toLowerCase() !== bookColor(bookInfoId).toLowerCase()) body.color = col;
  if (Object.keys(body).length && !(await saveBookInfo(body))) return;
  $("#book-info-dialog").close();
});
$("#bi-color-auto").addEventListener("click", async () => {
  if (await saveBookInfo({ color: "" })) openBookInfo(bookInfoId);
});
$("#bi-close").addEventListener("click", () => $("#book-info-dialog").close());


async function loadBooks() {
  try {
    booksData = await api("/api/books");
    renderBooksMenu();
  } catch (e) {}
}

function openBooksMenu() {
  const menu = $("#books-menu"), btn = $("#books-btn");
  const r = btn.getBoundingClientRect();
  menu.style.top = r.bottom + 6 + "px";
  menu.style.left = Math.max(8, Math.min(r.left, window.innerWidth - 380)) + "px";
  // the list scrolls inside the menu: never taller than the space under the button
  menu.style.maxHeight = Math.max(200, window.innerHeight - r.bottom - 16) + "px";
  sortBooksForMenu();
  if ($("#bm-filter")) $("#bm-filter").value = "";
  renderBooksMenu();
  menu.classList.remove("hidden");
  btn.setAttribute("aria-expanded", "true");
  if ($("#bm-filter")) $("#bm-filter").focus();
}
function closeBooksMenu() { $("#books-menu").classList.add("hidden"); $("#books-btn").setAttribute("aria-expanded", "false"); }
$("#books-btn").addEventListener("click", (ev) => { ev.stopPropagation(); $("#books-menu").classList.contains("hidden") ? openBooksMenu() : closeBooksMenu(); });
// clicks and Esc inside a dialog opened from the menu («О книге», «Добавить книги») close only that dialog
document.addEventListener("click", (ev) => { if (!ev.target.closest("#book-picker, dialog")) closeBooksMenu(); });
document.addEventListener("keydown", (ev) => { if (ev.key === "Escape" && !document.querySelector("dialog[open]")) closeBooksMenu(); });

async function makeMainBook(book_id) {
  try {
    const res = await api("/api/books/select", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ book_id }),
    });
    if (res.db_error) alert("Не удалось открыть книгу: " + res.db_error);
    $("#results").innerHTML = "";
    refreshDbBanner();
    if (document.querySelector("#tab-prepare.active")) refreshDbStatus();
  } catch (e) {
    alert("Не удалось переключить книгу: " + e.message);
  }
  loadBooks();
}

async function deleteBook(book_id) {
  const name = bookNameOf(book_id);
  if (!confirm(`Удалить книгу «${name}» из библиотеки?\n\nБудет удалена её база (папка library/${book_id}); исходный файл в папке corpus останется, книгу можно проиндексировать заново.`)) return;
  try {
    await api(`/api/books/${encodeURIComponent(book_id)}`, { method: "DELETE" });
  } catch (e) { alert("Не удалось удалить книгу: " + e.message); }
  loadBooks();
}
loadBooks();

// ------------------------------------------------------------ add a book ---
const AB = { file: null, meta: null, busy: false, jobId: null };
const abDlg = () => $("#add-book-dialog");
const abShow = (step) => ["file", "form", "progress", "batch"].forEach((n) => $(`#ab-step-${n}`).classList.toggle("hidden", n !== step));
function abError(msg) { const el = $("#ab-error"); el.textContent = msg || ""; el.classList.toggle("hidden", !msg); }

function abReset() {
  AB.file = null; AB.meta = null; AB.jobId = null;
  abShow("file"); abError("");
  $("#ab-upload-msg").textContent = ""; $("#ab-file").value = "";
  $("#ab-start").classList.add("hidden"); $("#ab-cancel-job").classList.add("hidden");
  $("#ab-start-all").classList.add("hidden");
  $("#ab-close").textContent = "Отмена";
  if (!ABB.running) ABB.items = [];
}

// while a batch is indexing, «+» reopens its list instead of starting over
$("#add-book-btn").addEventListener("click", () => { closeBooksMenu(); if (!ABB.running) abReset(); abDlg().showModal(); });
$("#ab-close").addEventListener("click", () => abDlg().close());

const abDrop = $("#ab-drop");
abDrop.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); $("#ab-file").click(); } });
["dragenter", "dragover"].forEach((n) => abDrop.addEventListener(n, (ev) => { ev.preventDefault(); abDrop.classList.add("drag"); }));
["dragleave", "drop"].forEach((n) => abDrop.addEventListener(n, (ev) => { ev.preventDefault(); abDrop.classList.remove("drag"); }));
abDrop.addEventListener("drop", (ev) => abHandleFiles(ev.dataTransfer.files));
$("#ab-file").addEventListener("change", (ev) => abHandleFiles(ev.target.files));

async function abUpload(file, force = false) {
  abError("");
  if (!/\.(epub|fb2|fb2\.zip|json|md)$/i.test(file.name)) {
    $("#ab-upload-msg").textContent = ""; abError("Этот формат не подходит. Нужны .epub, .fb2, .fb2.zip (или .json / .md)."); return;
  }
  $("#ab-upload-msg").textContent = `Загружаю «${file.name}» (${(file.size / 1e6).toFixed(1)} МБ)…`;
  try {
    const meta = await api(`/api/books/upload?filename=${encodeURIComponent(file.name)}${force ? "&force=true" : ""}`, {
      method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: file,
    });
    if (meta.duplicate_of) {
      const d = meta.duplicate_of;
      $("#ab-upload-msg").textContent = "";
      if (!confirm(`Эта книга уже есть в библиотеке: «${d.book_name}» (${d.reason}).\n\nВсё равно добавить её ещё раз отдельной книгой?`)) {
        abError(`Книга «${d.book_name}» уже добавлена раньше — повторно не добавляю.`);
        return;
      }
      return abUpload(file, true);
    }
    AB.file = meta.file; AB.meta = meta;
    $("#ab-name").value = meta.title || "";
    $("#ab-id").value = meta.book_id || "";
    $("#ab-fileline").textContent = [file.name, meta.author, meta.language ? `язык: ${meta.language}` : ""].filter(Boolean).join(" · ");
    $("#ab-preview").textContent = "Читаю книгу…";
    abShow("form");
    $("#ab-start").classList.remove("hidden");
    abEmbedHint();
    abPreview(meta.file);
  } catch (e) {
    $("#ab-upload-msg").textContent = ""; abError(e.message);
  }
}

async function abPreview(file) {
  try {
    const p = await api(`/api/books/preview?file=${encodeURIComponent(file)}`);
    const lang = { ru: "русский", en: "английский" }[p.language] || "русский + английский";
    $("#ab-preview").innerHTML = (p.trial ? `<div class="ab-trial">⚠ Это ознакомительный фрагмент (в конце: «Конец ознакомительного фрагмента»), а не вся книга.</div>` : "") +
      `Глав: <b>${p.chapters}</b> · абзацев: <b>${p.paragraphs.toLocaleString("ru-RU")}</b> · текст: ${lang}` +
      (p.sample ? `<span class="sample">«${escapeHtml(p.sample)}…»</span>` : "");
  } catch (e) { $("#ab-preview").textContent = "Не удалось прочитать книгу: " + e.message; }
}

// The embedding setup of the main book: vectors of all books must be comparable, so a new book is built the same way.
function abEmbedSetup() {
  const act = booksData.books.find((b) => b.active) || booksData.books[0] || {};
  return { model_name: act.embedding_model || "bge-m3", runtime: act.embedding_runtime || "auto",
           scheme: act.index_scheme || "D", vector_dtype: act.vector_dtype === "f32" ? "float32" : "int8" };
}
function abEmbedHint() {
  const e = abEmbedSetup();
  $("#ab-embed-hint").textContent = `Индексация займёт несколько минут (модель смысла: ${e.model_name}, как у основной книги).`;
}

$("#ab-start").addEventListener("click", async () => {
  abError("");
  const name = $("#ab-name").value.trim(), id = $("#ab-id").value.trim();
  if (!name) { abError("Укажите название книги."); return; }
  if (!/^[A-Za-z0-9_.-]+$/.test(id)) { abError("ID: только латинские буквы, цифры, «_», «-» и «.»."); return; }
  if (booksData.books.some((b) => b.book_id === id) &&
      !confirm(`Книга с ID «${id}» уже есть. Её база будет заменена после успешной сборки. Продолжить?`)) return;
  const setup = abEmbedSetup();
  let settings = {};
  try { settings = await api("/api/settings"); } catch (e) {}
  try {
    const { job_id } = await api("/api/jobs/index", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ json_path: AB.file, book_id: id, book_name: name, ...setup, device: settings.device || "auto", embed_batch_size: 32 }),
    });
    AB.jobId = job_id;
    abShow("progress");
    $("#ab-start").classList.add("hidden");
    $("#ab-cancel-job").classList.remove("hidden");
    $("#ab-close").textContent = "Свернуть";
    pollJob(job_id, $("#ab-step-progress .status-panel"), (job) => {
      $("#ab-cancel-job").classList.add("hidden");
      $("#ab-close").textContent = "Закрыть";
      const msg = $("#ab-step-progress .msg");
      if (job.status === "done" && job.result && !job.result.cancelled) {
        msg.textContent = `Готово: книга «${name}» добавлена в список.`;
        loadBooks();
        if (typeof refreshDbBanner === "function") refreshDbBanner();
      } else if (job.status === "error") { msg.textContent = ""; abShow("form"); abError("Индексация не удалась: " + (job.error || "неизвестная ошибка")); $("#ab-start").classList.remove("hidden"); }
      else msg.textContent = "Индексация отменена, книга не добавлена.";
    });
  } catch (e) { abError(e.message); }
});

$("#ab-cancel-job").addEventListener("click", async () => {
  if (!AB.jobId) return;
  ABB.stop = true;   // a batch stops after the current book
  try { await api(`/api/jobs/${AB.jobId}/cancel`, { method: "POST" }); $("#ab-step-progress .msg").textContent = "Отмена… дождитесь остановки."; }
  catch (e) { abError("Не удалось отменить: " + e.message); }
});

// ---- several books at once: upload all, edit names/IDs in a list, then index them one after another ----
const ABB = { items: [], running: false, stop: false };   // item: {fileName, file, title, id, info, status, jobId}

function abHandleFiles(files) {
  const list = [...files];
  if (!list.length) return;
  if (list.length === 1) { abUpload(list[0]); return; }
  abBatchUpload(list);
}

function abUniqueId(id) {
  const taken = new Set([...booksData.books.map((b) => b.book_id), ...ABB.items.map((x) => x.id)]);
  if (!taken.has(id)) return id;
  let n = 2;
  while (taken.has(`${id}_${n}`)) n++;
  return `${id}_${n}`;
}

async function abBatchUpload(files) {
  abError("");
  ABB.items = []; ABB.stop = false;
  abShow("batch");
  $("#ab-start").classList.add("hidden");
  const bad = files.filter((f) => !/\.(epub|fb2|fb2\.zip|json|md)$/i.test(f.name));
  for (const f of files) {
    if (bad.includes(f)) continue;
    const it = { fileName: f.name, file: null, title: "", id: "", info: "Загружаю…", status: "wait" };
    ABB.items.push(it);
    abBatchRender();
    try {
      const meta = await api(`/api/books/upload?filename=${encodeURIComponent(f.name)}`, {
        method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: f,
      });
      if (meta.duplicate_of) {
        const d = meta.duplicate_of;
        it.status = "dup"; it.title = d.book_name;
        it.info = `Эта книга уже есть в библиотеке: «${d.book_name}» (${d.reason}) — пропускаю.`;
        abBatchRender();
        continue;
      }
      const normT = (s) => (s || "").toLowerCase().replace(/ё/g, "е").replace(/[^\p{L}\p{N}]+/gu, " ").trim();   // \W is ASCII-only in JS: Cyrillic must be kept
      const twin = meta.title && ABB.items.find((x) => x !== it && x.file && normT(x.title) === normT(meta.title));
      if (twin) {
        it.status = "dup"; it.title = meta.title;
        it.info = `Эта книга уже есть в списке выше («${twin.fileName}») — пропускаю.`;
        abBatchRender();
        continue;
      }
      it.file = meta.file; it.title = meta.title || f.name.replace(/\.[^.]+$/, "");
      it.id = ""; it.id = abUniqueId(meta.book_id || "book");
      it.info = [meta.author, "читаю…"].filter(Boolean).join(" · ");
      abBatchRender();
      try {
        const p = await api(`/api/books/preview?file=${encodeURIComponent(meta.file)}`);
        it.info = [meta.author, `глав: ${p.chapters}, абзацев: ${p.paragraphs.toLocaleString("ru-RU")}`].filter(Boolean).join(" · ");
        if (p.trial) it.info += " · ⚠ это ознакомительный фрагмент, а не вся книга (можно убрать ✕)";
      } catch (e) { it.info = "не удалось прочитать: " + e.message; it.status = "bad"; }
    } catch (e) { it.info = "не загрузилась: " + e.message; it.status = "bad"; }
    abBatchRender();
  }
  const notes = [];
  if (bad.length) notes.push(`Пропущены файлы неподходящего формата: ${bad.map((f) => f.name).join(", ")}.`);
  const dups = ABB.items.filter((x) => x.status === "dup");
  if (dups.length) notes.push(`Пропущены повторы (книга уже есть): ${dups.map((x) => `«${x.title}»`).join(", ")}.`);
  if (notes.length) abError(notes.join(" "));
  abBatchRender();
}

const AB_STATUS = { wait: "", bad: "✖ ошибка", dup: "↺ уже есть", queued: "в очереди", run: "⏳ индексируется…", done: "✔ готово", fail: "✖ не удалось", cancel: "отменено" };

function abBatchRender() {
  const box = $("#ab-batch-list");
  box.innerHTML = ABB.items.map((it, i) => `
    <div class="abb-row abb-${it.status}" data-i="${i}">
      <div class="abb-file" title="${escapeHtml(it.fileName)}">${escapeHtml(it.fileName)}</div>
      <input class="abb-title" type="text" value="${escapeHtml(it.title)}" placeholder="Название" ${ABB.running || !it.file ? "disabled" : ""} />
      <input class="abb-id" type="text" value="${escapeHtml(it.id)}" placeholder="ID" spellcheck="false" ${ABB.running || !it.file ? "disabled" : ""} />
      <button type="button" class="bm-btn abb-del" title="Убрать из списка" ${ABB.running ? "disabled" : ""}>✕</button>
      <div class="abb-info">${escapeHtml(it.info)} <b class="abb-status">${AB_STATUS[it.status] || ""}</b></div>
    </div>`).join("");
  box.querySelectorAll(".abb-row").forEach((row) => {
    const it = ABB.items[Number(row.dataset.i)];
    row.querySelector(".abb-title").addEventListener("input", (e) => { it.title = e.target.value; });
    row.querySelector(".abb-id").addEventListener("input", (e) => { it.id = e.target.value.trim(); });
    row.querySelector(".abb-del").addEventListener("click", () => { ABB.items.splice(Number(row.dataset.i), 1); abBatchRender(); });
  });
  const ready = ABB.items.filter((x) => x.file && !["bad", "dup", "done"].includes(x.status));
  const btn = $("#ab-start-all");
  btn.textContent = `Добавить и проиндексировать все (${ready.length})`;
  btn.classList.toggle("hidden", ABB.running || !ready.length);
  $("#ab-cancel-job").classList.toggle("hidden", !ABB.running);
  $("#ab-close").textContent = ABB.running ? "Свернуть" : (ABB.items.some((x) => x.status === "done") ? "Закрыть" : "Отмена");
}

// Overall progress of a batch: «Книга 2 из 5», bar = finished books + the share of the current one.
function abBatchTotal(curFrac = 0) {
  const todo = ABB.items.filter((x) => ["queued", "run", "done", "fail", "cancel"].includes(x.status) && x.batch);
  const box = $("#ab-batch-total");
  if (!todo.length) { box.classList.add("hidden"); return; }
  const finished = todo.filter((x) => ["done", "fail", "cancel"].includes(x.status)).length;
  const cur = todo.find((x) => x.status === "run");
  box.classList.remove("hidden");
  box.querySelector(".abb-total-line").innerHTML = cur
    ? `Книга <b>${finished + 1}</b> из <b>${todo.length}</b>: «${escapeHtml(cur.title)}»`
    : `Обработано книг: <b>${finished}</b> из <b>${todo.length}</b>`;
  box.querySelector(".abb-total-bar").style.width = `${Math.round(100 * (finished + (cur ? curFrac : 0)) / todo.length)}%`;
}

function abWaitJob(jobId) {
  const panel = $("#ab-step-progress .status-panel");
  // follow the current book's own bar to move the overall one smoothly
  const watch = setInterval(() => {
    const w = parseFloat(panel.querySelector(".progress-bar-inner").style.width) || 0;
    abBatchTotal(Math.min(1, w / 100));
  }, 500);
  return new Promise((resolve) => pollJob(jobId, panel, (job) => { clearInterval(watch); resolve(job); }));
}

$("#ab-start-all").addEventListener("click", async () => {
  abError("");
  const todo = ABB.items.filter((x) => x.file && !["bad", "dup", "done"].includes(x.status));
  const ids = new Set();
  for (const it of todo) {
    if (!it.title.trim()) { abError(`Укажите название для «${it.fileName}».`); return; }
    if (!/^[A-Za-z0-9_.-]+$/.test(it.id)) { abError(`ID «${it.id}» (${it.fileName}): только латинские буквы, цифры, «_», «-» и «.».`); return; }
    if (ids.has(it.id)) { abError(`ID «${it.id}» повторяется в списке.`); return; }
    ids.add(it.id);
  }
  const clash = todo.filter((it) => booksData.books.some((b) => b.book_id === it.id));
  if (clash.length && !confirm(`Книги с такими ID уже есть: ${clash.map((x) => x.id).join(", ")}.\nИх базы будут заменены. Продолжить?`)) return;
  const setup = abEmbedSetup();
  let settings = {};
  try { settings = await api("/api/settings"); } catch (e) {}
  ABB.running = true; ABB.stop = false;
  ABB.items.forEach((it) => { it.batch = false; });
  todo.forEach((it) => { it.status = "queued"; it.batch = true; });
  $("#ab-step-progress").classList.remove("hidden");
  abBatchRender();
  for (const it of todo) {
    if (ABB.stop) { it.status = "cancel"; continue; }
    it.status = "run"; abBatchRender(); abBatchTotal(0);
    try {
      const { job_id } = await api("/api/jobs/index", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ json_path: it.file, book_id: it.id, book_name: it.title.trim(), ...setup, device: settings.device || "auto", embed_batch_size: 32 }),
      });
      it.jobId = AB.jobId = job_id;
      const job = await abWaitJob(job_id);
      if (job.status === "done" && job.result && !job.result.cancelled) it.status = "done";
      else if (job.status === "error") { it.status = "fail"; it.info += " · " + (job.error || "ошибка"); }
      else { it.status = "cancel"; ABB.stop = true; }
    } catch (e) { it.status = "fail"; it.info += " · " + e.message; }
    await loadBooks();
    abBatchRender();
  }
  ABB.running = false; AB.jobId = null;
  const done = ABB.items.filter((x) => x.status === "done").length;
  $("#ab-step-progress .msg").textContent = `Готово: добавлено книг — ${done} из ${todo.length}.`;
  abBatchTotal();
  abBatchRender();
  if (typeof refreshDbBanner === "function") refreshDbBanner();
});


// Red banner when the active book database is missing/corrupted (clear message instead of silent failures).
async function refreshDbBanner() {
  const el = $("#db-banner");
  try {
    const st = await api("/api/status");
    if (st.db_error) {
      el.textContent = "База книги недоступна: " + st.db_error + " — поиск по тексту не работает. Выберите другую книгу или постройте базу во вкладке «База данных».";
      el.classList.remove("hidden");
    } else {
      el.classList.add("hidden");
    }
  } catch (e) {}
}
refreshDbBanner();


// ======================================================================= history ===
const MODE_LABEL = { search: "Поиск", rag: "Вопрос", translation: "Анализ перевода" };
const MODE_ICON = { search: "🔍", rag: "💬", translation: "🔤" };
let historyOpen = false;
let recentQueries = [];   // [{query, mode}] newest first, unique by text
let recentIdx = -1;       // position while cycling with the ↑ / ↓ keys
let settingQuery = false; // true while the code (not the user) edits the query box
let currentHistoryEntry = null;

const pad2 = (n) => String(n).padStart(2, "0");
function parseLocal(iso) { return new Date(iso); }
function fmtTime(iso) { const d = parseLocal(iso); return `${pad2(d.getHours())}:${pad2(d.getMinutes())}`; }
function fmtStamp(iso) { const d = parseLocal(iso); return `${pad2(d.getDate())}.${pad2(d.getMonth() + 1)} ${fmtTime(iso)}`; }
function dayLabel(iso) {
  const d = parseLocal(iso);
  const today = new Date(); today.setHours(0, 0, 0, 0);
  const that = new Date(d); that.setHours(0, 0, 0, 0);
  const diff = Math.round((today - that) / 86400000);
  if (diff === 0) return "Сегодня";
  if (diff === 1) return "Вчера";
  return `${pad2(d.getDate())}.${pad2(d.getMonth() + 1)}.${d.getFullYear()}`;
}

// History lives in a left sidebar (rail + panel) on the Search tab. Whether it is open is remembered
// between launches; leaving the Search tab or a narrow-screen auto-close hides it without changing that.
const HISTORY_OPEN_KEY = "history_open";
function historyPref() { try { return localStorage.getItem(HISTORY_OPEN_KEY) === "1"; } catch (e) { return false; } }
function saveHistoryPref(open) { try { localStorage.setItem(HISTORY_OPEN_KEY, open ? "1" : "0"); } catch (e) {} }

function openHistoryPanel(remember = true) {
  historyOpen = true;
  if (remember) saveHistoryPref(true);
  $("#history-panel").classList.remove("hidden");
  document.body.classList.add("history-open");
  $("#history-rail").setAttribute("aria-expanded", "true");
  loadHistory();
}
function closeHistory(remember = true) {
  historyOpen = false;
  if (remember) saveHistoryPref(false);
  $("#history-panel").classList.add("hidden");
  document.body.classList.remove("history-open");
  $("#history-rail").setAttribute("aria-expanded", "false");
  $("#history-clear-confirm").classList.add("hidden");
}
$("#history-rail").addEventListener("click", () => (historyOpen ? closeHistory() : openHistoryPanel()));
$("#history-close").addEventListener("click", () => closeHistory());
document.addEventListener("keydown", (ev) => { if (ev.key === "Escape" && historyOpen) closeHistory(); });
$$(".tab-btn[data-tab]").forEach((b) => b.addEventListener("click", () => {
  const onSearch = b.dataset.tab === "search";
  document.body.classList.toggle("search-tab", onSearch);
  if (!onSearch) closeHistory(false);
  else if (historyPref() && !historyOpen) openHistoryPanel(false);
}));
document.body.classList.add("search-tab");
if (historyPref()) openHistoryPanel(false);

async function loadHistory() {
  const list = $("#history-list");
  try {
    const p = new URLSearchParams({ limit: "300" });
    if ($("#history-mode").value) p.set("mode", $("#history-mode").value);
    if ($("#history-q").value.trim()) p.set("q", $("#history-q").value.trim());
    if ($("#history-rating").value) p.set("rating", $("#history-rating").value);
    if ($("#history-sort").value !== "date") p.set("sort", $("#history-sort").value);
    const data = await api("/api/history?" + p.toString());
    renderHistoryList(data.items, data.total);
  } catch (e) {
    console.error("history:", e);
    list.innerHTML = `<div class="msg-box error">Не удалось загрузить историю: ${escapeHtml(e.message)}</div>`;
  }
}

function fmtDuration(s) {
  s = Math.round(s);
  return s < 60 ? `${s} с` : `${Math.floor(s / 60)} мин ${s % 60} с`;
}

function historyRow(e) {
  return `<div class="h-row${currentHistoryEntry && currentHistoryEntry.id === e.id ? " active" : ""}" data-id="${e.id}" tabindex="0">
    <div class="h-main">
      <div class="h-meta"><span class="h-mode">${MODE_ICON[e.mode] || ""} ${MODE_LABEL[e.mode] || e.mode}${e.params && e.params.overview ? " · обзор" : ""}<span class="h-rate" title="${e.note ? escapeHtml(e.note) : "оценка"}">${e.rating > 0 ? "👍" : e.rating < 0 ? "👎" : ""}${e.note ? " 📝" : ""}</span>${e.params && e.params.cloud ? `<span class="h-cloud" title="Облако: ${escapeHtml(e.params.provider || "")} — тексты отправлялись на внешний сервер">☁</span>` : ""}</span><span class="h-time">${fmtTime(e.created_at)}</span></div>
      <div class="h-query" title="${escapeHtml(e.query)}">${escapeHtml(e.query)}</div>
      ${e.llm_model ? `<div class="h-model" title="Модель, которая отвечала">🤖 ${escapeHtml(e.llm_model)}</div>` : ""}
      ${e.duration_s || e.tps ? `<div class="h-model" title="Время выполнения запроса и скорость генерации ответа">⏱ ${e.duration_s ? fmtDuration(e.duration_s) : "—"}${e.tps ? ` · ${Number(e.tps).toFixed(1)} t/s` : ""}</div>` : ""}
      <div class="h-sum">${escapeHtml(e.summary || "")}</div>
    </div>
    <div class="h-btns">
      <button type="button" class="h-pin${e.pinned ? " on" : ""}" title="${e.pinned ? "Открепить" : "Закрепить"}">${e.pinned ? "★" : "☆"}</button>
      <button type="button" class="h-del" title="Удалить из истории">✕</button>
    </div>
  </div>`;
}

function renderHistoryList(items, total) {
  const list = $("#history-list");
  if (!items.length) {
    list.innerHTML = `<div class="hint-text h-empty">${$("#history-q").value || $("#history-mode").value || $("#history-rating").value
      ? "Ничего не найдено." : "История пока пуста: ваши поиски, вопросы и проверки переводов появятся здесь."}</div>`;
    return;
  }
  let html = "";
  const pinned = items.filter((e) => e.pinned);
  if (pinned.length) html += `<div class="h-group">Закреплённые</div>` + pinned.map(historyRow).join("");
  let last = null;
  for (const e of items.filter((x) => !x.pinned)) {
    const label = dayLabel(e.created_at);
    if (label !== last) { html += `<div class="h-group">${label}</div>`; last = label; }
    html += historyRow(e);
  }
  if (total > items.length) html += `<div class="hint-text h-empty">Показаны последние ${items.length} из ${total}.</div>`;
  list.innerHTML = html;
  list.querySelectorAll(".h-row").forEach((row) => {
    const id = Number(row.dataset.id);
    row.addEventListener("click", (ev) => { if (!ev.target.closest(".h-btns")) openHistoryEntry(id); });
    row.addEventListener("keydown", (ev) => { if (ev.key === "Enter") openHistoryEntry(id); });
    row.querySelector(".h-pin").addEventListener("click", () => toggleHistoryPin(id));
    row.querySelector(".h-del").addEventListener("click", () => askDeleteHistoryEntry(row, id));
  });
}

// Deleting one entry: inline two-step confirmation under the row (same style as «Очистить историю»).
function askDeleteHistoryEntry(row, id) {
  const open = row.nextElementSibling && row.nextElementSibling.classList.contains("h-del-confirm");
  $$("#history-list .h-del-confirm").forEach((b) => b.remove());
  if (open) return;  // second click on ✕ just closes the question
  const pinned = row.querySelector(".h-pin.on");
  row.insertAdjacentHTML("afterend", `<div class="confirm-bar h-del-confirm">
    <span>Удалить эту запись${pinned ? " (она закреплена)" : ""}? Отменить будет нельзя.</span>
    <button type="button" class="small h-del-yes">Да, удалить</button>
    <button type="button" class="secondary small h-del-no">Отмена</button></div>`);
  const bar = row.nextElementSibling;
  bar.querySelector(".h-del-yes").addEventListener("click", () => { bar.remove(); deleteHistoryEntry(id); });
  bar.querySelector(".h-del-no").addEventListener("click", () => bar.remove());
  bar.querySelector(".h-del-yes").focus();
}

async function toggleHistoryPin(id) {
  try {
    const r = await api(`/api/history/${id}/pin`, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    if (currentHistoryEntry && currentHistoryEntry.id === id) { currentHistoryEntry.pinned = r.pinned; updateHistoryNote(); }
    loadHistory();
  } catch (e) { console.error("pin:", e); alert("Не удалось изменить закрепление: " + e.message); }
}

async function deleteHistoryEntry(id) {
  try {
    await api(`/api/history/${id}`, { method: "DELETE" });
    if (currentHistoryEntry && currentHistoryEntry.id === id) { currentHistoryEntry = null; updateHistoryNote(); }
    loadHistory();
    loadRecent();
  } catch (e) { console.error("delete history:", e); alert("Не удалось удалить запись: " + e.message); }
}

$("#history-mode").addEventListener("change", loadHistory);
$("#history-rating").addEventListener("change", loadHistory);
$("#history-sort").addEventListener("change", loadHistory);
let historyFilterTimer = null;
$("#history-q").addEventListener("input", () => { clearTimeout(historyFilterTimer); historyFilterTimer = setTimeout(loadHistory, 250); });

$("#history-clear").addEventListener("click", () => $("#history-clear-confirm").classList.remove("hidden"));
$("#history-clear-no").addEventListener("click", () => $("#history-clear-confirm").classList.add("hidden"));
$("#history-clear-yes").addEventListener("click", async () => {
  $("#history-clear-confirm").classList.add("hidden");
  try {
    const r = await api("/api/history", { method: "DELETE" });
    currentHistoryEntry = null;
    updateHistoryNote();
    loadHistory();
    loadRecent();
    $("#history-list").insertAdjacentHTML("afterbegin", `<div class="msg-box ok">Удалено записей: ${r.deleted}. Закреплённые сохранены.</div>`);
  } catch (e) { console.error("clear history:", e); alert("Не удалось очистить историю: " + e.message); }
});

// --- put a history entry's settings back into the query form ---
function applyEntryToForm(e) {
  settingQuery = true;
  $("#query-input").value = e.query;
  settingQuery = false;
  autosizeQuery();
  const sel = $("#qmode-select");
  const m = e.mode === "rag" && e.params && e.params.overview ? "overview" : e.mode;
  if ([...sel.options].some((o) => o.value === m)) { sel.value = m; sel.dispatchEvent(new Event("change")); }
  if (e.params && Array.isArray(e.params.book_ids) && e.params.book_ids.length) setSelectedBooks(e.params.book_ids);
  if (e.mode === "search" && e.params && e.params.mode) {
    const sm = $$('input[name="smode"]').find((r) => r.value === e.params.mode);
    if (sm) sm.checked = true;
  }
}

async function openHistoryEntry(id) {
  try {
    const e = await api(`/api/history/${id}`);
    currentHistoryEntry = e;
    applyEntryToForm(e);
    setLastQuery(e.query);
    $("#status-panel").classList.add("hidden");
    if (e.mode === "search") renderResults((e.result && e.result.results) || []);
    else if (e.mode === "rag") renderRagAnswer(e.result);
    else renderTranslationReport(e.result, null, `/api/history/${e.id}/report.md`);
    updateHistoryNote();
    renderHistoryExtra(e);
    document.querySelectorAll(".h-row").forEach((r) => r.classList.toggle("active", Number(r.dataset.id) === id));
    if (window.innerWidth < 1200) closeHistory(false);  // on narrow screens the drawer would cover the result
    window.scrollTo({ top: 0, behavior: "smooth" });
  } catch (err) {
    console.error("open history entry:", err);
    $("#history-note").className = "history-note error";
    $("#history-note").textContent = "Не удалось открыть запись истории: " + err.message;
  }
}

function updateHistoryNote() {
  const e = currentHistoryEntry;
  const box = $("#history-note");
  if (!e) { box.className = "history-note hidden"; $("#history-extra").className = "history-extra hidden"; return; }
  let warn = "";
  if (!e.book_loaded) warn = ` Книга «${escapeHtml(e.book_id || "?")}» сейчас не открыта — тексты абзацев могут отсутствовать.`;
  else if (e.missing_paragraphs) warn = ` Некоторых абзацев (${e.missing_paragraphs}) в базе больше нет — база могла быть пересобрана.`;
  box.className = "history-note";
  box.innerHTML = `<span>📋 Результат от <strong>${fmtStamp(e.created_at)}</strong> (${MODE_LABEL[e.mode] || e.mode}${e.summary ? ", " + escapeHtml(e.summary) : ""}) — показан из истории, запрос заново не выполнялся.${warn}</span>
    <span class="h-note-btns"><button type="button" id="hn-pin" class="secondary">${e.pinned ? "Открепить" : "Закрепить"}</button></span>`;
  $("#hn-pin").addEventListener("click", () => toggleHistoryPin(e.id));
}

// --- recent queries: datalist dropdown + the ↑ / ↓ keys in the query box ---
async function loadRecent() {
  try {
    const data = await api("/api/history?limit=80");
    const seen = new Set();
    recentQueries = [];
    for (const e of data.items) {
      if (seen.has(e.query)) continue;
      seen.add(e.query);
      recentQueries.push({ query: e.query, mode: e.mode, params: e.params });
      if (recentQueries.length >= 15) break;
    }
  } catch (e) { console.error("recent queries:", e); }
}

$("#query-input").addEventListener("input", () => { if (!settingQuery) recentIdx = -1; });
$("#query-input").addEventListener("keydown", (ev) => {
  const box = ev.target;
  if (ev.key === "ArrowUp" && (box.value === "" || recentIdx >= 0)) {
    if (!recentQueries.length) return;
    ev.preventDefault();
    recentIdx = Math.min(recentIdx + 1, recentQueries.length - 1);
    const r = recentQueries[recentIdx];
    applyEntryToForm({ query: r.query, mode: r.mode, params: r.params });
  } else if (ev.key === "ArrowDown" && recentIdx >= 0) {
    ev.preventDefault();
    recentIdx -= 1;
    if (recentIdx < 0) { settingQuery = true; box.value = ""; settingQuery = false; autosizeQuery(); }
    else { const r = recentQueries[recentIdx]; applyEntryToForm({ query: r.query, mode: r.mode, params: r.params }); }
  }
});

// called by runQuery() after a search/question/analysis finished (the server has saved it by then)
function afterQuery() {
  recentIdx = -1;
  loadRecent();
  if (historyOpen) loadHistory();
  if (typeof refreshLlmBadge === "function") refreshLlmBadge();   // the model in memory may have changed
}
loadRecent();

// Back from the reading page: «Поиск» there links to /#q=<query>; put the query back into the box.
// The whole search page view (query, mode, shown result/answer, scroll position) is kept in sessionStorage
// when leaving the page (e.g. for «Чтение») and put back on return, so the user lands where they were.
const VIEW_KEY = "search_view";
let viewReady = false;   // don't overwrite the saved view before it has been restored
function saveView() {
  if (!viewReady) return;
  try {
    sessionStorage.setItem(VIEW_KEY, JSON.stringify({
      q: $("#query-input").value, qmode: $("#qmode-select").value,
      html: $("#results").innerHTML, scrollY: window.scrollY,
      items: lastSearchItems, badges: badgeBooks, historyId: currentHistoryEntry ? currentHistoryEntry.id : null,
    }));
  } catch (e) { /* storage full / unavailable: nothing to restore later */ }
}
// Saved while scrolling and before following a link too: in the desktop window (WebView2) «pagehide» is not reliable.
let viewTimer = null;
window.addEventListener("scroll", () => { clearTimeout(viewTimer); viewTimer = setTimeout(saveView, 250); }, { passive: true });
document.addEventListener("click", (e) => { if (e.target.closest("a[href]")) saveView(); }, true);
window.addEventListener("pagehide", saveView);
document.addEventListener("visibilitychange", saveView);
(function restoreView() {
  const q = new URLSearchParams(location.hash.slice(1)).get("q");
  if (q) history.replaceState(null, "", location.pathname);
  let v = null;
  try { v = JSON.parse(sessionStorage.getItem(VIEW_KEY) || "null"); } catch (e) {}
  if (v && (!q || v.q === q)) {
    $("#query-input").value = v.q || "";
    if (v.qmode && $("#qmode-select").value !== v.qmode) {
      $("#qmode-select").value = v.qmode;
      $("#qmode-select").dispatchEvent(new Event("change"));
    }
    const results = $("#results");
    results.innerHTML = v.html || "";
    lastSearchItems = v.items || null;
    badgeBooks = !!v.badges;
    bindCardHandlers(results);
    results.querySelectorAll(".cite").forEach((el) => el.addEventListener("click", () => jumpToSource(el.dataset.id)));
    // restore the scroll once layout settles (fonts, the books list, etc.)
    const y = v.scrollY || 0, t0 = Date.now();
    (function tick() {   // retry while the page is still growing, up to ~3 s
      window.scrollTo(0, y);
      if (Math.abs(window.scrollY - y) < 4 || Date.now() - t0 > 3000) { viewReady = true; return; }
      setTimeout(tick, 100);
    })();
  } else {
    if (q) $("#query-input").value = q;
    viewReady = true;
  }
  autosizeQuery();
})();

try { const bad = Md.selfTest(); if (bad.length) console.warn("[md.js] self-test failed:", bad); } catch (e) { console.warn("[md.js]", e); }

// ------------------------------------------------------------ restart -----
$("#restart-btn").addEventListener("click", async () => {
  const status = $("#restart-status");
  let info;
  try {
    info = await api("/api/restart");
    if (!info.available) { status.textContent = "Перезапуск отсюда недоступен: запустите программу через start.bat."; return; }
    const warn = info.running_jobs.length ? `\n\nСейчас выполняются задачи (${info.running_jobs.length}) — они будут прерваны.` : "";
    if (!confirm("Перезапустить программу?" + warn)) return;
    $("#restart-btn").disabled = true;
    status.textContent = "Перезапуск…";
    await api("/api/restart", { method: "POST" });
  } catch (e) { status.textContent = "Ошибка: " + e.message; $("#restart-btn").disabled = false; return; }
  // wait until a NEW server process answers (its boot_id differs from the old one)
  const t0 = Date.now();
  while (Date.now() - t0 < 120000) {
    await new Promise((r) => setTimeout(r, 1000));
    try {
      const r = await fetch("/api/restart", { cache: "no-store" });
      if (r.ok && (await r.json()).boot_id !== info.boot_id) { location.reload(); return; }
    } catch (e) { /* not up yet */ }
    status.textContent = `Перезапуск… ${Math.round((Date.now() - t0) / 1000)} с`;
  }
  status.textContent = "Программа не ответила за 2 минуты. Посмотрите окно программы или запустите start.bat заново.";
});

// the top bar is pinned; the settings section bar sticks right below it (its height can change when tabs wrap)
(() => {
  const bar = document.querySelector(".topbar");
  if (!bar) return;
  const sync = () => document.documentElement.style.setProperty("--topbar-h", bar.offsetHeight + "px");
  sync();
  if (window.ResizeObserver) new ResizeObserver(sync).observe(bar); else window.addEventListener("resize", sync);
})();
