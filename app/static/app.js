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
  "Сравнение EN/RU": "Сравнение EN/RU",
  "Анализ смысловых расхождений": "Анализ расхождений",
  "Формирование отчёта": "Формирование отчёта",
  // real app.core.indexer stage names
  parsing: "Разбор JSON", database: "Запись абзацев", lemmatize: "Лемматизация RU",
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
$("#query-input").addEventListener("keydown", (e) => { if (e.key === "Enter") runQuery(); });

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
      const data = await api(`/api/search?q=${encodeURIComponent(q)}&mode=${smode}&k=10`);
      renderStages(panel, { stages: [{ name: "fts", state: "done" }], progress: 1, message: `Найдено: ${data.results.length}`, status: "done" });
      renderResults(data.results);
      afterQuery();
    } catch (e) {
      renderStages(panel, { stages: [], progress: 1, message: "Ошибка: " + e.message, status: "error" });
    }
  } else if (mode === "rag") {
    try {
      const { job_id } = await api("/api/jobs/rag", {
        method: "POST", headers: { "Content-Type": "application/json" },
        // no `k`: the server uses the «Сколько абзацев передавать модели» setting
        body: JSON.stringify({ question: q, ...replay }),
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
  const pa = paper ? `<a class="read-link" href="${readHref({ paper: p.paper })}" title="Читать весь документ"><b>${paper}</b></a>` : "";
  const se = section ? `<a class="read-link" href="${readHref({ paper: p.paper, section: p.section })}" title="Читать с начала раздела">${section}</a>` : "";
  return `<span class="crumbs">${pa}${paper && section ? " · " : ""}${se}</span>`;
}

function readHref({ paper, section = null, para = null }) {
  const h = new URLSearchParams();
  h.set("p", paper);
  if (section != null) h.set("s", section);
  if (para) h.set("para", para);
  if (lastQuery) h.set("q", lastQuery);
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
function langBlock(cls, snippet, full) {
  if (snippet) {
    return `<div class="${cls}"><span class="snip">${snippet}</span><span class="full">${highlight(full, lastQuery)}</span></div>`;
  }
  return `<div class="${cls}">${highlight(full, lastQuery)}</div>`;
}

function cardHtml(p) {
  const hasSnip = !!(p.snippet_en || p.snippet_ru);
  return `
    <div class="card" data-id="${p.id}">
      <div class="card-head">
        <a class="ref read-link" href="${readHref({ paper: p.paper, para: p.id })}" title="Открыть абзац в тексте документа (документ ${p.paper}, раздел ${p.section}, абзац ${p.paragraph})">${p.id}</a>
        ${crumbs(p)}
        <span class="ranks" title="score: ${(p.score ?? 0).toFixed?.(3) ?? p.score}">${rankBadge("слова", p.fts_rank)}${rankBadge("смысл", p.sem_rank)}</span>
      </div>
      <div class="card-langs">
        ${langBlock("en", p.snippet_en, p.en)}
        ${langBlock("ru", p.snippet_ru, p.ru)}
      </div>
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

function renderResults(items, prefixHtml = "") {
  const results = $("#results");
  lastView = null;
  lastSearchItems = items;
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
  const data = await api(`/api/paragraph/${id}/context?before=2&after=2`);
  block.innerHTML = data.context.map((p) => `
    <div class="card-langs" data-id="${p.id}" style="margin-top:6px;${String(p.id) === id ? "font-weight:600" : ""}">
      <div class="en">${escapeHtml(p.en)}</div><div class="ru">${escapeHtml(p.ru)}</div>
    </div>`).join("");
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

function renderAnswerHtml(text) {
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

function renderRagAnswer(result, errText) {
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
  const planHtml = result.plan
    ? `<div class="plan-line">${escapeHtml(result.plan.summary || "")}` +
      (result.plan.semantic_query ? ` · по смыслу: «${escapeHtml(result.plan.semantic_query)}»` : "") + `</div>` +
      ((result.plan.terms_lines || []).length
        ? `<details class="terms-details"><summary>Как проверены слова запроса (по тексту книги)</summary><div class="terms-lines">${result.plan.terms_lines.map(escapeHtml).join("<br>")}</div></details>` : "")
    : `<div class="plan-line">Поиск по тексту вопроса (подготовка запроса не использовалась).</div>`;
  const unknownHtml = (result.unknown_citations || []).length
    ? `<div class="hint-warning">Внимание: ответ ссылается на номера, которых нет среди найденных абзацев (${escapeHtml(result.unknown_citations.join(", "))}) — проверьте их вручную.</div>`
    : "";
  const noCiteHtml = (!cited.length && !result.no_answer)
    ? `<div class="hint-warning">В ответе нет ссылок на найденные абзацы — относитесь к нему осторожно.</div>` : "";

  let html = noteHtml + `<div class="rag-answer${mdOn ? " md-on" : ""}">${renderAnswerHtml(result.answer || "")}</div>` + unknownHtml + noCiteHtml +
    `<div class="card rag-meta">${planHtml}` +
    `<div class="stats-line">Найдено кандидатов: по словам ${result.candidates?.fts ?? "-"}, по смыслу ${result.candidates?.semantic ?? "-"}; передано модели абзацев: ${sources.length}</div>` +
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
  const head = noteHtml + `<div class="card"><strong>${escapeHtml(result.summary || "")}</strong>
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
    $("#book-select").focus();
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
async function loadBooks() {
  try {
    const data = await api("/api/books");
    const sel = $("#book-select");
    sel.innerHTML = data.books.length
      ? data.books.map((b) => `<option value="${escapeHtml(b.book_id)}" ${b.active ? "selected" : ""}>${escapeHtml(b.book_id)}</option>`).join("")
      : `<option value="">(нет баз)</option>`;
  } catch (e) {}
}
$("#book-select").addEventListener("change", async (ev) => {
  const book_id = ev.target.value;
  if (!book_id) return;
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
});
loadBooks();


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

function openHistoryPanel() {
  historyOpen = true;
  $("#history-panel").classList.remove("hidden");
  document.body.classList.add("history-open");
  loadHistory();
}
function closeHistory() {
  historyOpen = false;
  $("#history-panel").classList.add("hidden");
  document.body.classList.remove("history-open");
  $("#history-clear-confirm").classList.add("hidden");
}
$("#history-btn").addEventListener("click", () => (historyOpen ? closeHistory() : openHistoryPanel()));
$("#history-close").addEventListener("click", closeHistory);
document.addEventListener("keydown", (ev) => { if (ev.key === "Escape" && historyOpen) closeHistory(); });
$$(".tab-btn[data-tab]").forEach((b) => b.addEventListener("click", () => { if (b.dataset.tab !== "search") closeHistory(); }));

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

function historyRow(e) {
  return `<div class="h-row${currentHistoryEntry && currentHistoryEntry.id === e.id ? " active" : ""}" data-id="${e.id}" tabindex="0">
    <div class="h-main">
      <div class="h-meta"><span class="h-mode">${MODE_ICON[e.mode] || ""} ${MODE_LABEL[e.mode] || e.mode}<span class="h-rate" title="${e.note ? escapeHtml(e.note) : "оценка"}">${e.rating > 0 ? "👍" : e.rating < 0 ? "👎" : ""}${e.note ? " 📝" : ""}</span>${e.params && e.params.cloud ? `<span class="h-cloud" title="Облако: ${escapeHtml(e.params.provider || "")} — тексты отправлялись на внешний сервер">☁</span>` : ""}</span><span class="h-time">${fmtTime(e.created_at)}</span></div>
      <div class="h-query" title="${escapeHtml(e.query)}">${escapeHtml(e.query)}</div>
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
    row.querySelector(".h-del").addEventListener("click", () => deleteHistoryEntry(id));
  });
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
  const sel = $("#qmode-select");
  if ([...sel.options].some((o) => o.value === e.mode)) { sel.value = e.mode; sel.dispatchEvent(new Event("change")); }
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
    if (window.innerWidth < 1200) closeHistory();  // on narrow screens the drawer would cover the result
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
    $("#recent-queries").innerHTML = recentQueries.map((r) => `<option value="${escapeHtml(r.query)}"></option>`).join("");
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
    if (recentIdx < 0) { settingQuery = true; box.value = ""; settingQuery = false; }
    else { const r = recentQueries[recentIdx]; applyEntryToForm({ query: r.query, mode: r.mode, params: r.params }); }
  }
});

// called by runQuery() after a search/question/analysis finished (the server has saved it by then)
function afterQuery() {
  recentIdx = -1;
  loadRecent();
  if (historyOpen) loadHistory();
}
loadRecent();

// Back from the reading page: «Поиск» there links to /#q=<query>; put the query back into the box.
(function restoreQueryFromHash() {
  const q = new URLSearchParams(location.hash.slice(1)).get("q");
  if (q) {
    $("#query-input").value = q;
    history.replaceState(null, "", location.pathname);
  }
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
