"use strict";
/* История: оценка, параметры запроса (снимок эксперимента), сравнение, повторы.
 * Зависит от app.js: $, $$, api, escapeHtml, fmtStamp, MODE_LABEL, openHistoryEntry, runQuery,
 * applyEntryToForm, loadHistory, renderAnswerHtml, mdOn. */

const fmtNum = (v) => (typeof v === "number" ? String(Math.round(v * 100) / 100).replace(".", ",") : String(v));

const RETRIEVAL_LABELS_UI = {
  k: "Сколько абзацев запрошено", fallback_plain_hybrid: "Поиск без подготовки запроса",
  candidates_fts_asked: "Кандидатов по словам (запрошено)", candidates_semantic_asked: "Кандидатов по смыслу (запрошено)",
  candidates_found: "Найдено кандидатов", rrf_k: "Константа объединения (RRF)", retrieved: "Найдено абзацев",
  after_relevance_filter: "Осталось после порога релевантности", with_neighbors: "С учётом соседних абзацев",
  sources_sent: "Абзацев реально передано модели", sources_lang: "Язык абзацев для модели", context_share: "Доля контекста под источники",
  paragraphs: "Абзацев проверено", batches: "Порций", batch_size: "Абзацев в порции", verify: "Второй проход проверки",
  dropped_candidates: "Отброшено кандидатов", truncated_batches: "Порций с обрезанным ответом", below_threshold: "Скрыто ниже порога",
  min_severity: "Порог важности", mode: "Режим поиска", lang: "Язык", fts_mode_used: "Режим слов (применён)",
};
const TIMING_LABELS = {
  plan_s: "Подготовка запроса", retrieval_s: "Поиск", context_s: "Сборка контекста", generation_s: "Генерация ответа",
  total_s: "Всего", fts_s: "Поиск по словам", semantic_s: "Поиск по смыслу", hybrid_s: "Объединение",
};
const STAT_LABELS = {
  tokens_per_sec: "Скорость генерации, токенов/с", overall_tokens_per_sec: "Скорость общая (с ожиданием первого токена), токенов/с",
  gen_time_s: "Время генерации (без ожидания первого токена), с", stats_source: "Откуда цифры скорости",
  requested_model: "Запрошенная модель (ответила запасная)", fallback_note: "Переключение на запасную модель",
  cost: "Стоимость, $", cost_estimated: "Стоимость оценена по прайсу", avg_tokens_per_sec: "Скорость (средняя), токенов/с", ttft_s: "До первого токена, с",
  elapsed_s: "Время всего, с", prompt_tokens: "Токенов в запросе", completion_tokens: "Токенов в ответе",
  reasoning_tokens: "Токенов «рассуждения»", total_tokens: "Токенов всего", stop_reason: "Причина остановки",
  total_prompt_tokens: "Токенов в запросах (всего)", total_completion_tokens: "Токенов в ответах (всего)", model: "Модель",
};
const PROMPT_SENT_LABELS = {
  system: "Системное сообщение (system), отправленное модели", user: "Сообщение с вопросом (user), отправленное модели",
  rewrite_system: "Подготовка запроса: system", rewrite_user: "Подготовка запроса: user",
  verify_system: "Перепроверка замечания: system", verify_user: "Перепроверка замечания: user",
};

let optionsLabelCache = null;
let promptTitleCache = null;
async function snapshotDictionaries() {
  if (!optionsLabelCache) { try { optionsLabelCache = (await api("/api/options")).options; } catch (e) { optionsLabelCache = []; } }
  if (!promptTitleCache) {
    try { promptTitleCache = Object.fromEntries((await api("/api/prompts")).prompts.map((p) => [p.id, p.title])); } catch (e) { promptTitleCache = {}; }
  }
}

function fmtAny(v) {
  if (v === null || v === undefined) return "—";
  if (typeof v === "boolean") return v ? "да" : "нет";
  if (typeof v === "number") return fmtNum(v);
  if (typeof v === "object") {
    if ("fts" in v || "semantic" in v) return `по словам ${v.fts ?? "—"}, по смыслу ${v.semantic ?? "—"}`;
    return JSON.stringify(v);
  }
  return String(v);
}

async function copyText(text, btn) {
  try { await navigator.clipboard.writeText(text); }
  catch (e) {
    const ta = document.createElement("textarea");
    ta.value = text; document.body.appendChild(ta); ta.select();
    try { document.execCommand("copy"); } catch (e2) { /* ignore */ }
    ta.remove();
  }
  if (btn) { const old = btn.textContent; btn.textContent = "Скопировано ✓"; setTimeout(() => { btn.textContent = old; }, 1500); }
}

// ------------------------------------------------------------------ snapshot ---
function snapRow(k, v) { return `<div class="snap-row"><div class="k">${escapeHtml(k)}</div><div class="v">${escapeHtml(fmtAny(v))}</div></div>`; }

function snapshotHtml(e) {
  const x = e.experiment;
  if (!x) return `<div class="hint-text">параметры не сохранены (запись сделана старой версией программы)</div>`;
  const kind = x.kind || e.mode;
  const general = [];
  if (x.provider) general.push(snapRow("Провайдер", `${x.provider.is_cloud ? "☁ облако" : "🖥 локально"}: ${x.provider.name} (${x.provider.kind}, ${x.provider.host})`));
  if (x.provider && x.provider.cost !== null && x.provider.cost !== undefined) general.push(snapRow("Стоимость запроса", x.provider.cost === 0 ? "бесплатно" : `≈ $${Number(x.provider.cost).toFixed(4)}`));
  if (x.rewrite_provider) general.push(snapRow("Подготовка запроса через", `${x.rewrite_provider.name} · ${x.rewrite_provider.model}`));
  if (x.model !== undefined) general.push(snapRow("Модель", x.model));
  if (x.temperature !== undefined) general.push(snapRow("Температура", x.temperature));
  if (x.max_tokens !== undefined) general.push(snapRow("Макс. токенов ответа", x.max_tokens));
  if (x.reasoning) general.push(snapRow("Рассуждение модели", `${x.reasoning.setting ?? "—"} (передано: ${x.reasoning.effort ?? "—"})`));
  if (x.context_length !== undefined) general.push(snapRow("Длина контекста модели", x.context_length));
  if (x.kind !== "search") general.push(snapRow("Набор промтов", x.preset || "— (не выбран)"));
  if (x.app) general.push(snapRow("Версия программы / дата", `${x.app.version ?? "—"} / ${x.app.date ?? "—"}`));
  if (x.embedder) general.push(snapRow("Модель поиска по смыслу", `${x.embedder.model ?? "—"} (${x.embedder.runtime ?? "—"})`));

  let html = `<div class="snap-sub">Общее</div><div class="snap-grid">${general.join("")}</div>`;
  if (x.plan && x.plan.summary) {
    html += `<div class="snap-sub">Поисковые слова (план поиска)</div><div class="snap-grid">${snapRow("План", x.plan.summary)}${x.plan.semantic_query ? snapRow("Фраза для поиска по смыслу", x.plan.semantic_query) : ""}</div>`;
  }
  const ret = Object.entries(x.retrieval || {}).map(([k, v]) => snapRow(RETRIEVAL_LABELS_UI[k] || k, v));
  if (ret.length) html += `<div class="snap-sub">Поиск и источники</div><div class="snap-grid">${ret.join("")}</div>`;
  const ss = Object.entries(x.search_settings || {}).filter(([, v]) => v !== null && v !== undefined)
    .map(([k, v]) => snapRow({ fts_weights: "Веса поиска по словам", fts_mode_hybrid: "Режим слов (смешанный)", fts_mode_standalone: "Режим слов («По словам»)", fts_stopwords: "Служебные слова отбрасываются" }[k] || k, v));
  if (ss.length) html += `<div class="snap-sub">Настройки поиска по словам</div><div class="snap-grid">${ss.join("")}</div>`;

  const prefix = kind === "translation" ? "translation_" : "rag_";
  const rKey = kind === "translation" ? "reasoning_translation" : "reasoning_rag";
  const opts = Object.entries(x.options || {}).filter(([k]) => k.startsWith(prefix) || k === rKey).map(([k, v]) => {
    const spec = (optionsLabelCache || []).find((o) => o.key === k);
    let shown = v;
    if (spec && spec.type === "choice") { const c = spec.choices.find((c2) => c2.value === v); if (c) shown = c.label; }
    return snapRow(spec ? spec.label : k, shown);
  });
  if (opts.length) html += `<div class="snap-sub">Параметры ответа</div><div class="snap-grid">${opts.join("")}</div>`;

  const tm = Object.entries(x.timings || {}).map(([k, v]) => snapRow(TIMING_LABELS[k] || k, typeof v === "number" ? fmtNum(v) + " с" : v));
  if (tm.length) html += `<div class="snap-sub">Время по этапам</div><div class="snap-grid">${tm.join("")}</div>`;
  const st = Object.entries(x.stats || {}).filter(([, v]) => v === null || typeof v !== "object").filter(([, v]) => v !== null)
    .map(([k, v]) => snapRow(STAT_LABELS[k] || k, v));
  if (st.length) html += `<div class="snap-sub">Работа модели</div><div class="snap-grid">${st.join("")}</div>`;

  const pv = Object.entries(x.prompt_versions || {}).map(([id, v]) => snapRow((promptTitleCache || {})[id] || id, v && typeof v === "object" ? v.label : v));
  if (pv.length) html += `<div class="snap-sub">Версии промтов</div><div class="snap-grid">${pv.join("")}</div>`;

  const texts = Object.entries(x.prompts || {}).filter(([k, v]) => typeof v === "string" && v && k !== "user_note");
  if (texts.length) {
    html += `<div class="snap-sub">Точные тексты, отправленные модели</div>`;
    if (x.prompts && x.prompts.user_note) html += `<div class="hint-text">${escapeHtml(x.prompts.user_note)}</div>`;
    html += texts.map(([k, v], i) => `<div class="snap-pre-head"><span>${escapeHtml(PROMPT_SENT_LABELS[k] || k)}</span>
      <button type="button" class="small secondary" data-copy="${i}">Копировать</button></div><pre class="snap-pre">${escapeHtml(v)}</pre>`).join("");
  }
  return html;
}

// ------------------------------------------------------------------ extra panel ---
function renderHistoryExtra(e) {
  const box = $("#history-extra");
  const px = e.experiment && e.experiment.prompt_texts && Object.keys(e.experiment.prompt_texts).length;
  const canExact = (e.mode === "rag" || e.mode === "translation") && !!px;
  box.className = "history-extra";
  box.innerHTML = `
    <div class="he-row">
      <strong>Оценка:</strong>
      <button type="button" class="rate-btn${e.rating > 0 ? " on-up" : ""}" data-rate="1" title="Результат хороший">👍 Хорошо</button>
      <button type="button" class="rate-btn${e.rating < 0 ? " on-down" : ""}" data-rate="-1" title="Результат плохой">👎 Плохо</button>
      <input type="text" class="rate-note" maxlength="4000" placeholder="Заметка к этому результату (необязательно)" value="${escapeHtml(e.note || "")}" />
      <button type="button" class="small secondary" data-act="note">Сохранить заметку</button>
      <span class="save-status" data-rstatus></span>
    </div>
    <div class="he-row">
      ${e.mode !== "search" ? `<button type="button" data-act="replay" ${canExact ? "" : "disabled"}
         title="Те же тексты промтов, модель, температура и параметры ответа, что были в тот раз. Веса и режимы поиска по словам — текущие.">Повторить запрос</button>` : ""}
      <button type="button" class="secondary" data-act="replay-cur" title="Заново выполнить запрос с теми настройками, что сейчас в программе">Повторить с текущими настройками</button>
      <button type="button" class="secondary" data-act="compare">Сравнить с…</button>
      ${e.mode !== "search" && !canExact ? `<span class="hint-text">Точный повтор недоступен: параметры не сохранены.</span>` : ""}
    </div>
    <div id="cmp-picker" class="hidden"></div>
    <details class="snap-details"><summary>Параметры этого запроса ▸</summary><div class="snap-body hint-text">Загрузка…</div></details>`;

  const status = box.querySelector("[data-rstatus]");
  const noteInput = box.querySelector(".rate-note");
  const post = async (body) => {
    try {
      const r = await api(`/api/history/${e.id}/rating`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      e.rating = r.rating; e.note = r.note;
      box.querySelectorAll(".rate-btn").forEach((b) => {
        b.classList.toggle("on-up", b.dataset.rate === "1" && e.rating > 0);
        b.classList.toggle("on-down", b.dataset.rate === "-1" && e.rating < 0);
      });
      setStatus(status, "✓ Сохранено", "ok");
      if (historyOpen) loadHistory();
    } catch (err) { setStatus(status, "✗ " + err.message, "err"); }
  };
  box.querySelectorAll(".rate-btn").forEach((b) => b.addEventListener("click", () => {
    const want = Number(b.dataset.rate);
    post({ rating: e.rating === want ? 0 : want });
  }));
  box.querySelector('[data-act="note"]').addEventListener("click", () => post({ note: noteInput.value }));
  noteInput.addEventListener("keydown", (ev) => { if (ev.key === "Enter") post({ note: noteInput.value }); });

  const rp = box.querySelector('[data-act="replay"]');
  if (rp) rp.addEventListener("click", () => { applyEntryToForm(e); runQuery(e.id); });
  box.querySelector('[data-act="replay-cur"]').addEventListener("click", () => { applyEntryToForm(e); runQuery(); });
  box.querySelector('[data-act="compare"]').addEventListener("click", () => togglePicker(e));

  const det = box.querySelector(".snap-details");
  const sum = det.querySelector("summary");
  det.addEventListener("toggle", async () => {
    sum.textContent = det.open ? "Параметры этого запроса ▾" : "Параметры этого запроса ▸";
    if (!det.open || det.dataset.done) return;
    det.dataset.done = "1";
    await snapshotDictionaries();
    const body = det.querySelector(".snap-body");
    body.className = "snap-body";
    body.innerHTML = snapshotHtml(e);
    const texts = Object.entries((e.experiment && e.experiment.prompts) || {}).filter(([k, v]) => typeof v === "string" && v && k !== "user_note");
    body.querySelectorAll("[data-copy]").forEach((b) => b.addEventListener("click", () => copyText(texts[Number(b.dataset.copy)][1], b)));
  });
}

// ------------------------------------------------------------------ compare ---
async function togglePicker(e) {
  const box = $("#cmp-picker");
  if (!box.classList.contains("hidden")) { box.classList.add("hidden"); return; }
  box.classList.remove("hidden");
  box.innerHTML = `<div class="hint-text">Загрузка списка…</div>`;
  try {
    const data = await api(`/api/history?mode=${encodeURIComponent(e.mode)}&limit=200`);
    const same = (x) => (x.query || "").trim().toLowerCase() === (e.query || "").trim().toLowerCase();
    const items = data.items.filter((x) => x.id !== e.id).sort((a, b) => (same(b) - same(a)));
    if (!items.length) { box.innerHTML = `<div class="hint-text">Нет других записей этого вида для сравнения.</div>`; return; }
    box.innerHTML = `<div class="hint-text" style="margin-bottom:4px">Выберите запись для сравнения (сначала — с тем же вопросом):</div><div class="picker-list">` +
      items.map((x) => `<button type="button" class="picker-item" data-id="${x.id}">${fmtStamp(x.created_at)} · ${escapeHtml(x.query)}
        <small>— ${escapeHtml(x.summary || "")}${same(x) ? " · тот же запрос" : ""}${x.rating > 0 ? " · 👍" : x.rating < 0 ? " · 👎" : ""}${x.has_experiment ? "" : " · без параметров"}</small></button>`).join("") + `</div>`;
    box.querySelectorAll(".picker-item").forEach((b) => b.addEventListener("click", () => showCompare(e.id, Number(b.dataset.id))));
  } catch (err) { box.innerHTML = `<div class="msg-inline err">${escapeHtml(err.message)}</div>`; }
}

function deltaWords(label, a, b) {
  if (typeof a !== "number" || typeof b !== "number") return "";
  const d = b - a;
  if (d === 0) return "одинаково";
  const pct = a ? ` (${Math.round(Math.abs(d) / Math.abs(a) * 100)}%)` : "";
  const amt = fmtNum(Math.abs(d)) + pct;
  const lowerBetter = /Время|первого токена/.test(label);
  const higherBetter = /Скорость/.test(label);
  if (lowerBetter) return `<span class="${d < 0 ? "cmp-better" : "cmp-worse"}">B ${d < 0 ? "быстрее" : "медленнее"} на ${amt}</span>`;
  if (higherBetter) return `<span class="${d > 0 ? "cmp-better" : "cmp-worse"}">B ${d > 0 ? "быстрее" : "медленнее"} на ${amt}</span>`;
  return `у B ${d > 0 ? "больше" : "меньше"} на ${amt}`;
}

function cmpAnswerHtml(entry) {
  const r = entry.result || {};
  if (entry.mode === "rag") {
    const ans = r.answer ? `<div class="rag-answer${mdOn ? " md-on" : ""}">${renderAnswerHtml(r.answer)}</div>` : `<div class="hint-text">Ответа нет.</div>`;
    return ans + `<div class="stats-line">Источников в ответе: ${(r.cited_ids || []).length}</div>`;
  }
  if (entry.mode === "translation") {
    const issues = r.issues || [];
    return `<div class="card compact"><strong>${escapeHtml(r.summary || "")}</strong></div>` + issues.slice(0, 40).map((i) =>
      `<div class="card compact issue-card ${escapeHtml(i.severity)}"><div class="card-head"><span>[${escapeHtml(i.id)}] ${escapeHtml(String(i.severity).toUpperCase())} — ${escapeHtml(i.category)}</span></div>${mdText(i.problem)}</div>`).join("");
  }
  const rs = r.results || [];
  return `<div class="card compact">${rs.length ? rs.slice(0, 15).map((p) => `<div>${p.paper != null ? `Paper ${p.paper}:${p.section}.${p.paragraph}` : "№" + p.id}</div>`).join("") : "Нет результатов."}</div>`;
}

function rateMark(en) { return en.rating > 0 ? " 👍" : en.rating < 0 ? " 👎" : ""; }

function renderCompare(c) {
  const results = $("#results");
  lastView = () => renderCompare(c);
  const warn = [];
  if (!c.same_mode) warn.push("Записи разных видов — сравнение может быть малоинформативным.");
  if (!c.same_query) warn.push(`Вопросы разные: A — «${c.a.query}», B — «${c.b.query}».`);
  if (!c.has_experiment.a || !c.has_experiment.b) warn.push(`У записи ${!c.has_experiment.a ? "A" : ""}${!c.has_experiment.a && !c.has_experiment.b ? " и " : ""}${!c.has_experiment.b ? "B" : ""} параметры не сохранены — различия настроек неизвестны, сравниваются только ответы и показатели.`);

  let html = `<div class="cmp-head"><strong>Сравнение двух результатов</strong>
    <button type="button" class="secondary small" id="cmp-back">← Назад к записи</button></div>` +
    warn.map((w) => `<div class="hint-warning">${escapeHtml(w)}</div>`).join("");

  html += `<div class="cmp-cols"><div><div class="cmp-title">A — эта запись · ${fmtStamp(c.a.created_at)}${rateMark(c.a)}</div>${cmpAnswerHtml(c.a)}</div>
    <div><div class="cmp-title">B — выбранная · ${fmtStamp(c.b.created_at)}${rateMark(c.b)}</div>${cmpAnswerHtml(c.b)}</div></div>`;

  if (c.metrics.length) {
    html += `<div class="card"><div class="cmp-section">Показатели</div><table class="cmp-table"><tr><th>Что</th><th>A</th><th>B</th><th>Итог</th></tr>` +
      c.metrics.map((m) => `<tr><td>${escapeHtml(m.label)}</td><td>${escapeHtml(fmtAny(m.a))}</td><td>${escapeHtml(fmtAny(m.b))}</td><td>${deltaWords(m.label, m.a, m.b)}</td></tr>`).join("") + `</table></div>`;
  }
  if (c.param_diffs.length) {
    html += `<div class="card"><div class="cmp-section">Чем отличаются настройки (${c.param_diffs.length})</div><table class="cmp-table"><tr><th>Параметр</th><th>A</th><th>B</th></tr>` +
      c.param_diffs.map((d) => `<tr><td>${escapeHtml(d.label)}</td><td>${escapeHtml(d.a)}</td><td>${escapeHtml(d.b)}</td></tr>`).join("") + `</table></div>`;
  } else if (c.has_experiment.a && c.has_experiment.b) {
    html += `<div class="card"><div class="cmp-section">Настройки</div><div class="hint-text">Все параметры совпадают — различия в ответах объясняются только случайностью модели.</div></div>`;
  }
  const changed = c.prompt_diffs.filter((p) => p.changed);
  const same = c.prompt_diffs.filter((p) => !p.changed);
  if (changed.length || same.length) {
    html += `<div class="card"><div class="cmp-section">Тексты промтов</div>`;
    for (const p of changed) {
      html += `<div class="snap-sub">${escapeHtml(p.title)} — изменён</div>`;
      if (p.only_in) html += `<div class="hint-text">Использовался только в записи ${p.only_in.toUpperCase()}.</div><pre class="snap-pre">${escapeHtml(p.text || "")}</pre>`;
      else html += `<div class="diff-view">` + (p.diff || []).map((l) =>
        `<div class="dl ${l.op === "+" ? "add" : l.op === "-" ? "del" : l.op === "@" ? "gap" : ""}">${l.op === "+" ? "+ " : l.op === "-" ? "− " : "  "}${escapeHtml(l.text)}</div>`).join("") + `</div>`;
    }
    if (changed.length) html += `<div class="hint-text" style="margin-top:6px">«−» — строка была в A, но её нет в B; «+» — строка есть только в B.</div>`;
    if (same.length) html += `<div class="hint-text" style="margin-top:6px">Без изменений: ${same.map((p) => escapeHtml(p.title)).join("; ")}.</div>`;
    html += `</div>`;
  }
  results.innerHTML = html;
  $("#cmp-back").addEventListener("click", () => openHistoryEntry(c.a.id));
}

async function showCompare(aId, bId) {
  try {
    const c = await api(`/api/history/compare?a=${aId}&b=${bId}`);
    $("#status-panel").classList.add("hidden");
    renderCompare(c);
    window.scrollTo({ top: 0, behavior: "smooth" });
  } catch (err) {
    $("#cmp-picker").innerHTML = `<div class="msg-inline err">Не удалось сравнить: ${escapeHtml(err.message)}</div>`;
  }
}
