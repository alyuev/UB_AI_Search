"use strict";
/* Настройки → «Модель LLM»: провайдеры (локальные программы и облачные сервисы), выбор модели,
 * «Облачные модели» с подтверждением, переопределения по задачам и значок «Локально / Облако» у строки запроса.
 * Зависит от app.js и settings-ui.js: $, $$, api, escapeHtml, setStatus, jsonPost.
 * Ключи API никогда не читаются из программы: сервер отдаёт только {has_key, last4}. */

let provState = null;
let provModels = { pid: null, list: [], total: 0 };
let provModelTimer = null;
let provSel = null;   // provider shown in the drop-down (id)

const TASK_LABELS = { rag: "Ответ на вопрос", translation: "Анализ перевода", rewrite: "Подготовка запроса (поисковые слова)" };

function fmtCtx(n) { return n ? (n >= 1000 ? Math.round(n / 1000) + "K" : String(n)) : "—"; }
function fmtPrice(p) { return p === null || p === undefined ? "?" : (p === 0 ? "0" : "$" + (p < 0.1 ? p.toFixed(3) : p.toFixed(2))); }
function provIcon(p) { return p.is_cloud ? "☁" : "🖥"; }
function cloudHosts() {
  const hosts = [...new Set((provState?.providers || []).filter((p) => p.is_cloud).map((p) => p.host))];
  return hosts.length ? hosts.join(", ") : "облачного провайдера (например, openrouter.ai)";
}

async function loadProvidersUi() {
  const root = $("#providers-root");
  if (!root) return;
  try {
    provState = await api("/api/llm/providers");
  } catch (e) {
    root.innerHTML = `<div class="msg-inline err">Не удалось загрузить провайдеров: ${escapeHtml(e.message)}${typeof OLD_SERVER_HINT !== "undefined" ? OLD_SERVER_HINT : ""}</div>`;
    return;
  }
  renderProviders();
  loadTaskLlmUi();
  refreshLlmBadge();
}

// ---------------------------------------------------------------- рендер ---
function renderProviders() {
  const root = $("#providers-root");
  const s = provState;
  const protection = s.protection && s.protection.notice ? `<div class="msg-inline warn">${escapeHtml(s.protection.notice)}</div>` : "";
  const cloudBlock = s.allow_cloud_llm
    ? `<div class="cloud-box on"><div class="cloud-line"><span class="cloud-ico">☁</span><div><strong>Облачные модели разрешены.</strong>
         <div class="hint-text">Тексты абзацев и ваши вопросы могут отправляться на внешние серверы выбранных облачных провайдеров.</div></div>
         <button type="button" class="secondary small" id="cloud-off-btn">Выключить (работать только локально)</button></div></div>`
    : `<div class="cloud-box"><div class="cloud-line"><span class="cloud-ico">🔒</span><div><strong>Режим «только локально».</strong>
         <div class="hint-text">Тексты никуда не отправляются, облачные провайдеры не работают.</div></div>
         <button type="button" class="secondary small" id="cloud-on-btn">Разрешить облачные модели…</button></div>
         <div id="cloud-confirm" class="cloud-confirm hidden">
           <div class="msg-inline warn">${escapeHtml(s.cloud_warning.replace("{host}", cloudHosts()))}</div>
           <div class="inline-actions"><button type="button" id="cloud-confirm-btn" class="danger-ish">Понимаю, разрешить</button>
             <button type="button" id="cloud-cancel-btn" class="secondary">Отмена</button></div></div></div>`;
  if (!s.providers.some((p) => p.id === provSel)) provSel = (s.providers.find((p) => p.active) || s.providers[0] || {}).id;
  const options = s.providers.map((p) => `<option value="${escapeHtml(p.id)}"${p.id === provSel ? " selected" : ""}>${provIcon(p)} ${escapeHtml(p.name)}${p.active ? " — активный" : ""}</option>`).join("");
  const fact = (k, v) => `<dt>${k}</dt><dd>${v}</dd>`;
  const cards = s.providers.map((p) => {
    const keyVal = p.has_key ? `сохранён (…${escapeHtml(p.last4 || "")}${p.key_source === "env" ? ", из переменной среды" : ""})` : `<span class="err-text">не задан</span>`;
    const blocked = p.is_cloud && !s.allow_cloud_llm;
    return `<div class="prov-card${p.active ? " active" : ""}${p.is_cloud ? " cloud" : ""}${p.id === provSel ? "" : " hidden"}" data-pid="${escapeHtml(p.id)}">
      <div class="prov-head"><span class="prov-ico" title="${p.is_cloud ? "облако" : "локально"}">${provIcon(p)}</span>
        <span class="prov-name">${escapeHtml(p.name)}</span>
        <span class="prov-dot unknown" data-dot title="не проверялось"></span>
        ${p.active ? `<span class="badge ok">активный</span>` : ""}
        ${blocked ? `<span class="badge warn" title="Включите «Облачные модели» ниже">выключен</span>` : ""}
        <span class="prov-kind">${escapeHtml(p.kind_title)}</span></div>
      <dl class="prov-facts">
        ${fact("Адрес", `<code>${escapeHtml(p.base_url || p.host)}</code>`)}
        ${fact("Модель", p.model ? `<code>${escapeHtml(p.model)}</code>` : `<span class="err-text">не выбрана</span>`)}
        ${p.needs_key ? fact("Ключ", keyVal) : ""}
        ${(p.fallback_models || []).length ? fact("Запасные", p.fallback_models.map((m) => `<code>${escapeHtml(m)}</code>`).join(" → ")) : ""}
        ${p.kind === "openrouter" && s.allow_cloud_llm && p.has_key ? fact("Счёт", `<span class="prov-account" data-account>запрашиваю…</span>`) : ""}
      </dl>
      <div class="inline-actions prov-btns">
        ${p.active ? "" : `<button type="button" class="small" data-act="activate">Сделать активным</button>`}
        <button type="button" class="secondary small" data-act="check">Проверить соединение</button>
        <button type="button" class="secondary small" data-act="models">Модели…</button>
        <button type="button" class="secondary small" data-act="edit">Изменить</button>
        <button type="button" class="secondary small" data-act="delete">Удалить</button>
        <span data-msg class="save-status"></span></div>
      <div class="prov-confirm hidden"></div></div>`;
  }).join("");
  const kinds = s.kinds.map((k) => `<button type="button" class="kind-card" data-kind="${escapeHtml(k.id)}" title="${escapeHtml(k.desc)}">
      <span class="kind-title">${k.icon} ${escapeHtml(k.title)}</span><span class="hint-text">${escapeHtml(k.desc)}</span></button>`).join("");
  root.innerHTML = `${protection}
    <div class="sub-card prov-panel">
      <div class="prov-pick">
        <label for="prov-select">Провайдер</label>
        <select id="prov-select">${options}</select>
        <button type="button" id="prov-add-btn" class="secondary small">＋ Добавить провайдера</button>
      </div>
      <div class="prov-list">${cards}</div>
      <div id="prov-kinds" class="kind-grid hidden"><div class="hint-text">Выберите, откуда будет браться модель:</div>${kinds}</div>
      <div id="prov-wizard" class="prov-wizard hidden"></div>
      <div id="prov-models" class="prov-models hidden"></div>
    </div>
    ${cloudBlock}`;
  loadAccounts();
}

// OpenRouter: what the account behind the key looks like (free tier, spent, limit) - plain words from the service's own answer
async function loadAccounts() {
  for (const card of $$("#providers-root .prov-card")) {
    const el = card.querySelector("[data-account]");
    if (!el) continue;
    try {
      const r = await api(`/api/llm/providers/${card.dataset.pid}/account`);
      el.textContent = r.available ? r.summary : (r.reason || "неизвестно");
    } catch (e) { el.textContent = "не удалось получить (" + e.message + ")"; }
  }
}

// -------------------------------------------------------------- действия ---
document.addEventListener("change", (ev) => {
  if (ev.target.id !== "prov-select") return;
  provSel = ev.target.value;
  $$("#providers-root .prov-card").forEach((c) => c.classList.toggle("hidden", c.dataset.pid !== provSel));
  const m = $("#prov-models"); if (m) { m.classList.add("hidden"); m.innerHTML = ""; }
  const w = $("#prov-wizard"); if (w) { w.classList.add("hidden"); w.innerHTML = ""; }
});

document.addEventListener("click", async (ev) => {
  const t = ev.target;
  if (!t.closest || !t.closest("#providers-root")) return;
  const card = t.closest(".prov-card");
  const act = t.dataset && t.dataset.act;
  if (t.id === "cloud-on-btn") { $("#cloud-confirm").classList.remove("hidden"); t.classList.add("hidden"); return; }
  if (t.id === "cloud-cancel-btn") { $("#cloud-confirm").classList.add("hidden"); $("#cloud-on-btn").classList.remove("hidden"); return; }
  if (t.id === "cloud-confirm-btn") return setCloud(true, true);
  if (t.id === "cloud-off-btn") return setCloud(false, false);
  if (t.id === "prov-add-btn") { $("#prov-kinds").classList.toggle("hidden"); return; }
  const kc = t.closest(".kind-card");
  if (kc) return openWizard(kc.dataset.kind, null);
  if (!card) return;
  const pid = card.dataset.pid, msg = card.querySelector("[data-msg]");
  if (act === "activate") return provCall(`/api/llm/providers/${pid}/activate`, "POST", msg, true);
  if (act === "check") return checkProvider(pid, card);
  if (act === "models") return openModels(pid);
  if (act === "edit") return openWizard(provState.providers.find((p) => p.id === pid).kind, pid);
  if (act === "delete") {
    const box = card.querySelector(".prov-confirm");
    box.classList.remove("hidden");
    box.innerHTML = `Удалить провайдера «${escapeHtml(card.querySelector(".prov-name").textContent)}» вместе с сохранённым ключом?
      <button type="button" class="small" data-act="delete-yes">Да, удалить</button> <button type="button" class="secondary small" data-act="delete-no">Нет</button>`;
    return;
  }
  if (act === "delete-no") { card.querySelector(".prov-confirm").classList.add("hidden"); return; }
  if (act === "delete-yes") return provCall(`/api/llm/providers/${pid}`, "DELETE", msg, true);
});

async function provCall(url, method, msgEl, reload) {
  try {
    setStatus(msgEl, "…");
    await api(url, jsonPost(method, {}));
    if (reload) await loadProvidersUi();
  } catch (e) { setStatus(msgEl, "✗ " + e.message, "err"); }
}

async function setCloud(allow, confirm) {
  try {
    await api("/api/llm/cloud-access", jsonPost("POST", { allow, confirm }));
    await loadProvidersUi();
  } catch (e) {
    const el = $("#cloud-confirm") || $("#providers-root");
    el.insertAdjacentHTML("beforeend", `<div class="msg-inline err">${escapeHtml(e.message)}</div>`);
  }
}

async function checkProvider(pid, card) {
  const msg = card.querySelector("[data-msg]"), dot = card.querySelector("[data-dot]");
  setStatus(msg, "Проверка…");
  try {
    const r = await api(`/api/llm/providers/${pid}/check`, jsonPost("POST", {}));
    dot.className = "prov-dot " + (r.ok ? "ok" : "bad");
    dot.title = r.message;
    setStatus(msg, (r.ok ? "✓ " : "✗ ") + r.message + (r.latency_ms ? ` (${r.latency_ms} мс)` : ""), r.ok ? "ok" : "err");
  } catch (e) { dot.className = "prov-dot bad"; setStatus(msg, "✗ " + e.message, "err"); }
}

// ------------------------------------------------------------------ мастер ---
function headersToText(h) { return Object.entries(h || {}).map(([k, v]) => `${k}: ${v}`).join("\n"); }
function textToHeaders(t) {
  const out = {};
  for (const line of t.split("\n")) {
    const s = line.trim();
    if (!s) continue;
    const i = s.indexOf(":");
    if (i < 1) throw new Error(`Заголовок «${s}»: нужен вид «Имя: значение»`);
    out[s.slice(0, i).trim()] = s.slice(i + 1).trim();
  }
  return out;
}

function openWizard(kind, pid) {
  const k = provState.kinds.find((x) => x.id === kind);
  const p = pid ? provState.providers.find((x) => x.id === pid) : null;
  const wiz = $("#prov-wizard");
  const needsKey = k.needs_key === true || k.needs_key === "optional";
  const keyHint = kind === "openrouter"
    ? `Ключ создаётся на сайте openrouter.ai (раздел Keys в настройках аккаунта) — откройте этот сайт сами в браузере. Ключ хранится на этом компьютере в зашифрованном виде, в настройки, историю и журналы он не попадает.`
    : kind === "huggingface"
    ? `Токен создаётся на сайте huggingface.co (Settings → Access Tokens): откройте сайт сами в браузере, при создании отметьте право «Make calls to Inference Providers». Токен хранится на этом компьютере в зашифрованном виде, в настройки, историю и журналы он не попадает.`
    : kind === "sharellm"
    ? `Токен создаётся на сайте sharellm.net в личном кабинете — откройте сайт сами в браузере. Токен хранится на этом компьютере в зашифрованном виде, в настройки, историю и журналы он не попадает.`
    : "Ключ нужен не всем серверам. Хранится зашифрованно в отдельном файле.";
  const generic = !["lmstudio", "openrouter", "huggingface"].includes(kind);
  const presetUrl = !!k.default_url && k.needs_key === true;
  wiz.classList.remove("hidden");
  $("#prov-kinds").classList.add("hidden");
  wiz.innerHTML = `<h4>${p ? "Изменить" : "Новый провайдер"}: ${k.icon} ${escapeHtml(k.title)}</h4>
    <div class="hint-text">${escapeHtml(k.desc)}</div>
    <div class="set-grid wiz-grid">
      ${wizRow("Название", `<input type="text" id="w-name" maxlength="60" />`)}
      ${wizRow("Адрес", `<input type="text" id="w-url" placeholder="${escapeHtml(k.default_url || "https://…/v1")}" />`, presetUrl ? "подставлен автоматически — менять нужно, только если сервис сменил адрес" : "")}
      ${needsKey ? wizRow("Ключ API",
        `<div class="key-line"><input type="password" id="w-key" autocomplete="off" spellcheck="false" placeholder="${p && p.has_key ? "оставьте пустым, чтобы не менять" : "вставьте ключ"}" />
           <button type="button" class="secondary small" id="w-key-show">Показать</button></div>
         ${p && p.has_key ? `<label class="check-inline"><input type="checkbox" id="w-key-clear" /> удалить сохранённый ключ (…${escapeHtml(p.last4 || "")})</label>` : ""}`, keyHint) : ""}
      ${wizRow("Модель по умолчанию", `<input type="text" id="w-model" placeholder="можно выбрать позже из списка" />`)}
      ${wizRow("Запасные модели", `<textarea id="w-fallback" rows="2" placeholder="по одному названию в строке"></textarea>`, kind === "openrouter"
        ? "необязательно: если основная модель занята или не отвечает, программа сама попробует следующую по списку. Подойдёт и «openrouter/free» — сервис сам выберет свободную бесплатную модель"
        : "необязательно: если основная модель не отвечает, программа попробует следующую по списку")}
      ${wizRow("Таймаут, секунд", `<input type="number" id="w-timeout" min="1" max="36000" placeholder="как в общих настройках" />`)}
      ${kind !== "lmstudio" ? wizRow("Размер контекста, токенов", `<input type="number" id="w-ctx" min="512" placeholder="определить автоматически" />`, "заполняйте, только если программа неверно определяет размер") : ""}
      ${wizRow("Свои параметры модели", `<div class="inline-actions"><input type="number" id="w-temp" step="0.05" min="0" max="2" placeholder="температура" style="width:120px" />
          <input type="number" id="w-maxtok" min="1" placeholder="макс. токенов" style="width:130px" /></div>`, "необязательно: заменяют общие настройки только для этого провайдера")}
      ${wizRow("Дополнительные заголовки", `<textarea id="w-headers" rows="2" placeholder="Имя: значение (по одному в строке)"></textarea>`, "необязательно")}
      ${generic ? wizRow("Рассуждение", `<label class="check-inline"><input type="checkbox" id="w-reason" /> передавать серверу режим рассуждения (reasoning_effort)</label>`, "включайте, только если сервер это поддерживает") : ""}
    </div>
    <div class="set-actions"><button type="button" id="w-save">${p ? "Сохранить" : "Добавить"}</button>
      <button type="button" class="secondary" id="w-cancel">Отмена</button><span id="w-status" class="save-status"></span></div>
    <div id="w-err" class="field-err"></div>`;
  $("#w-name").value = p ? p.name : k.title;
  $("#w-url").value = p ? p.base_url : k.default_url;
  $("#w-model").value = p ? (p.default_model || "") : "";
  if (p) {
    $("#w-timeout").value = p.timeout || "";
    $("#w-fallback").value = (p.fallback_models || []).join("\n");
    if ($("#w-ctx")) $("#w-ctx").value = p.context_length || "";
    $("#w-temp").value = p.params?.temperature ?? "";
    $("#w-maxtok").value = p.params?.max_tokens ?? "";
    $("#w-headers").value = headersToText(p.extra_headers);
    if ($("#w-reason")) $("#w-reason").checked = !!p.send_reasoning_effort;
  }
  if ($("#w-key-show")) $("#w-key-show").addEventListener("click", () => {
    const i = $("#w-key"); const show = i.type === "password";
    i.type = show ? "text" : "password"; $("#w-key-show").textContent = show ? "Скрыть" : "Показать";
  });
  $("#w-cancel").addEventListener("click", () => { wiz.classList.add("hidden"); wiz.innerHTML = ""; });
  $("#w-save").addEventListener("click", () => saveWizard(kind, pid));
  wiz.scrollIntoView({ behavior: "smooth", block: "nearest" });
  (needsKey && !p ? $("#w-key") : $("#w-name")).focus();
}

function wizRow(label, control, hint) {
  return `<div class="set-row"><div class="set-label">${escapeHtml(label)}${hint ? `<span class="hint-text">${escapeHtml(hint)}</span>` : ""}</div><div class="set-field">${control}</div></div>`;
}

async function saveWizard(kind, pid) {
  const err = $("#w-err"); err.textContent = "";
  try {
    const num = (id) => { const v = $(id) && $(id).value.trim(); return v ? Number(v.replace(",", ".")) : null; };
    const params = {};
    if (num("#w-temp") !== null) params.temperature = num("#w-temp");
    if (num("#w-maxtok") !== null) params.max_tokens = num("#w-maxtok");
    const body = {
      name: $("#w-name").value.trim(), kind, base_url: $("#w-url").value.trim(), default_model: $("#w-model").value.trim(),
      timeout: num("#w-timeout"), extra_headers: textToHeaders($("#w-headers").value), params,
      fallback_models: $("#w-fallback").value.split("\n").map((x) => x.trim()).filter(Boolean),
    };
    if ($("#w-ctx")) body.context_length = num("#w-ctx");
    if ($("#w-reason")) body.send_reasoning_effort = $("#w-reason").checked;
    const key = $("#w-key") ? $("#w-key").value.trim() : "";
    if (key) body.api_key = key;
    if ($("#w-key-clear") && $("#w-key-clear").checked) body.clear_key = true;
    for (const f of ["timeout", "context_length"]) if (body[f] !== undefined && body[f] !== null && Number.isNaN(body[f])) throw new Error("Введите число");
    setStatus($("#w-status"), "Сохранение…");
    const r = await api(pid ? `/api/llm/providers/${pid}` : "/api/llm/providers", jsonPost(pid ? "PUT" : "POST", body));
    await loadProvidersUi();
    if (!pid && k.default_url && k.needs_key === true && !provState.allow_cloud_llm) {
      $("#prov-wizard").classList.remove("hidden");
      $("#prov-wizard").innerHTML = `<div class="msg-inline warn">Провайдер добавлен, но облачные модели пока выключены. Нажмите «Разрешить облачные модели…» выше и подтвердите — после этого можно проверить соединение и выбрать модель.</div>`;
    } else if (!pid) { provSel = r.id; renderProviders(); openModels(r.id); }
  } catch (e) { err.textContent = e.message; setStatus($("#w-status"), ""); }
}

// ------------------------------------------------------------ список моделей ---
async function openModels(pid) {
  const p = provState.providers.find((x) => x.id === pid);
  const box = $("#prov-models");
  box.classList.remove("hidden");
  provModels = { pid, list: [], total: 0 };
  const isOr = p.kind === "openrouter";
  box.innerHTML = `<h4>Модели: ${provIcon(p)} ${escapeHtml(p.name)}</h4>
    <div class="inline-actions">
      <input type="text" id="m-q" placeholder="поиск по названию…" style="min-width:220px" />
      ${p.kind === "lmstudio" ? "" : `<label class="check-inline"><input type="checkbox" id="m-free" ${p.kind === "huggingface" ? "checked" : ""} /> ${p.kind === "huggingface" ? "только дешёвые (хватит бесплатного кредита)" : "только бесплатные"}</label>`}
      <label class="check-inline">контекст от <select id="m-ctx"><option value="0">любой</option><option value="16000">16K</option><option value="32000">32K</option><option value="100000">100K</option><option value="200000">200K</option></select></label>
      <button type="button" class="secondary small" id="m-refresh">Обновить список</button></div>
    ${p.kind === "huggingface" ? `<div class="hint-text">Полностью бесплатных моделей у Hugging Face нет: каждому аккаунту даётся небольшой ежемесячный кредит (около $0,10). Поэтому показаны самые дешёвые модели — цена до $0,10 за 1 млн токенов на входе и до $0,30 на выходе; цены указаны в долларах за 1 млн токенов (вход / выход). Когда кредит кончится, сервис ответит ошибкой 402 до начала следующего месяца.</div>` : ""}
    ${isOr ? `<div class="hint-text">Цены указаны в долларах за 1 миллион токенов (вход / выход). Токен — примерно 3–4 буквы. Бесплатные модели могут быть медленнее и иметь лимиты.</div>` : ""}
    <div id="m-status" class="hint-text">Загрузка списка…</div>
    <div id="m-list" class="model-list"></div>
    <div class="inline-actions"><button type="button" class="secondary small" id="m-test">Проверить выбранную модель</button><span id="m-test-out" class="save-status"></span></div>`;
  const reload = (refresh) => loadModels(refresh);
  $("#m-q").addEventListener("input", () => { clearTimeout(provModelTimer); provModelTimer = setTimeout(() => reload(false), 250); });
  for (const id of ["#m-free", "#m-ctx"]) if ($(id)) $(id).addEventListener("change", () => reload(false));
  $("#m-refresh").addEventListener("click", () => reload(true));
  $("#m-test").addEventListener("click", () => testModel(pid));
  box.scrollIntoView({ behavior: "smooth", block: "nearest" });
  reload(false);
}

async function loadModels(refresh) {
  const pid = provModels.pid;
  const q = encodeURIComponent($("#m-q").value.trim());
  const free = $("#m-free") && $("#m-free").checked ? "&free_only=true" : "";
  try {
    const d = await api(`/api/llm/models?provider=${encodeURIComponent(pid)}&q=${q}${free}&min_context=${$("#m-ctx").value}${refresh ? "&refresh=true" : ""}`);
    provModels = { pid, list: d.models, total: d.total };
    renderModelList();
  } catch (e) {
    $("#m-list").innerHTML = "";
    $("#m-status").innerHTML = `<span class="err-text">${escapeHtml(e.message)}</span>`;
  }
}

function renderModelList() {
  const p = provState.providers.find((x) => x.id === provModels.pid);
  const cur = p.model;
  const shown = provModels.list.slice(0, 200);
  $("#m-status").textContent = `Найдено: ${provModels.list.length} из ${provModels.total}` + (provModels.list.length > shown.length ? ` (показаны первые ${shown.length} — уточните поиск)` : "");
  const autoRow = p.kind === "openrouter" && !$("#m-q").value.trim() ? `<div class="model-row${"openrouter/free" === cur ? " sel" : ""}" data-mid="openrouter/free" tabindex="0">
      <span class="m-id">openrouter/free</span><span class="hint-text">автоподбор: сервис сам выберет любую свободную бесплатную модель (удобно, когда конкретная модель перегружена)</span>
      <span class="m-badges"><span class="mbadge">бесплатно</span><button type="button" class="secondary small" data-fb="openrouter/free">в запасные</button></span></div>` : "";
  $("#m-list").innerHTML = autoRow + shown.map((m) => {
    const price = m.price_in !== null && m.price_in !== undefined ? `<span class="mbadge">${m.free ? "бесплатно" : `$${fmtPrice(m.price_in).replace("$", "")} / $${fmtPrice(m.price_out).replace("$", "")}`}</span>` : "";
    return `<div class="model-row${m.id === cur ? " sel" : ""}" data-mid="${escapeHtml(m.id)}" tabindex="0">
      <span class="m-id">${escapeHtml(m.id)}</span>${m.name && m.name !== m.id ? `<span class="hint-text">${escapeHtml(m.name)}</span>` : ""}
      <span class="m-badges">${m.loaded === true ? `<span class="mbadge ok">загружена</span>` : ""}${m.loaded === false ? `<span class="mbadge">не загружена</span>` : ""}
        ${m.context_length ? `<span class="mbadge" title="размер контекста">${fmtCtx(m.context_length)}</span>` : ""}${price}
        ${m.supports_reasoning ? `<span class="mbadge" title="умеет рассуждать">рассуждает</span>` : ""}
        <button type="button" class="secondary small" data-fb="${escapeHtml(m.id)}" title="добавить в список запасных моделей">в запасные</button></span></div>`;
  }).join("");
  $$("#m-list [data-fb]").forEach((b) => b.addEventListener("click", (ev) => { ev.stopPropagation(); addFallback(b.dataset.fb, b); }));
  $$("#m-list .model-row").forEach((row) => {
    row.addEventListener("click", () => pickModel(row.dataset.mid));
    row.addEventListener("keydown", (ev) => { if (ev.key === "Enter") pickModel(row.dataset.mid); });
  });
}

async function addFallback(mid, btn) {
  const pid = provModels.pid;
  const p = provState.providers.find((x) => x.id === pid);
  const cur = p.fallback_models || [];
  if (cur.includes(mid) || mid === p.model) { btn.textContent = "уже есть"; return; }
  if (cur.length >= 5) { btn.textContent = "максимум 5"; return; }
  try {
    await api(`/api/llm/providers/${pid}`, jsonPost("PUT", { fallback_models: [...cur, mid] }));
    p.fallback_models = [...cur, mid];
    btn.textContent = "добавлена ✓";
    $("#m-test-out").textContent = `Запасные модели: ${p.fallback_models.join(" → ")}`;
    loadProvidersUi().then(() => openModels(pid));
  } catch (e) { $("#m-test-out").textContent = "✗ " + e.message; }
}

async function pickModel(mid) {
  const pid = provModels.pid;
  try {
    await api(`/api/llm/providers/${pid}`, jsonPost("PUT", { default_model: mid }));
    const p = provState.providers.find((x) => x.id === pid);
    if (p && p.active) await api("/api/settings", jsonPost("POST", { llm_model: mid }));
    const keepQ = $("#m-q").value;
    await loadProvidersUi();
    await openModels(pid);
    $("#m-q").value = keepQ;
    $("#m-test-out").textContent = `Выбрана модель: ${mid}`;
  } catch (e) { $("#m-test-out").textContent = "✗ " + e.message; }
}

async function testModel(pid) {
  const out = $("#m-test-out"), p = provState.providers.find((x) => x.id === pid);
  const model = p.model || "";
  setStatus(out, "Проверка…");
  try {
    const r = await api(`/api/llm/providers/${pid}/test`, jsonPost("POST", { model }));
    setStatus(out, (r.ok ? "✓ " : "✗ ") + r.message, r.ok ? "ok" : "err");
  } catch (e) { setStatus(out, "✗ " + e.message, "err"); }
}

// ------------------------------------------------------- переопределения по задачам ---
async function loadTaskLlmUi() {
  const root = $("#task-llm-root");
  if (!root) return;
  const st = await api("/api/settings");
  const opts = (sel) => `<option value="">как в активном</option>` + provState.providers.map((p) =>
    `<option value="${escapeHtml(p.id)}"${p.id === sel ? " selected" : ""}>${provIcon(p)} ${escapeHtml(p.name)}</option>`).join("");
  root.innerHTML = `<h3>Какая модель что делает</h3>
    <div class="hint-text">По умолчанию всё делает активный провайдер. Можно, например, оставить подготовку запроса на маленькой локальной модели, а ответы получать из облака.
    «Подготовка запроса» — это лишний запрос к модели перед каждым вопросом: у бесплатных облачных моделей он тратит ваш лимит запросов, поэтому его выгоднее оставить на локальной модели.</div>
    <div class="set-grid">` + Object.entries(TASK_LABELS).map(([t, label]) => `<div class="set-row" data-task="${t}">
      <div class="set-label">${escapeHtml(label)}</div>
      <div class="set-field"><select data-tprov>${opts(st["llm_provider_" + t] || "")}</select>
        <input type="text" data-tmodel list="tm-list-${t}" placeholder="модель (пусто — выбранная у провайдера)" value="${escapeHtml(st["llm_model_" + t] || "")}" />
        <datalist id="tm-list-${t}"></datalist><div class="field-err"></div></div></div>`).join("") + `</div>
    <div class="set-actions"><button type="button" id="save-task-llm-btn">Сохранить</button><span id="task-llm-status" class="save-status"></span></div>`;
  $$("#task-llm-root .set-row[data-task]").forEach((row) => {
    // the model list is fetched only when the user focuses the field (no automatic requests to a cloud provider)
    row.querySelector("[data-tprov]").addEventListener("change", () => { row.querySelector("datalist").innerHTML = ""; row.dataset.filled = ""; });
    row.querySelector("[data-tmodel]").addEventListener("focus", () => { if (!row.dataset.filled) { row.dataset.filled = "1"; fillTaskModels(row); } });
  });
  $("#save-task-llm-btn").addEventListener("click", saveTaskLlm);
}

async function fillTaskModels(row) {
  const pid = row.querySelector("[data-tprov]").value || provState.active;
  const dl = row.querySelector("datalist"), err = row.querySelector(".field-err");
  err.textContent = "";
  try {
    const d = await api(`/api/llm/models?provider=${encodeURIComponent(pid)}`);
    dl.innerHTML = d.models.slice(0, 300).map((m) => `<option value="${escapeHtml(m.id)}"></option>`).join("");
  } catch (e) { dl.innerHTML = ""; err.textContent = "Список моделей недоступен: " + e.message; }
}

async function saveTaskLlm() {
  const st = $("#task-llm-status"), payload = {};
  $$("#task-llm-root .set-row[data-task]").forEach((row) => {
    payload["llm_provider_" + row.dataset.task] = row.querySelector("[data-tprov]").value;
    payload["llm_model_" + row.dataset.task] = row.querySelector("[data-tmodel]").value.trim();
  });
  try {
    await api("/api/settings", jsonPost("POST", payload));
    setStatus(st, "✓ Сохранено", "ok");
    refreshLlmBadge();
  } catch (e) { setStatus(st, "✗ " + e.message, "err"); }
}

// ---------------------------------------------- значок «Локально / Облако» ---
async function refreshLlmBadge() {
  const el = $("#llm-badge");
  if (!el) return;
  try {
    const [pv, st] = await Promise.all([api("/api/llm/providers"), api("/api/settings")]);
    const byId = Object.fromEntries(pv.providers.map((p) => [p.id, p]));
    const act = byId[pv.active];
    const taskProv = (t) => byId[st["llm_provider_" + t] || ""] || act;
    const main = taskProv("rag");
    const model = st["llm_model_rag"] || (main.id === pv.active ? st.llm_model : "") || main.default_model || "модель не выбрана";
    const others = ["translation", "rewrite"].filter((t) => taskProv(t).is_cloud && !main.is_cloud).map((t) => t === "translation" ? "перевод" : "подготовка запроса");
    const blocked = [main, taskProv("translation"), taskProv("rewrite")].some((p) => p.is_cloud) && !pv.allow_cloud_llm;
    el.classList.remove("hidden", "cloud", "local", "blocked");
    if (blocked) { el.classList.add("blocked"); el.textContent = "⚠ Облачный провайдер выбран, но облачные модели выключены"; }
    else if (main.is_cloud) { el.classList.add("cloud"); el.textContent = `☁ Облако: ${main.name} · ${model}`; }
    else {
      el.classList.add(others.length ? "cloud" : "local");
      el.textContent = `🖥 Локально` + (model ? ` · ${model}` : "") + (others.length ? ` (в облаке: ${others.join(", ")})` : "");
    }
    el.title = main.is_cloud ? `Тексты абзацев и вопросы отправляются на сервер ${main.host}` : "Тексты остаются на этом компьютере";
  } catch (e) { el.classList.add("hidden"); }
}
refreshLlmBadge();
