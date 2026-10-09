"use strict";
/* Настройки: «Параметры ответа» (форма строится из GET /api/options) и «Промты» (редакторы + наборы).
 * Зависит от app.js: $, $$, api, escapeHtml, setStatus. */

const OLD_SERVER_HINT = " (если ошибка 404 — программа ещё работает на старой версии, перезапустите её)";
const jsonPost = (method, body) => ({ method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body ?? {}) });

// ================================================================== Параметры ответа =====
let optionsSpec = null;

function optNumFmt(v) { return String(v).replace(".", ","); }
function optDefaultText(o) {
  if (o.type === "bool") return o.default ? "включено" : "выключено";
  if (o.type === "choice") { const c = o.choices.find((x) => x.value === o.default); return c ? c.label : String(o.default); }
  return optNumFmt(o.default);
}

function optControlHtml(o) {
  const k = escapeHtml(o.key);
  if (o.type === "bool") return `<label class="check-inline"><input type="checkbox" data-opt="${k}" /> включено</label>`;
  if (o.type === "choice") {
    return `<div class="radio-list">` + o.choices.map((c) =>
      `<label><input type="radio" name="opt_${k}" value="${escapeHtml(c.value)}" data-opt="${k}" /><span>${escapeHtml(c.label)}</span></label>`).join("") + `</div>`;
  }
  return `<input type="number" data-opt="${k}" min="${o.min}" max="${o.max}" step="${o.step || (o.type === "int" ? 1 : "any")}" />`;
}

function optRowHtml(o) {
  const range = (o.type === "int" || o.type === "float") ? `от ${optNumFmt(o.min)} до ${optNumFmt(o.max)} · ` : "";
  return `<div class="set-row" data-optrow="${escapeHtml(o.key)}">
    <div class="set-label">${escapeHtml(o.label)}<span class="hint-text">${escapeHtml(o.hint || "")}</span></div>
    <div class="set-field">${optControlHtml(o)}
      <div class="opt-meta"><span>${range}по умолчанию: ${escapeHtml(optDefaultText(o))}</span>
        <button type="button" class="link-btn" data-optdef="${escapeHtml(o.key)}">по умолчанию</button><span class="changed"></span></div>
      <div class="field-err"></div></div></div>`;
}

function optGetValue(o, row) {
  if (o.type === "bool") return row.querySelector("input[type=checkbox]").checked;
  if (o.type === "choice") { const r = row.querySelector("input[type=radio]:checked"); return r ? r.value : o.default; }
  const raw = row.querySelector("input[type=number]").value.trim().replace(",", ".");
  if (raw === "") return null;
  return o.type === "int" ? (Number.isInteger(Number(raw)) ? parseInt(raw, 10) : Number(raw)) : parseFloat(raw);
}

function optSetValue(o, row, v) {
  if (o.type === "bool") row.querySelector("input[type=checkbox]").checked = !!v;
  else if (o.type === "choice") row.querySelectorAll("input[type=radio]").forEach((r) => { r.checked = r.value === String(v); });
  else row.querySelector("input[type=number]").value = v;
}

function optRefreshMeta(o, row) {
  const v = optGetValue(o, row);
  const isDef = v === o.default || (typeof v === "number" && v === Number(o.default));
  row.querySelector("[data-optdef]").disabled = isDef;
  row.querySelector(".changed").textContent = isDef ? "" : "изменено (не стандартное)";
}

async function loadOptionsForm() {
  const root = $("#options-root");
  try {
    optionsSpec = await api("/api/options");
  } catch (e) {
    root.innerHTML = `<div class="msg-inline err">Не удалось загрузить параметры: ${escapeHtml(e.message)}${OLD_SERVER_HINT}</div>`;
    return;
  }
  root.innerHTML = optionsSpec.groups.map((g) => {
    const items = optionsSpec.options.filter((o) => o.group === g.id);
    if (!items.length) return "";
    return `<div class="sub-card"><h3>${escapeHtml(g.title)}</h3><div class="set-grid">${items.map(optRowHtml).join("")}</div></div>`;
  }).join("");
  for (const o of optionsSpec.options) {
    const row = root.querySelector(`[data-optrow="${o.key}"]`);
    if (!row) continue;
    optSetValue(o, row, o.value);
    optRefreshMeta(o, row);
    row.addEventListener("input", () => { optRefreshMeta(o, row); clearOptErrors(row); });
    row.addEventListener("change", () => { optRefreshMeta(o, row); clearOptErrors(row); });
    row.querySelector("[data-optdef]").addEventListener("click", () => { optSetValue(o, row, o.default); optRefreshMeta(o, row); clearOptErrors(row); });
  }
}

function clearOptErrors(row) {
  if (row) { row.querySelector(".field-err").textContent = ""; row.querySelectorAll("input").forEach((i) => i.classList.remove("invalid")); return; }
  $$("#options-root .field-err").forEach((e) => { e.textContent = ""; });
  $$("#options-root input.invalid").forEach((i) => i.classList.remove("invalid"));
}

$("#options-defaults-btn").addEventListener("click", () => {
  if (!optionsSpec) return;
  for (const o of optionsSpec.options) {
    const row = $(`#options-root [data-optrow="${o.key}"]`);
    if (row) { optSetValue(o, row, o.default); optRefreshMeta(o, row); }
  }
  setStatus($("#options-status"), "Подставлены стандартные значения — нажмите «Сохранить параметры».");
});

$("#save-options-btn").addEventListener("click", async () => {
  if (!optionsSpec) return;
  const st = $("#options-status");
  clearOptErrors();
  const payload = {};
  for (const o of optionsSpec.options) {
    const row = $(`#options-root [data-optrow="${o.key}"]`);
    if (!row) continue;
    const v = optGetValue(o, row);
    if (v === null) {
      row.querySelector(".field-err").textContent = "Введите число.";
      row.querySelector("input").classList.add("invalid");
      setStatus(st, "✗ Не сохранено: есть незаполненные поля", "err");
      return;
    }
    payload[o.key] = v;
  }
  setStatus(st, "Сохранение…");
  try {
    await api("/api/settings", jsonPost("POST", payload));
    setStatus(st, "✓ Сохранено — применится со следующего вопроса", "ok");
    loadOptionsForm();
  } catch (e) {
    // the server message names the offending field: «Подпись поля»: …
    const bad = optionsSpec.options.find((o) => e.message.includes("«" + o.label + "»"));
    if (bad) {
      const row = $(`#options-root [data-optrow="${bad.key}"]`);
      row.querySelector(".field-err").textContent = e.message;
      const inp = row.querySelector("input");
      if (inp) inp.classList.add("invalid");
      row.scrollIntoView({ block: "center", behavior: "smooth" });
      setStatus(st, "✗ Не сохранено: проверьте выделенное поле", "err");
    } else {
      setStatus(st, "✗ " + e.message, "err");
    }
  }
});

// ================================================================== Промты ===============
let promptsData = null;      // last GET /api/prompts
let presetsData = null;      // {presets, active}
const promptItems = {};      // id -> server item

const PH_RE = /\{([A-Za-z_][A-Za-z0-9_]*)\}/g;

function estTokens(t) { return Math.max(1, Math.floor(t.length / 3)); }
function cardEl(id) { return document.getElementById("pc-" + id); }
function cardDirty(id) { const c = cardEl(id); return !!c && c.querySelector("textarea").value !== promptItems[id].text; }
function anyDirty() { return Object.keys(promptItems).filter(cardDirty); }

// ---- line diff (LCS) for «Показать стандартный» ----
function lineDiff(a, b) {
  const n = a.length, m = b.length;
  const dp = Array.from({ length: n + 1 }, () => new Int32Array(m + 1));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--)
    dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  const ops = [];
  let i = 0, j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) { ops.push({ op: "=", a: a[i], b: b[j] }); i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) ops.push({ op: "-", a: a[i++] });
    else ops.push({ op: "+", b: b[j++] });
  }
  while (i < n) ops.push({ op: "-", a: a[i++] });
  while (j < m) ops.push({ op: "+", b: b[j++] });
  return ops;
}

function diffViewHtml(defText, curText) {
  if (defText === curText) return `<div class="diff-title">Текст совпадает со стандартным.</div><pre class="snap-pre">${escapeHtml(defText)}</pre>`;
  const ops = lineDiff(defText.split("\n"), curText.split("\n"));
  const cells = ops.map((o) => {
    if (o.op === "=") return `<div class="dl">${escapeHtml(o.a)}</div><div class="dl">${escapeHtml(o.b)}</div>`;
    if (o.op === "-") return `<div class="dl del">${escapeHtml(o.a)}</div><div class="dl gap"></div>`;
    return `<div class="dl gap"></div><div class="dl add">${escapeHtml(o.b)}</div>`;
  }).join("");
  return `<div class="diff-title">Слева — стандартный текст (только чтение), справа — то, что сейчас в поле. Красным — строки, которых вы лишились; зелёным — добавленные.</div>
    <div class="diff-cols" style="display:grid"><h5>Стандартный</h5><h5>Сейчас в поле</h5>${cells}</div>`;
}

function promptCardHtml(p) {
  const ph = Object.entries(p.placeholders || {});
  const chips = ph.length
    ? `<div class="chips">Подстановки (нажмите, чтобы вставить в место курсора): ` + ph.map(([name, desc]) =>
        `<button type="button" class="chip${(p.required || []).includes(name) ? " req" : ""}" data-ph="${escapeHtml(name)}" title="${escapeHtml(desc)}">{${escapeHtml(name)}}</button>`).join("") +
      `</div><div class="hint-text" style="margin-bottom:6px">${ph.map(([n, d]) => `<code>{${escapeHtml(n)}}</code> — ${escapeHtml(d)}`).join("; ")}. Звёздочка — обязательная.</div>`
    : `<div class="hint-text" style="margin-bottom:6px">В этом промте нет подстановок — это обычный текст.</div>`;
  return `<details class="prompt-card" id="pc-${escapeHtml(p.id)}" data-pid="${escapeHtml(p.id)}">
    <summary class="prompt-head"><h4>${escapeHtml(p.title)}</h4><span class="pill-wrap"></span></summary>
    <div class="prompt-desc">${escapeHtml(p.description)}</div>
    ${chips}
    <textarea class="prompt-text" spellcheck="false"></textarea>
    <div class="live-check"></div>
    <div class="prompt-foot">
      <span class="prompt-count"></span>
      <button type="button" class="small" data-act="save">Сохранить</button>
      <button type="button" class="small secondary" data-act="reset">Сбросить к стандартному</button>
      <button type="button" class="small secondary" data-act="default">Показать стандартный</button>
    </div>
    <div class="srv-msg"></div>
    <div class="diff-view hidden"></div>
  </details>`;
}

function refreshCard(id) {
  const c = cardEl(id); if (!c) return;
  const p = promptItems[id];
  const ta = c.querySelector("textarea");
  const dirty = ta.value !== p.text;
  c.classList.toggle("dirty", dirty);
  if (dirty) c.open = true;
  c.querySelector(".pill-wrap").innerHTML =
    `<span class="pill${p.custom ? " custom" : ""}" title="Короткий код текста: меняется при любой правке">${escapeHtml(p.info.label)}</span>` +
    (dirty ? ` <span class="pill dirty">не сохранено</span>` : "");
  c.querySelector(".prompt-count").textContent =
    `${ta.value.length} знаков · ≈${estTokens(ta.value)} токенов (стандартный: ${p.default_info.chars} знаков, ≈${p.default_info.tokens})`;
  // live placeholder check (the server still has the last word)
  const found = new Set(); let m; PH_RE.lastIndex = 0;
  while ((m = PH_RE.exec(ta.value))) found.add(m[1]);
  const allowed = Object.keys(p.placeholders || {});
  const unknown = [...found].filter((x) => !allowed.includes(x));
  const missing = (p.required || []).filter((x) => !found.has(x));
  const msgs = [];
  if (unknown.length) msgs.push("Лишние подстановки: " + unknown.map((x) => "{" + x + "}").join(", ") + " — такие значения программа не знает.");
  if (missing.length) msgs.push("Не хватает обязательных: " + missing.map((x) => "{" + x + "}").join(", ") + ".");
  c.querySelector(".live-check").innerHTML = msgs.map((t) => `<div class="msg-inline warn">${escapeHtml(t)}</div>`).join("");
  const dv = c.querySelector(".diff-view");
  if (!dv.classList.contains("hidden")) dv.innerHTML = diffViewHtml(p.default, ta.value);
  renderPresetState();
}

function setCardItem(item, keepText) {
  promptItems[item.id] = item;
  const c = cardEl(item.id); if (!c) return;
  if (!keepText) c.querySelector("textarea").value = item.text;
  refreshCard(item.id);
}

function srvMsg(id, text, kind) {
  const c = cardEl(id); if (!c) return;
  c.querySelector(".srv-msg").innerHTML = text ? `<div class="msg-inline ${kind}">${escapeHtml(text)}</div>` : "";
}

function renderPromptsAll(data) {
  promptsData = data;
  Object.keys(promptItems).forEach((k) => delete promptItems[k]);
  data.prompts.forEach((p) => { promptItems[p.id] = p; });
  $("#prompts-global-warn").innerHTML = (data.warnings || []).concat(data.load_warnings || [])
    .map((w) => `<div class="msg-inline warn">${escapeHtml(w)}</div>`).join("");
  $("#prompts-root").innerHTML = data.groups.map((g) => {
    const items = data.prompts.filter((p) => p.group === g.id);
    return items.length ? `<div class="prompt-group"><div class="prompt-group-title">${escapeHtml(g.title)}</div>` + items.map(promptCardHtml).join("") + `</div>` : "";
  }).join("");
  for (const p of data.prompts) {
    const c = cardEl(p.id);
    c.querySelector("textarea").value = p.text;
    bindCard(c, p.id);
    refreshCard(p.id);
  }
  renderPresetState();
}

function bindCard(c, id) {
  const ta = c.querySelector("textarea");
  ta.addEventListener("input", () => { refreshCard(id); srvMsg(id, ""); });
  c.querySelectorAll(".chip").forEach((b) => b.addEventListener("click", () => {
    const t = "{" + b.dataset.ph + "}";
    ta.focus();
    ta.setRangeText(t, ta.selectionStart, ta.selectionEnd, "end");
    ta.dispatchEvent(new Event("input"));
  }));
  c.querySelector('[data-act="save"]').addEventListener("click", () => savePrompt(id));
  c.querySelector('[data-act="reset"]').addEventListener("click", () => resetPrompt(id));
  c.querySelector('[data-act="default"]').addEventListener("click", (ev) => {
    const dv = c.querySelector(".diff-view");
    const show = dv.classList.contains("hidden");
    dv.classList.toggle("hidden", !show);
    ev.target.textContent = show ? "Скрыть стандартный" : "Показать стандартный";
    if (show) dv.innerHTML = diffViewHtml(promptItems[id].default, ta.value);
  });
}

async function refreshPromptMeta() {
  // pull fresh warnings / preset flags without touching the text fields being edited
  try {
    const [d, pr] = await Promise.all([api("/api/prompts"), api("/api/prompt-presets")]);
    promptsData = d; presetsData = pr;
    $("#prompts-global-warn").innerHTML = (d.warnings || []).map((w) => `<div class="msg-inline warn">${escapeHtml(w)}</div>`).join("");
    renderPresetBar();
  } catch (e) { /* non-critical */ }
}

async function savePrompt(id) {
  const ta = cardEl(id).querySelector("textarea");
  srvMsg(id, "Сохранение…", "ok");
  try {
    const item = await api("/api/prompts/" + encodeURIComponent(id), jsonPost("PUT", { text: ta.value }));
    setCardItem(item, false);
    srvMsg(id, "✓ Сохранено — действует со следующего вопроса.", "ok");
    refreshPromptMeta();
  } catch (e) {
    srvMsg(id, e.message, "err");
  }
}

async function resetPrompt(id) {
  try {
    const item = await api(`/api/prompts/${encodeURIComponent(id)}/reset`, jsonPost("POST"));
    setCardItem(item, false);
    srvMsg(id, "✓ Возвращён стандартный текст.", "ok");
    refreshPromptMeta();
  } catch (e) { srvMsg(id, e.message, "err"); }
}

async function loadPromptsUi(force) {
  if (!force && anyDirty().length) return;   // never overwrite unsaved edits
  const root = $("#prompts-root");
  try {
    const [d, pr] = await Promise.all([api("/api/prompts"), api("/api/prompt-presets")]);
    presetsData = pr;
    renderPromptsAll(d);
    renderPresetBar();
  } catch (e) {
    root.innerHTML = `<div class="msg-inline err">Не удалось загрузить промты: ${escapeHtml(e.message)}${OLD_SERVER_HINT}</div>`;
  }
}

// ---------------------------------------------------------------- presets -----
let presetAction = null;   // {kind, ...} while an inline action panel is open

function activePresetName() { return presetsData ? presetsData.active : null; }
function presetByName(n) { return presetsData && presetsData.presets.find((p) => p.name === n); }

function renderPresetBar() {
  const bar = $("#prompt-presets");
  if (!presetsData) { bar.innerHTML = ""; return; }
  const list = presetsData.presets;
  const opts = list.length
    ? list.map((p) => `<option value="${escapeHtml(p.name)}">${escapeHtml(p.name)} (${p.count} изм.)${p.active ? " — активный" : ""}</option>`).join("")
    : `<option value="">(наборов пока нет)</option>`;
  const keep = ($("#preset-select") || {}).value;
  bar.innerHTML = `
    <div class="preset-line"><strong>Наборы промтов</strong><span id="preset-state" class="preset-state"></span></div>
    <div class="preset-line">
      <select id="preset-select" title="Сохранённые наборы промтов">${opts}</select>
      <button type="button" class="small secondary" data-pa="load">Загрузить</button>
      <button type="button" class="small secondary" data-pa="rename">Переименовать</button>
      <button type="button" class="small secondary" data-pa="delete">Удалить</button>
      <button type="button" class="small secondary" data-pa="export">Экспорт в файл</button>
    </div>
    <div class="preset-line">
      <button type="button" class="small" data-pa="saveas">Сохранить текущий набор как…</button>
      <button type="button" class="small secondary" data-pa="import">Импорт из файла</button>
      <button type="button" class="small secondary" data-pa="resetall">Сбросить всё к стандартным</button>
      <input type="file" id="preset-file" accept=".json,application/json" class="hidden" />
    </div>
    <div id="preset-action" class="preset-action hidden"></div>`;
  const sel = $("#preset-select");
  if (keep && list.some((p) => p.name === keep)) sel.value = keep;
  else if (presetsData.active && list.some((p) => p.name === presetsData.active)) sel.value = presetsData.active;
  bar.querySelectorAll("[data-pa]").forEach((b) => b.addEventListener("click", () => startPresetAction(b.dataset.pa)));
  $("#preset-file").addEventListener("change", importPresetFile);
  syncSelectTitle(sel);
  sel.addEventListener("change", () => syncSelectTitle(sel));
  renderPresetState();
}

function renderPresetState() {
  const el = $("#preset-state"); if (!el || !presetsData) return;
  const active = presetsData.active;
  const custom = promptsData ? promptsData.prompts.filter((p) => promptItems[p.id] && promptItems[p.id].custom).length : 0;
  const unsaved = anyDirty().length;
  const changed = (promptsData && promptsData.preset_dirty) || unsaved > 0;
  let html = active
    ? ` · активный набор: <span class="pill custom">${escapeHtml(active)}</span>${changed ? ' <span class="pill dirty">изменён</span>' : ""}`
    : ` · набор не выбран${custom ? "" : " (используются стандартные тексты)"}`;
  html += custom ? ` · свои тексты: ${custom} из ${Object.keys(promptItems).length}` : "";
  if (unsaved) html += ` · <span class="pill dirty">несохранённых правок: ${unsaved}</span>`;
  el.innerHTML = html;
}

function openActionPanel(html) {
  const p = $("#preset-action");
  p.innerHTML = html;
  p.classList.remove("hidden");
  return p;
}
function closeActionPanel() { presetAction = null; const p = $("#preset-action"); if (p) { p.classList.add("hidden"); p.innerHTML = ""; } }

function actionMsg(text, kind) {
  const p = $("#preset-action");
  p.classList.remove("hidden");
  let m = p.querySelector(".pa-msg");
  if (!m) { m = document.createElement("div"); m.className = "pa-msg"; m.style.flexBasis = "100%"; p.appendChild(m); }
  m.innerHTML = text ? `<div class="msg-inline ${kind || "err"}">${escapeHtml(text)}</div>` : "";
}

function selectedPreset() { const s = $("#preset-select"); return s ? s.value : ""; }

function startPresetAction(kind) {
  closeActionPanel();
  const sel = selectedPreset();
  const needSel = ["load", "rename", "delete", "export"].includes(kind);
  if (needSel && !sel) { openActionPanel(`<span>Сначала выберите набор в списке.</span><button type="button" class="small secondary" data-cancel>Закрыть</button>`).querySelector("[data-cancel]").addEventListener("click", closeActionPanel); return; }
  presetAction = { kind, name: sel };
  if (kind === "export") {
    const a = document.createElement("a");
    a.href = "/api/prompt-presets-export?name=" + encodeURIComponent(sel);
    a.download = "prompt-preset.json";
    document.body.appendChild(a); a.click(); a.remove();
    presetAction = null;
    return;
  }
  if (kind === "import") { $("#preset-file").value = ""; $("#preset-file").click(); presetAction = null; return; }

  let panel;
  if (kind === "saveas") {
    panel = openActionPanel(`<span>Название набора:</span><input type="text" id="pa-name" maxlength="80" value="${escapeHtml(activePresetName() || "")}" />
      <input type="text" id="pa-note" placeholder="заметка (необязательно)" />
      <button type="button" class="small" data-ok>Сохранить набор</button><button type="button" class="small secondary" data-cancel>Отмена</button>`);
  } else if (kind === "rename") {
    panel = openActionPanel(`<span>Новое название для «${escapeHtml(sel)}»:</span><input type="text" id="pa-name" maxlength="80" value="${escapeHtml(sel)}" />
      <button type="button" class="small" data-ok>Переименовать</button><button type="button" class="small secondary" data-cancel>Отмена</button>`);
  } else if (kind === "delete") {
    panel = openActionPanel(`<span>Удалить набор «${escapeHtml(sel)}»? Текущие тексты промтов не изменятся.</span>
      <button type="button" class="small" data-ok>Да, удалить</button><button type="button" class="small secondary" data-cancel>Отмена</button>`);
  } else if (kind === "load") {
    const n = anyDirty().length;
    const warn = (n || (promptsData && promptsData.preset_dirty))
      ? `Ваши текущие правки промтов будут заменены содержимым набора «${escapeHtml(sel)}». `
      : "";
    panel = openActionPanel(`<span>${warn}Загрузить набор «${escapeHtml(sel)}»?</span>
      <button type="button" class="small" data-ok>Загрузить</button><button type="button" class="small secondary" data-cancel>Отмена</button>`);
  } else if (kind === "resetall") {
    panel = openActionPanel(`<span>Вернуть ВСЕ промты к стандартным текстам (ваши правки будут потеряны, сохранённые наборы останутся)?</span>
      <button type="button" class="small" data-ok>Да, сбросить всё</button><button type="button" class="small secondary" data-cancel>Отмена</button>`);
  }
  panel.querySelector("[data-cancel]").addEventListener("click", closeActionPanel);
  const nameInput = panel.querySelector("#pa-name");
  if (nameInput) { nameInput.focus(); nameInput.select(); nameInput.addEventListener("keydown", (ev) => { if (ev.key === "Enter") panel.querySelector("[data-ok]").click(); }); }
  panel.querySelector("[data-ok]").addEventListener("click", () => doPresetAction(kind, sel));
}

async function doPresetAction(kind, sel) {
  try {
    if (kind === "saveas") {
      const name = $("#pa-name").value.trim();
      const dirty = anyDirty();
      if (dirty.length) { actionMsg(`Есть несохранённые правки в промтах (${dirty.length}). Сначала нажмите «Сохранить» в их карточках, затем сохраните набор.`); return; }
      if (presetByName(name) && !presetAction.confirmedOverwrite) {
        presetAction.confirmedOverwrite = true;
        actionMsg(`Набор «${name}» уже есть. Нажмите «Сохранить набор» ещё раз, чтобы заменить его.`, "warn");
        return;
      }
      presetsData = await api("/api/prompt-presets", jsonPost("POST", { name, note: ($("#pa-note").value || "").trim() }));
      closeActionPanel(); await refreshPromptMeta(); renderPresetBar();
    } else if (kind === "rename") {
      presetsData = await api("/api/prompt-presets/" + encodeURIComponent(sel), jsonPost("PUT", { new_name: $("#pa-name").value.trim() }));
      closeActionPanel(); await refreshPromptMeta(); renderPresetBar();
    } else if (kind === "delete") {
      presetsData = await api("/api/prompt-presets/" + encodeURIComponent(sel), { method: "DELETE" });
      closeActionPanel(); await refreshPromptMeta(); renderPresetBar();
    } else if (kind === "load") {
      const d = await api(`/api/prompt-presets/${encodeURIComponent(sel)}/load`, jsonPost("POST"));
      presetsData = await api("/api/prompt-presets");
      renderPromptsAll(d); renderPresetBar();
      if (d.load_warnings && d.load_warnings.length) actionMsg(d.load_warnings.join(" "), "warn");
    } else if (kind === "resetall") {
      const d = await api("/api/prompts/reset-all", jsonPost("POST"));
      presetsData = await api("/api/prompt-presets");
      renderPromptsAll(d); renderPresetBar();
    }
  } catch (e) {
    if ($("#preset-action")) actionMsg(e.message);
  }
}

async function importPresetFile(ev) {
  const f = ev.target.files && ev.target.files[0];
  if (!f) return;
  let data;
  try { data = JSON.parse(await f.text()); }
  catch (e) { openActionPanel(`<span>Файл не похож на JSON.</span>`); return; }
  try {
    const res = await api("/api/prompt-presets-import", jsonPost("POST", { data }));
    presetsData = await api("/api/prompt-presets");
    renderPresetBar();
    const imp = res.imported || [];
    const prob = res.problems || [];
    openActionPanel(`<span>✓ Импортировано наборов: ${imp.length}${imp.length ? " (" + imp.map((x) => "«" + escapeHtml(typeof x === "string" ? x : x.name || "") + "»").join(", ") + ")" : ""}.</span>` +
      (prob.length ? `<div class="pa-msg" style="flex-basis:100%">${prob.map((p) => `<div class="msg-inline warn">${escapeHtml(p)}</div>`).join("")}</div>` : "") +
      `<button type="button" class="small secondary" data-cancel>Закрыть</button>`).querySelector("[data-cancel]").addEventListener("click", closeActionPanel);
  } catch (e) {
    openActionPanel(`<span class="msg-inline err">${escapeHtml(e.message)}</span><button type="button" class="small secondary" data-cancel>Закрыть</button>`)
      .querySelector("[data-cancel]").addEventListener("click", closeActionPanel);
  }
}

window.loadOptionsForm = loadOptionsForm;
window.loadPromptsUi = loadPromptsUi;
