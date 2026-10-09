"use strict";
// Reading page: one paper (document) at a time, bilingual, with a hideable table of contents,
// quick title filter, previous/next, reader settings and the search-word highlight carried over
// from the search page. Address: read.html#p=2&para=2:5.10&s=5&q=<search query>.

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const WORD_RE = /[\p{L}]+/gu;
const normWord = (w) => w.toLowerCase().replace(/ё/g, "е");

function escapeHtml(s) {
  return (s || "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function api(path) {
  let res;
  try { res = await fetch(path); } catch (e) {
    throw new Error("нет связи с программой — похоже, она закрыта или перезапускается. Запустите start.bat и обновите страницу.");
  }
  if (!res.ok) {
    let msg = res.statusText;
    try { const j = await res.json(); msg = j.detail || msg; } catch (e) {}
    throw new Error(msg);
  }
  return res.json();
}

// ------------------------------------------------------------ settings -----
const DEFAULTS = {
  primary: "ru", second: "columns", secondSize: 90, pairs: true,
  font: "serif", size: 18, weight: 400, lineHeight: 1.65, letter: 0, word: 0, paraGap: 1.1, width: 1100,
  align: "justify", nums: "full", theme: "light",
};
const FONTS = {
  serif: '"Cambria", "Georgia", "Times New Roman", serif',
  georgia: '"Georgia", "Cambria", serif',
  sans: '"Segoe UI Variable Text", "Segoe UI", system-ui, sans-serif',
};
const FORMAT = {
  secondSize: (v) => `${v}%`, size: (v) => `${v}px`, weight: (v) => v, lineHeight: (v) => Number(v).toFixed(2),
  letter: (v) => `${Number(v).toFixed(2)} em`, word: (v) => `${Number(v).toFixed(2)} em`,
  paraGap: (v) => `${Number(v).toFixed(1)} em`, width: (v) => `${v}px`,
};

let S = loadSettings();

function loadSettings() {
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem("reader_settings") || "{}"); } catch (e) {}
  const out = { ...DEFAULTS, ...saved };
  try { out.theme = localStorage.getItem("theme") || out.theme; } catch (e) {}
  return out;
}

function saveSettings() {
  try {
    localStorage.setItem("reader_settings", JSON.stringify(S));
    localStorage.setItem("theme", S.theme);   // shared with the search page
  } catch (e) {}
}

function applySettings() {
  const root = document.documentElement;
  root.setAttribute("data-theme", S.theme);
  const t = $("#text");
  t.style.setProperty("--r-font", FONTS[S.font] || FONTS.serif);
  t.style.setProperty("--r-size", `${S.size}px`);
  t.style.setProperty("--r-weight", S.weight);
  t.style.setProperty("--r-lh", S.lineHeight);
  t.style.setProperty("--r-letter", `${S.letter}em`);
  t.style.setProperty("--r-word", `${S.word}em`);
  t.style.setProperty("--r-gap", `${S.paraGap}em`);
  t.style.setProperty("--r-second", `${S.secondSize / 100}`);
  document.documentElement.style.setProperty("--r-width", `${S.width}px`);
  t.classList.remove("mode-columns", "mode-inter", "mode-off", "nums-full", "nums-seq", "nums-off", "align-left");
  t.classList.add(`mode-${S.second}`, `nums-${S.nums}`);
  if (S.align === "left") t.classList.add("align-left");
  document.body.classList.toggle("ru-first", S.primary === "ru");
  document.body.classList.toggle("no-pairs", !S.pairs);
  document.body.classList.toggle("second-off", S.second === "off");
  syncSettingsUi();
}

function syncSettingsUi() {
  $$("#settings-panel .seg").forEach((seg) => {
    seg.querySelectorAll("button").forEach((b) => b.classList.toggle("on", String(S[seg.dataset.key]) === b.dataset.v));
  });
  $$("#settings-panel input[type=range]").forEach((inp) => {
    inp.value = S[inp.dataset.key];
    const out = inp.closest("label").querySelector("output");
    if (out) out.textContent = (FORMAT[inp.dataset.key] || String)(S[inp.dataset.key]);
  });
  $$("#settings-panel input[type=checkbox]").forEach((inp) => { inp.checked = !!S[inp.dataset.key]; });
}

function initSettingsPanel() {
  const panel = $("#settings-panel");
  const open = (on) => {
    panel.classList.toggle("hidden", !on);
    $("#settings-btn").setAttribute("aria-expanded", String(on));
  };
  $("#settings-btn").addEventListener("click", () => open(panel.classList.contains("hidden")));
  $("#settings-close").addEventListener("click", () => open(false));
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") open(false); });
  panel.addEventListener("click", (e) => {
    const b = e.target.closest(".seg button");
    if (!b) return;
    const key = b.closest(".seg").dataset.key;
    S[key] = b.dataset.v;
    const rerender = key === "primary" || key === "second";
    applySettings(); saveSettings();
    if (rerender) { renderToc(); markCurrentInToc(); }
  });
  panel.addEventListener("input", (e) => {
    const inp = e.target;
    if (inp.type === "range") S[inp.dataset.key] = Number(inp.value);
    else if (inp.type === "checkbox") S[inp.dataset.key] = inp.checked;
    else return;
    applySettings(); saveSettings();
  });
  $("#settings-reset").addEventListener("click", () => {
    S = { ...DEFAULTS, theme: S.theme };
    applySettings(); saveSettings(); renderToc(); markCurrentInToc();
  });
}

// ----------------------------------------------------------- addresses -----
function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  return {
    paper: h.has("p") ? Number(h.get("p")) : null,
    section: h.has("s") ? Number(h.get("s")) : null,
    para: h.get("para") || null,
    q: h.get("q") || "",
  };
}

function hashFor({ paper, section = null, para = null, q = state.q }) {
  const h = new URLSearchParams();
  h.set("p", paper);
  if (section != null) h.set("s", section);
  if (para) h.set("para", para);
  if (q) h.set("q", q);
  return "#" + h.toString();
}

function go(target) {
  const next = hashFor(target);
  if (next === location.hash) onHash(); else location.hash = next;
}

// ------------------------------------------------------------------ TOC -----
const state = { toc: [], paper: null, data: null, q: "", flat: [] };

// Titles: the primary language first, the other one under it (unless the second language is hidden).
function titleHtml(t, numLabel) {
  if (!t || (!t.ru && !t.en)) return `<span class="t1">${escapeHtml(numLabel)}</span>`;
  const first = S.primary === "ru" ? t.ru || t.en : t.en || t.ru;
  const second = S.primary === "ru" ? t.en : t.ru;
  return `<span class="t1">${escapeHtml(first)}</span>` +
    (S.second !== "off" && second && second !== first ? `<span class="t2">${escapeHtml(second)}</span>` : "");
}

const hasTitle = (t) => t && (t.ru || t.en);

function renderToc() {
  const filter = normWord($("#toc-filter").value.trim());
  const match = (t, extra) => !filter ||
    normWord(`${t?.ru || ""} ${t?.en || ""} ${extra}`).includes(filter);
  // The sections of the current paper unfold smoothly when it changes; the previous paper's list stays
  // for one more render to fold up. No animation while filtering (the tree is redrawn on every key).
  const changed = !filter && state.paper !== state.tocOpenPaper;
  const closing = changed ? state.tocOpenPaper : null;
  if (!filter) state.tocOpenPaper = state.paper;
  const html = [];
  for (const part of state.toc) {
    const papers = part.papers.map((p) => {
      const paperHit = match(p.title, `${p.paper}`);
      const secs = p.sections.filter((s) => hasTitle(s.title) && (paperHit || match(s.title, `${p.paper}:${s.section}`)));
      if (!paperHit && !secs.length) return "";
      const showSecs = filter ? secs : (p.paper === state.paper || p.paper === closing ? secs : []);
      const anim = !changed ? "" : p.paper === state.paper ? " opening" : p.paper === closing ? " closing" : "";
      return `<div class="toc-paper${p.paper === state.paper ? " current" : ""}">
        <a href="${hashFor({ paper: p.paper })}" data-paper="${p.paper}"><span class="toc-num">${p.paper}</span><span class="toc-t">${titleHtml(p.title, `Документ ${p.paper}`)}</span></a>
        ${showSecs.length ? `<div class="toc-secs${anim}">${showSecs.map((s) =>
          `<a href="${hashFor({ paper: p.paper, section: s.section })}" data-paper="${p.paper}" data-section="${s.section}"><span class="toc-num">${s.section}</span><span class="toc-t">${titleHtml(s.title, "")}</span></a>`).join("")}</div>` : ""}
      </div>`;
    }).join("");
    if (!papers) continue;
    html.push(`<div class="toc-part">${hasTitle(part.title) ? `<div class="toc-part-title">${part.part != null ? `Часть ${part.part}. ` : ""}${titleHtml(part.title, "")}</div>` : ""}${papers}</div>`);
  }
  $("#toc-tree").innerHTML = html.join("") || `<div class="hint-text toc-empty">Ничего не найдено</div>`;
}

function markCurrentInToc() {
  const cur = $("#toc-tree .toc-paper.current");
  if (cur && !$("#toc-filter").value) cur.scrollIntoView({ block: "nearest" });
}

function initToc() {
  const toggle = $("#toc-toggle");
  const setOpen = (on) => {
    document.body.classList.toggle("toc-hidden", !on);
    toggle.textContent = on ? "‹" : "›";
    toggle.setAttribute("aria-expanded", String(on));
    try { localStorage.setItem("reader_toc", on ? "1" : "0"); } catch (e) {}
  };
  let open = window.innerWidth > 900;
  try { const v = localStorage.getItem("reader_toc"); if (v !== null && window.innerWidth > 900) open = v === "1"; } catch (e) {}
  setOpen(open);
  toggle.addEventListener("click", () => setOpen(document.body.classList.contains("toc-hidden")));
  let t = null;
  $("#toc-filter").addEventListener("input", () => { clearTimeout(t); t = setTimeout(renderToc, 120); });
  $("#toc-filter").addEventListener("keydown", (e) => {
    if (e.key === "Enter") { const a = $("#toc-tree a"); if (a) { location.hash = a.getAttribute("href"); } }
  });
  $("#toc-tree").addEventListener("click", (e) => {
    // on a narrow screen the TOC covers the text: close it after a choice
    if (e.target.closest("a") && window.innerWidth <= 900) setOpen(false);
  });
}

// ----------------------------------------------------------------- text -----
// Highlight the search words: `marks` = word forms chosen by the server (lemma-aware, both languages).
function markText(text, set) {
  const esc = escapeHtml(text);
  if (!set || !set.size) return esc;
  return esc.replace(WORD_RE, (w) => (set.has(normWord(w)) ? `<mark>${w}</mark>` : w));
}

function paraHtml(p, marks) {
  return `<div class="para card-langs" data-id="${p.id}" id="para-${p.id}">
    <a class="pnum" href="${hashFor({ paper: p.paper, para: p.id })}" title="Ссылка на абзац ${p.id}"><span class="pn-full">${p.id}</span><span class="pn-seq">${p.n}</span></a>
    <div class="ru" lang="ru">${markText(p.ru, marks.ru)}</div>
    <div class="en" lang="en">${markText(p.en, marks.en)}</div>
  </div>`;
}

function headingHtml(level, t, num) {
  const ru = t?.ru || "", en = t?.en || "";
  return `<div class="read-h ${level}">
    ${num ? `<span class="h-num">${escapeHtml(num)}</span>` : ""}
    <div class="ru" lang="ru">${escapeHtml(ru)}</div><div class="en" lang="en">${escapeHtml(en)}</div>
  </div>`;
}

function renderText() {
  const d = state.data;
  const marks = { ru: new Set(d.marks?.ru || []), en: new Set(d.marks?.en || []) };
  const out = [];
  if (d.part) out.push(`<div class="read-part">${titleHtml({ ru: d.part.ru, en: d.part.en }, "")}</div>`);
  if (d.paper !== 0) out.push(`<div class="read-kicker">Документ ${d.paper}</div>`);
  out.push(headingHtml("h-paper", d.title, ""));
  let sec = null;
  for (const p of d.paragraphs) {
    if (p.section !== sec) {
      sec = p.section;
      if (hasTitle(p.section_title)) out.push(`<div class="sec-anchor" id="sec-${p.section}"></div>` + headingHtml("h-sec", p.section_title, `${p.section}.`));
      else out.push(`<div class="sec-anchor" id="sec-${p.section}"></div>`);
    }
    out.push(paraHtml(p, marks));
  }
  $("#text").innerHTML = out.join("");
  document.title = `${(S.primary === "ru" ? d.title?.ru : d.title?.en) || `Документ ${d.paper}`} — Чтение`;

  const pos = state.flat.indexOf(d.paper);
  const label = d.paper === 0 ? "Предисловие" : `Документ ${d.paper}`;
  $$(".chapter-pos").forEach((el) => { el.textContent = `${label} · ${pos + 1} из ${state.flat.length}`; });
  for (const [btn, target] of [["prev", d.prev], ["next", d.next]]) {
    $$(`.nav-${btn}`).forEach((b) => {
      b.disabled = target == null;
      b.dataset.paper = target ?? "";
      const t = state.toc.flatMap((x) => x.papers).find((x) => x.paper === target);
      b.title = t ? `${target}. ${(S.primary === "ru" ? t.title?.ru : t.title?.en) || ""}` : "";
    });
  }
  // query chip
  $("#query-chip").classList.toggle("hidden", !state.q);
  $("#query-chip-text").textContent = state.q ? `«${state.q}»` : "";
}

function scrollToTarget({ section, para }) {
  let el = null;
  if (para) el = document.getElementById(`para-${para}`);
  else if (section != null) el = document.getElementById(`sec-${section}`);
  if (!el) { window.scrollTo(0, 0); return; }
  el.scrollIntoView({ block: para ? "center" : "start" });
  if (para) {
    el.classList.remove("flash");
    void el.offsetWidth;
    el.classList.add("flash");
  }
}

async function onHash() {
  const h = readHash();
  state.q = h.q;
  const paper = h.paper ?? state.flat[0] ?? 0;
  try {
    $("#read-msg").classList.add("hidden");
    if (paper !== state.paper || !state.data || state.data.q !== state.q) {
      const d = await api(`/api/read/${paper}${state.q ? `?q=${encodeURIComponent(state.q)}` : ""}`);
      d.q = state.q;
      state.data = d;
      state.paper = paper;
      renderText();
      renderToc();
      markCurrentInToc();
    }
    scrollToTarget(h);
  } catch (e) {
    $("#read-msg").textContent = "Не удалось открыть документ: " + e.message;
    $("#read-msg").className = "msg-box error";
  }
  // the search page link keeps the query so «Поиск» returns to the same search
  $("#back-to-search").href = state.q ? `/#q=${encodeURIComponent(state.q)}` : "/";
}

function initNav() {
  const step = (dir) => {
    const target = state.data && (dir < 0 ? state.data.prev : state.data.next);
    if (target != null) go({ paper: target });
  };
  $$(".nav-prev").forEach((b) => b.addEventListener("click", () => step(-1)));
  $$(".nav-next").forEach((b) => b.addEventListener("click", () => step(1)));
  document.addEventListener("keydown", (e) => {
    if (e.altKey && (e.key === "ArrowLeft" || e.key === "ArrowRight")) {
      e.preventDefault();
      step(e.key === "ArrowLeft" ? -1 : 1);
    }
  });
  $("#query-clear").addEventListener("click", () => {
    const h = readHash();
    go({ paper: state.paper, section: h.section, para: h.para, q: "" });
  });
  window.addEventListener("hashchange", onHash);
}

// The top bar wraps to two rows on a narrow screen: the sticky TOC and drawers start right under it.
function syncTopbarHeight() {
  document.body.style.setProperty("--topbar", $(".topbar").offsetHeight + "px");
}

(async function main() {
  syncTopbarHeight();
  window.addEventListener("resize", syncTopbarHeight);
  initSettingsPanel();
  applySettings();
  initToc();
  initNav();
  try {
    const { parts } = await api("/api/toc");
    state.toc = parts;
    state.flat = parts.flatMap((x) => x.papers.map((p) => p.paper));
  } catch (e) {
    $("#read-msg").textContent = "Не удалось загрузить оглавление: " + e.message;
    $("#read-msg").className = "msg-box error";
  }
  renderToc();
  onHash();
})();
