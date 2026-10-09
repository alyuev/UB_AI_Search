// Hyphenation of book text (EN/RU columns) via the bundled Hyphenopoly (vendor/hyphenopoly,
// MIT, works offline). Browsers' own `hyphens: auto` depends on installed dictionaries and
// often lacks Russian, so FORCEHYPHENOPOLY makes the result the same everywhere.
// Results are rendered dynamically, so a MutationObserver hyphenates every new text column.
(function () {
  const TEXT_SEL = ".en, .ru, .en-col, .ru-col";
  const LANG = { en: "en-us", "en-col": "en-us", ru: "ru", "ru-col": "ru" };

  // v6 API: Hyphenopoly_Loader.js (loaded just before this file) defines Hyphenopoly.config().
  window.Hyphenopoly.config({
    require: { "ru": "FORCEHYPHENOPOLY", "en-us": "FORCEHYPHENOPOLY" },
    paths: { maindir: "vendor/hyphenopoly/", patterndir: "vendor/hyphenopoly/patterns/" },
    setup: {
      hide: "none",
      selectors: { ".hyph": { minWordLength: 6, leftmin: 3, rightmin: 2 } },
    },
    handleEvent: { error: (e) => { e.preventDefault?.(); console.warn("Hyphenopoly:", e.msg || e); } },
  });

  const done = new WeakSet();
  let pending = [];

  function langOf(el) {
    for (const c of el.classList) if (LANG[c]) return LANG[c];
    return null;
  }

  function prepare(el) {
    if (done.has(el)) return;
    const lang = langOf(el);
    if (!lang) return;
    done.add(el);
    el.lang = lang;
    el.classList.add("hyph");
    pending.push(el);
  }

  async function flush() {
    if (!pending.length) return;
    if (!window.Hyphenopoly.hyphenators) { setTimeout(flush, 100); return; }  // loader not ready yet
    const els = pending; pending = [];
    // v6: the polyglot "HTML" hyphenator picks the pattern from each element's lang attribute
    const h = await window.Hyphenopoly.hyphenators.HTML;
    for (const el of els) {
      try { h(el, ".hyph"); } catch (e) { console.warn("Hyphenopoly:", e); }
    }
  }

  function scan(root) {
    if (root.nodeType !== 1) return;
    if (root.matches(TEXT_SEL)) prepare(root);
    root.querySelectorAll(TEXT_SEL).forEach(prepare);
  }

  document.addEventListener("DOMContentLoaded", () => {
    scan(document.body);
    flush();
    new MutationObserver((muts) => {
      for (const m of muts) m.addedNodes.forEach(scan);
      if (pending.length) flush();
    }).observe(document.body, { childList: true, subtree: true });
  });
})();
