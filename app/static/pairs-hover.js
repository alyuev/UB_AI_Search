// Hover highlight of parallel terms: point at a word in one language and its counterpart in the
// other column is underlined. Pairs come from the RU<->EN glossary (/api/paragraph/{id}/pairs);
// words outside the glossary are not highlighted. Works on every .card-langs block that knows its
// paragraph id (its own data-id, or the data-id of the result card around it).
(function () {
  const WORD_RE = /[\p{L}­]+/gu;
  const norm = (s) => s.replace(/­/g, "").toLowerCase().replace(/ё/g, "е");
  const loaded = new WeakMap();   // .card-langs -> Promise

  function paraId(langs) {
    return langs.dataset.id || langs.closest(".card[data-id]")?.dataset.id || null;
  }

  // Wrap every glossary word of one text column in <span class="pw" data-g="ids">.
  function wrapColumn(col, map) {
    const walker = document.createTreeWalker(col, NodeFilter.SHOW_TEXT);
    const nodes = [];
    for (let n = walker.nextNode(); n; n = walker.nextNode()) nodes.push(n);
    for (const node of nodes) {
      const text = node.data;
      let last = 0, frag = null;
      for (const m of text.matchAll(WORD_RE)) {
        const ids = map.get(norm(m[0]));
        if (!ids) continue;
        frag = frag || document.createDocumentFragment();
        if (m.index > last) frag.append(text.slice(last, m.index));
        const span = document.createElement("span");
        span.className = "pw";
        span.dataset.g = ids.join(" ");
        span.textContent = m[0];
        frag.append(span);
        last = m.index + m[0].length;
      }
      if (!frag) continue;
      if (last < text.length) frag.append(text.slice(last));
      node.replaceWith(frag);
    }
  }

  function prepare(langs) {
    if (loaded.has(langs)) return loaded.get(langs);
    const id = paraId(langs);
    const p = !id ? Promise.resolve() : fetch(`/api/paragraph/${encodeURIComponent(id)}/pairs`)
      .then((r) => {
        if (!r.ok) throw new Error(`pairs: HTTP ${r.status}`);
        return r.json();
      })
      .then(({ pairs }) => {
        if (!pairs || !pairs.length) return;
        const ru = new Map(), en = new Map();
        const add = (map, w, g) => { const k = norm(w); map.set(k, [...(map.get(k) || []), g]); };
        pairs.forEach((pr, g) => { pr.ru.forEach((w) => add(ru, w, g)); pr.en.forEach((w) => add(en, w, g)); });
        langs.querySelectorAll(":scope > .ru").forEach((c) => wrapColumn(c, ru));
        langs.querySelectorAll(":scope > .en").forEach((c) => wrapColumn(c, en));
      })
      .catch(() => { loaded.delete(langs); });   // server busy/restarting: try again on the next hover
    loaded.set(langs, p);
    return p;
  }

  function clear(langs) {
    langs.querySelectorAll(".pw.hl, .pw.hl2").forEach((s) => s.classList.remove("hl", "hl2"));
  }

  // The visible text block a word belongs to: a card shows either the snippet or the full paragraph
  // (both are in the DOM), so positions are measured inside the one that is shown.
  const visibleBase = (col) => [...col.querySelectorAll(":scope > .snip, :scope > .full")].find((b) => b.offsetParent) || col;

  // Relative position of a word in its block: 0 = start, 1 = end.
  function relPos(span, base) {
    const r = document.createRange();
    r.setStart(base, 0);
    r.setEndBefore(span);
    const total = base.textContent.length || 1;
    return r.toString().length / total;
  }

  // Of the counterpart's occurrences in the other column pick the one in "the same place": the k-th of k
  // when both sides have the same number of occurrences, otherwise the nearest by relative position.
  function counterpart(word, sameSide, otherSide, otherBase) {
    if (!otherSide.length) return null;
    const k = sameSide.indexOf(word);
    if (sameSide.length === otherSide.length && k >= 0) return otherSide[k];
    const p = relPos(word, visibleBase(word.closest(".ru, .en")));
    let best = null, bestD = Infinity;
    for (const s of otherSide) {
      const d = Math.abs(relPos(s, otherBase) - p);
      if (d < bestD) { best = s; bestD = d; }
    }
    return best;
  }

  document.addEventListener("mouseover", (ev) => {
    const langs = ev.target.closest?.(".card-langs");
    if (!langs) return;
    prepare(langs);
    const w = ev.target.closest(".pw");
    clear(langs);
    if (!w) return;
    const ids = new Set(w.dataset.g.split(" "));
    const related = (s) => s.offsetParent && s.dataset.g.split(" ").some((g) => ids.has(g));
    const myCol = w.closest(".ru, .en");
    const otherCol = [...langs.querySelectorAll(":scope > .ru, :scope > .en")].find((c) => c !== myCol);
    const same = [...myCol.querySelectorAll(".pw")].filter(related);
    const other = otherCol ? [...otherCol.querySelectorAll(".pw")].filter(related) : [];
    // all occurrences of the pair: faint; the word under the mouse and its counterpart: bright
    same.concat(other).forEach((s) => s.classList.add("hl2"));
    const twin = otherCol ? counterpart(w, same, other, visibleBase(otherCol)) : null;
    [w, twin].forEach((s) => { if (s) { s.classList.remove("hl2"); s.classList.add("hl"); } });
  });
  document.addEventListener("mouseout", (ev) => {
    const langs = ev.target.closest?.(".card-langs");
    if (langs && !langs.contains(ev.relatedTarget)) clear(langs);
  });
})();
