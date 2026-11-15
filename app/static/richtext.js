"use strict";
// Display of book paragraphs with their inline formatting (italics, bold, line breaks, super/subscript).
// The server already sends sanitized light markup (`ru_html` / `en_html`: only <i> <b> <br> <sup> <sub>);
// this file additionally whitelists the tags on the client, so that nothing else can ever get into the DOM.
// Search-word highlighting works on the TEXT NODES of that markup only (tags are never touched), so a
// highlighted word never breaks the formatting. A word cut by a formatting tag in the middle is not
// highlighted; that is rare and harmless.
(function () {
  const ALLOWED = new Set(["I", "B", "BR", "SUP", "SUB"]);
  const DROP = new Set(["SCRIPT", "STYLE", "TEMPLATE", "NOSCRIPT"]);

  // markup -> detached <div> holding only whitelisted elements (no attributes) and text. Parsed by
  // DOMParser: an inert document, so nothing in the markup is executed or loaded.
  function sanitize(markup) {
    const src = new DOMParser().parseFromString("<body>" + markup + "</body>", "text/html").body;
    const out = document.createElement("div");
    (function walk(from, to) {
      for (const n of Array.from(from.childNodes)) {
        if (n.nodeType === 3) to.appendChild(document.createTextNode(n.data));
        else if (n.nodeType === 1) {
          if (DROP.has(n.tagName)) continue;
          if (ALLOWED.has(n.tagName)) {
            const el = document.createElement(n.tagName.toLowerCase());
            to.appendChild(el);
            walk(n, el);
          } else walk(n, to);   // unknown tag: keep its text
        }
      }
    })(src, out);
    return out;
  }

  // HTML of one paragraph column. `markFn(text)` turns a plain text piece into escaped HTML (with
  // <mark> highlighting if wanted). Without markup this is simply markFn(plain).
  function render(plain, markup, markFn) {
    if (!markup) return markFn(plain || "");
    const root = sanitize(markup);
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const nodes = [];
    for (let n = walker.nextNode(); n; n = walker.nextNode()) nodes.push(n);
    for (const n of nodes) {
      const t = document.createElement("template");
      t.innerHTML = markFn(n.data);
      n.replaceWith(t.content);
    }
    return root.innerHTML;
  }

  window.RichText = { sanitize, render };
})();
