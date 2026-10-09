"use strict";
/* Небольшой безопасный разбор Markdown (без библиотек и CDN).
 *
 * Принцип безопасности: весь текст СНАЧАЛА экранируется (< > & " '), и только потом поверх него
 * ставятся наши собственные теги. «Сырой» HTML из ответа модели не проходит никогда.
 * Картинок нет. Ссылки — только http:// и https:// (rel="noopener noreferrer", target=_blank).
 *
 * Поддержано: **жирный**, *курсив* / _курсив_, `код`, ```блок кода```, списки (маркированные и
 * нумерованные, один-два уровня вложенности), абзацы и переносы строк, заголовки #…####,
 * цитата (> …), горизонтальная линия (---).
 *
 * Использование: Md.render(text, { inline: fn, assumption: true })
 *   inline(escapedHtml) -> html : дополнительная обработка уже безопасного текста
 *                                 (в app.js так делаются кликабельные ссылки вида [50:7.2]);
 *   assumption: true — строки, начинающиеся с «Допущение:», оборачиваются в <em class="assumption">.
 * Md.selfTest() -> массив описаний проваленных проверок (пустой = всё хорошо).
 */
(function (global) {
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const ASSUMPTION_RE = /^[\s*_>-]*Допущение\s*:/;

  // ---- inline ---------------------------------------------------------------------------------
  function inline(raw, opts) {
    // служебные символы-заместители не должны приходить снаружи
    let s = esc(String(raw).replace(/[\u0000-\u0002]/g, ""));
    const slots = [];
    const hold = (html) => { slots.push(html); return "\u0001" + (slots.length - 1) + "\u0002"; };

    // `код` (внутри ничего не разбираем)
    s = s.replace(/`([^`\n]+)`/g, (m, c) => hold("<code>" + c + "</code>"));
    // [текст](http...) — только http/https; экранирование уже убрало кавычки из адреса
    s = s.replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)<>]+)\)/gi, (m, text, url) =>
      hold('<a href="' + url + '" target="_blank" rel="noopener noreferrer">' + text + "</a>"));
    // **жирный** / __жирный__
    s = s.replace(/\*\*(?=\S)([^\n]+?)(?<=\S)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/(^|[^\w_])__(?=\S)([^\n]+?)(?<=\S)__(?![\w_])/g, "$1<strong>$2</strong>");
    // *курсив* / _курсив_
    s = s.replace(/(^|[^*\w])\*(?=[^\s*])([^*\n]+?)(?<=[^\s*])\*(?!\*)/g, "$1<em>$2</em>");
    s = s.replace(/(^|[^\w_])_(?=[^\s_])([^_\n]+?)(?<=[^\s_])_(?![\w_])/g, "$1<em>$2</em>");

    if (opts && typeof opts.inline === "function") s = opts.inline(s);
    for (let pass = 0; pass < 3 && /\u0001\d+\u0002/.test(s); pass++) {
      s = s.replace(/\u0001(\d+)\u0002/g, (m, i) => slots[Number(i)]);
    }
    return s;
  }

  function inlineLine(text, opts) {
    const html = inline(text, opts);
    if (opts && opts.assumption && ASSUMPTION_RE.test(text)) return '<em class="assumption">' + html + "</em>";
    return html;
  }

  // ---- blocks ---------------------------------------------------------------------------------
  const BULLET_RE = /^(\s*)([-*+])\s+(.*)$/;
  const NUM_RE = /^(\s*)(\d{1,3})[.)]\s+(.*)$/;
  const HEAD_RE = /^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$/;
  const HR_RE = /^\s{0,3}([-*_])(\s*\1){2,}\s*$/;
  const isItem = (l) => BULLET_RE.test(l) || NUM_RE.test(l);
  const indentOf = (l) => l.match(/^\s*/)[0].replace(/\t/g, "    ").length;

  function renderList(lines, opts) {
    const first = BULLET_RE.exec(lines[0]) || NUM_RE.exec(lines[0]);
    const ordered = !!NUM_RE.exec(lines[0]) && !BULLET_RE.exec(lines[0]);
    const base = indentOf(lines[0]);
    const items = [];
    for (const line of lines) {
      const m = BULLET_RE.exec(line) || NUM_RE.exec(line);
      if (m && indentOf(line) <= base + 1) items.push({ text: m[3], sub: [] });
      else if (items.length) items[items.length - 1].sub.push(line);
    }
    let startAttr = "";
    if (ordered) {
      const n = Number(first[2]);
      if (n !== 1) startAttr = ' start="' + n + '"';
    }
    const body = items.map((it) => {
      let sub = "";
      if (it.sub.length) {
        const cut = Math.min(...it.sub.filter((l) => l.trim()).map(indentOf));
        sub = renderBlocks(it.sub.map((l) => l.slice(Math.min(cut, indentOf(l)))), opts);
      }
      return "<li>" + inlineLine(it.text, opts) + sub + "</li>";
    }).join("");
    return (ordered ? "<ol" + startAttr + ">" : "<ul>") + body + (ordered ? "</ol>" : "</ul>");
  }

  function renderBlocks(lines, opts) {
    const out = [];
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (!line.trim()) { i++; continue; }

      // ``` блок кода
      if (/^\s*```/.test(line)) {
        const code = [];
        i++;
        while (i < lines.length && !/^\s*```/.test(lines[i])) code.push(lines[i++]);
        i++;
        out.push("<pre class=\"md-code\"><code>" + esc(code.join("\n")) + "</code></pre>");
        continue;
      }
      const h = HEAD_RE.exec(line);
      if (h) {
        const lvl = Math.min(6, h[1].length + 2);   // # -> h3, ## -> h4, ...
        out.push("<h" + lvl + ' class="md-h">' + inline(h[2], opts) + "</h" + lvl + ">");
        i++; continue;
      }
      if (HR_RE.test(line)) { out.push("<hr>"); i++; continue; }
      if (/^\s{0,3}>/.test(line)) {
        const q = [];
        while (i < lines.length && /^\s{0,3}>/.test(lines[i])) q.push(lines[i++].replace(/^\s{0,3}>\s?/, ""));
        out.push("<blockquote>" + renderBlocks(q, opts) + "</blockquote>");
        continue;
      }
      if (isItem(line)) {
        const block = [];
        while (i < lines.length) {
          const l = lines[i];
          if (isItem(l)) { block.push(l); i++; }
          else if (l.trim() && /^\s+\S/.test(l) && block.length) { block.push(l); i++; }
          else if (!l.trim() && i + 1 < lines.length && isItem(lines[i + 1]) &&
                   indentOf(lines[i + 1]) <= indentOf(block[0]) + 1) { i++; }   // пустая строка между пунктами
          else break;
        }
        out.push(renderList(block, opts));
        continue;
      }
      // абзац: строки подряд до пустой строки / другого блока
      const para = [];
      while (i < lines.length && lines[i].trim() && !/^\s*```/.test(lines[i]) && !HEAD_RE.test(lines[i]) &&
             !HR_RE.test(lines[i]) && !/^\s{0,3}>/.test(lines[i]) && !(para.length && isItem(lines[i]))) {
        para.push(lines[i++]);
      }
      if (!para.length) { para.push(lines[i++]); }
      out.push("<p>" + para.map((l) => inlineLine(l, opts)).join("<br>") + "</p>");
    }
    return out.join("");
  }

  function render(text, opts) {
    const src = String(text == null ? "" : text).replace(/\r\n?/g, "\n");
    return renderBlocks(src.split("\n"), opts || {});
  }

  // ---- self-test (XSS и основные случаи) ------------------------------------------------------
  function selfTest() {
    const fails = [];
    const t = (name, ok) => { if (!ok) fails.push(name); };
    const r = (s, o) => render(s, o);
    const noTag = (html, tag) => !new RegExp("<" + tag + "\\b", "i").test(html);

    let h = r("<script>alert(1)</script>");
    t("script экранирован", noTag(h, "script") && h.includes("&lt;script&gt;"));
    h = r('<img src=x onerror=alert(1)>');
    t("img/onerror экранирован", noTag(h, "img") && !/<[^>]*onerror/i.test(h));
    h = r("[клик](javascript:alert(1))");
    t("javascript: не ссылка", noTag(h, "a") && !/href=/i.test(h));
    h = r("[клик](JaVaScRiPt:alert(1)) и [data](data:text/html,x)");
    t("javascript: (регистр) и data: не ссылки", noTag(h, "a"));
    h = r("[ok](https://example.com/a?b=1&c=2)");
    t("https-ссылка с rel", /<a href="https:\/\/example\.com\/a\?b=1&amp;c=2" target="_blank" rel="noopener noreferrer">ok<\/a>/.test(h));
    h = r('[x](https://a.com/"onmouseover="alert(1))');
    t("кавычка в адресе не ломает атрибут", !/"\s*onmouseover/i.test(h));
    h = r("![картинка](https://a.com/x.png)");
    t("картинки не рисуются", noTag(h, "img"));
    h = r("**<b onclick=x>жирный</b>**");
    t("теги внутри жирного экранированы", noTag(h, "b") && h.includes("<strong>"));
    h = r("`<script>` и ```\n<script>x</script>\n```");
    t("код экранирован", noTag(h, "script"));
    h = r("# <svg onload=x>");
    t("заголовок экранирован", noTag(h, "svg"));
    h = r("\u0001" + "0" + "\u0002 [a](javascript:1)");
    t("служебные символы не подделывают вставки", noTag(h, "a"));
    // форматирование
    t("жирный", r("a **b** c") === "<p>a <strong>b</strong> c</p>");
    t("курсив", r("a *b* c") === "<p>a <em>b</em> c</p>");
    t("код", r("a `b` c") === "<p>a <code>b</code> c</p>");
    t("список", r("- a\n- b") === "<ul><li>a</li><li>b</li></ul>");
    t("нумерованный", r("1. a\n2. b") === "<ol><li>a</li><li>b</li></ol>");
    t("вложенный список", /<ul><li>a<ul><li>b<\/li><\/ul><\/li><\/ul>/.test(r("- a\n  - b")));
    t("заголовок", r("### Тема") === '<h5 class="md-h">Тема</h5>');
    t("цитата", r("> q") === "<blockquote><p>q</p></blockquote>");
    t("перенос строки", r("a\nb") === "<p>a<br>b</p>");
    t("два абзаца", r("a\n\nb") === "<p>a</p><p>b</p>");
    t("умножение со звёздочками не курсив", !r("2 * 3 * 4").includes("<em>"));
    t("snake_case не курсив", !r("my_var_name").includes("<em>"));
    h = r("Факт [50:7.2].", { inline: (s) => s.replace(/\[(\d+:\d+\.\d+)\]/g, '[<span class="cite">$1</span>]') });
    t("ссылка на абзац через inline-хук", h.includes('<span class="cite">50:7.2</span>'));
    h = r("- Допущение: год", { assumption: true });
    t("допущение выделено", h.includes('<em class="assumption">'));
    return fails;
  }

  global.Md = { render, inline, esc, selfTest };
})(window);
