/*
 * A small syntax highlighter for the code samples on this site.
 *
 * Written rather than pulled in because the pages carry no other dependency and
 * the grammar needed here is narrow: TypeScript, plus shell and JSON. It reads
 * each block's textContent, so any hand-written markup inside is discarded and
 * the result is consistent across every sample.
 */
(function () {
  var KEYWORDS = [
    'abstract','as','async','await','break','case','catch','class','const','constructor',
    'continue','declare','default','delete','do','else','enum','export','extends','extends',
    'false','finally','for','from','function','get','if','implements','import','in','instanceof',
    'interface','keyof','let','new','null','of','private','protected','public','readonly',
    'return','satisfies','set','static','super','switch','this','throw','true','try','type',
    'typeof','undefined','var','void','while','yield',
  ];

  // Built-in and ambient names worth marking as types rather than plain text.
  var TYPES = [
    'Array','Promise','Record','Object','String','Number','Boolean','Date','Error','JSON',
    'Math','Set','Map','console','process','string','number','boolean','any','unknown','never',
  ];

  var kw = new RegExp('^(?:' + KEYWORDS.join('|') + ')$');
  var ty = new RegExp('^(?:' + TYPES.join('|') + ')$');

  function esc(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function span(cls, text) {
    return '<span class="' + cls + '">' + esc(text) + '</span>';
  }

  /* ── TypeScript / JavaScript ─────────────────────────── */

  // One pass, longest-match-first. Order is the grammar here.
  var TS = new RegExp([
    '(\\/\\/[^\\n]*)',                              // 1 line comment
    '(\\/\\*[\\s\\S]*?\\*\\/)',                     // 2 block comment
    '(`(?:\\\\.|[^`\\\\])*`)',                      // 3 template string
    "('(?:\\\\.|[^'\\\\])*')",                      // 4 single-quoted
    '("(?:\\\\.|[^"\\\\])*")',                      // 5 double-quoted
    '(@[A-Za-z_$][\\w$]*)',                         // 6 decorator
    '(\\b\\d[\\d_]*(?:\\.\\d+)?\\b)',               // 7 number
    '([A-Za-z_$][\\w$]*)',                          // 8 identifier
    '([{}()\\[\\];,.:]|=>|[=+\\-*/<>!?&|]+)',       // 9 punctuation / operator
  ].join('|'), 'g');

  function highlightTS(src) {
    var out = '';
    var last = 0;
    var m;
    TS.lastIndex = 0;

    while ((m = TS.exec(src)) !== null) {
      if (m.index > last) out += esc(src.slice(last, m.index));
      last = TS.lastIndex;

      if (m[1] || m[2]) { out += span('c-com', m[1] || m[2]); continue; }
      if (m[3] || m[4] || m[5]) { out += span('c-str', m[3] || m[4] || m[5]); continue; }
      if (m[6]) { out += span('c-dec', m[6]); continue; }
      if (m[7]) { out += span('c-num', m[7]); continue; }

      if (m[8]) {
        var word = m[8];
        var before = src.slice(0, m.index);
        var after = src.slice(TS.lastIndex);

        if (kw.test(word)) out += span('c-kw', word);
        else if (ty.test(word) || /^[A-Z]/.test(word)) out += span('c-typ', word);
        else if (/^\s*\(/.test(after)) out += span('c-fn', word);       // call or definition
        else if (/\.\s*$/.test(before)) out += span('c-prop', word);    // member access
        else out += esc(word);
        continue;
      }

      out += span('c-pun', m[9]);
    }

    if (last < src.length) out += esc(src.slice(last));
    return out;
  }

  /* ── shell ───────────────────────────────────────────── */

  var SH = /(#[^\n]*)|('(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*")|(^|\n)(\s*)([A-Za-z][\w.-]*)|(\s-{1,2}[\w-]+)/g;

  function highlightShell(src) {
    var out = '';
    var last = 0;
    var m;
    SH.lastIndex = 0;

    while ((m = SH.exec(src)) !== null) {
      if (m.index > last) out += esc(src.slice(last, m.index));
      last = SH.lastIndex;

      if (m[1]) { out += span('c-com', m[1]); continue; }
      if (m[2]) { out += span('c-str', m[2]); continue; }
      if (m[5]) { out += esc(m[3] || '') + esc(m[4] || '') + span('c-fn', m[5]); continue; }
      if (m[6]) { out += span('c-kw', m[6]); continue; }
    }

    if (last < src.length) out += esc(src.slice(last));
    return out;
  }

  /* ── JSON ────────────────────────────────────────────── */

  var JS_ON = /("(?:\\.|[^"\\])*")(\s*:)?|(\b(?:true|false|null)\b)|(\b\d[\d.]*\b)|([{}[\],:])/g;

  function highlightJSON(src) {
    var out = '';
    var last = 0;
    var m;
    JS_ON.lastIndex = 0;

    while ((m = JS_ON.exec(src)) !== null) {
      if (m.index > last) out += esc(src.slice(last, m.index));
      last = JS_ON.lastIndex;

      if (m[1]) {
        out += span(m[2] ? 'c-prop' : 'c-str', m[1]);
        if (m[2]) out += span('c-pun', m[2]);
        continue;
      }
      if (m[3]) { out += span('c-kw', m[3]); continue; }
      if (m[4]) { out += span('c-num', m[4]); continue; }
      if (m[5]) { out += span('c-pun', m[5]); continue; }
    }

    if (last < src.length) out += esc(src.slice(last));
    return out;
  }

  /* ── apply ───────────────────────────────────────────── */

  var byLang = { ts: highlightTS, js: highlightTS, bash: highlightShell, sh: highlightShell, json: highlightJSON };

  [].slice.call(document.querySelectorAll('pre.code > code')).forEach(function (el) {
    var lang = el.parentNode.getAttribute('data-lang') || 'ts';
    var fn = byLang[lang] || highlightTS;
    el.innerHTML = fn(el.textContent);
  });
})();
