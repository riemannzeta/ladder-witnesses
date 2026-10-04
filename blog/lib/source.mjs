// Source tokenizer + Ghost converter for post.src.md.
//
// post.src.md is ordinary GitHub-flavoured Markdown with $…$ inline math and
// $$…$$ display math. processSource() turns it into Markdown for a Ghost
// Markdown card (see README.md for the rules and the reasons behind them).
//
// The SAME scanner is used by build_ghost.mjs (to convert) and by lint.mjs (to
// list the math the author intended), so they cannot disagree about what is math.

import {HTML_TAG_RE, HTML_OPEN_CLOSE_TAG_RE} from 'markdown-it/lib/common/html_re.mjs';
import HTML_BLOCK_NAMES from 'markdown-it/lib/common/html_blocks.mjs';

export const DEFAULT_IMAGE_BASE =
  'https://raw.githubusercontent.com/riemannzeta/ladder-witnesses/main/blog/';

// ---------------------------------------------------------------------------
// Escaping

// CommonMark "ASCII punctuation": exactly the characters markdown-it's escape
// rule turns `\X` into a literal X for.
const ASCII_PUNCT = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~';

export function isAsciiPunct(ch) {
  return typeof ch === 'string' && ch.length === 1 && ASCII_PUNCT.includes(ch);
}

/**
 * Backslash-escape every ASCII punctuation character. markdown-it turns each
 * `\X` back into a literal `X` (as an inert text token), so the LaTeX reaches
 * the HTML byte-for-byte and none of its characters can start emphasis (_ *),
 * <sup> (^), <sub> (~), <mark> (==), links ([ ]), HTML (<), entities (&),
 * code spans (`) or table cells (|). Letters, digits, spaces and non-ASCII
 * characters are left alone.
 */
export function escapeInlineLatex(latex) {
  let out = '';
  for (const ch of latex) out += isAsciiPunct(ch) ? '\\' + ch : ch;
  return out;
}

export function escapeHtmlText(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

/** Inline math for Ghost: `\\(` E(latex) `\\)`; markdown-it renders `\(latex\)`. */
export function inlineMathToGhost(latex) {
  return '\\\\(' + escapeInlineLatex(latex) + '\\\\)';
}

/** Display math for Ghost: one raw HTML block (no blank lines inside). */
export function displayMathToGhost(latexLines) {
  const h = latexLines.map(escapeHtmlText);
  if (h.length === 1) return ['<div class="math">$$' + h[0] + '$$</div>'];
  return [
    '<div class="math">$$' + h[0],
    ...h.slice(1, -1),
    h[h.length - 1] + '$$</div>',
  ];
}

// ---------------------------------------------------------------------------
// Block-level helpers (CommonMark / markdown-it 14 rules, simplified)

const isBlank = (line) => /^[ \t]*$/.test(line);

function indentOf(line) {
  let width = 0;
  let k = 0;
  for (; k < line.length; k++) {
    const c = line[k];
    if (c === ' ') width++;
    else if (c === '\t') width += 4 - (width % 4);
    else break;
  }
  return {width, length: k};
}

function fenceOpener(content) {
  const m = /^(`{3,}|~{3,})(.*)$/.exec(content);
  if (!m) return null;
  if (m[1][0] === '`' && m[2].includes('`')) return null;
  return {ch: m[1][0], len: m[1].length};
}

function isFenceCloser(content, fence) {
  const m = /^(`{3,}|~{3,})[ \t]*$/.exec(content);
  return !!m && m[1][0] === fence.ch && m[1].length >= fence.len;
}

const ATX_RE = /^#{1,6}(?:[ \t]|$)/;
const HR_RE = /^(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,})$/;
const SETEXT_RE = /^(?:=+|-+)[ \t]*$/;
const FOOTNOTE_DEF_RE = /^\[\^([^\]\s]+)\]:/;
const REF_DEF_RE = /^\[(?!\^)(?:[^[\]\\]|\\.)+\]:(?:[ \t]|$)/;
const QUOTE_PREFIX_RE = /^((?:[ \t]*>[ \t]?)+)/;

function listMarker(content) {
  const m = /^([-+*]|\d{1,9}[.)])([ \t]+|$)/.exec(content);
  if (!m) return null;
  const marker = m[1];
  const ordered = /^\d/.test(marker);
  const text = content.slice(m[0].length);
  return {
    marker,
    ordered,
    num: ordered ? parseInt(marker, 10) : null,
    pad: m[2].length,
    text,
    empty: text.trim() === '',
  };
}

// markdown-it html_block start conditions; [open, close, canInterruptParagraph]
const HTML_SEQUENCES = [
  [/^<(script|pre|style|textarea)(?=(\s|>|$))/i, /<\/(script|pre|style|textarea)>/i, true],
  [/^<!--/, /-->/, true],
  [/^<\?/, /\?>/, true],
  [/^<![A-Z]/, />/, true],
  [/^<!\[CDATA\[/, /\]\]>/, true],
  [new RegExp('^</?(' + HTML_BLOCK_NAMES.join('|') + ')(?=(\\s|/?>|$))', 'i'), /^$/, true],
  [new RegExp(HTML_OPEN_CLOSE_TAG_RE.source + '\\s*$'), /^$/, false],
];

function htmlBlockType(content) {
  if (content[0] !== '<') return 0;
  for (let k = 0; k < HTML_SEQUENCES.length; k++) {
    if (HTML_SEQUENCES[k][0].test(content)) return k + 1;
  }
  return 0;
}

// markdown-it's table cell splitter (a `\|` is an escaped pipe).
function escapedSplit(str) {
  const result = [];
  let isEscaped = false;
  let lastPos = 0;
  let current = '';
  for (let pos = 0; pos < str.length; pos++) {
    const ch = str.charCodeAt(pos);
    if (ch === 0x7c) {
      if (!isEscaped) {
        result.push(current + str.substring(lastPos, pos));
        current = '';
        lastPos = pos + 1;
      } else {
        current += str.substring(lastPos, pos - 1);
        lastPos = pos;
      }
    }
    isEscaped = ch === 0x5c;
  }
  result.push(current + str.substring(lastPos));
  return result;
}

function hardBreakAtEnd(line) {
  if (/ {2,}$/.test(line)) return true;
  const m = /(\\+)$/.exec(line);
  return !!m && m[1].length % 2 === 1;
}

// Ghost loads markdown-it-lazy-headers, which makes `#foo` (no space) a
// heading. GitHub does not, so escape a `#` that would start a block.
function escapeLeadingHash(s) {
  return s.startsWith('#') ? '\\' + s : s;
}

// ---------------------------------------------------------------------------
// URLs

export function normalizeBase(base) {
  if (!base) return base;
  return base.endsWith('/') ? base : base + '/';
}

export function isRelativeUrl(url) {
  return url !== '' && !/^(?:[a-zA-Z][a-zA-Z0-9+.-]*:|\/|#|\?)/.test(url);
}

function rewriteDest(dest, base) {
  if (!base) return dest;
  if (dest.startsWith('<') && dest.endsWith('>')) {
    const inner = dest.slice(1, -1);
    return isRelativeUrl(inner) ? '<' + base + inner.replace(/^\.\//, '') + '>' : dest;
  }
  return isRelativeUrl(dest) ? base + dest.replace(/^\.\//, '') : dest;
}

function rewriteImgTag(tag, base) {
  if (!base || !/^<img\b/i.test(tag)) return tag;
  return tag.replace(/(\ssrc\s*=\s*)(["']?)([^"'\s>]+)\2/i, (m, pre, q, url) =>
    isRelativeUrl(url) ? pre + q + base + url.replace(/^\.\//, '') + q : m);
}

const IMAGE_EXT_RE = /\.(?:png|jpe?g|gif|svg|webp|avif)(?:[?#].*)?$/i;

// ---------------------------------------------------------------------------
// Inline scanner

const AUTOLINK_RE = /^([a-zA-Z][a-zA-Z0-9+.-]{1,31}):([^<>\x00-\x20]*)$/;
const EMAIL_RE = /^([a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*)$/;
// Elements whose text KaTeX auto-render ignores: math inside them is not math.
const OPAQUE_ELEMENT_RE = /^<(script|noscript|style|textarea|pre|code|option)(?=[\s/>])/i;

const runLength = (s, i, ch) => {
  let k = i;
  while (s[k] === ch) k++;
  return k - i;
};

// `(` at p: parse `(dest "title")`; returns {end, destStart, destEnd} or null
function scanParenDest(s, p) {
  const n = s.length;
  let k = p + 1;
  while (k < n && /[ \t\n]/.test(s[k])) k++;
  const destStart = k;
  if (s[k] === '<') {
    const e = s.indexOf('>', k + 1);
    if (e < 0 || s.slice(k + 1, e).includes('\n')) return null;
    k = e + 1;
  } else {
    let depth = 0;
    while (k < n) {
      const ch = s[k];
      if (ch === '\\' && isAsciiPunct(s[k + 1])) { k += 2; continue; }
      if (ch === '(') depth++;
      else if (ch === ')') { if (depth === 0) break; depth--; }
      else if (/[\s\x00-\x1f]/.test(ch)) break;
      k++;
    }
  }
  const destEnd = k;
  while (k < n && /[ \t\n]/.test(s[k])) k++;
  if (k > destEnd && (s[k] === '"' || s[k] === "'" || s[k] === '(')) {
    const close = s[k] === '(' ? ')' : s[k];
    let t = k + 1;
    while (t < n && s[t] !== close) t += s[t] === '\\' ? 2 : 1;
    if (t >= n) return null;
    k = t + 1;
    while (k < n && /[ \t\n]/.test(s[k])) k++;
  }
  if (s[k] !== ')') return null;
  return {end: k + 1, destStart, destEnd};
}

/**
 * Split one block's text into segments. mode 'md' = Markdown paragraph text
 * (code spans, images, links, inline HTML, `\` escapes); mode 'raw' = inside a
 * raw HTML block (markdown does not run there).
 */
function scanInline(s, mode, rec) {
  const segs = [];
  const n = s.length;
  let i = 0;
  let textStart = 0;
  const push = (seg) => {
    if (seg.start > textStart) segs.push({type: 'text', start: textStart, end: seg.start});
    segs.push(seg);
    i = textStart = seg.end;
  };

  while (i < n) {
    const c = s[i];

    if (c === '\\') {
      if (mode === 'md') i += isAsciiPunct(s[i + 1]) ? 2 : 1;
      else i += s[i + 1] === '$' ? 2 : 1;
      continue;
    }

    if (c === '`' && mode === 'md') {
      const len = runLength(s, i, '`');
      let j = i + len;
      let close = -1;
      while (j < n) {
        if (s[j] === '`') {
          const m = runLength(s, j, '`');
          if (m === len) { close = j; break; }
          j += m;
        } else {
          j++;
        }
      }
      if (close >= 0) push({type: 'code', start: i, end: close + len});
      else i += len;
      continue;
    }

    if (c === '<') {
      const rest = s.slice(i);
      if (rest.startsWith('<!--')) {
        const e = s.indexOf('-->', i + 4);
        if (e >= 0) { push({type: 'opaque', start: i, end: e + 3}); continue; }
      }
      const tag = HTML_TAG_RE.exec(rest);
      if (tag) {
        const el = OPAQUE_ELEMENT_RE.exec(tag[0]);
        if (el && !tag[0].endsWith('/>')) {
          const closeRe = new RegExp('</' + el[1] + '\\s*>', 'i');
          const after = s.slice(i + tag[0].length);
          const m = closeRe.exec(after);
          if (m) {
            push({type: 'opaque', start: i, end: i + tag[0].length + m.index + m[0].length});
            continue;
          }
        }
        push({type: 'tag', start: i, end: i + tag[0].length});
        continue;
      }
      if (mode === 'md') {
        const gt = s.indexOf('>', i + 1);
        if (gt > i) {
          const inner = s.slice(i + 1, gt);
          if (!inner.includes('<') && (AUTOLINK_RE.test(inner) || EMAIL_RE.test(inner))) {
            push({type: 'opaque', start: i, end: gt + 1});
            continue;
          }
        }
      }
      i++;
      continue;
    }

    if (mode === 'md' && c === '!' && s[i + 1] === '[') {
      // Image: alt text becomes an attribute (never typeset), so it is opaque.
      let k = i + 2;
      let depth = 1;
      while (k < n) {
        if (s[k] === '\\') { k += 2; continue; }
        if (s[k] === '[') depth++;
        else if (s[k] === ']' && --depth === 0) break;
        k++;
      }
      if (k < n) {
        if (s[k + 1] === '(') {
          const d = scanParenDest(s, k + 1);
          if (d) {
            push({type: 'image', start: i, end: d.end, destStart: d.destStart, destEnd: d.destEnd});
            continue;
          }
        }
        if (s[i + 2] !== '^') {
          push({type: 'opaque', start: i, end: k + 1});
          continue;
        }
      }
      i++;
      continue;
    }

    if (mode === 'md' && c === ']' && s[i + 1] === '(') {
      // Link destination/title: attributes, never typeset.
      const d = scanParenDest(s, i + 1);
      if (d) { push({type: 'opaque', start: i + 1, end: d.end}); continue; }
      i++;
      continue;
    }

    if (mode === 'md' && c === '[' && s[i + 1] === '^') {
      const m = /^\[\^([^\]\s]+)\]/.exec(s.slice(i, i + 256));
      if (m) {
        rec.ref(m[1], i);
        i += m[0].length;
        continue;
      }
      i++;
      continue;
    }

    if (c === '$') {
      if (s[i + 1] === '$') {
        if (mode === 'raw') {
          let k = i + 2;
          while (k < n && !(s[k] === '$' && s[k + 1] === '$')) k += s[k] === '\\' ? 2 : 1;
          if (k >= n) {
            rec.error(i, 'unclosed `$$` inside a raw HTML block');
            push({type: 'bad', start: i, end: i + 2});
          } else if (s.slice(i + 2, k).includes('\n')) {
            rec.error(i, 'display math inside a raw HTML block must stay on one line; move it out of the HTML block (blank lines around the tags) and write it as a `$$` block');
            push({type: 'bad', start: i, end: k + 2});
          } else if (s.slice(i + 2, k).trim() === '') {
            rec.error(i, 'empty display math `$$$$`');
            push({type: 'bad', start: i, end: k + 2});
          } else {
            push({type: 'math', display: true, latex: s.slice(i + 2, k), start: i, end: k + 2});
          }
          continue;
        }
        rec.error(i, '`$$` display math must be on its own line(s), separated from surrounding text (KaTeX on the site would treat a stray `$$` as a display delimiter)');
        const k = s.indexOf('$$', i + 2);
        push({type: 'bad', start: i, end: k >= 0 ? k + 2 : i + 2});
        continue;
      }

      // Inline $…$: opener not followed by whitespace; closer not preceded by
      // whitespace and not followed by a digit (pandoc's rule). `\$` never
      // closes. Math never contains a backtick (that would be a code span).
      const next = s[i + 1];
      if (next === undefined || /\s/.test(next)) { i++; continue; }
      let k = i + 1;
      let found = -1;
      while (k < n) {
        const ch = s[k];
        if (ch === '\\') { k += 2; continue; }
        if (ch === '`') break;
        if (ch === '$' && !/\s/.test(s[k - 1]) && !/[0-9]/.test(s[k + 1] || '')) { found = k; break; }
        k++;
      }
      if (found < 0) {
        rec.warn(i, 'unpaired `$` left as a literal dollar sign (write `\\$` to make that explicit)');
        i++;
        continue;
      }
      const latex = s.slice(i + 1, found);
      if (latex.includes('\n')) {
        rec.error(i, 'inline math `$…$` contains a line break; put it on one line (or, if the first `$` is a literal dollar, write it as `\\$`)');
        push({type: 'bad', start: i, end: found + 1});
        continue;
      }
      push({type: 'math', display: false, latex, start: i, end: found + 1});
      continue;
    }

    i++;
  }
  if (n > textStart) segs.push({type: 'text', start: textStart, end: n});
  return segs;
}

// ---------------------------------------------------------------------------
// Converter

class Converter {
  constructor(src, opts) {
    this.lines = src.replace(/^﻿/, '').replace(/\r\n?/g, '\n').split('\n');
    if (this.lines.length && this.lines[this.lines.length - 1] === '') this.lines.pop();
    this.imageBase = opts.imageBase === null ? null : normalizeBase(opts.imageBase ?? DEFAULT_IMAGE_BASE);
    this.emitInlineMath = opts.emitInlineMath ?? inlineMathToGhost;
    this.out = [];
    this.outSrc = [];
    this.math = [];
    this.refs = [];
    this.defs = [];
    this.images = [];
    this.errors = [];
    this.warnings = [];
    this.ctx = []; // containers: {kind: 'list' | 'footnote', col, label}
    this.para = null; // open paragraph being collected for joining
    this.prevKind = 'blank';
    this.tableActive = false;
  }

  // -- bookkeeping ----------------------------------------------------------
  emit(text, srcLine) {
    this.out.push(text);
    this.outSrc.push(srcLine);
  }
  lastOutBlank() {
    return this.out.length === 0 || this.out[this.out.length - 1].trim() === '';
  }
  emitBlank(srcLine) {
    if (!this.lastOutBlank()) this.emit('', srcLine);
  }
  error(line, message) {
    this.errors.push({line, message});
  }
  warn(line, message) {
    this.warnings.push({line, message});
  }
  baseCol() {
    return this.ctx.length ? this.ctx[this.ctx.length - 1].col : 0;
  }
  popTo(ind) {
    while (this.ctx.length && this.ctx[this.ctx.length - 1].col > ind) this.ctx.pop();
  }
  footnote() {
    for (let k = this.ctx.length - 1; k >= 0; k--) {
      if (this.ctx[k].kind === 'footnote') return this.ctx[k].label;
    }
    return null;
  }

  /** Convert inline text (may contain `\n` between source lines). */
  convertInline(s, mode, lineOf) {
    const footnote = this.footnote();
    const rec = {
      ref: (label, pos) => this.refs.push({label, line: lineOf(pos)}),
      error: (pos, msg) => this.error(lineOf(pos), msg),
      warn: (pos, msg) => this.warn(lineOf(pos), msg),
    };
    const segs = scanInline(s, mode, rec);
    let out = '';
    for (const seg of segs) {
      const raw = s.slice(seg.start, seg.end);
      if (seg.type === 'math') {
        this.math.push({latex: seg.latex, display: seg.display, line: lineOf(seg.start), footnote, mode});
        if (mode === 'md') out += this.emitInlineMath(seg.latex);
        else if (seg.display) out += '$$' + escapeHtmlText(seg.latex) + '$$';
        else out += '\\(' + escapeHtmlText(seg.latex) + '\\)';
      } else if (seg.type === 'image') {
        const dest = s.slice(seg.destStart, seg.destEnd);
        this.images.push({url: dest, line: lineOf(seg.start)});
        out += s.slice(seg.start, seg.destStart) + rewriteDest(dest, this.imageBase) + s.slice(seg.destEnd, seg.end);
      } else if (seg.type === 'tag') {
        out += rewriteImgTag(raw, this.imageBase);
      } else {
        out += raw;
      }
    }
    return out;
  }

  emitMd(line, lineNo) {
    this.emit(this.convertInline(line, 'md', () => lineNo), lineNo);
  }

  // -- paragraphs (joined into one line) ------------------------------------
  startPara(prefix, text, lineNo, line) {
    const hb = hardBreakAtEnd(line);
    this.para = {prefix, parts: [{text: hb ? text : text.trimEnd(), lineNo, breakBefore: false, indent: '', hardBreak: hb}]};
    this.prevKind = 'para';
  }

  addToPara(line, lineNo, wsLen) {
    const prev = this.para.parts[this.para.parts.length - 1];
    const hb = hardBreakAtEnd(line);
    const text = line.slice(wsLen);
    this.para.parts.push({text: hb ? text : text.trimEnd(), lineNo, breakBefore: prev.hardBreak, indent: line.slice(0, wsLen), hardBreak: hb});
  }

  flushPara() {
    const p = this.para;
    if (!p) return;
    this.para = null;
    const joined = p.parts.map((x) => x.text).join('\n');
    const starts = [];
    let off = 0;
    for (const x of p.parts) {
      starts.push(off);
      off += x.text.length + 1;
    }
    const lineOf = (pos) => {
      let k = 0;
      while (k + 1 < starts.length && starts[k + 1] <= pos) k++;
      return p.parts[k].lineNo;
    };
    const pieces = this.convertInline(joined, 'md', lineOf).split('\n');
    if (pieces.length !== p.parts.length) throw new Error('internal: line count changed while converting a paragraph');
    let cur = p.prefix + escapeLeadingHash(pieces[0]);
    let curLine = p.parts[0].lineNo;
    for (let k = 1; k < pieces.length; k++) {
      const part = p.parts[k];
      if (part.breakBefore) {
        // the author asked for a hard break (two trailing spaces or `\`): keep it
        this.emit(cur, curLine);
        cur = part.indent + escapeLeadingHash(pieces[k]);
        curLine = part.lineNo;
      } else {
        cur += ' ' + pieces[k];
      }
    }
    this.emit(cur, curLine);
  }

  // Would this line end the open paragraph / table / blockquote paragraph?
  interrupts(i, ind, content, mode) {
    const base = this.baseCol();
    if (ind - base >= 4) return false;
    if (content.startsWith('$$')) return true;
    if (fenceOpener(content)) return true;
    if (content[0] === '>') return true;
    if (HR_RE.test(content)) return true;
    const lm = listMarker(content);
    if (lm) {
      if (mode === 'table' || ind < base) return true;
      if (lm.empty) return false;
      if (lm.ordered && lm.num !== 1) return false;
      return true;
    }
    if (mode !== 'table' && FOOTNOTE_DEF_RE.test(content)) return true;
    const ht = htmlBlockType(content);
    if (ht && HTML_SEQUENCES[ht - 1][2]) return true;
    if (ATX_RE.test(content)) return true;
    if (mode === 'para' && (content[0] === '|' || this.isTableStart(i))) return true;
    return false;
  }

  isTableStart(i) {
    const head = this.lines[i];
    const delim = this.lines[i + 1];
    if (delim === undefined || isBlank(delim)) return false;
    const base = this.baseCol();
    const dInd = indentOf(delim).width;
    if (dInd < base || dInd - base >= 4) return false;
    if (indentOf(head).width - base >= 4) return false;
    const d = delim.trim();
    if (d.length < 2 || !/^[|\-:]/.test(d) || !/^[|\-:\s]/.test(d[1])) return false;
    if (d[0] === '-' && /\s/.test(d[1])) return false;
    if (!/^[|\-:\s]+$/.test(d)) return false;
    const cols = d.split('|');
    let aligns = 0;
    for (let k = 0; k < cols.length; k++) {
      const t = cols[k].trim();
      if (!t) {
        if (k === 0 || k === cols.length - 1) continue;
        return false;
      }
      if (!/^:?-+:?$/.test(t)) return false;
      aligns++;
    }
    const h = head.trim();
    if (!h.includes('|')) return false;
    const hc = escapedSplit(h);
    if (hc.length && hc[0] === '') hc.shift();
    if (hc.length && hc[hc.length - 1] === '') hc.pop();
    return hc.length > 0 && hc.length === aligns;
  }

  // -- main loop --------------------------------------------------------------
  run() {
    let i = 0;
    while (i < this.lines.length) i = this.step(i);
    this.flushPara();
    while (this.out.length && this.out[this.out.length - 1].trim() === '') {
      this.out.pop();
      this.outSrc.pop();
    }
  }

  step(i) {
    const line = this.lines[i];
    const lineNo = i + 1;

    if (isBlank(line)) {
      this.flushPara();
      this.tableActive = false;
      this.emitBlank(lineNo);
      this.prevKind = 'blank';
      return i + 1;
    }

    const {width: ind, length: wsLen} = indentOf(line);
    const content = line.slice(wsLen);
    let lazy = false;

    if (this.para) {
      const base = this.baseCol();
      if (ind >= base && ind - base < 4 && SETEXT_RE.test(content)) {
        this.flushPara();
        this.emit(line, lineNo);
        this.prevKind = 'heading';
        return i + 1;
      }
      if (!this.interrupts(i, ind, content, 'para')) {
        this.addToPara(line, lineNo, wsLen);
        return i + 1;
      }
      this.flushPara();
      lazy = true;
    } else if (this.prevKind === 'quote' && content[0] !== '>') {
      if (!this.interrupts(i, ind, content, 'quote')) {
        this.emitMd(line, lineNo); // lazy continuation of the quoted paragraph
        return i + 1;
      }
    } else if (this.tableActive) {
      if (!this.interrupts(i, ind, content, 'table')) {
        this.emitMd(line, lineNo); // table row
        return i + 1;
      }
      this.tableActive = false;
    }

    // A new block starts here.
    const isMath = content.startsWith('$$');
    if (!(isMath && lazy)) this.popTo(ind);
    const base = this.baseCol();

    if (isMath) {
      return this.displayMath(i, {firstPrefix: ' '.repeat(base), contPrefix: ' '.repeat(base), firstContent: content});
    }

    if (ind - base >= 4) return this.indentedCode(i, base);

    const fence = fenceOpener(content);
    if (fence) return this.fenced(i, fence, null);

    if (content[0] === '>') return this.quote(i);

    if (HR_RE.test(content)) {
      this.emit(line, lineNo);
      this.prevKind = 'hr';
      return i + 1;
    }

    const lm = listMarker(content);
    if (lm) return this.listItem(i, ind, wsLen, lm);

    const fd = FOOTNOTE_DEF_RE.exec(content);
    if (fd) return this.footnoteDef(i, wsLen, fd);

    if (REF_DEF_RE.test(content)) {
      this.emit(line.replace(/^(\s*\[(?:[^[\]\\]|\\.)+\]:[ \t]*)(<[^>]*>|\S+)/, (m, pre, url) =>
        IMAGE_EXT_RE.test(url.replace(/^<|>$/g, '')) ? pre + rewriteDest(url, this.imageBase) : m), lineNo);
      this.prevKind = 'ref';
      return i + 1;
    }

    const ht = htmlBlockType(content);
    if (ht) return this.htmlBlock(i, ht);

    if (ATX_RE.test(content)) {
      this.emitMd(line, lineNo);
      this.prevKind = 'heading';
      return i + 1;
    }

    if (this.isTableStart(i)) {
      this.tableActive = true;
      this.emitMd(line, lineNo);
      this.prevKind = 'table';
      return i + 1;
    }
    if (content[0] === '|') {
      // pipe-led line outside a valid table: never joined to its neighbours
      this.emitMd(line, lineNo);
      this.prevKind = 'pipe';
      return i + 1;
    }

    this.startPara(line.slice(0, wsLen), content, lineNo, line);
    return i + 1;
  }

  indentedCode(i, base) {
    let last = i;
    let j = i;
    while (j < this.lines.length) {
      const l = this.lines[j];
      if (isBlank(l)) { j++; continue; }
      if (indentOf(l).width - base >= 4) { last = j; j++; continue; }
      break;
    }
    for (let k = i; k <= last; k++) this.emit(this.lines[k], k + 1);
    this.prevKind = 'code';
    return last + 1;
  }

  // Fenced code: verbatim. quotePrefix: when the fence is inside a blockquote.
  fenced(i, fence, quoted) {
    this.emit(this.lines[i], i + 1);
    let j = i + 1;
    for (; j < this.lines.length; j++) {
      const l = this.lines[j];
      let inner = l;
      if (quoted) {
        const m = QUOTE_PREFIX_RE.exec(l);
        if (!m) break; // the blockquote ended, and with it the fence
        inner = l.slice(m[1].length);
      }
      this.emit(l, j + 1);
      if (isFenceCloser(inner.trimStart(), fence)) {
        this.prevKind = quoted ? 'quote-fence' : 'fence';
        return j + 1;
      }
    }
    if (!quoted) this.warn(i + 1, 'unclosed code fence: everything after it is code');
    this.prevKind = 'fence';
    return j;
  }

  quote(i) {
    const line = this.lines[i];
    const m = QUOTE_PREFIX_RE.exec(line);
    const inner = line.slice(m[1].length).trimStart();
    const fence = fenceOpener(inner);
    if (fence) return this.fenced(i, fence, true);
    if (inner.startsWith('$$')) {
      const prefix = m[1].endsWith(' ') ? m[1] : m[1] + ' ';
      return this.displayMath(i, {firstPrefix: prefix, contPrefix: prefix, firstContent: inner, quoted: true});
    }
    this.emitMd(line, i + 1);
    this.prevKind = 'quote';
    return i + 1;
  }

  listItem(i, ind, wsLen, lm) {
    const line = this.lines[i];
    const lineNo = i + 1;
    const pad = lm.empty || lm.pad > 4 ? 1 : lm.pad;
    this.ctx.push({kind: 'list', col: ind + lm.marker.length + pad});
    if (lm.empty || lm.pad > 4) {
      this.emit(line, lineNo);
      this.prevKind = 'list';
      return i + 1;
    }
    const prefix = line.slice(0, wsLen + lm.marker.length + lm.pad);
    return this.blockStartText(i, prefix, lm.text);
  }

  footnoteDef(i, wsLen, fd) {
    const line = this.lines[i];
    const lineNo = i + 1;
    const label = fd[1];
    this.defs.push({label, line: lineNo});
    const base = this.baseCol();
    this.ctx.push({kind: 'footnote', col: base + 4, label});
    const after = line.slice(wsLen + fd[0].length);
    const pad = /^[ \t]*/.exec(after)[0].length;
    const text = after.slice(pad);
    if (text === '') {
      this.emit(line, lineNo);
      this.prevKind = 'fndef';
      return i + 1;
    }
    return this.blockStartText(i, line.slice(0, wsLen + fd[0].length + pad), text);
  }

  // The text right after a list marker or footnote label.
  blockStartText(i, prefix, text) {
    const line = this.lines[i];
    const lineNo = i + 1;
    const contPrefix = ' '.repeat(this.baseCol());
    if (text.startsWith('$$')) {
      return this.displayMath(i, {firstPrefix: prefix, contPrefix, firstContent: text, noBlankBefore: true});
    }
    const fence = fenceOpener(text);
    if (fence) return this.fenced(i, fence, null);
    if (ATX_RE.test(text) || text[0] === '>' || text[0] === '|' || htmlBlockType(text) || listMarker(text)) {
      // a block construct as the item's first line: leave the line as it is
      this.emitMd(line, lineNo);
      this.prevKind = 'list';
      return i + 1;
    }
    this.startPara(prefix, text, lineNo, line);
    return i + 1;
  }

  htmlBlock(i, type) {
    const seq = HTML_SEQUENCES[type - 1];
    let end = i + 1; // exclusive
    const first = this.lines[i].trimStart();
    if (type <= 5) {
      if (!seq[1].test(first)) {
        let j = i + 1;
        while (j < this.lines.length && !seq[1].test(this.lines[j].trimStart())) j++;
        end = Math.min(j + 1, this.lines.length);
      }
      for (let k = i; k < end; k++) this.emit(this.lines[k], k + 1);
      this.prevKind = 'html';
      return end;
    }
    while (end < this.lines.length && !isBlank(this.lines[end])) end++;
    const block = this.lines.slice(i, end);
    const starts = [];
    let off = 0;
    for (const l of block) {
      starts.push(off);
      off += l.length + 1;
    }
    const lineOf = (pos) => {
      let k = 0;
      while (k + 1 < starts.length && starts[k + 1] <= pos) k++;
      return i + 1 + k;
    };
    const converted = this.convertInline(block.join('\n'), 'raw', lineOf).split('\n');
    converted.forEach((l, k) => this.emit(l, i + 1 + k));
    this.prevKind = 'html';
    return end;
  }

  /**
   * Display math starting on line i. firstContent starts with `$$`.
   * Emits a blank line, the <div class="math"> block (every line prefixed so
   * it stays inside its list item / footnote / blockquote), and a blank line.
   */
  displayMath(i, {firstPrefix, contPrefix, firstContent, quoted = false, noBlankBefore = false}) {
    const lineNo = i + 1;
    const strip = (l) => {
      if (quoted) {
        const m = QUOTE_PREFIX_RE.exec(l);
        return (m ? l.slice(m[1].length) : l).trim();
      }
      return l.trim();
    };
    const c0 = firstContent.trim();
    const parts = [];
    let endLine = i;
    if (c0.length >= 4 && c0.endsWith('$$')) {
      const inner = c0.slice(2, -2);
      if (inner.includes('$$')) {
        this.error(lineNo, 'display math: text between two `$$…$$` blocks on one line; each `$$` block needs its own line(s)');
      }
      parts.push(inner);
    } else {
      parts.push(c0.slice(2));
      let j = i + 1;
      let closed = false;
      for (; j < this.lines.length; j++) {
        const t = strip(this.lines[j]);
        if (t.endsWith('$$')) {
          const before = t.slice(0, -2);
          if (before.includes('$$')) this.error(j + 1, 'display math: unexpected `$$` inside the block');
          parts.push(before);
          closed = true;
          break;
        }
        if (t.includes('$$')) {
          this.error(j + 1, 'display math: the closing `$$` must end its line (no text after it); put following text on the next line');
        }
        parts.push(t);
      }
      if (!closed) {
        this.error(lineNo, 'display math: no closing `$$` line before the end of the file');
        this.emit(this.lines[i], lineNo);
        this.prevKind = 'para';
        return i + 1;
      }
      endLine = j;
    }
    const latexLines = parts.map((p) => p.trim()).filter((p) => p !== '');
    if (latexLines.length === 0) {
      this.error(lineNo, 'empty display math');
      this.prevKind = 'blank';
      return endLine + 1;
    }
    this.math.push({latex: latexLines.join('\n'), display: true, line: lineNo, footnote: this.footnote(), mode: 'block'});

    const blank = quoted ? contPrefix.trimEnd() : '';
    if (!noBlankBefore) {
      if (quoted) {
        if (this.out.length && this.out[this.out.length - 1].trim() !== blank) this.emit(blank, lineNo);
      } else {
        this.emitBlank(lineNo);
      }
    }
    displayMathToGhost(latexLines).forEach((h, k) => this.emit((k === 0 ? firstPrefix : contPrefix) + h, lineNo));
    this.emit(blank, endLine + 1);
    this.prevKind = quoted ? 'quote' : 'blank';
    return endLine + 1;
  }

  result() {
    return {
      output: this.out.length ? this.out.join('\n') + '\n' : '',
      lineMap: this.outSrc, // output line k (0-based) came from source line lineMap[k] (1-based)
      math: this.math,
      footnoteRefs: this.refs,
      footnoteDefs: this.defs,
      images: this.images,
      errors: this.errors,
      warnings: this.warnings,
    };
  }
}

/**
 * Convert post.src.md text to Ghost Markdown.
 * opts.imageBase: prefix for relative image paths (null = leave them alone).
 * opts.emitInlineMath: override the inline-math emitter (tests use this to
 *   build a deliberately naive conversion).
 */
export function processSource(src, opts = {}) {
  const c = new Converter(src, opts);
  c.run();
  return c.result();
}

/** The math spans the author intended, in source order. */
export function extractMath(src) {
  return processSource(src, {imageBase: null}).math;
}
