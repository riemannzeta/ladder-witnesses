#!/usr/bin/env node
// Lint post.ghost.md against post.src.md the way symmetrybroken.com will see it.
//
//   node lint.mjs [--src post.src.md] [--ghost post.ghost.md] [--preview preview.html]
//                 [--image-base URL]
//
// 1. Render post.ghost.md with Ghost's exact markdown-it configuration.
// 2. Walk the HTML text nodes like KaTeX auto-render (same delimiters, same
//    ignored tags) and collect every math span, in document order.
// 3. Require that list to equal the math in post.src.md (same tokenizer as the
//    build): same count, order, display flag and byte-identical LaTeX.
// 4. Typeset every span with KaTeX 0.16.9 (throwOnError: true).
// 5. Check footnotes, <br> inside paragraphs, div.math placement, figures.
// Writes preview.html and exits non-zero on any error.

import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
import {parse} from 'parse5';
import katex from 'katex';
import {createGhostMarkdownIt, renderWithLineMap} from './lib/ghost-markdown.mjs';
import {collectRenderedMath, findSplitSpans, splitAtDelimiters, walkLikeAutoRender, SITE_DELIMITERS} from './lib/autorender.mjs';
import {processSource, DEFAULT_IMAGE_BASE, normalizeBase, isRelativeUrl} from './lib/source.mjs';
import {parseArgs, isMain} from './lib/cli.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
export const MARKDOWN_IT_VERSION = require('markdown-it/package.json').version;
export const KATEX_VERSION = katex.version;

const KATEX_CDN = 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/';
// Official jsDelivr SRI hashes from the katex@0.16.9 README (test/run.mjs
// re-derives them from node_modules/katex).
export const KATEX_SRI = {
  'katex.min.css': 'sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV',
  'katex.min.js': 'sha384-XjKyOOlGwcjNTAIQHIpgOno0Hl1YQqzUOEleOLALmuqehneUG+vnGctmUb0ZY0l8',
  'contrib/auto-render.min.js': 'sha384-+VBxd3r6XgURycqtZ117nYw44OOcIax56Z4dCRWbxyPt0Koah1uHoK0o4+/RRE05',
};

const DOC_PREFIX = '<!DOCTYPE html><html><head></head><body>';
const DOC_SUFFIX = '</body></html>';

const show = (s, max = 160) => {
  const one = s.replace(/\n/g, '⏎');
  return '`' + (one.length > max ? one.slice(0, max - 1) + '…' : one) + '`';
};

// What a backslash-escape loss does to common LaTeX.
const ESCAPE_HINT = {
  '{': 'braces', '}': 'braces', ',': 'thin space', ';': 'thick space', ':': 'medium space',
  '!': 'negative space', '|': 'norm bars', '\\': 'line break / backslash', '_': 'literal underscore',
  '#': 'hash', '%': 'percent', '&': 'ampersand', '$': 'dollar', ' ': 'control space',
};

const SPLIT_CAUSE = {
  sup: ['caret', 'markdown-it-sup paired two unescaped carets `^…^` into <sup>'],
  sub: ['tilde', 'markdown-it-sub paired two unescaped tildes `~…~` into <sub>'],
  em: ['underscore', 'Markdown emphasis paired unescaped underscores/asterisks `_…_` into <em>'],
  strong: ['underscore', 'Markdown strong emphasis paired `__…__` / `**…**` into <strong>'],
  mark: ['equals', 'markdown-it-mark paired `==…==` into <mark>'],
  s: ['tilde', '`~~…~~` became strikethrough <s>'],
  a: ['link', 'a link (linkify or `[…](…)`) was created inside the math'],
  br: ['newline', 'a line break <br> inside the math (a hard-wrapped line)'],
  code: ['backtick', 'backticks inside the math became <code>'],
  img: ['image', 'image syntax inside the math'],
};

// Brace depth the way auto-render's findEndOfMath counts it (`\x` skipped).
function braceDepthAtEnd(latex) {
  let depth = 0;
  for (let k = 0; k < latex.length; k++) {
    if (latex[k] === '\\') k++;
    else if (latex[k] === '{') depth++;
    else if (latex[k] === '}') depth--;
  }
  return depth;
}

function markdownUnescape(s) {
  return s.replace(/\\([!-/:-@[-`{-~])/g, '$1');
}

function explainLatexChange(expected, got) {
  if (markdownUnescape(expected) === got) {
    const eaten = [...new Set(expected.match(/\\[!-/:-@[-`{-~]/g) || [])];
    const hints = [...new Set(eaten.map((e) => ESCAPE_HINT[e[1]]).filter(Boolean))];
    return {
      code: 'backslash-eaten',
      message: `Markdown ate the backslash in ${eaten.map((e) => '`' + e + '`').join(', ')}` +
        (hints.length ? ` (${hints.join(', ')})` : '') +
        `: KaTeX gets ${show(got)} instead of ${show(expected)}. A backslash before ASCII punctuation is a Markdown escape, so each one needs its own escaping backslash.`,
    };
  }
  const squash = (s) => s.replace(/\s+/g, ' ').trim();
  if (squash(expected) === squash(got)) {
    return {
      code: 'math-whitespace',
      message: `rendered LaTeX differs from the source only in whitespace: expected ${show(expected)}, KaTeX gets ${show(got)}`,
    };
  }
  let k = 0;
  while (k < expected.length && k < got.length && expected[k] === got[k]) k++;
  return {
    code: 'math-changed',
    message: `rendered LaTeX differs from the source at character ${k}: expected ${show(expected)}, KaTeX gets ${show(got)}`,
  };
}

// LCS alignment of intended vs rendered math, with del/ins runs paired up.
function alignMath(a, b) {
  const key = (x) => (x.display ? 'D' : 'I') + '\u0000' + x.latex;
  const n = a.length;
  const m = b.length;
  const dp = Array.from({length: n + 1}, () => new Int32Array(m + 1));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      dp[i][j] = key(a[i]) === key(b[j]) ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }
  const ops = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (key(a[i]) === key(b[j])) ops.push({op: 'equal', a: a[i++], b: b[j++]});
    else if (dp[i + 1][j] >= dp[i][j + 1]) ops.push({op: 'del', a: a[i++]});
    else ops.push({op: 'ins', b: b[j++]});
  }
  while (i < n) ops.push({op: 'del', a: a[i++]});
  while (j < m) ops.push({op: 'ins', b: b[j++]});
  const out = [];
  for (let k = 0; k < ops.length;) {
    if (ops[k].op === 'equal') {
      out.push(ops[k++]);
      continue;
    }
    const dels = [];
    const ins = [];
    while (k < ops.length && ops[k].op !== 'equal') (ops[k].op === 'del' ? dels : ins).push(ops[k++]);
    const p = Math.min(dels.length, ins.length);
    for (let q = 0; q < p; q++) out.push({op: 'change', a: dels[q].a, b: ins[q].b});
    for (let q = p; q < dels.length; q++) out.push(dels[q]);
    for (let q = p; q < ins.length; q++) out.push(ins[q]);
  }
  return out;
}

function findBody(doc) {
  const html = doc.childNodes.find((n) => n.tagName === 'html');
  return html.childNodes.find((n) => n.tagName === 'body');
}

/**
 * Lint. Returns {errors, warnings, stats, html, rendered, intended, fresh}.
 * Each problem: {code, message, ghostLine?, srcLine?}.
 */
export function lintPost({src, ghost, imageBase = DEFAULT_IMAGE_BASE, srcDir = null}) {
  imageBase = normalizeBase(imageBase);
  ghost = ghost.replace(/\r\n?/g, '\n');
  const errors = [];
  const warnings = [];

  // -- what the author wrote ------------------------------------------------
  const source = processSource(src, {imageBase});
  for (const e of source.errors) errors.push({code: 'source', message: e.message, srcLine: e.line});
  for (const w of source.warnings) warnings.push({code: 'source', message: w.message, srcLine: w.line});
  const fresh = source.output === ghost;
  if (!fresh) {
    warnings.push({code: 'stale', message: 'post.ghost.md is not what build_ghost.mjs makes from post.src.md now (stale or hand-edited): run `npm run build`'});
  }
  const srcLineOf = (ghostLine) => (fresh && ghostLine ? source.lineMap[ghostLine - 1] ?? null : null);

  // -- what Ghost renders ---------------------------------------------------
  const md = createGhostMarkdownIt();
  const {html, tokens, env, lineAt} = renderWithLineMap(md, ghost);
  const ghostLineAtHtml = (offset) => {
    const l = lineAt(offset);
    return l === null ? null : l + 1;
  };
  const at = (ghostLine) => ({ghostLine, srcLine: srcLineOf(ghostLine)});

  // token-level checks
  const ghostLines = ghost.split('\n');
  let divMathBlocks = 0;
  let quoteDepth = 0;
  for (const tok of tokens) {
    const tokLine = tok.map ? tok.map[0] + 1 : null;
    if (tok.type === 'blockquote_open') quoteDepth++;
    if (tok.type === 'blockquote_close') quoteDepth--;
    if (tok.type === 'html_block' && /<div\b[^>]*\bclass\s*=\s*["']?math\b/i.test(tok.content)) {
      divMathBlocks++;
      const c = tok.content.trim();
      if (!/^<div class="math">\$\$[\s\S]*\$\$<\/div>$/.test(c)) {
        warnings.push({code: 'div-math-mixed', message: `HTML block holding <div class="math"> has other content: ${show(c, 100)}`, ...at(tokLine)});
      }
    }
    if (tok.type === 'heading_open' && tokLine && /^\s{0,3}#{1,6}[^#\s]/.test(ghostLines[tokLine - 1] || '')) {
      warnings.push({code: 'lazy-heading', message: 'a `#` without a following space starts a heading on Ghost (markdown-it-lazy-headers) but not on GitHub', ...at(tokLine)});
    }
    if (tok.type !== 'inline' || !tok.children) continue;
    let breaks = 0;
    for (const child of tok.children) {
      const childLine = tokLine === null ? null : tokLine + breaks;
      if (child.type === 'softbreak' || child.type === 'hardbreak') breaks++;
      if (child.type === 'html_inline' && /<div\b[^>]*\bclass\s*=\s*["']?math\b/i.test(child.content)) {
        errors.push({code: 'div-math-inline', message: '<div class="math"> is inside a paragraph, not its own HTML block, so Markdown processes the math inside it (e.g. `\\,` → `,`). It must start its own line with a blank line before and after.', ...at(childLine)});
      }
      if (child.type === 'softbreak') {
        warnings.push({
          code: 'br-in-paragraph',
          message: quoteDepth > 0
            ? 'a newline inside a blockquote paragraph becomes <br> on Ghost (blockquote lines are kept as written; join them in the source if the break is unintended)'
            : 'a newline inside a paragraph becomes <br> on Ghost (breaks: true); unjoined hard wrap?',
          ...at(childLine),
        });
      }
      if (child.type === 'sup_open' || child.type === 'sub_open' || child.type === 'mark_open') {
        const tag = child.tag;
        warnings.push({code: `${tag}-in-text`, message: `Ghost turned ${tag === 'sup' ? '`^…^`' : tag === 'sub' ? '`~…~`' : '`==…==`'} into <${tag}>; GitHub shows it literally`, ...at(childLine)});
      }
      if (child.type === 'link_open') {
        const href = child.attrGet('href') || '';
        if (isRelativeUrl(href)) warnings.push({code: 'relative-link', message: `relative link ${show(href)} will not resolve on symmetrybroken.com`, ...at(childLine)});
      }
      if (child.type === 'image') {
        const s = child.attrGet('src') || '';
        if (isRelativeUrl(s)) {
          warnings.push({code: 'relative-image', message: `relative image ${show(s)} will not resolve on Ghost (build with --image-base)`, ...at(childLine)});
        } else if (srcDir && imageBase && s.startsWith(imageBase)) {
          const local = path.join(srcDir, decodeURIComponent(s.slice(imageBase.length)));
          if (!fs.existsSync(local)) warnings.push({code: 'missing-figure', message: `figure ${show(s)} has no local file ${show(path.relative(process.cwd(), local))}`, ...at(childLine)});
        }
      }
    }
  }

  // -- KaTeX auto-render emulation on the parsed HTML -------------------------
  const doc = parse(DOC_PREFIX + html + DOC_SUFFIX, {sourceCodeLocationInfo: true});
  const body = findBody(doc);
  const lineOfNode = (node) => {
    const loc = node && node.sourceCodeLocation;
    return loc ? ghostLineAtHtml(Math.max(0, loc.startOffset - DOC_PREFIX.length)) : null;
  };
  const {math: renderedRaw, problems} = collectRenderedMath(body);
  const rendered = renderedRaw.map((r) => ({...r, ghostLine: lineOfNode(r.node)}));

  // spans that markdown broke apart by inserting tags
  const splits = findSplitSpans(body);
  const explained = new Set();
  for (const sp of splits) {
    for (const n of sp.nodes) explained.add(n);
    const tags = [...sp.elements].sort();
    for (const tag of tags) {
      const [what, cause] = SPLIT_CAUSE[tag] || ['html', `an HTML tag <${tag}> inside the math`];
      errors.push({
        code: `split-by-${tag}`,
        message: `math span ${show(sp.flatText)} is broken by <${tag}> (${what} problem): ${cause}, so KaTeX never sees the span`,
        ...at(lineOfNode(sp.node)),
      });
    }
  }
  for (const p of problems) {
    if (explained.has(p.node)) continue;
    errors.push({
      code: p.kind === 'unmatched-open' ? 'unmatched-delimiter' : 'stray-delimiter',
      message: p.kind === 'unmatched-open'
        ? `opening ${show(p.delimiter)} has no closing delimiter in the same text node (${show(p.text, 80)}); KaTeX leaves it as raw text`
        : `stray closing ${show(p.delimiter)} in text ${show(p.text, 80)}`,
      ...at(lineOfNode(p.node)),
    });
  }

  // -- intended math, in rendered order (footnotes move to the end) -----------
  const fnOrder = ((env.footnotes && env.footnotes.list) || []).map((f) => f.label).filter((l) => l !== undefined);
  const intended = source.math.filter((m) => m.footnote === null);
  for (const label of fnOrder) intended.push(...source.math.filter((m) => m.footnote === label));
  const definedLabels = new Set(source.footnoteDefs.map((d) => d.label));
  const seenDefs = new Set();
  for (const d of source.footnoteDefs) {
    if (seenDefs.has(d.label)) warnings.push({code: 'footnote-duplicate', message: `footnote [^${d.label}] is defined twice`, srcLine: d.line});
    seenDefs.add(d.label);
    if (!fnOrder.includes(d.label)) {
      const lost = source.math.filter((m) => m.footnote === d.label).length;
      warnings.push({code: 'footnote-unused', message: `footnote [^${d.label}] is never referenced, so Ghost drops it${lost ? ` (with ${lost} math span(s))` : ''}`, srcLine: d.line});
    }
  }
  for (const r of source.footnoteRefs) {
    if (!definedLabels.has(r.label)) errors.push({code: 'footnote-undefined', message: `footnote reference [^${r.label}] has no definition; it would render literally`, srcLine: r.line});
  }
  walkLikeAutoRender(body, (text, nodes) => {
    for (const piece of splitAtDelimiters(text, SITE_DELIMITERS)) {
      const m = piece.type === 'text' && /\[\^[^\]\s]+\]/.exec(piece.data);
      if (m) {
        errors.push({code: 'footnote-unresolved', message: `unresolved footnote reference ${show(m[0])} in the rendered page`, ...at(lineOfNode(nodes[0]))});
        break;
      }
    }
  });

  // -- compare --------------------------------------------------------------
  const ok = new Set(); // rendered items that match the source
  for (const op of alignMath(intended, rendered)) {
    if (op.op === 'equal') {
      ok.add(op.b);
      continue;
    }
    if (op.op === 'change') {
      let ex;
      if (op.a.latex === op.b.latex) {
        ex = {code: 'math-display-flag', message: `${show(op.a.latex)} should be ${op.a.display ? 'display' : 'inline'} math but renders as ${op.b.display ? 'display' : 'inline'}`};
      } else {
        ex = explainLatexChange(op.a.latex, op.b.latex);
      }
      errors.push({...ex, srcLine: op.a.line, ghostLine: op.b.ghostLine});
    } else if (op.op === 'del') {
      let why = '';
      if (braceDepthAtEnd(op.a.latex) > 0) {
        why = ' (its braces are unbalanced, and auto-render only accepts a closing delimiter at brace depth 0)';
      } else {
        try {
          katex.renderToString(op.a.latex, {displayMode: op.a.display, throwOnError: true, strict: 'ignore'});
        } catch (e) {
          why = ` (and KaTeX could not parse it anyway: ${e.message})`;
        }
      }
      errors.push({code: 'math-missing', message: `${op.a.display ? 'display' : 'inline'} math ${show(op.a.latex)} from the source never reaches KaTeX${why}`, srcLine: op.a.line});
    } else {
      errors.push({code: 'math-unexpected', message: `KaTeX would typeset ${show(op.b.rawData)}, which is not math in the source`, ...at(op.b.ghostLine)});
    }
  }

  // -- KaTeX 0.16.9, same macro sharing as renderMathInElement ----------------
  const macros = {};
  const katexBad = new Set();
  for (const r of rendered) {
    const strictWarnings = [];
    try {
      const out = katex.renderToString(r.latex, {
        displayMode: r.display,
        throwOnError: true,
        macros,
        strict: (code, msg) => {
          strictWarnings.push(`${code}: ${msg}`);
          return 'ignore';
        },
      });
      if (out.includes('color:#cc0000')) {
        warnings.push({code: 'katex-untrusted', message: `KaTeX renders part of ${show(r.latex)} in red (command needs trust:true, which the site does not set)`, ...at(r.ghostLine)});
      }
    } catch (e) {
      katexBad.add(r);
      errors.push({code: 'katex-error', message: `KaTeX ${KATEX_VERSION} cannot parse ${show(r.latex)}: ${e.message}`, ...at(r.ghostLine)});
    }
    for (const w of strictWarnings) warnings.push({code: 'katex-strict', message: `${show(r.latex, 80)}: ${w}`, ...at(r.ghostLine)});
  }

  const good = rendered.filter((r) => ok.has(r) && !katexBad.has(r));
  const stats = {
    inlineOk: good.filter((r) => !r.display).length,
    displayOk: good.filter((r) => r.display).length,
    inlineTotal: intended.filter((m) => !m.display).length,
    displayTotal: intended.filter((m) => m.display).length,
    renderedTotal: rendered.length,
    divMathBlocks,
  };
  return {errors, warnings, stats, html, rendered, intended, fresh};
}

export function formatProblem(p, severity, names = {src: 'post.src.md', ghost: 'post.ghost.md'}) {
  const where = [];
  if (p.ghostLine) where.push(`${names.ghost}:${p.ghostLine}`);
  if (p.srcLine) where.push(`${names.src}:${p.srcLine}`);
  return `${severity} [${p.code}]${where.length ? ' ' + where.join(' ← ') : ''}: ${p.message}`;
}

export function summaryLine(report) {
  const s = report.stats;
  const spans = `${s.inlineOk} inline, ${s.displayOk} display math spans OK`;
  if (report.errors.length === 0) return spans;
  return `FAIL: ${report.errors.length} error(s), ${report.warnings.length} warning(s); ` +
    `${s.inlineOk}/${s.inlineTotal} inline, ${s.displayOk}/${s.displayTotal} display math spans OK`;
}

const escapeAttr = (s) => s.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');
const escapeText = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

/**
 * A standalone page that loads KaTeX 0.16.9 exactly the way the site does.
 * `expected` is the lint's predicted list of typeset LaTeX; after rendering,
 * the page compares it with KaTeX's own <annotation> of every span.
 */
export function previewPage({html, title, status, expected = [], imageMap = (s) => s}) {
  const body = html.replace(/(<img\b[^>]*?\ssrc=")([^"]*)(")/g, (m, a, url, b) => a + escapeAttr(imageMap(url.replace(/&amp;/g, '&'))) + b);
  const delimiters = JSON.stringify(SITE_DELIMITERS);
  const expectedJson = JSON.stringify(expected).replace(/</g, '\\u003c');
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${escapeText(title)}</title>
<link rel="stylesheet" href="${KATEX_CDN}katex.min.css" integrity="${KATEX_SRI['katex.min.css']}" crossorigin="anonymous">
<script defer src="${KATEX_CDN}katex.min.js" integrity="${KATEX_SRI['katex.min.js']}" crossorigin="anonymous"></script>
<script defer src="${KATEX_CDN}contrib/auto-render.min.js" integrity="${KATEX_SRI['contrib/auto-render.min.js']}" crossorigin="anonymous"></script>
<script>
  // Same call as symmetrybroken.com: KaTeX 0.16.9 auto-render, these delimiters only.
  document.addEventListener("DOMContentLoaded", function () {
    var failures = [];
    renderMathInElement(document.body, {
      delimiters: ${delimiters},
      errorCallback: function (msg, err) { failures.push(msg + " " + err); console.error(msg, err); }
    });
    // Compare what this browser typeset with what lint.mjs predicted.
    var expected = JSON.parse(document.getElementById("lint-expected").textContent);
    var got = Array.prototype.map.call(
      document.querySelectorAll("article .katex-mathml annotation"),
      function (a) { return a.textContent; });
    var same = got.length === expected.length &&
      got.every(function (x, k) { return x === expected[k]; });
    if (!same) console.error("lint predicted", expected, "browser typeset", got);
    var bar = document.getElementById("preview-katex");
    bar.textContent = "KaTeX in this browser: " + got.length + " spans typeset (" +
      document.querySelectorAll("article .katex-display").length + " display), " +
      failures.length + " failures, " +
      (same ? "identical to the lint's prediction" : "DIFFERENT from the lint's prediction (see console)");
    if (failures.length || !same) bar.className = "bad";
  });
</script>
<script type="application/json" id="lint-expected">${expectedJson}</script>
<style>
  :root { --ink: #1f1f1f; --muted: #6b6b6b; --rule: #e4e0d8; --paper: #fdfcf9; --accent: #8a3b12; --code: #f3f0ea; }
  @media (prefers-color-scheme: dark) {
    :root { --ink: #e8e6e1; --muted: #a29e96; --rule: #3a3833; --paper: #1b1a18; --accent: #e3a27c; --code: #2a2825; }
  }
  html { background: var(--paper); }
  body { margin: 0; color: var(--ink); background: var(--paper); font: 20px/1.6 Georgia, "Iowan Old Style", "Palatino Linotype", Palatino, "Times New Roman", serif; }
  .preview-bar { position: sticky; top: 0; z-index: 2; font: 13px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--code); border-bottom: 1px solid var(--rule); padding: 8px 16px; color: var(--muted); }
  .preview-bar code { background: none; padding: 0; }
  .preview-bar .bad { color: #b3261e; font-weight: 600; }
  article { max-width: 720px; margin: 0 auto; padding: 32px 16px 96px; overflow-wrap: break-word; }
  h1, h2, h3, h4 { line-height: 1.25; margin: 1.6em 0 0.5em; }
  h1 { font-size: 2em; margin-top: 0.4em; }
  a { color: var(--accent); }
  img { max-width: 100%; height: auto; display: block; margin: 1.5em auto; }
  code, pre { font: 0.8em/1.5 "SFMono-Regular", Menlo, Consolas, monospace; background: var(--code); }
  code { padding: 0.1em 0.3em; border-radius: 3px; }
  pre { padding: 12px 16px; overflow-x: auto; border-radius: 4px; }
  pre code { padding: 0; background: none; font-size: 1em; }
  blockquote { margin: 1.2em 0; padding: 0 0 0 1em; border-left: 3px solid var(--rule); color: var(--muted); }
  table { border-collapse: collapse; margin: 1.5em 0; font-size: 0.9em; display: block; overflow-x: auto; }
  th, td { border-bottom: 1px solid var(--rule); padding: 6px 12px; }
  .math { overflow-x: auto; overflow-y: hidden; }
  .katex-display { margin: 1.2em 0; }
  hr.footnotes-sep { border: 0; border-top: 1px solid var(--rule); margin: 3em 0 1em; }
  .footnotes { font-size: 0.85em; color: var(--muted); }
</style>
</head>
<body>
<div class="preview-bar"><code>${escapeText(status)}</code> · <span id="preview-katex">KaTeX not loaded yet (needs network for cdn.jsdelivr.net)</span></div>
<article>
${body}
</article>
</body>
</html>
`;
}

function main() {
  let args;
  try {
    args = parseArgs(process.argv.slice(2), {src: 'value', ghost: 'value', preview: 'value', 'image-base': 'value'});
  } catch (e) {
    console.error(`lint: ${e.message}`);
    process.exit(2);
  }
  if (args.help) {
    console.log('usage: node lint.mjs [--src post.src.md] [--ghost post.ghost.md] [--preview preview.html] [--image-base URL]');
    return;
  }
  const srcPath = args.src ? path.resolve(args.src) : path.join(HERE, 'post.src.md');
  const ghostPath = args.ghost ? path.resolve(args.ghost) : path.join(HERE, 'post.ghost.md');
  const previewPath = args.preview ? path.resolve(args.preview) : path.join(HERE, 'preview.html');
  const imageBase = normalizeBase(args['image-base'] ?? DEFAULT_IMAGE_BASE);
  const rel = (p) => path.relative(process.cwd(), p) || p;
  for (const p of [srcPath, ghostPath]) {
    if (!fs.existsSync(p)) {
      console.error(`lint: ${rel(p)} not found${p === ghostPath ? ' (run `npm run build` first)' : ''}`);
      process.exit(2);
    }
  }
  const srcDir = path.dirname(srcPath);
  const report = lintPost({
    src: fs.readFileSync(srcPath, 'utf8'),
    ghost: fs.readFileSync(ghostPath, 'utf8'),
    imageBase,
    srcDir,
  });
  const names = {src: rel(srcPath), ghost: rel(ghostPath)};
  for (const e of report.errors) console.error(formatProblem(e, 'error', names));
  for (const w of report.warnings) console.warn(formatProblem(w, 'warning', names));

  const summary = summaryLine(report);
  const previewDir = path.dirname(previewPath);
  const imageMap = (url) => {
    if (!url.startsWith(imageBase)) return url;
    const local = path.join(srcDir, decodeURIComponent(url.slice(imageBase.length)));
    return fs.existsSync(local) ? path.relative(previewDir, local).split(path.sep).join('/') : url;
  };
  const status = `lint: ${summary}${report.warnings.length && !report.errors.length ? `, ${report.warnings.length} warning(s)` : ''} · markdown-it ${MARKDOWN_IT_VERSION} + Ghost plugins · KaTeX ${KATEX_VERSION} · figures shown from local files`;
  fs.writeFileSync(previewPath, previewPage({
    html: report.html,
    title: `Preview: ${path.basename(ghostPath)}`,
    status,
    expected: report.rendered.map((r) => r.latex),
    imageMap,
  }));

  console.log(`lint: ${summary}` +
    `${report.errors.length ? '' : `; ${report.warnings.length} warning(s)`}` +
    ` (markdown-it ${MARKDOWN_IT_VERSION} + Ghost plugins, KaTeX ${KATEX_VERSION})`);
  console.log(`lint: preview written to ${rel(previewPath)}`);
  process.exit(report.errors.length ? 1 : 0);
}

if (isMain(import.meta.url)) main();
