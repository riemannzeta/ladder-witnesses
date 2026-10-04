// npm test: build the fixture, lint it, and prove the lint catches the
// failure modes when the escaping is left out.

import {test} from 'node:test';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {parse} from 'parse5';

import {processSource, extractMath, inlineMathToGhost, escapeInlineLatex} from '../lib/source.mjs';
import {createGhostMarkdownIt} from '../lib/ghost-markdown.mjs';
import {splitAtDelimiters, SITE_DELIMITERS, collectRenderedMath} from '../lib/autorender.mjs';
import {lintPost, formatProblem, summaryLine, KATEX_SRI} from '../lint.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const BLOG = path.resolve(HERE, '..');
const FIXTURE = path.join(HERE, 'fixtures', 'sample.src.md');
const FIXTURE_DIR = path.dirname(FIXTURE);
const SRC = fs.readFileSync(FIXTURE, 'utf8');

const lint = (src, ghost) => lintPost({src, ghost, srcDir: FIXTURE_DIR});
const codes = (problems) => problems.map((p) => p.code);
const dump = (r) => [...r.errors.map((e) => formatProblem(e, 'error')), ...r.warnings.map((w) => formatProblem(w, 'warning'))].join('\n');

// Deliberately naive conversions -------------------------------------------

// Same pipeline, but inline math is NOT escaped: `$x$` -> `\\(x\\)` verbatim.
const naiveInline = (src) => processSource(src, {emitInlineMath: (latex) => '\\\\(' + latex + '\\\\)'}).output;

// The "obvious" regex job on the raw source: no tokenizer, no escaping.
const naiveRegex = (src) => src
  .replace(/\$\$([\s\S]+?)\$\$/g, (m, tex) => '<div class="math">$$' + tex + '$$</div>')
  .replace(/\$([^$\n]+?)\$/g, (m, tex) => '\\\\(' + tex + '\\\\)');

// Helpers ------------------------------------------------------------------

function textNodes(html) {
  const out = [];
  const walk = (n) => {
    for (const c of n.childNodes || []) {
      if (c.nodeName === '#text') out.push(c.value);
      else walk(c);
    }
  };
  walk(parse('<!DOCTYPE html><html><body>' + html + '</body></html>'));
  return out;
}

function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const ASCII = Array.from({length: 0x7f - 0x20}, (_, k) => String.fromCharCode(0x20 + k)).join('');
const ALPHABET = ASCII + '\\\\\\{}^_~|*=`[]<>&' + 'αé∑\t';

function randomStrings(n, seed, {maxLen = 40, exclude = () => false} = {}) {
  const rnd = mulberry32(seed);
  const out = [];
  while (out.length < n) {
    const len = 1 + Math.floor(rnd() * maxLen);
    let s = '';
    for (let k = 0; k < len; k++) s += ALPHABET[Math.floor(rnd() * ALPHABET.length)];
    if (!exclude(s)) out.push(s);
  }
  return out;
}

const ADVERSARIAL = [
  ASCII,
  '\\{V_i^A\\}', 'e^{-x^2}', '\\hat{\\theta}_{zx}', 'a_1 b_2 c_3 * d*', '^a^ ~b~ ==c== ~~d~~',
  '\\|x\\|', '|x|', 'a \\\\ b', '\\', '\\\\', 'x\\', '[^1]', '[a](b)', '![a](b)', '<b>x</b>', '<!-- c -->',
  '&amp;', '&#x41;', '`code`', '``', 'http://example.org/x_y', 'mailto:a@b.co', 'a@b.co', 'www.example.org',
  '# h', '#h', '> q', '- l', '1. l', '   lead', 'trail   ', 'a\tb', '\\(x\\)', '\\[x\\]', '$x$', '\\$',
  '\\text{do}(x)', 'P(Y=1 \\mid do(X=1))', 'x_{t+1}', '\\,\\;\\:\\!', '\\_\\#\\%\\&\\$',
];

// ---------------------------------------------------------------------------

test('fixture: build has no errors or warnings', () => {
  const res = processSource(SRC);
  assert.deepEqual(res.errors, []);
  assert.deepEqual(res.warnings, []);
  assert.equal(res.math.filter((m) => !m.display).length, 26);
  assert.equal(res.math.filter((m) => m.display).length, 3);
});

test('fixture: lint passes, every span byte-identical and KaTeX-clean', () => {
  const ghost = processSource(SRC).output;
  const r = lint(SRC, ghost);
  assert.deepEqual(r.errors, [], dump(r));
  assert.ok(r.fresh);
  // the only warning: the blockquote keeps its line break (by design)
  assert.deepEqual(codes(r.warnings), ['br-in-paragraph'], dump(r));
  assert.match(r.warnings[0].message, /blockquote/);
  assert.equal(summaryLine(r), '26 inline, 3 display math spans OK');
  assert.equal(r.rendered.length, 29);
  assert.equal(r.stats.divMathBlocks, 3);
});

test('fixture: generated Markdown for the tricky cases', () => {
  const lines = processSource(SRC).output.split('\n');
  const has = (s) => assert.ok(lines.includes(s), `missing line:\n${s}`);
  has('Failure mode 1, braces: the set of potential outcomes \\\\(\\\\\\{V\\_i\\^A\\\\\\}\\\\) must keep its escaped braces.');
  has('Failure mode 2, carets: the Gaussian kernel \\\\(e\\^\\{\\-x\\^2\\}\\\\) has two carets with no space between them.');
  has('Failure mode 3, underscores across spans: compare \\\\(\\\\hat\\{\\\\theta\\}\\_\\{zx\\}\\\\) vs \\\\(n\\_\\{z\\}\\\\) in the same paragraph.');
  has('<div class="math">$$\\begin{aligned}');
  has('\\hat{\\tau}_{\\text{naive}} &amp;&lt; \\tau_{\\max} \\quad \\text{if} \\quad a \\,\\&amp;\\, b');
  has('\\end{aligned}$$</div>');
  has('     <div class="math">$$\\mathbb{E}[Y_x] = \\sum_u P(u)\\, Y_x(u)$$</div>');
  assert.ok(lines.some((l) => l.includes('A literal price is \\$12, and `$x$` in inline code is not math.')));
  has('![A relative figure, alt text with $x$ is not typeset](https://raw.githubusercontent.com/riemannzeta/ladder-witnesses/main/blog/figures/example.svg)');
  has('\\#hashtags are not headings on GitHub, so they must not become one on Ghost.');
  has('[^2]: A short footnote with \\\\(\\\\lambda\\_\\{\\\\min\\}\\\\).');
  // fenced code untouched
  has('# fenced code: $$ and $x$ are not math here');
  has('price = "$$"');
});

test('NEGATIVE: unescaped inline math fails, with brace, caret and underscore diagnostics', () => {
  const r = lint(SRC, naiveInline(SRC));
  assert.ok(r.errors.length > 0, 'lint must fail');
  const brace = r.errors.find((e) => e.code === 'backslash-eaten' && e.message.includes('`\\{`') && e.message.includes('braces'));
  assert.ok(brace, 'brace problem not reported:\n' + dump(r));
  assert.match(brace.message, /KaTeX gets `\{V_i\^A\}` instead of `\\\{V_i\^A\\\}`/);
  assert.equal(brace.srcLine, 9);
  const caret = r.errors.find((e) => e.code === 'split-by-sup' && e.message.includes('caret'));
  assert.ok(caret, 'caret problem not reported:\n' + dump(r));
  assert.match(caret.message, /`\\\(e\{-x2\}\\\)`/);
  const under = r.errors.filter((e) => e.code === 'split-by-em' && e.message.includes('underscore'));
  assert.equal(under.length, 2, 'underscore problem not reported for both spans:\n' + dump(r));
  assert.match(under[0].message, /\\hat\{\\theta\}/);
  assert.match(under[1].message, /`\\\(n\{z\}\\\)`/);
  assert.match(summaryLine(r), /^FAIL: /);
});

test('NEGATIVE: naive regex conversion of the raw source fails, with the same diagnostics', () => {
  const r = lint(SRC, naiveRegex(SRC));
  assert.ok(r.errors.length > 0, 'lint must fail');
  const c = codes(r.errors);
  assert.ok(r.errors.some((e) => e.code === 'backslash-eaten' && e.message.includes('braces')), 'brace:\n' + dump(r));
  assert.ok(c.includes('split-by-sup'), 'caret:\n' + dump(r));
  assert.ok(c.includes('split-by-em'), 'underscore:\n' + dump(r));
  assert.ok(c.includes('math-missing'));
  assert.ok(r.stats.displayOk < r.stats.displayTotal, 'display math must also be flagged');
});

test('NEGATIVE: failure mode 4, <div class="math"> inside a paragraph', () => {
  const src = 'Some text\n\n$$\na \\, b\n$$\n';
  assert.equal(processSource(src).output, 'Some text\n\n<div class="math">$$a \\, b$$</div>\n');
  const r = lint(src, 'Some text <div class="math">$$a \\, b$$</div>\n');
  const c = codes(r.errors);
  assert.ok(c.includes('div-math-inline'), dump(r));
  const eaten = r.errors.find((e) => e.code === 'backslash-eaten');
  assert.ok(eaten && eaten.message.includes('`\\,`') && eaten.message.includes('thin space'), dump(r));
  // and the builder's form is fine
  assert.deepEqual(lint(src, processSource(src).output).errors, []);
});

test('escaping property: inline LaTeX survives Ghost markdown-it byte-for-byte in every inline context', () => {
  const md = createGhostMarkdownIt();
  const samples = [...ADVERSARIAL, ...randomStrings(400, 1234)];
  const contexts = [
    (m) => `x ${m} y`,
    (m) => `| h |\n|---|\n| x ${m} y |`,
    (m) => `- x ${m} y`,
    (m) => `1. x ${m} y`,
    (m) => `## x ${m} y`,
    (m) => `> x ${m} y`,
    (m) => `*x ${m} y*`,
    (m) => `[x ${m} y](https://example.org)`,
    (m) => `ref[^n]\n\n[^n]: x ${m} y`,
  ];
  for (const latex of samples) {
    for (const ctx of contexts) {
      const html = md.render(ctx(inlineMathToGhost(latex)));
      const want = '\\(' + latex + '\\)';
      assert.ok(textNodes(html).some((t) => t.includes(want)),
        `not preserved: ${JSON.stringify(latex)} in ${JSON.stringify(ctx('M'))}\nmarkdown: ${ctx(inlineMathToGhost(latex))}\nhtml: ${html}`);
    }
  }
  // two spans in one paragraph cannot pair up across the prose between them
  for (let k = 0; k + 1 < samples.length; k++) {
    const [a, b] = [samples[k], samples[k + 1]];
    const html = md.render(`x ${inlineMathToGhost(a)} vs ${inlineMathToGhost(b)} y`);
    // both spans and the prose between them must sit in ONE text node
    assert.ok(textNodes(html).some((t) => t.includes('\\(' + a + '\\) vs \\(' + b + '\\)')),
      `pair not preserved: ${JSON.stringify([a, b])}\nhtml: ${html}`);
  }
});

test('display property: display LaTeX survives as a div.math block in every block context', () => {
  const md = createGhostMarkdownIt();
  const lines = randomStrings(300, 99, {maxLen: 30, exclude: (s) => s.includes('$') || s.trim() === '' || /^\s*>/.test(s)});
  const contexts = [
    (ls) => `Lead.\n\n$$\n${ls.join('\n')}\n$$\n\nTail.`,
    (ls) => `Lead.\n$$\n${ls.join('\n')}\n$$\nTail.`,
    (ls) => `- item\n\n  $$\n${ls.map((l) => '  ' + l).join('\n')}\n  $$\n- next`,
    (ls) => `- $$\n${ls.map((l) => '  ' + l).join('\n')}\n  $$\n- next`,
    (ls) => `See[^d].\n\n[^d]: Note.\n\n    $$\n${ls.map((l) => '    ' + l).join('\n')}\n    $$`,
    (ls) => `> quote\n> $$\n${ls.map((l) => '> ' + l).join('\n')}\n> $$\n> after`,
  ];
  for (let k = 0; k + 3 <= lines.length; k += 3) {
    const ls = lines.slice(k, k + 3);
    for (const ctx of contexts) {
      const src = ctx(ls);
      const res = processSource(src);
      assert.deepEqual(res.errors, [], src);
      const display = res.math.filter((m) => m.display);
      assert.equal(display.length, 1, src);
      const want = '$$' + display[0].latex + '$$';
      assert.equal(display[0].latex, ls.map((l) => l.trim()).join('\n'));
      const html = md.render(res.output);
      assert.ok(textNodes(html).some((t) => t === want),
        `display not preserved for ${JSON.stringify(ls)} in context\n${src}\n--- ghost:\n${res.output}\n--- html:\n${html}`);
    }
  }
});

test('tokenizer: code, fences, indented code, comments, image alt and link titles are not math', () => {
  const src = [
    'Inline `$x$` and ``$$y$$`` code, \\$5 escaped, then $z$.',
    '',
    '```',
    '$a$ and $$b$$',
    '```',
    '',
    '~~~~',
    '$$',
    '~~~~',
    '',
    '    $indented$ code',
    '',
    '<!-- $comment$ -->',
    '',
    '![alt $x$](figures/a.png "t $y$") [l](https://x.org "title $w$") $q$',
    '',
    '<pre>$pre$</pre>',
  ].join('\n');
  const res = processSource(src);
  assert.deepEqual(res.errors, []);
  assert.deepEqual(res.math.map((m) => m.latex), ['z', 'q']);
  assert.ok(res.output.includes('Inline `$x$` and ``$$y$$`` code, \\$5 escaped, then \\\\(z\\\\).'));
  assert.deepEqual(lint(src, res.output).errors, []);
});

test('tokenizer: dollar rules', () => {
  const m = (s) => extractMath(s).map((x) => x.latex);
  assert.deepEqual(m('a $x$ b'), ['x']);
  assert.deepEqual(m('costs $5 and $6'), []); // closing $ must not follow a space
  assert.deepEqual(m('$x$5'), []); // or precede a digit
  assert.deepEqual(m('$ x$'), []); // opening $ must not precede a space
  assert.deepEqual(m('$\\$5 + x$'), ['\\$5 + x']); // \$ inside math does not close it
  assert.deepEqual(m('\\\\$x$'), ['x']); // escaped backslash, then math
  assert.deepEqual(m('$a$$b$'), ['a', 'b']);
  const w = processSource('costs $5 today').warnings;
  assert.equal(w.length, 1);
  assert.match(w[0].message, /unpaired `\$`/);
});

test('tokenizer: errors for display math inside a line, math across a line break, unclosed display', () => {
  const err = (s) => processSource(s).errors.map((e) => `${e.line}: ${e.message}`);
  assert.match(err('text $$x$$ text')[0], /^1: `\$\$` display math must be on its own line/);
  assert.match(err('a $x +\ny$ b')[0], /^1: inline math `\$…\$` contains a line break/);
  assert.match(err('a\n\n$$\nx\n')[0], /^3: display math: no closing `\$\$`/);
  assert.match(err('$$\nx\n$$ and more')[0], /closing `\$\$` must end its line/);
  assert.match(err('$$$$')[0], /empty display math/);
});

test('joining: wrapped paragraphs, list items, lazy lines and footnotes become one line', () => {
  const src = [
    'One',
    'two',
    '   three.',
    '',
    '- item',
    '  continued',
    'lazily',
    '- next',
    '  1. nested',
    '     wrapped',
    '',
    '[^f]: note',
    '    wrapped',
    '[^g]: second',
    '',
    'see[^f][^g]',
    '',
    '| a | b |',
    '|---|---|',
    '| 1 | 2 |',
    '',
    '> q1',
    '> q2',
    '',
    '# Heading',
    'para after heading',
    'continues',
    '',
    'hard  ',
    'break',
    '',
    'The year',
    '2019. ok',
    '#5 not a heading',
  ].join('\n');
  const out = processSource(src).output.split('\n');
  for (const line of [
    'One two three.', '- item continued lazily', '- next', '  1. nested wrapped', '[^f]: note wrapped', '[^g]: second',
    '| a | b |', '|---|---|', '| 1 | 2 |', '> q1', '> q2', '# Heading', 'para after heading continues',
    'hard  ', 'break', 'The year 2019. ok #5 not a heading',
  ]) {
    assert.ok(out.includes(line), `expected line ${JSON.stringify(line)} in:\n${out.join('\n')}`);
  }
  const r = lint(src, processSource(src).output);
  assert.deepEqual(r.errors, []);
  // only the blockquote keeps a soft break
  assert.deepEqual(codes(r.warnings), ['br-in-paragraph']);
});

test('images: relative paths get --image-base, absolute URLs are untouched', () => {
  const src = '![a](figures/x.png) ![b](./figures/y.png "T") ![c](https://e.org/z.png) ![d](/abs.png)\n\n<img src="figures/w.png">\n';
  const out = processSource(src, {imageBase: 'https://cdn.example/blog'}).output;
  assert.equal(out,
    '![a](https://cdn.example/blog/figures/x.png) ![b](https://cdn.example/blog/figures/y.png "T") ![c](https://e.org/z.png) ![d](/abs.png)\n\n' +
    '<img src="https://cdn.example/blog/figures/w.png">\n');
});

test('footnotes: rendered order, unused and undefined footnotes', () => {
  const src = 'First[^b] then[^a] $x$.\n\n[^a]: A $\\alpha$.\n[^b]: B $\\beta$.\n[^c]: unused $\\gamma$.\n';
  const r = lint(src, processSource(src).output);
  assert.deepEqual(r.errors, [], dump(r));
  assert.deepEqual(r.rendered.map((m) => m.latex), ['x', '\\beta', '\\alpha']);
  assert.ok(r.warnings.some((w) => w.code === 'footnote-unused' && w.message.includes('[^c]') && w.message.includes('1 math span')));
  const bad = 'Missing[^zz] here.\n';
  const r2 = lint(bad, processSource(bad).output);
  assert.ok(codes(r2.errors).includes('footnote-undefined'));
  assert.ok(codes(r2.errors).includes('footnote-unresolved'));
});

test('lint: stale post.ghost.md, KaTeX errors and stray site delimiters are reported', () => {
  const src = 'a $x$ b\n';
  const r = lint(src, 'a \\\\(x\\\\) b changed\n');
  assert.deepEqual(r.errors, []);
  assert.ok(codes(r.warnings).includes('stale'));
  const bad = 'a $\\fraq{1}{2}$ b\n';
  const rb = lint(bad, processSource(bad).output);
  assert.deepEqual(codes(rb.errors), ['katex-error'], dump(rb));
  assert.match(rb.errors[0].message, /Undefined control sequence: \\fraq/);
  // unbalanced braces: auto-render cannot even find the closing delimiter
  const unbalanced = 'a $\\frac{1}{2$ b\n';
  const ru = lint(unbalanced, processSource(unbalanced).output);
  assert.ok(ru.errors.some((e) => e.code === 'math-missing' && /unbalanced/.test(e.message)), dump(ru));
  // `\\(` typed into the source renders as a site delimiter that is not source math
  const sneaky = 'literal \\\\(y\\\\) here\n';
  assert.ok(codes(lint(sneaky, processSource(sneaky).output).errors).includes('math-unexpected'));
});

test('auto-render port matches katex@0.16.9 contrib/auto-render/splitAtDelimiters.js', async (t) => {
  let real;
  try {
    real = (await import('katex/contrib/auto-render/splitAtDelimiters.js')).default;
  } catch (e) {
    t.skip(`cannot import the original module on this Node version (${e.message})`);
    return;
  }
  const pieces = ['$$', '\\(', '\\)', '\\[', '\\]', '{', '}', '\\', 'x', ' ', '$'];
  const rnd = mulberry32(7);
  for (let k = 0; k < 3000; k++) {
    let s = '';
    const len = Math.floor(rnd() * 14);
    for (let j = 0; j < len; j++) s += pieces[Math.floor(rnd() * pieces.length)];
    assert.deepEqual(splitAtDelimiters(s, SITE_DELIMITERS), real(s, SITE_DELIMITERS), JSON.stringify(s));
  }
  // and a whole page
  const html = createGhostMarkdownIt().render(processSource(SRC).output);
  const body = parse('<!DOCTYPE html><html><body>' + html + '</body></html>').childNodes[1].childNodes[1];
  assert.equal(collectRenderedMath(body).math.length, 29);
});

test('preview: SRI hashes match node_modules/katex', () => {
  for (const [file, sri] of Object.entries(KATEX_SRI)) {
    const data = fs.readFileSync(path.join(BLOG, 'node_modules', 'katex', 'dist', file));
    assert.equal('sha384-' + crypto.createHash('sha384').update(data).digest('base64'), sri, file);
  }
});

test('CLI: build + lint the fixture (exit 0), lint the naive file (exit 1)', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'blog-test-'));
  try {
    const ghost = path.join(tmp, 'post.ghost.md');
    const preview = path.join(tmp, 'preview.html');
    const node = process.execPath;
    const b = spawnSync(node, [path.join(BLOG, 'build_ghost.mjs'), '--in', FIXTURE, '--out', ghost], {encoding: 'utf8'});
    assert.equal(b.status, 0, b.stderr);
    assert.match(b.stdout, /26 inline, 3 display math spans/);
    const l = spawnSync(node, [path.join(BLOG, 'lint.mjs'), '--src', FIXTURE, '--ghost', ghost, '--preview', preview], {encoding: 'utf8'});
    assert.equal(l.status, 0, l.stdout + l.stderr);
    assert.match(l.stdout, /26 inline, 3 display math spans OK/);
    const page = fs.readFileSync(preview, 'utf8');
    assert.ok(page.includes('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css'));
    assert.ok(page.includes('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js'));
    assert.ok(page.includes('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js'));
    assert.ok(page.includes('renderMathInElement(document.body'));
    assert.ok(page.includes('"DOMContentLoaded"'));
    assert.ok(page.includes(JSON.stringify(SITE_DELIMITERS)));
    assert.ok(page.includes('max-width: 720px'));
    assert.ok(page.includes('\\(e^{-x^2}\\)'));

    fs.writeFileSync(ghost, naiveInline(SRC));
    const bad = spawnSync(node, [path.join(BLOG, 'lint.mjs'), '--src', FIXTURE, '--ghost', ghost, '--preview', preview], {encoding: 'utf8'});
    assert.equal(bad.status, 1);
    assert.match(bad.stdout, /lint: FAIL: /);
    assert.match(bad.stderr, /error \[split-by-sup\]/);

    const e = spawnSync(node, [path.join(BLOG, 'build_ghost.mjs'), '--in', path.join(tmp, 'nope.md')], {encoding: 'utf8'});
    assert.equal(e.status, 2);
  } finally {
    fs.rmSync(tmp, {recursive: true, force: true});
  }
});

test('escapeInlineLatex escapes exactly the 32 ASCII punctuation characters', () => {
  const escaped = [...ASCII].filter((ch) => escapeInlineLatex(ch) !== ch);
  assert.equal(escaped.join(''), '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~');
  assert.equal(escapeInlineLatex('aZ9 αé'), 'aZ9 αé');
});
