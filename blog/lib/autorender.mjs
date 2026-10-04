// What KaTeX 0.16.9 auto-render does on the site, reproduced on a parse5 tree.
//
// findEndOfMath / splitAtDelimiters are copied verbatim (logic and order) from
// katex@0.16.9 contrib/auto-render/splitAtDelimiters.js (MIT licence);
// test/run.mjs cross-checks this copy against the installed package.
// walkLikeAutoRender mirrors renderElem() in contrib/auto-render/auto-render.js:
// it visits text nodes in document order, concatenates adjacent text siblings,
// and skips the default ignoredTags.

/** The site's renderMathInElement delimiters, in the site's order. */
export const SITE_DELIMITERS = [
  {left: '$$', right: '$$', display: true},
  {left: '\\[', right: '\\]', display: true},
  {left: '\\(', right: '\\)', display: false},
];

/** auto-render's default ignoredTags. */
export const IGNORED_TAGS = ['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'option'];

export const findEndOfMath = function (delimiter, text, startIndex) {
  let index = startIndex;
  let braceLevel = 0;

  const delimLength = delimiter.length;

  while (index < text.length) {
    const character = text[index];

    if (braceLevel <= 0 &&
        text.slice(index, index + delimLength) === delimiter) {
      return index;
    } else if (character === '\\') {
      index++;
    } else if (character === '{') {
      braceLevel++;
    } else if (character === '}') {
      braceLevel--;
    }

    index++;
  }

  return -1;
};

const escapeRegex = function (string) {
  return string.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&');
};

const amsRegex = /^\\begin{/;

export const splitAtDelimiters = function (text, delimiters) {
  let index;
  const data = [];

  const regexLeft = new RegExp(
    '(' + delimiters.map((x) => escapeRegex(x.left)).join('|') + ')'
  );

  while (true) {
    index = text.search(regexLeft);
    if (index === -1) {
      break;
    }
    if (index > 0) {
      data.push({
        type: 'text',
        data: text.slice(0, index),
      });
      text = text.slice(index); // now text starts with delimiter
    }
    // ... so this always succeeds:
    const i = delimiters.findIndex((delim) => text.startsWith(delim.left));
    index = findEndOfMath(delimiters[i].right, text, delimiters[i].left.length);
    if (index === -1) {
      break;
    }
    const rawData = text.slice(0, index + delimiters[i].right.length);
    const math = amsRegex.test(rawData)
      ? rawData
      : text.slice(delimiters[i].left.length, index);
    data.push({
      type: 'math',
      data: math,
      rawData,
      display: delimiters[i].display,
    });
    text = text.slice(index + delimiters[i].right.length);
  }

  if (text !== '') {
    data.push({
      type: 'text',
      data: text,
    });
  }

  return data;
};

// ---------------------------------------------------------------------------
// Tree walking (parse5 default tree adapter: elements have tagName/childNodes,
// text nodes have nodeName '#text' and an entity-decoded .value).

const isText = (n) => n.nodeName === '#text';
const isElement = (n) => typeof n.tagName === 'string';

/**
 * Visit every text run auto-render would process, in document order.
 * cb(text, nodes) gets the concatenated text of adjacent sibling text nodes.
 */
export function walkLikeAutoRender(elem, cb, ignoredTags = IGNORED_TAGS) {
  const kids = elem.childNodes || [];
  for (let i = 0; i < kids.length; i++) {
    const child = kids[i];
    if (isText(child)) {
      const nodes = [child];
      let text = child.value;
      let j = i + 1;
      while (j < kids.length && isText(kids[j])) {
        text += kids[j].value;
        nodes.push(kids[j]);
        j++;
      }
      cb(text, nodes);
      i = j - 1;
    } else if (isElement(child)) {
      if (!ignoredTags.includes(child.tagName.toLowerCase())) {
        walkLikeAutoRender(child, cb, ignoredTags);
      }
    }
  }
}

/**
 * Collect the math spans auto-render would typeset, plus delimiter problems
 * (an opening delimiter with no closer in the same text node — auto-render then
 * leaves the rest of that text node untouched — or a stray closing delimiter).
 */
export function collectRenderedMath(root, delimiters = SITE_DELIMITERS) {
  const math = [];
  const problems = [];
  const regexLeft = new RegExp('(' + delimiters.map((x) => escapeRegex(x.left)).join('|') + ')');
  const closers = delimiters.map((d) => d.right).filter((r) => !delimiters.some((d) => d.left === r));
  walkLikeAutoRender(root, (text, nodes) => {
    const data = splitAtDelimiters(text, delimiters);
    data.forEach((piece, k) => {
      if (piece.type === 'math') {
        math.push({latex: piece.data, display: piece.display, rawData: piece.rawData, node: nodes[0]});
        return;
      }
      const isLast = k === data.length - 1;
      if (isLast && regexLeft.test(piece.data)) {
        // splitAtDelimiters only leaves a left delimiter in the final text piece
        // when findEndOfMath failed: an unmatched opening delimiter.
        const delim = delimiters.find((d) => piece.data.startsWith(d.left));
        problems.push({kind: 'unmatched-open', delimiter: delim ? delim.left : '?', text: piece.data, fullText: text, node: nodes[0]});
        return;
      }
      for (const c of closers) {
        if (piece.data.includes(c)) {
          problems.push({kind: 'stray-close', delimiter: c, text: piece.data, fullText: text, node: nodes[0]});
          break;
        }
      }
    });
  });
  return {math, problems};
}

// ---------------------------------------------------------------------------
// Diagnosis: find math spans that markdown broke apart by inserting tags.
// We flatten each block's inline content (text across <em>, <sup>, <br>, …)
// and look for delimiter pairs whose span crosses element boundaries.

const INLINE_TAGS = new Set([
  'a', 'abbr', 'b', 'bdi', 'bdo', 'br', 'cite', 'code', 'data', 'del', 'dfn', 'em', 'i', 'img',
  'ins', 'kbd', 'mark', 'q', 's', 'samp', 'small', 'span', 'strong', 'sub', 'sup', 'time', 'u',
  'var', 'wbr', 'label',
]);
const OPAQUE_INLINE = new Set(['br', 'img', 'wbr', 'code', 'kbd', 'samp']);

/**
 * Returns [{flatText, display, elements: Set<tagName>, nodes, node}] for every
 * delimiter pair that is only complete when inline markup is ignored.
 */
export function findSplitSpans(root, delimiters = SITE_DELIMITERS, ignoredTags = IGNORED_TAGS) {
  const results = [];

  function analyzeRun(chars) {
    if (!chars.length) return;
    const flat = chars.map((c) => c.ch).join('');
    let offset = 0;
    let rest = flat;
    // Same scanning as splitAtDelimiters, but on the flattened text.
    const regexLeft = new RegExp('(' + delimiters.map((x) => escapeRegex(x.left)).join('|') + ')');
    while (true) {
      const idx = rest.search(regexLeft);
      if (idx === -1) break;
      const start = offset + idx;
      rest = rest.slice(idx);
      const d = delimiters.find((x) => rest.startsWith(x.left));
      const end = findEndOfMath(d.right, rest, d.left.length);
      if (end === -1) {
        // skip this opener and keep looking
        rest = rest.slice(d.left.length);
        offset = start + d.left.length;
        continue;
      }
      const stop = start + end + d.right.length;
      const spanChars = chars.slice(start, stop);
      const origins = new Set(spanChars.map((c) => c.origin));
      if (origins.size > 1) {
        // Elements that wrap only part of the span (or are opaque tokens inside it).
        const counts = new Map();
        for (const c of spanChars) {
          for (const el of c.path) counts.set(el, (counts.get(el) || 0) + 1);
        }
        const elements = new Set();
        for (const [el, n] of counts) {
          if (n < spanChars.length || OPAQUE_INLINE.has(el.tagName)) elements.add(el.tagName);
        }
        for (const c of spanChars) if (c.opaque) elements.add(c.opaque);
        const shown = spanChars.map((c) => (c.opaque ? (c.opaque === 'br' ? '\n' : `<${c.opaque}>`) : c.ch)).join('');
        results.push({flatText: shown, display: d.display, elements, nodes: origins, node: spanChars[0].node});
      }
      rest = rest.slice(end + d.right.length);
      offset = stop;
    }
  }

  function visitBlock(block) {
    let run = [];
    const pushInline = (node, path) => {
      if (isText(node)) {
        for (const ch of node.value) run.push({ch, origin: node, path, node});
        return;
      }
      if (!isElement(node)) return;
      const tag = node.tagName.toLowerCase();
      if (!INLINE_TAGS.has(tag)) {
        analyzeRun(run);
        run = [];
        visitBlock(node);
        return;
      }
      if (OPAQUE_INLINE.has(tag) || ignoredTags.includes(tag)) {
        const ch = tag === 'br' ? '\n' : '￼';
        run.push({ch, origin: node, path: [...path, node], opaque: tag, node});
        return;
      }
      for (const kid of node.childNodes || []) pushInline(kid, [...path, node]);
    };
    if (ignoredTags.includes((block.tagName || '').toLowerCase())) return;
    for (const kid of block.childNodes || []) pushInline(kid, []);
    analyzeRun(run);
  }

  visitBlock(root);
  return results;
}
