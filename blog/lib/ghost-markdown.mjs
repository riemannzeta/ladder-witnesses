// Ghost's Markdown-card renderer, reproduced exactly.
//
// TryGhost/Koenig `kg-markdown-html-renderer` (Ghost 6.68) does:
//
//   new MarkdownIt({html: true, breaks: true, linkify: true})
//       .use(markdownItFootnote).use(markdownItLazyHeaders).use(markdownItMark)
//       .use(markdownItImageLazyLoading).use(namedHeaders)
//       .use(markdownItSub).use(markdownItSup);
//   markdownIt.linkify.set({fuzzyLink: false});
//
// image-lazy-loading (adds loading="lazy") and namedHeaders (adds heading ids)
// only touch attributes, never text, so they cannot affect math and are omitted.
// Versions are pinned in package.json to the ones Ghost ships.

import MarkdownIt from 'markdown-it';
import markdownItFootnote from 'markdown-it-footnote';
import markdownItLazyHeaders from 'markdown-it-lazy-headers';
import markdownItMark from 'markdown-it-mark';
import markdownItSub from 'markdown-it-sub';
import markdownItSup from 'markdown-it-sup';

export function createGhostMarkdownIt() {
  const md = new MarkdownIt({html: true, breaks: true, linkify: true})
    .use(markdownItFootnote)
    .use(markdownItLazyHeaders)
    .use(markdownItMark)
    .use(markdownItSub)
    .use(markdownItSup);
  md.linkify.set({fuzzyLink: false});
  return md;
}

/**
 * Render like md.render(), but also return the token stream, the footnote env,
 * and a map from HTML output offsets to (0-based) Markdown source lines, so that
 * problems found in the HTML can be reported with a line number.
 *
 * The loop below is markdown-it 14's Renderer.prototype.render verbatim, with
 * one bookkeeping line added; the result is checked against md.render().
 */
export function renderWithLineMap(md, src) {
  const env = {};
  const tokens = md.parse(src, env);
  const renderer = md.renderer;
  const offsets = []; // [{offset, line}] sorted by offset
  let result = '';
  for (let i = 0, len = tokens.length; i < len; i++) {
    const tok = tokens[i];
    if (tok.map && tok.nesting >= 0) offsets.push({offset: result.length, line: tok.map[0]});
    const type = tok.type;
    if (type === 'inline') {
      result += renderer.renderInline(tok.children, md.options, env);
    } else if (typeof renderer.rules[type] !== 'undefined') {
      result += renderer.rules[type](tokens, i, md.options, env, renderer);
    } else {
      result += renderer.renderToken(tokens, i, md.options, env);
    }
  }
  const reference = md.render(src, {});
  if (reference !== result) {
    throw new Error('internal: line-mapped render differs from md.render()');
  }
  const lineAt = (offset) => {
    let lo = 0;
    let hi = offsets.length - 1;
    let best = null;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (offsets[mid].offset <= offset) {
        best = offsets[mid].line;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    return best; // 0-based line or null
  };
  return {html: result, tokens, env, lineAt};
}
