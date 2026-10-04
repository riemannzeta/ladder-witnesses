# Blog pipeline: `post.src.md` → Ghost Markdown card

You write the post as ordinary Markdown with `$…$` and `$$…$$` math, which GitHub renders as-is.
`build_ghost.mjs` converts it into `post.ghost.md`, which you paste into a Ghost **Markdown card** on
symmetrybroken.com. `lint.mjs` then renders that file exactly as Ghost and KaTeX will. It fails unless every
math span reaches KaTeX byte-for-byte and parses.

## Workflow

```sh
cd blog
npm install          # once (versions are pinned to what Ghost 6.68 / the site use)
npm run build        # post.src.md  -> post.ghost.md
npm run lint         # must end with "N inline, M display math spans OK"; writes preview.html
npm test             # the toolchain's own tests (fixture + negative tests)
```

1. Write `post.src.md`. `test/fixtures/sample.src.md` shows every supported construct.
2. Run `npm run build`. If there are conversion errors, nothing is written and each error is listed with its line number.
3. Run `npm run lint` and fix every **error**. Warnings are worth reading too. Each problem gives `post.ghost.md:LINE ← post.src.md:LINE`.
4. Open `preview.html` in a browser (it needs internet access for cdn.jsdelivr.net).
   - It loads KaTeX 0.16.9 and auto-render with the site's delimiters.
   - The bar at the top shows how many spans the browser typeset. It also says whether that list is identical to what the lint predicted.
   - Figures are shown from your local `figures/`.
5. In Ghost, add a **Markdown card** (`+` → Markdown, or type `/markdown`), and paste all of `post.ghost.md` into it.
   - Don't paste it into the rich-text editor. That editor does not run this Markdown renderer.
6. Figures: `![alt](figures/x.png)` is rewritten to `https://raw.githubusercontent.com/riemannzeta/ladder-witnesses/main/blog/figures/x.png`, which only resolves after the repo is pushed.
   - Either push first, or upload the figures to Ghost image cards and delete those lines.
   - Use `--image-base URL` on both `build` and `lint` to point somewhere else.

CLI options:
`node build_ghost.mjs [--in post.src.md] [--out post.ghost.md] [--image-base URL]` and
`node lint.mjs [--src post.src.md] [--ghost post.ghost.md] [--preview preview.html] [--image-base URL]`.
Defaults resolve inside `blog/`, so `node blog/lint.mjs` also works from the repo root.

## Writing `post.src.md`

- **Inline math** `$…$`.
  - The opening `$` must not be followed by a space.
  - The closing `$` must not be preceded by a space or followed by a digit (pandoc's rule).
  - It must stay on one line. Math that is split by a hard wrap is an error.
- **Display math** `$$…$$`. The opening `$$` starts a line and the closing `$$` ends one, possibly lines later.
  - Blank lines inside are dropped.
  - Display math can sit in a list item or footnote: indent it to the item's text.
  - It can also go in a blockquote (`> $$`), or directly after a list marker (`- $$x$$`).
  - Display math in the middle of a line is an error. On the site a stray `$$` is a delimiter.
- **A literal dollar** is `\$`. An unpaired `$` is left alone, with a warning.
- **Code** is never math: `` `…` `` spans, fenced blocks and indented blocks.
- **Never math, also:** HTML comments, image alt text, link URLs/titles, and `<code>`/`<pre>` elements.
- **Hard-wrap freely.** Paragraphs, list items (indented or lazy continuation lines) and footnote definitions are joined into one line.
  - End a line with two spaces or `\` when you want a real line break.
  - Blockquote, table, heading and HTML lines are kept as written.
- **Tables:** inside a math cell, prefer `\mid` / `\vert` to a bare `|`. GitHub splits cells on `|` before it renders math. The Ghost output is correct either way.
- **Footnotes:** `[^id]` references and `[^id]: …` definitions pass through. Inline `^[…]` footnotes are not supported for math.
- **`#` at the start of a paragraph** is escaped. Ghost loads markdown-it-lazy-headers, which turns `#tag` into a heading; GitHub does not.

## What the build does, and why

Ghost renders the card with `markdown-it` 14.3.0 (`html`, `breaks`, `linkify`) plus the footnote, lazy-headers, mark, sub and sup plugins.
KaTeX 0.16.9 auto-render then scans the page's text nodes for `$$…$$`, `\[…\]` and `\(…\)`. A single `$` is not a delimiter.
Each of these four failures was reproduced with the exact config:

1. **Backslashes vanish.** `\\(\{V_i^A\}\\)` renders as `\({V_i^A}\)`.
   - Cause: `\` followed by ASCII punctuation is a Markdown escape, so `\{`, `\}`, `\,`, `\|`, `\_` and `\\` lose their backslash.
   - Build: inline math becomes `\\(` + the LaTeX with **every** ASCII punctuation character backslash-escaped + `\\)`. For example, `$\{V_i^A\}$` → `\\(\\\{V\_i\^A\\\}\\)`.
2. **Carets become superscript.** `\\(e^{-x^2}\\)` renders as `\(e<sup>{-x</sup>2}\)`.
   - Cause: markdown-it-sup pairs carets that have no whitespace between them. sub does the same with `~`, and mark with `==`.
   - Build: the same escaping rule, so `^` → `\^`, `~` → `\~`, `=` → `\=`.
3. **Underscores pair up across spans.** `\\(\hat{\theta}_{zx}\\) vs \\(n_{z}\\)` gets an `<em>` from one span to the next.
   - Cause: emphasis pairs `_` and `*` across separate math spans.
   - Build: the same escaping rule, so `_` → `\_` and `*` → `\*`.
4. **A div inside a paragraph is Markdown.** In `text <div class="math">$$a \, b$$</div>`, the `\,` is lost.
   - Cause: a `<div>` is raw HTML only when it starts its own HTML block. Inside a paragraph, its content is Markdown.
   - Build: display math becomes its own block:
     - a blank line;
     - `<div class="math">$$…$$</div>`, with `&`, `<` and `>` as entities and no blank lines inside;
     - another blank line.

Once every ASCII punctuation character is escaped, markdown-it can only emit the LaTeX as literal text.
That rules out links, HTML, entities, code spans and table-cell splits too. The HTML text is then exactly `\(` + LaTeX + `\)`.
Because `breaks: true` turns every newline into `<br>`, hard-wrapped paragraphs are joined.
Relative image paths get `--image-base`.

## What the lint checks

- It renders `post.ghost.md` with Ghost's markdown-it config, parses the HTML (parse5), and walks the text nodes like auto-render: same delimiters, same ignored tags, same left-to-right matching.
  - It flags any opening delimiter left without a closer, and any stray closer.
  - When markdown split a span, it names the tag that did it (`<sup>` = carets, `<em>` = underscores, `<br>` = a hard wrap, …).
- The typeset list must equal the math in `post.src.md`, found by the build's own tokenizer: same count, order, display flag and bytes.
  - Footnote math is compared in rendered order, at the end.
  - Mismatches are diffed and explained. For example: "Markdown ate the backslash in `\{`, `\}` (braces)".
- Every span goes through `katex.renderToString(…, {displayMode, throwOnError: true})`, sharing `\gdef` macros like the site. KaTeX strict-mode notes become warnings.
- **Footnotes:** a reference without a definition is an error. A definition that is never referenced is a warning, because Ghost drops it.
- **Line breaks:** a newline inside a paragraph (`<br>` from `breaks: true`) is a warning.
- **`<div class="math">`:** inside a paragraph it is an error. It must be its own HTML block.
- **Freshness:** a `post.ghost.md` that is not the current build of `post.src.md` is a warning.
- **Links and figures:** relative links/images are warnings, and so are figures whose local file is missing.

Exit status: 0 if there are no errors, 1 on errors, 2 on usage errors.

## Files

- `build_ghost.mjs`: the build CLI.
- `lint.mjs`: the lint CLI, plus `preview.html`.
- `lib/source.mjs`: the shared tokenizer and converter.
- `lib/ghost-markdown.mjs`: Ghost's markdown-it config.
- `lib/autorender.mjs`: KaTeX auto-render's delimiter scanner, vendored from katex@0.16.9 and checked against it in the tests.
- `lib/cli.mjs`: shared CLI helpers.
- `test/run.mjs`, `test/fixtures/`: the tests and the fixture.
- `post.src.md` is yours, and `post.ghost.md` is generated. `preview.html` and `node_modules/` are git-ignored.
