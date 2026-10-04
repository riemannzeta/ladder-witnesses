#!/usr/bin/env node
// Convert post.src.md ($…$ / $$…$$ Markdown that renders on GitHub) into
// post.ghost.md, for pasting into a Ghost *Markdown card* on symmetrybroken.com.
//
//   node build_ghost.mjs [--in post.src.md] [--out post.ghost.md]
//                        [--image-base https://raw.githubusercontent.com/riemannzeta/ladder-witnesses/main/blog/]
//
// Default paths are relative to this script's directory (blog/); paths given on
// the command line are relative to the current directory. On any conversion
// error nothing is written and the exit code is 1.

import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {processSource, DEFAULT_IMAGE_BASE} from './lib/source.mjs';
import {parseArgs, isMain} from './lib/cli.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));

function main() {
  let args;
  try {
    args = parseArgs(process.argv.slice(2), {in: 'value', out: 'value', 'image-base': 'value'});
  } catch (e) {
    console.error(`build_ghost: ${e.message}`);
    process.exit(2);
  }
  if (args.help) {
    console.log('usage: node build_ghost.mjs [--in post.src.md] [--out post.ghost.md] [--image-base URL]');
    return;
  }
  const inPath = args.in ? path.resolve(args.in) : path.join(HERE, 'post.src.md');
  const outPath = args.out ? path.resolve(args.out) : path.join(HERE, 'post.ghost.md');
  const imageBase = args['image-base'] ?? DEFAULT_IMAGE_BASE;
  const rel = (p) => path.relative(process.cwd(), p) || p;
  if (!fs.existsSync(inPath)) {
    console.error(`build_ghost: ${rel(inPath)} not found (write the post there first; see README.md)`);
    process.exit(2);
  }

  const res = processSource(fs.readFileSync(inPath, 'utf8'), {imageBase});
  for (const w of res.warnings) console.warn(`${rel(inPath)}:${w.line}: warning: ${w.message}`);
  for (const e of res.errors) console.error(`${rel(inPath)}:${e.line}: error: ${e.message}`);
  if (res.errors.length) {
    console.error(`build_ghost: ${res.errors.length} error(s); ${rel(outPath)} not written`);
    process.exit(1);
  }
  fs.writeFileSync(outPath, res.output);
  const inline = res.math.filter((m) => !m.display).length;
  const display = res.math.length - inline;
  console.log(`build_ghost: wrote ${rel(outPath)} (${inline} inline, ${display} display math spans; ${res.warnings.length} warning(s))`);
}

if (isMain(import.meta.url)) main();
