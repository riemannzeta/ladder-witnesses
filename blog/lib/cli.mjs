// Tiny shared CLI helpers for build_ghost.mjs and lint.mjs.

import fs from 'node:fs';
import {pathToFileURL} from 'node:url';

/** Parse `--name value` / `--name=value` / `--flag` against spec {name: 'value'|'flag'}. */
export function parseArgs(argv, spec) {
  const out = {};
  for (let k = 0; k < argv.length; k++) {
    const a = argv[k];
    if (a === '-h' || a === '--help') {
      out.help = true;
      continue;
    }
    const m = /^--([a-z-]+)(?:=(.*))?$/.exec(a);
    if (!m || !(m[1] in spec)) throw new Error(`unknown argument: ${a}`);
    if (spec[m[1]] === 'flag') {
      out[m[1]] = true;
      continue;
    }
    const v = m[2] !== undefined ? m[2] : argv[++k];
    if (v === undefined) throw new Error(`--${m[1]} needs a value`);
    out[m[1]] = v;
  }
  return out;
}

/** True when the module at metaUrl is the script node was started with. */
export function isMain(metaUrl) {
  if (!process.argv[1]) return false;
  try {
    return metaUrl === pathToFileURL(fs.realpathSync(process.argv[1])).href;
  } catch {
    return false;
  }
}
