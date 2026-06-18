import { describe, it, expect } from 'vitest';
import { readFileSync, readdirSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';
import allowlistJson from './colorGuard.allowlist.json';

/**
 * Color-system CI guard (UI/UX #7421). Fails when a `ui/src` file introduces a
 * non-canonical color family. Canonical neutrals are `gray-*`; the canonical
 * primary accent is `blue-*`. New `slate-`, `zinc-`, `neutral-`, or `purple-`
 * utility classes are rejected unless the file is in the reviewed allowlist.
 */
const SRC_ROOT = resolve(__dirname, '..');
const BANNED = /\b(?:slate|zinc|neutral|purple)-/;
const ALLOW = new Set(Object.keys((allowlistJson as { allow: Record<string, string> }).allow));
const COLOR_GUARD_TIMEOUT_MS = 30000;

function walk(dir: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      out.push(...walk(full));
    } else if (/\.(tsx?|css)$/.test(entry.name)) {
      out.push(full);
    }
  }
  return out;
}

const sourceFiles = walk(SRC_ROOT);
const sourceTypeScriptFiles = sourceFiles.filter((file) => /\.tsx?$/.test(file));

describe('color-system guard (#7421)', () => {
  it('has no non-canonical color families outside the allowlist', () => {
    const offenders: string[] = [];
    for (const file of sourceFiles) {
      const rel = relative(SRC_ROOT, file).replace(/\\/g, '/');
      if (ALLOW.has(rel) || rel.startsWith('utils/colorGuard')) continue;
      const source = readFileSync(file, 'utf8');
      if (BANNED.test(source)) {
        offenders.push(rel);
      }
    }
    expect(offenders, `Use gray-* (neutrals) / blue-* (primary). Offending files: ${offenders.join(', ')}`).toEqual([]);
  }, COLOR_GUARD_TIMEOUT_MS);
});

/**
 * Contrast guard (UI/UX #7439): `text-gray-500` (#6B7280) is ~2.8:1 on the
 * app's gray-900/950 surfaces — below the WCAG AA 4.5:1 minimum for body text.
 * Foreground text must use `text-gray-400` or lighter. This locks the sweep so
 * a regression can't silently reintroduce the failing class.
 */
describe('contrast guard (#7439)', () => {
  it('uses no text-gray-500 foreground class in tsx/ts sources', () => {
    const offenders: string[] = [];
    for (const file of sourceTypeScriptFiles) {
      const rel = relative(SRC_ROOT, file).replace(/\\/g, '/');
      if (rel.startsWith('utils/colorGuard')) continue;
      if (/text-gray-500/.test(readFileSync(file, 'utf8'))) {
        offenders.push(rel);
      }
    }
    expect(
      offenders,
      `text-gray-500 fails AA on dark surfaces — use text-gray-400+. Offending: ${offenders.join(', ')}`,
    ).toEqual([]);
  }, COLOR_GUARD_TIMEOUT_MS);
});
