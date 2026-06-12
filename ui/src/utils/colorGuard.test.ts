import { describe, it, expect } from 'vitest';
import { readFileSync, readdirSync, statSync } from 'node:fs';
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

function walk(dir: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) {
      out.push(...walk(full));
    } else if (/\.(tsx?|css)$/.test(entry)) {
      out.push(full);
    }
  }
  return out;
}

describe('color-system guard (#7421)', () => {
  it('has no non-canonical color families outside the allowlist', () => {
    const offenders: string[] = [];
    for (const file of walk(SRC_ROOT)) {
      const rel = relative(SRC_ROOT, file).replace(/\\/g, '/');
      if (ALLOW.has(rel) || rel.startsWith('utils/colorGuard')) continue;
      const source = readFileSync(file, 'utf8');
      if (BANNED.test(source)) {
        offenders.push(rel);
      }
    }
    expect(offenders, `Use gray-* (neutrals) / blue-* (primary). Offending files: ${offenders.join(', ')}`).toEqual([]);
  });
});
