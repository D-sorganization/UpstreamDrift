import { describe, it, expect } from 'vitest';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';

/**
 * 12px text floor (#14281). Arbitrary font sizes below `text-xs` (12px) are
 * unreadable on phones. `Scene3D.tsx` is excluded while UpstreamDrift#14250
 * holds it; drop the exemption when that lands.
 */
const SRC = resolve(__dirname);
const EXEMPT = new Set(['components/visualization/Scene3D.tsx']);
const BELOW_FLOOR = /\btext-\[(?:[0-9]|1[01])px\]/g;

function sourceFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) return sourceFiles(path);
    return /\.tsx?$/.test(name) && !/\.test\.tsx?$/.test(name) ? [path] : [];
  });
}

describe('12px text floor (#14281)', () => {
  it('no source file uses an arbitrary font size below 12px', () => {
    const offenders = sourceFiles(SRC)
      .map((path) => relative(SRC, path))
      .filter((path) => !EXEMPT.has(path))
      .flatMap((path) =>
        [...readFileSync(join(SRC, path), 'utf-8').matchAll(BELOW_FLOOR)].map(
          (match) => `${path}: ${match[0]}`
        )
      );
    expect(offenders).toEqual([]);
  });
});
