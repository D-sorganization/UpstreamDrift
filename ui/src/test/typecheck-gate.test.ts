/**
 * Regression guard for the `type-check` npm script (issue #7971).
 *
 * `ui/tsconfig.json` is a *solution-style* config: it has `"files": []` and
 * delegates to project `references`. Plain `tsc --noEmit` ignores
 * `references`, so it compiled **zero** files and the CI type-safety gate
 * could never fail. The script must therefore run in build mode (`tsc -b`).
 */

import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, it, expect } from 'vitest';

const uiRoot = resolve(__dirname, '../..');

function readJson(relativePath: string): Record<string, unknown> {
  const raw = readFileSync(resolve(uiRoot, relativePath), 'utf-8');
  // tsconfig files may contain comments; strip line comments before parsing.
  const stripped = raw.replace(/^\s*\/\/.*$/gm, '').replace(/\/\*[\s\S]*?\*\//g, '');
  return JSON.parse(stripped) as Record<string, unknown>;
}

describe('type-check script (issue #7971)', () => {
  const pkg = readJson('package.json');
  const scripts = pkg.scripts as Record<string, string>;
  const rootTsconfig = readJson('tsconfig.json');

  it('root tsconfig is solution-style (files: [] + references)', () => {
    expect(rootTsconfig.files).toEqual([]);
    expect(Array.isArray(rootTsconfig.references)).toBe(true);
    expect((rootTsconfig.references as unknown[]).length).toBeGreaterThan(0);
  });

  it('type-check runs tsc in build mode so referenced projects are checked', () => {
    const script = scripts['type-check'];
    expect(script).toBeDefined();
    // `tsc -b` / `tsc --build` is the only mode that follows `references`.
    expect(script).toMatch(/\btsc\b.*(\s-b\b|\s--build\b)/);
  });

  it('type-check does not emit and is not incremental-cached', () => {
    const script = scripts['type-check'];
    expect(script).toContain('--noEmit');
    // Without --force a stale .tsbuildinfo can short-circuit the gate.
    expect(script).toContain('--force');
  });

  it('every referenced project actually includes source files', () => {
    const references = rootTsconfig.references as Array<{ path: string }>;
    for (const ref of references) {
      const child = readJson(ref.path.replace(/^\.\//, ''));
      const include = child.include as unknown[] | undefined;
      expect(include, `${ref.path} must declare an include list`).toBeDefined();
      expect((include ?? []).length).toBeGreaterThan(0);
    }
  });
});
