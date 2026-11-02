import { readdirSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const sourceRoot = resolve('src');

function sourceFiles(directory: string): string[] {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) return sourceFiles(path);
    if (!/\.(?:[cm]?[jt]sx?)$/.test(entry.name)) return [];
    if (/\.(?:test|spec)\.[^.]+$/.test(entry.name)) return [];
    return [path];
  });
}

describe('Scene environment source contract', () => {
  it('does not select a CDN preset on any Environment component', () => {
    const presetUsage = sourceFiles(sourceRoot).filter((path) =>
      /<Environment\b[^>]*\bpreset\s*=/s.test(readFileSync(path, 'utf8')),
    );

    expect(presetUsage).toEqual([]);
  });
});
