import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

/**
 * Phone-width layout guard (#14282). At 390px a multi-column grid with no
 * breakpoint squeezes label/value pairs past the viewport, and the page
 * shells use `overflow-hidden`, so the overflow is clipped instead of
 * scrolling. A multi-column grid on these pages must also carry a
 * breakpoint column variant, so its phone layout is a deliberate choice.
 */
const PAGES = ['AnalysisTools.tsx', 'DataExplorer.tsx', 'Terrain.tsx'];
const CLASS_NAME = /className="([^"]*)"/g;
const MULTI_COLUMN_BASE = /(?<![:\w-])grid-cols-[2-9]\b/;
const BREAKPOINT_COLUMNS = /\b(?:sm|md|lg|xl|2xl):grid-cols-/;

function bareMultiColumnGrids(source: string): string[] {
  return [...source.matchAll(CLASS_NAME)]
    .map((match) => match[1])
    .filter((classes) => MULTI_COLUMN_BASE.test(classes) && !BREAKPOINT_COLUMNS.test(classes));
}

describe('phone-width grids (#14282)', () => {
  it.each(PAGES)('%s has no multi-column grid without a breakpoint', (page) => {
    const source = readFileSync(resolve(__dirname, page), 'utf-8');
    expect(bareMultiColumnGrids(source)).toEqual([]);
  });
});
