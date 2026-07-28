import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

describe('useAnalysisTools lifecycle guard', () => {
  it('clears the mounted ref on unmount before async completions can set state', () => {
    const source = readFileSync(join(__dirname, 'useAnalysisTools.ts'), 'utf-8');

    expect(source).toMatch(/import\s+\{[^}]*useEffect[^}]*\}\s+from 'react';/);
    expect(source).toMatch(
      /useEffect\(\(\) => \{\s*isMountedRef\.current = true;\s*return \(\) => \{\s*isMountedRef\.current = false;\s*\};\s*\}, \[\]\);/,
    );
  });
});
