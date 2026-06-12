import { describe, it, expect } from 'vitest';
import { titleForPath } from './routeTitles';

describe('titleForPath (#7432)', () => {
  it('maps the root path to Dashboard', () => {
    expect(titleForPath('/')).toBe('Dashboard');
  });

  it('matches nested tool routes by prefix', () => {
    expect(titleForPath('/tools/model-explorer')).toBe('Model Explorer');
    expect(titleForPath('/tools/canonical-core/estimation')).toBe(
      'Canonical Core — Estimation'
    );
  });

  it('does not match the root prefix for non-root paths', () => {
    expect(titleForPath('/simulation')).toBe('Simulation');
  });

  it('returns null for unknown paths so the 404 page sets its own title', () => {
    expect(titleForPath('/nope/whatever')).toBeNull();
  });
});
