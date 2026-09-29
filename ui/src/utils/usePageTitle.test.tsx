import { describe, it, expect, afterEach } from 'vitest';
import { render } from '@testing-library/react';
import { usePageTitle } from './usePageTitle';

function Probe({ title }: { title: string }) {
  usePageTitle(title);
  return null;
}

describe('usePageTitle (#7432)', () => {
  afterEach(() => {
    document.title = '';
  });

  it('suffixes the app name', () => {
    render(<Probe title="Simulation" />);
    expect(document.title).toBe('Simulation — Golf Modeling Suite');
  });

  it('falls back to the bare app name for empty titles', () => {
    render(<Probe title="   " />);
    expect(document.title).toBe('Golf Modeling Suite');
  });
});
