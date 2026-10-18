import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import { VariabilityView } from './VariabilityView';
import type { NSXPlotModel } from '@/api/loopNullSpace';

// The shared fixtures, read in place (never copied): NSX-99 golden and the
// strings both UIs must show (tests/unit/loop_nullspace/variability_view_text.py).
const GOLDENS = resolve(__dirname, '../../../../../tests/fixtures/loop_nullspace/goldens');
const golden = (
  JSON.parse(readFileSync(resolve(GOLDENS, 'report_v2_variability.json'), 'utf8')) as {
    plot_model: NSXPlotModel;
  }
).plot_model;
const text = JSON.parse(readFileSync(resolve(GOLDENS, 'variability_view_text.json'), 'utf8')) as {
  complete: { expected: string[] };
  unavailable: { plot_model: NSXPlotModel; expected: string[] };
};

const norm = (s: string | null) => (s ?? '').replace(/\s+/g, ' ');

afterEach(cleanup);

describe('VariabilityView on the shared v2 golden', () => {
  it('shows every string of the shared text list', () => {
    const { container } = render(<VariabilityView sensitivity={golden.sensitivity!} />);
    const shown = norm(container.textContent);
    const missing = text.complete.expected.filter((s) => !shown.includes(norm(s)));
    expect(missing).toEqual([]);
  });

  it('shows UNVALIDATED badges and hypothesis chips from the payload', () => {
    render(<VariabilityView sensitivity={golden.sensitivity!} />);
    const kernel = screen.getByTestId('nsx-var-kernel_heatmap');
    expect(kernel).toHaveTextContent('UNVALIDATED');
    expect(screen.getByTestId('nsx-var-strategy_pareto')).toHaveTextContent(
      'Hypothesis: H-NOISE-SIGNAL-DEPENDENT',
    );
  });

  it('keeps a withheld arm to its reason and gives it no number', () => {
    render(<VariabilityView sensitivity={golden.sensitivity!} />);
    const arm = golden.sensitivity!.strategy_pareto.points.find((p) => !p.shown)!;
    const row = screen.getByTestId(`nsx-var-point-${arm.label}`);
    expect(row).toHaveTextContent('Withheld');
    expect(row).toHaveTextContent(arm.reason);
    expect(row).toHaveTextContent('unavailable');
  });

  it('lists skipped arms with their reasons', () => {
    render(<VariabilityView sensitivity={golden.sensitivity!} />);
    for (const s of golden.sensitivity!.strategy_pareto.skipped) {
      expect(screen.getByTestId(`nsx-var-skipped-${s.label}`)).toHaveTextContent(s.reason);
    }
  });

  it('a capability-limited engine shows its UNAVAILABLE reason', () => {
    const limited = text.unavailable.plot_model;
    const { container } = render(<VariabilityView sensitivity={limited.sensitivity!} />);
    const shown = norm(container.textContent);
    expect(shown).toContain('UNAVAILABLE: ');
    for (const s of text.unavailable.expected) expect(shown).toContain(norm(s));
  });
});
