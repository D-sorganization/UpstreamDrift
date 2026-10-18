import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { VariabilityControls, type ControlsState } from './VariabilityControls';
import type { SensitivityCatalogue } from '@/api/loopNullSpace';

const CAT: SensitivityCatalogue = {
  stages: [
    { id: 'torque', label: 'TORQUE' },
    { id: 'timing', label: 'TIMING' },
  ],
  spectra: [{ id: 'white', label: 'WHITE' }],
  mechanisms: [{ id: 'H-NOISE-SIGNAL-DEPENDENT', label: 'M1', note: 'needs a signal law' }],
  strategies: [
    { id: 'S1', name: 'free', mechanism: 'm', availability: 'implemented', reason: '' },
    { id: 'S4', name: 'share', mechanism: 'm', availability: 'not_implemented', reason: 'no model' },
  ],
  outcomes: [{ id: 'face_angle', label: 'face angle' }],
  default_outcome_priority: ['face_angle'],
  engines: [
    { id: 'pinocchio', method: 'analytic', available: true, reason: '' },
    { id: 'simscape', method: 'none', available: false, reason: 'snapshot grid only' },
  ],
};

afterEach(cleanup);

function setup() {
  const onChange = vi.fn<(s: ControlsState) => void>();
  render(<VariabilityControls catalogue={CAT} onChange={onChange} />);
  return onChange;
}

describe('VariabilityControls', () => {
  it('lists exactly the catalogue values', () => {
    setup();
    const opts = (label: string) =>
      Array.from((screen.getByLabelText(label) as HTMLSelectElement).options).map((o) => o.value);
    expect(opts('Stage')).toEqual(['torque', 'timing']);
    expect(opts('Spectrum')).toEqual(['white']);
    expect(opts('Primary outcome')).toEqual(['face_angle']);
    expect(opts('Engine')).toEqual(['pinocchio', 'simscape']);
    expect(screen.getByLabelText(/H-NOISE-SIGNAL-DEPENDENT/)).toBeInTheDocument();
    expect(screen.getByLabelText(/S4/)).toBeInTheDocument();
  });

  it('defaults to face angle first and the whole strategy catalogue', () => {
    const onChange = setup();
    expect((screen.getByLabelText('Primary outcome') as HTMLSelectElement).value).toBe('face_angle');
    expect(onChange).toHaveBeenCalled();
    const state = onChange.mock.calls.at(-1)![0];
    expect(state.enabled).toBe(false);
    expect(state.choices.strategies).toBeNull();
    expect(state.choices.outcomePriority).toEqual(['face_angle']);
  });

  it('shows the UNAVAILABLE reason of a capability-limited engine', () => {
    setup();
    expect(screen.getByRole('option', { name: /simscape/ })).toHaveTextContent(
      'UNAVAILABLE: snapshot grid only',
    );
  });

  it('reports the chosen options, an unchecked strategy and a parse error', () => {
    const onChange = setup();
    fireEvent.click(screen.getByLabelText('Run variability analysis'));
    fireEvent.change(screen.getByLabelText('Channels and amplitudes (variance)'), {
      target: { value: 'shoulder=0.004, elbow=0.009' },
    });
    fireEvent.click(screen.getByLabelText(/S4/));
    fireEvent.change(screen.getByLabelText('Engine'), { target: { value: 'simscape' } });
    const state = onChange.mock.calls.at(-1)![0];
    expect(state.enabled).toBe(true);
    expect(state.error).toBeNull();
    expect(state.engine).toBe('simscape');
    expect(state.choices.strategies).toEqual(['S1']);
    expect(state.choices.disturbances[0].channels).toEqual([
      ['shoulder', 0.004],
      ['elbow', 0.009],
    ]);
    fireEvent.change(screen.getByLabelText('Channels and amplitudes (variance)'), {
      target: { value: 'shoulder' },
    });
    expect(onChange.mock.calls.at(-1)![0].error).not.toBeNull();
  });
});
