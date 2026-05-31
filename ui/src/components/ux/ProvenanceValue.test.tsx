/**
 * ProvenanceValue render + behaviour tests (epic #5968).
 */

import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { ProvenanceValue } from './ProvenanceValue';
import {
  describeProvenance,
  isLinked,
  type ProvenanceValueData,
} from '../../ux/provenance';

function sample(): ProvenanceValueData {
  return {
    value: 500,
    displayUnits: 'fps',
    label: 'Frame rate',
    record: {
      formula: 'fps = 1.0 / timestep',
      inputs: ['simulation.timestep'],
      source: 'mujoco:run-42',
      computedAt: '2026-05-30T12:00:00+00:00',
      engine: 'mujoco',
      runId: 'run-42',
    },
  };
}

describe('ProvenanceValue', () => {
  it('renders the value with display units', () => {
    render(<ProvenanceValue data={sample()} />);
    expect(screen.getByLabelText('Frame rate')).toHaveTextContent('500 fps');
  });

  it('exposes the provenance via aria-describedby', () => {
    render(<ProvenanceValue data={sample()} />);
    const valueEl = screen.getByLabelText('Frame rate');
    const descId = valueEl.getAttribute('aria-describedby');
    expect(descId).toBeTruthy();
    const desc = document.getElementById(descId!);
    expect(desc!.textContent).toContain('fps = 1.0 / timestep');
    expect(desc!.textContent).toContain('simulation.timestep');
  });

  it('renders a link badge for derived values', () => {
    render(<ProvenanceValue data={sample()} />);
    expect(screen.getByTestId('provenance-link-badge')).toBeInTheDocument();
  });

  it('omits the link badge for constants', () => {
    const constant = sample();
    constant.record.inputs = [];
    render(<ProvenanceValue data={constant} />);
    expect(screen.queryByTestId('provenance-link-badge')).toBeNull();
  });

  it('describeProvenance mirrors the Python describe() layout', () => {
    const text = describeProvenance(sample());
    expect(text).toContain('value: 500 fps');
    expect(text).toContain('source: mujoco:run-42');
    expect(isLinked(sample())).toBe(true);
  });
});
