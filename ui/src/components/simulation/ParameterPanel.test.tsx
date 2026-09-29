/**
 * ParameterPanel tests (issue #7424).
 *
 * The panel is a controlled view over the simulation store; it must not own
 * or reset parameter values on its own.
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ParameterPanel } from './ParameterPanel';
import {
  DEFAULT_PARAMETERS,
  type SimulationParameters,
} from '@/stores/useSimulationStore';

function make(overrides: Partial<SimulationParameters> = {}): SimulationParameters {
  return { ...DEFAULT_PARAMETERS, ...overrides };
}

describe('ParameterPanel', () => {
  it('renders the controlled value from props, not a hardcoded default', () => {
    render(
      <ParameterPanel engine="mujoco" value={make({ duration: 10 })} onChange={vi.fn()} />,
    );
    expect(screen.getByLabelText(/Duration/)).toHaveValue(10);
  });

  it('does not push any value to the store on mount (#7424)', () => {
    const onChange = vi.fn();
    render(
      <ParameterPanel engine="mujoco" value={make({ duration: 10 })} onChange={onChange} />,
    );
    // The old panel fired onChange on mount, clobbering user values. It must not.
    expect(onChange).not.toHaveBeenCalled();
  });

  it('remounting with the same store value keeps it (no reset)', () => {
    const onChange = vi.fn();
    const value = make({ duration: 10 });
    const { unmount } = render(
      <ParameterPanel engine="mujoco" value={value} onChange={onChange} />,
    );
    unmount();
    render(<ParameterPanel engine="drake" value={value} onChange={onChange} />);
    // Even though the engine differs, the panel shows the store value untouched.
    expect(screen.getByLabelText(/Duration/)).toHaveValue(10);
    expect(onChange).not.toHaveBeenCalled();
  });

  it('clearing the duration field then typing yields the typed value, not a default', () => {
    const onChange = vi.fn();
    render(
      <ParameterPanel engine="mujoco" value={make({ duration: 10 })} onChange={onChange} />,
    );
    const input = screen.getByLabelText(/Duration/);

    // Clear mid-edit: must NOT commit 3.0 (the old "|| 3.0" bug).
    fireEvent.change(input, { target: { value: '' } });
    const committedAfterClear = onChange.mock.calls.map((c) => c[0]);
    expect(committedAfterClear).not.toContainEqual({ duration: 3.0 });

    // Type a new value.
    fireEvent.change(input, { target: { value: '12' } });
    expect(onChange).toHaveBeenLastCalledWith({ duration: 12 });
  });

  it('forwards a reset-to-defaults click', () => {
    const onResetDefaults = vi.fn();
    render(
      <ParameterPanel
        engine="drake"
        value={make()}
        onChange={vi.fn()}
        onResetDefaults={onResetDefaults}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: /reset to engine defaults/i }));
    expect(onResetDefaults).toHaveBeenCalled();
  });
});
