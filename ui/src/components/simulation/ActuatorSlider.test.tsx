/**
 * ActuatorSlider tests (issue #7425) — debounced send + failure revert.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { ActuatorSlider } from './ActuatorSlider';
import type { ActuatorInfo } from './ActuatorPanel';

const ACT: ActuatorInfo = {
  index: 2,
  name: 'hip',
  control_type: 'constant',
  value: 0,
  min_value: -10,
  max_value: 10,
  units: 'Nm',
  joint_type: 'hinge',
};

describe('ActuatorSlider', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('debounces drag ticks into a single send with the final value', async () => {
    const onValueChange = vi.fn().mockResolvedValue({ success: true });
    render(
      <ActuatorSlider
        actuator={ACT}
        onValueChange={onValueChange}
        onControlTypeChange={vi.fn()}
        availableTypes={['constant']}
      />,
    );
    const slider = screen.getByLabelText(/hip control value/i);

    act(() => {
      fireEvent.change(slider, { target: { value: '1' } });
      fireEvent.change(slider, { target: { value: '2' } });
      fireEvent.change(slider, { target: { value: '3' } });
    });
    expect(onValueChange).not.toHaveBeenCalled();

    await act(async () => {
      vi.advanceTimersByTime(200);
    });
    expect(onValueChange).toHaveBeenCalledTimes(1);
    expect(onValueChange).toHaveBeenCalledWith(2, 3);
  });

  it('reverts to the actuator value and reports when a send fails', async () => {
    const onValueChange = vi
      .fn()
      .mockResolvedValue({ success: false, error: 'engine not loaded' });
    const onError = vi.fn();
    render(
      <ActuatorSlider
        actuator={ACT}
        onValueChange={onValueChange}
        onControlTypeChange={vi.fn()}
        onError={onError}
        availableTypes={['constant']}
      />,
    );
    const slider = screen.getByLabelText(/hip control value/i) as HTMLInputElement;

    act(() => {
      fireEvent.change(slider, { target: { value: '7' } });
    });
    await act(async () => {
      vi.advanceTimersByTime(200);
    });

    expect(onError).toHaveBeenCalledWith('engine not loaded');
    // Reverted to the confirmed actuator value (0).
    expect(Number(slider.value)).toBe(0);
  });
});
