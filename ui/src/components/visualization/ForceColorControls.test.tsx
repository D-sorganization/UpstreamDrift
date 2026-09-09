import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { ForceColorControls } from './ForceColorControls';
import { defaultForceColorScale } from './forceColors';

describe('force color controls', () => {
  it('toggles immediately and applies custom settings', () => {
    const onChange = vi.fn();
    render(<ForceColorControls scale={defaultForceColorScale} onChange={onChange} />);
    fireEvent.click(screen.getByLabelText('Color Segments by Axial Force'));
    expect(onChange.mock.lastCall?.[0].enabled).toBe(true);
    fireEvent.change(screen.getByLabelText('Tension Color'), { target: { value: '#00ff00' } });
    fireEvent.click(screen.getByText('Apply Colors and Ranges'));
    expect(onChange.mock.lastCall?.[0].tension_color).toBe('#00ff00');
  });
  it('rejects invalid ranges without emitting a scale', () => {
    const onChange = vi.fn();
    render(<ForceColorControls scale={defaultForceColorScale} onChange={onChange} />);
    fireEvent.change(screen.getByLabelText('Tension Saturation (N)'), { target: { value: '0' } });
    fireEvent.click(screen.getByText('Apply Colors and Ranges'));
    expect(onChange).not.toHaveBeenCalled();
    expect(screen.getByRole('alert')).toHaveTextContent('limits');
  });
});
