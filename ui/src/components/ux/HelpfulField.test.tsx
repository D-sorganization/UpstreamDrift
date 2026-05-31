/**
 * HelpfulField render + behaviour tests (epic #5968).
 */

import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { HelpfulField } from './HelpfulField';
import { getFieldMetadata } from '../../ux/fieldMetadata';
import { isInRange } from '../../ux/fieldHelpers';

describe('HelpfulField', () => {
  it('links the input to its help text via aria-describedby', () => {
    render(
      <HelpfulField fieldId="simulation.duration" value={3} onChange={() => {}} />,
    );
    const meta = getFieldMetadata('simulation.duration');
    const input = screen.getByLabelText(/Duration/);
    const describedBy = input.getAttribute('aria-describedby');
    expect(describedBy).toBeTruthy();
    const help = document.getElementById(describedBy!);
    expect(help).not.toBeNull();
    expect(help!.textContent).toBe(meta.shortHelp);
  });

  it('fires onViolation when a numeric value breaches the range', () => {
    const onViolation = vi.fn();
    render(
      <HelpfulField
        fieldId="simulation.duration"
        value={3}
        onChange={() => {}}
        onViolation={onViolation}
      />,
    );
    fireEvent.change(screen.getByLabelText(/Duration/), {
      target: { value: '1000' },
    });
    expect(onViolation).toHaveBeenCalledWith('simulation.duration', 1000);
  });

  it('does not fire onViolation for an in-range value', () => {
    const onViolation = vi.fn();
    render(
      <HelpfulField
        fieldId="simulation.duration"
        value={3}
        onChange={() => {}}
        onViolation={onViolation}
      />,
    );
    fireEvent.change(screen.getByLabelText(/Duration/), {
      target: { value: '5' },
    });
    expect(onViolation).not.toHaveBeenCalled();
  });

  it('renders an enum field as a select', () => {
    render(
      <HelpfulField
        fieldId="simulation.engine"
        value="mujoco"
        onChange={() => {}}
      />,
    );
    const select = screen.getByLabelText(/Physics engine/);
    expect(select.tagName).toBe('SELECT');
    expect(screen.getByRole('option', { name: 'drake' })).toBeInTheDocument();
  });

  it('isInRange respects the metadata bounds', () => {
    const meta = getFieldMetadata('simulation.duration');
    expect(isInRange(meta, 3)).toBe(true);
    expect(isInRange(meta, 1000)).toBe(false);
  });
});
