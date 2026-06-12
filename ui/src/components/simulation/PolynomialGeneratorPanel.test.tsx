/**
 * PolynomialGeneratorPanel tests (issue #7429).
 *
 * Guards that an empty or NaN-containing coefficient list can never be
 * submitted from the UI.
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { PolynomialGeneratorPanel } from './PolynomialGeneratorPanel';
import type { ActuatorInfo } from './ActuatorPanel';

const ACTUATORS: ActuatorInfo[] = [
  {
    index: 0,
    name: 'hip',
    control_type: 'torque',
    value: 0,
    min_value: -1,
    max_value: 1,
    units: 'Nm',
    joint_type: 'hinge',
  },
];

function renderExpanded(onApply = vi.fn().mockResolvedValue(undefined)) {
  render(
    <PolynomialGeneratorPanel actuators={ACTUATORS} onApplyPolynomial={onApply} />,
  );
  fireEvent.click(screen.getByRole('button', { name: /Polynomial Generator/ }));
  return onApply;
}

describe('PolynomialGeneratorPanel', () => {
  it('renders nothing when there are no actuators', () => {
    const { container } = render(
      <PolynomialGeneratorPanel actuators={[]} onApplyPolynomial={vi.fn()} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it('disables Remove when only one coefficient remains', () => {
    renderExpanded();
    const removeButtons = screen.getAllByRole('button', { name: 'Remove' });
    // Default starts with two coefficients -> both removable.
    expect(removeButtons[0]).toBeEnabled();

    fireEvent.click(removeButtons[0]);

    const remaining = screen.getAllByRole('button', { name: 'Remove' });
    expect(remaining).toHaveLength(1);
    expect(remaining[0]).toBeDisabled();
  });

  it('cannot remove the last coefficient (no empty list)', () => {
    const onApply = renderExpanded();
    // Remove the second coefficient ('0'), leaving the first ('1').
    fireEvent.click(screen.getAllByRole('button', { name: 'Remove' })[1]);
    const lastRemove = screen.getByRole('button', { name: 'Remove' });
    fireEvent.click(lastRemove); // disabled, should be a no-op
    expect(screen.getAllByLabelText(/Coefficient \d/)).toHaveLength(1);

    fireEvent.click(screen.getByRole('button', { name: /Apply Polynomial/ }));
    expect(onApply).toHaveBeenCalledWith(0, [1]);
  });

  it('disables Apply and flags the field when a coefficient is empty/NaN', () => {
    const onApply = renderExpanded();
    const firstInput = screen.getAllByLabelText(/Coefficient \d/)[0];
    fireEvent.change(firstInput, { target: { value: '' } });

    const apply = screen.getByRole('button', { name: /Apply Polynomial/ });
    expect(apply).toBeDisabled();
    expect(firstInput).toHaveAttribute('aria-invalid', 'true');
    expect(screen.getByRole('alert')).toBeInTheDocument();

    fireEvent.click(apply);
    expect(onApply).not.toHaveBeenCalled();
  });

  it('submits the parsed numeric coefficients when all are valid', () => {
    const onApply = renderExpanded();
    const inputs = screen.getAllByLabelText(/Coefficient \d/);
    fireEvent.change(inputs[0], { target: { value: '2.5' } });
    fireEvent.change(inputs[1], { target: { value: '-1' } });

    fireEvent.click(screen.getByRole('button', { name: /Apply Polynomial/ }));
    expect(onApply).toHaveBeenCalledWith(0, [2.5, -1]);
  });
});
