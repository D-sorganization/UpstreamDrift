import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { PlateImportForm } from './PlateImportForm';
import { fillPlateForm as fill, submitPlateForm } from './plateImportForm.fixture';

afterEach(cleanup);

describe('PlateImportForm', () => {
  it('starts with empty physical facts and refuses to submit until they are filled', () => {
    const onSubmit = vi.fn();
    render(<PlateImportForm onSubmit={onSubmit} />);
    expect(screen.getByTestId('plate-field-capacity_n')).toHaveValue('');
    expect(screen.getByTestId('plate-field-target_dir')).toHaveValue('0, -1, 0');
    submitPlateForm();
    expect(onSubmit).not.toHaveBeenCalled();
    expect(screen.getByTestId('plate-import-error')).toHaveTextContent('trial: required');
  });

  it('offers only the alignments a person can declare', () => {
    render(<PlateImportForm onSubmit={vi.fn()} />);
    const kinds = Array.from(
      (screen.getByTestId('plate-field-alignment_kind') as HTMLSelectElement).options,
    ).map((o) => o.value);
    expect(kinds).toEqual(['manual', 'shot_impact', 'absolute_utc']);
  });

  it('submits the overlay request built from a complete form', () => {
    const onSubmit = vi.fn();
    render(<PlateImportForm onSubmit={onSubmit} />);
    fill();
    submitPlateForm();
    expect(onSubmit).toHaveBeenCalledTimes(1);
    const request = onSubmit.mock.calls[0][0];
    expect(request.trial).toBe('session-1/swing-0001.json');
    expect(request.declared.capacity_n).toBe(3923);
    expect(request.alignment.kind).toBe('manual');
    expect(screen.queryByTestId('plate-import-error')).toBeNull();
  });

  it('names a refused field and does not submit', () => {
    const onSubmit = vi.fn();
    render(<PlateImportForm onSubmit={onSubmit} />);
    fill({ origin_m: '1, 2' });
    submitPlateForm();
    expect(onSubmit).not.toHaveBeenCalled();
    expect(screen.getByTestId('plate-import-error')).toHaveTextContent('origin_m');
  });

  it('shows the server refusal next to the form', () => {
    render(<PlateImportForm onSubmit={vi.fn()} serverError="capacity_n must be positive" />);
    expect(screen.getByTestId('plate-import-error')).toHaveTextContent(
      'Plate overlay refused: capacity_n must be positive',
    );
  });

  it('clears an active overlay', () => {
    const onClear = vi.fn();
    render(<PlateImportForm onSubmit={vi.fn()} onClear={onClear} active />);
    fireEvent.click(screen.getByRole('button', { name: 'Clear Overlay' }));
    expect(onClear).toHaveBeenCalledTimes(1);
  });
});
