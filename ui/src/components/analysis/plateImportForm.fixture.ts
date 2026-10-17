/** Test-only helper: fill the plate declarations form with a complete set of facts. */
import { fireEvent, screen } from '@testing-library/react';

export const COMPLETE_PLATE_FORM: Readonly<Record<string, string>> = {
  trial: 'session-1/swing-0001.json',
  uncertainty_source: 'datasheet rev 3',
  force_floor_n: '2.0',
  force_fraction: '0.01',
  cop_m: '0.010',
  sensing_depth_m: '0.015',
  sensing_depth_sigma_m: '0.002',
  capacity_n: '3923',
  origin_m: '0, 0, 0',
  offset_s: '0',
  sigma_offset_s: '0.001',
  alignment_note: 'clap on both clocks',
};

export function fillPlateForm(changes: Record<string, string> = {}): void {
  for (const [key, value] of Object.entries({ ...COMPLETE_PLATE_FORM, ...changes })) {
    fireEvent.change(screen.getByTestId(`plate-field-${key}`), { target: { value } });
  }
}

export function submitPlateForm(): void {
  fireEvent.click(screen.getByRole('button', { name: 'Overlay Plate Trial' }));
}
