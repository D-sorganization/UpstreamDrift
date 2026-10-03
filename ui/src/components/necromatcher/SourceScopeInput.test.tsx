import { expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { SourceScopeInput } from './SourceScopeInput';
const registration = vi.hoisted(() => vi.fn());
vi.mock('@/api/necromatcher', () => ({registerReviewedWindow: registration}));

const record = {
  schema: 'necromatcher/source-fit-scope/1', capture_id: 'capture',
  capture_hash: 'sha256:abc', source_clock_sha256: 'sha256:clock',
  first_frame: 0, end_exclusive_frame: 191, purpose: 'both_hands_on_club',
  review: {reason: 'Exclude ambiguous transition', uncertainty_policy: 'Conservative authored window', contact_calibrated: false},
};

function file(text: Promise<string>) {
  const selected = new File([''], 'window.json', {type: 'application/json'});
  selected.text = vi.fn().mockReturnValue(text);
  return selected;
}

it('registers a raw receipt for the selected fit before returning its portable window', async () => {
  const onChange = vi.fn();
  registration.mockResolvedValue(record);
  render(<SourceScopeInput fit="parent" disabled={false} onChange={onChange} />);
  const selected = file(Promise.resolve(JSON.stringify({schema: 'necromatcher/source-fit-scope-review/1'})));
  await userEvent.upload(screen.getByLabelText('Import Reviewed Window'), selected);
  expect(await screen.findByText(/Reviewed Original Frames: 0 to 191/)).toBeInTheDocument();
  expect(registration).toHaveBeenCalledWith('parent', selected);
  expect(onChange).toHaveBeenLastCalledWith(record, false);
});

it('rejects an oversized file before reading or registering it', async () => {
  const onChange = vi.fn();
  render(<SourceScopeInput fit="parent" disabled={false} onChange={onChange} />);
  const selected = new File(['x'.repeat(1024 * 1024 + 1)], 'window.json', {type: 'application/json'});
  selected.text = vi.fn();
  await userEvent.upload(screen.getByLabelText('Import Reviewed Window'), selected);
  expect(await screen.findByRole('alert')).toHaveTextContent('at Most 1 MiB');
  expect(selected.text).not.toHaveBeenCalled();
  expect(onChange).toHaveBeenLastCalledWith(null, true);
});

it('blocks an imported window that widens the inherited review', async () => {
  const onChange = vi.fn();
  const inherited = {...record, first_frame: 10} as Parameters<typeof SourceScopeInput>[0]['inherited'];
  render(<SourceScopeInput inherited={inherited} disabled={false} onChange={onChange} />);
  await userEvent.upload(screen.getByLabelText('Import Reviewed Window'), file(Promise.resolve(JSON.stringify(record))));
  expect(await screen.findByRole('alert')).toHaveTextContent('Cannot Widen');
  expect(onChange).toHaveBeenLastCalledWith(null, true);
});

it('passes the imported record unchanged and removes only the imported choice', async () => {
  const onChange = vi.fn();
  render(<SourceScopeInput disabled={false} onChange={onChange} />);
  await userEvent.upload(screen.getByLabelText('Import Reviewed Window'), file(Promise.resolve(JSON.stringify(record))));
  expect(await screen.findByText(/Reviewed Original Frames: 0 to 191/)).toBeInTheDocument();
  expect(onChange).toHaveBeenLastCalledWith(record, false);
  await userEvent.click(screen.getByRole('button', {name: 'Remove Imported Window'}));
  expect(onChange).toHaveBeenLastCalledWith(null, false);
  expect(screen.queryByText(/Reviewed Original Frames/)).not.toBeInTheDocument();
});

it('ignores a stale file read after the import is removed', async () => {
  const onChange = vi.fn();
  let finish!: (text: string) => void;
  render(<SourceScopeInput disabled={false} onChange={onChange} />);
  await userEvent.upload(screen.getByLabelText('Import Reviewed Window'), file(new Promise((resolve) => {finish = resolve;})));
  await userEvent.click(screen.getByRole('button', {name: 'Remove Imported Window'}));
  finish(JSON.stringify(record));
  expect(onChange).toHaveBeenLastCalledWith(null, false);
  expect(screen.queryByText(/Reviewed Original Frames/)).not.toBeInTheDocument();
});

it.each([true, '0', -1])('blocks malformed first-frame value %s', async (first_frame) => {
  const onChange = vi.fn();
  render(<SourceScopeInput disabled={false} onChange={onChange} />);
  await userEvent.upload(screen.getByLabelText('Import Reviewed Window'), file(Promise.resolve(JSON.stringify({...record, first_frame}))));
  expect(await screen.findByRole('alert')).toHaveTextContent('Reviewed Fitting Window');
  expect(onChange).toHaveBeenLastCalledWith(null, true);
});
