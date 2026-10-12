import { expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ShaftEvidenceInput } from './ShaftEvidenceInput';

function file(record: unknown) {
  const selected = new File([JSON.stringify(record)], 'shaft.json', {type: 'application/json'});
  selected.text = vi.fn().mockResolvedValue(JSON.stringify(record));
  return selected;
}
const record = {schema: 'necromatcher/shaft-axis-evidence/1', capture_id: 'capture', frames: [{frame_index: 0}]};

it('loads an explicit reviewed record and allows returning to body-only fitting', async () => {
  const changed = vi.fn();
  render(<ShaftEvidenceInput onChange={changed} />);
  const user = userEvent.setup();
  await user.upload(screen.getByLabelText('Reviewed Shaft Evidence'), file(record));
  await waitFor(() => expect(changed).toHaveBeenLastCalledWith(record, false));
  expect(screen.getByText(/1 Reviewed Frame/)).toBeInTheDocument();
  await user.click(screen.getByRole('button', {name: 'Remove Shaft Evidence'}));
  expect(changed).toHaveBeenLastCalledWith(null, false);
});

it('blocks the optional input on malformed or unsupported evidence', async () => {
  const changed = vi.fn();
  render(<ShaftEvidenceInput onChange={changed} />);
  await userEvent.setup().upload(screen.getByLabelText('Reviewed Shaft Evidence'), file({schema: 'wrong'}));
  await screen.findByRole('alert');
  expect(changed).toHaveBeenLastCalledWith(null, true);
});

it('ignores a late file read after removal', async () => {
  const changed = vi.fn();
  let finish!: (value: string) => void;
  const selected = file(record);
  selected.text = () => new Promise((resolve) => {finish = resolve;});
  render(<ShaftEvidenceInput onChange={changed} />);
  const user = userEvent.setup();
  await user.upload(screen.getByLabelText('Reviewed Shaft Evidence'), selected);
  await user.click(screen.getByRole('button', {name: 'Remove Shaft Evidence'}));
  finish(JSON.stringify(record));
  await waitFor(() => expect(changed).toHaveBeenLastCalledWith(null, false));
  expect(screen.queryByText(/1 Reviewed Frame/)).not.toBeInTheDocument();
});

it('does not publish a pending record after the source form unmounts', async () => {
  const changed = vi.fn();
  let finish!: (value: string) => void;
  const selected = file(record);
  selected.text = () => new Promise((resolve) => {finish = resolve;});
  const view = render(<ShaftEvidenceInput onChange={changed} />);
  await userEvent.setup().upload(screen.getByLabelText('Reviewed Shaft Evidence'), selected);
  expect(changed).toHaveBeenLastCalledWith(null, true);
  const count = changed.mock.calls.length;
  view.unmount();
  finish(JSON.stringify(record));
  await Promise.resolve();
  await Promise.resolve();
  expect(changed).toHaveBeenCalledTimes(count);
});

it('rejects an oversized record before reading its contents', async () => {
  const changed = vi.fn();
  const selected = file(record);
  Object.defineProperty(selected, 'size', {value: 2 * 1024 * 1024 + 1});
  render(<ShaftEvidenceInput onChange={changed} />);
  await userEvent.setup().upload(screen.getByLabelText('Reviewed Shaft Evidence'), selected);
  await screen.findByText('Evidence File Exceeds 2 MiB.');
  expect(selected.text).not.toHaveBeenCalled();
  expect(changed).toHaveBeenLastCalledWith(null, true);
});
