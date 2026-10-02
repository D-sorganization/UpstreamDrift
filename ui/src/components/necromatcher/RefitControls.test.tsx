import { useState } from 'react';
import { beforeEach, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { RefitControls } from './RefitControls';

const api = vi.hoisted(() => ({ fetchRefitPlan: vi.fn(), submitRefit: vi.fn(), fetchRefit: vi.fn(), cancelRefit: vi.fn() }));
vi.mock('@/api/necromatcher', () => api);
beforeEach(() => {
  Object.values(api).forEach((mock) => mock.mockReset());
  api.fetchRefitPlan.mockResolvedValue({source_fit_id: 'old', frame_indices: [0, 2], coordinate_order: ['hip'], coordinate_units: ['rad'], recorded_options: null});
});

it('requires explicit scales and displays computational success separately from rejection', async () => {
  const changed = vi.fn();
  const user = userEvent.setup();
  api.submitRefit.mockResolvedValue({run_id: 'run', source_fit_id: 'old', new_fit_id: 'new', status: 'running', acceptance: 'partial', blockers: [], message: 'Computing'});
  api.fetchRefit.mockResolvedValue({run_id: 'run', source_fit_id: 'old', new_fit_id: 'new', status: 'succeeded', acceptance: 'rejected', blockers: ['physical_clock_unknown'], message: 'Research stored'});
  render(<RefitControls fit="old" onStored={changed} />);
  await screen.findByLabelText('Coordinate Prior Scales');
  await user.type(screen.getByLabelText('New Fit Version'), 'new');
  expect(screen.getByRole('button', {name: 'Start Research Refit'})).toBeDisabled();
  await user.type(screen.getByLabelText('Coordinate Prior Scales'), '1');
  await user.click(screen.getByRole('button', {name: 'Start Research Refit'}));
  await waitFor(() => expect(changed).toHaveBeenCalledOnce());
  expect(screen.getByRole('status')).toHaveTextContent('succeeded · rejected');
  expect(screen.getByText('physical_clock_unknown')).toBeInTheDocument();
  expect(api.submitRefit.mock.calls[0][1].coordinate_scales).toEqual([1]);
});

it('discards a submission response after changing the selected source fit', async () => {
  let finish!: (value: unknown) => void;
  api.submitRefit.mockImplementation(() => new Promise((resolve) => {finish = resolve;}));
  const changed = vi.fn();
  const user = userEvent.setup();
  const view = render(<RefitControls fit="old" onStored={changed} />);
  await screen.findByLabelText('Coordinate Prior Scales');
  await user.type(screen.getByLabelText('New Fit Version'), 'new');
  await user.type(screen.getByLabelText('Coordinate Prior Scales'), '1');
  await user.click(screen.getByRole('button', {name: 'Start Research Refit'}));
  api.fetchRefitPlan.mockResolvedValue({source_fit_id: 'other', frame_indices: [0, 2], coordinate_order: ['hip'], coordinate_units: ['rad'], recorded_options: null});
  view.rerender(<RefitControls fit="other" onStored={changed} />);
  finish({run_id: 'old-run', source_fit_id: 'old', new_fit_id: 'new', status: 'succeeded', acceptance: 'rejected', blockers: [], message: 'Old result'});
  await screen.findByLabelText('Coordinate Prior Scales');
  expect(screen.queryByText(/Old result/)).not.toBeInTheDocument();
  expect(changed).not.toHaveBeenCalled();
});

it('reopens a terminal run without repeatedly refreshing the library', async () => {
  api.fetchRefit.mockResolvedValue({run_id: 'saved', source_fit_id: 'old', new_fit_id: 'new', status: 'succeeded', acceptance: 'rejected', blockers: [], message: 'Saved research run'});
  const changed = vi.fn();
  render(<RefitControls fit="old" initialRunId="saved" onStored={changed} />);
  expect(await screen.findByRole('status')).toHaveTextContent('succeeded · rejected');
  expect(changed).not.toHaveBeenCalled();
});


it('refreshes a newly submitted run that finishes while its URL is being saved', async () => {
  const changed = vi.fn();
  api.submitRefit.mockResolvedValue({run_id: 'new-run', source_fit_id: 'old', new_fit_id: 'new', status: 'running', acceptance: 'partial', blockers: [], message: 'Computing'});
  api.fetchRefit.mockResolvedValue({run_id: 'new-run', source_fit_id: 'old', new_fit_id: 'new', status: 'succeeded', acceptance: 'rejected', blockers: [], message: 'Stored'});
  function Harness() {
    const [run, setRun] = useState('');
    return <RefitControls fit="old" initialRunId={run} onRun={setRun} onStored={changed} />;
  }
  render(<Harness />);
  const user = userEvent.setup();
  await screen.findByLabelText('Coordinate Prior Scales');
  await user.type(screen.getByLabelText('New Fit Version'), 'new');
  await user.type(screen.getByLabelText('Coordinate Prior Scales'), '1');
  await user.click(screen.getByRole('button', {name: 'Start Research Refit'}));
  await waitFor(() => expect(changed).toHaveBeenCalledOnce());
});
