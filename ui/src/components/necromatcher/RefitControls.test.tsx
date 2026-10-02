import { useState } from 'react';
import { beforeEach, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { RefitControls } from './RefitControls';

const boundedConfig = {
  max_iterations: 30, prior_weight: 0.2, smoothness_weight: 0.03, closure_weight: 120,
  coordinate_bounds: [['hip', -1, 1]], interior_fractions: [0.25, 0.5, 0.75],
  initialization_policy: 'authored_range_project_zero_slopes',
  constraint_options: {
    ground: {normal: [0, 0, 1], height_m: 0}, position_weight: 100, rotation_weight: 50,
    ground_weight: 100, position_scale_m: 0.01, rotation_scale_rad: 0.1, ground_scale_m: 0.01,
    pinned_spheres: ['heel_r', 'heel_l'],
  },
};
function boundedPlan() {
  return {source_fit_id: 'old', frame_indices: [0, 1, 2, 3], coordinate_order: ['hip'], coordinate_units: ['rad'],
    recorded_options: {frame_indices: [0, 2, 3], knot_count: 3, coordinate_scales: [1], unknown_visibility_weight: 0.4, budget_wall_s: 300, config: boundedConfig},
    preserved_spline: {available: true, knot_count: 3, source_interval: [10, 12], reason: 'Saved spline verified'},
  };
}

const api = vi.hoisted(() => ({ fetchRefitPlan: vi.fn(), submitRefit: vi.fn(), fetchRefit: vi.fn(), cancelRefit: vi.fn() }));
vi.mock('@/api/necromatcher', () => api);
beforeEach(() => {
  Object.values(api).forEach((mock) => mock.mockReset());
  api.fetchRefitPlan.mockResolvedValue({source_fit_id: 'old', frame_indices: [0, 2], coordinate_order: ['hip'], coordinate_units: ['rad'], recorded_options: null});
});

it('transports the complete nested bounded contact recipe without flattened config keys', async () => {
  api.fetchRefitPlan.mockResolvedValue(boundedPlan());
  api.submitRefit.mockResolvedValue({run_id: 'r', source_fit_id: 'old', new_fit_id: 'new', status: 'failed', acceptance: 'rejected', blockers: [], message: 'Finished'});
  render(<RefitControls fit="old" onStored={vi.fn()} />);
  const user = userEvent.setup();
  await screen.findByLabelText('Coordinate Prior Scales');
  await user.type(screen.getByLabelText('New Fit Version'), 'new');
  await user.click(screen.getByRole('button', {name: 'Start Research Refit'}));
  const submitted = api.submitRefit.mock.calls[0][1];
  expect(submitted.config).toEqual(boundedConfig);
  expect(submitted).not.toHaveProperty('coordinate_bounds');
  expect(submitted).not.toHaveProperty('max_iterations');
  expect(submitted.initialization_source).toBe('sampled_parent');
  expect(submitted.operation).toBe('fit');
  expect(screen.getByText(/1 Authored Range.*2 Authored Heel Pins/)).toBeInTheDocument();
});

it('explicitly resumes the saved spline with strict policy and its full source interval', async () => {
  api.fetchRefitPlan.mockResolvedValue(boundedPlan());
  api.submitRefit.mockResolvedValue({run_id: 'r', source_fit_id: 'old', new_fit_id: 'new', status: 'failed', acceptance: 'rejected', blockers: [], message: 'Finished'});
  render(<RefitControls fit="old" onStored={vi.fn()} />);
  const user = userEvent.setup();
  await screen.findByLabelText('Coordinate Prior Scales');
  await user.selectOptions(screen.getByLabelText('Initialization Source'), 'preserved_spline');
  await user.type(screen.getByLabelText('New Fit Version'), 'new');
  expect(screen.getByLabelText('Spline Knots')).toBeDisabled();
  expect(screen.getByText(/Full Saved Source Interval: 10.*12/)).toBeInTheDocument();
  await user.click(screen.getByRole('button', {name: 'Start Research Refit'}));
  expect(api.submitRefit.mock.calls[0][1]).toMatchObject({
    initialization_source: 'preserved_spline', knot_count: 3, operation: 'fit',
    config: {...boundedConfig, initialization_policy: 'strict'},
  });
});

it('resets source-specific recipe and initialization choice when the selected fit changes', async () => {
  api.fetchRefitPlan.mockResolvedValue(boundedPlan());
  const view = render(<RefitControls fit="old" onStored={vi.fn()} />);
  const user = userEvent.setup();
  await screen.findByLabelText('Coordinate Prior Scales');
  await user.selectOptions(screen.getByLabelText('Initialization Source'), 'preserved_spline');
  api.fetchRefitPlan.mockResolvedValue({source_fit_id: 'other', frame_indices: [0, 2], coordinate_order: ['hip'], coordinate_units: ['rad'], recorded_options: null});
  view.rerender(<RefitControls fit="other" onStored={vi.fn()} />);
  await screen.findByLabelText('Coordinate Prior Scales');
  expect(screen.getByLabelText('Initialization Source')).toHaveValue('sampled_parent');
  expect(screen.queryByText(/2 Authored Heel Pins/)).not.toBeInTheDocument();
  expect(screen.getByLabelText('New Fit Version')).toHaveValue('');
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


it('retains the canonical baseline recipe when previous request options are absent', async () => {
  api.fetchRefitPlan.mockResolvedValue({...boundedPlan(), recorded_options: null, baseline_config: boundedConfig,
    preserved_spline: {available: false, knot_count: null, source_interval: null, reason: 'No saved spline'}});
  render(<RefitControls fit="old" onStored={vi.fn()} />);
  const user = userEvent.setup();
  await screen.findByLabelText('Coordinate Prior Scales');
  expect(screen.getByRole('option', {name: 'Resume Saved Spline'})).toBeDisabled();
  expect(screen.getByText('Saved Spline Unavailable: No saved spline')).toBeInTheDocument();
  expect(screen.getByLabelText('Evaluation Budget')).toHaveValue(30);
  await user.type(screen.getByLabelText('New Fit Version'), 'new');
  await user.type(screen.getByLabelText('Coordinate Prior Scales'), '1');
  await user.clear(screen.getByLabelText('Evaluation Budget'));
  await user.type(screen.getByLabelText('Evaluation Budget'), '40');
  api.submitRefit.mockResolvedValue({run_id: 'r', source_fit_id: 'old', new_fit_id: 'new', status: 'failed', acceptance: 'rejected', blockers: [], message: 'Finished'});
  await user.click(screen.getByRole('button', {name: 'Start Research Refit'}));
  expect(api.submitRefit.mock.calls[0][1].config).toEqual({...boundedConfig, max_iterations: 40});
});

it('requires both saved interval endpoint frames when resuming and restores sampled policy', async () => {
  api.fetchRefitPlan.mockResolvedValue(boundedPlan());
  render(<RefitControls fit="old" onStored={vi.fn()} />);
  const user = userEvent.setup();
  await screen.findByLabelText('Coordinate Prior Scales');
  await user.type(screen.getByLabelText('New Fit Version'), 'new');
  await user.selectOptions(screen.getByLabelText('Initialization Source'), 'preserved_spline');
  await user.clear(screen.getByLabelText('Source Frame Indices'));
  await user.type(screen.getByLabelText('Source Frame Indices'), '0, 1, 2');
  expect(screen.getByRole('button', {name: 'Start Research Refit'})).toBeDisabled();
  await user.selectOptions(screen.getByLabelText('Initialization Source'), 'sampled_parent');
  api.submitRefit.mockResolvedValue({run_id: 'r', source_fit_id: 'old', new_fit_id: 'new', status: 'failed', acceptance: 'rejected', blockers: [], message: 'Finished'});
  await user.click(screen.getByRole('button', {name: 'Start Research Refit'}));
  expect(api.submitRefit.mock.calls[0][1].config.initialization_policy).toBe('authored_range_project_zero_slopes');
});
