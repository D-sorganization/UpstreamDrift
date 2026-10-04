import { beforeEach, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ReplayControls } from './ReplayControls';

const api = vi.hoisted(() => ({fetchAssets: vi.fn(), fetchReplaySummary: vi.fn(), replayDataUrl: vi.fn()}));
vi.mock('@/api/necromatcher', () => api);
const hash = `sha256:${'a'.repeat(64)}`;
const asset = {dataset_id:'replay-1', session_id:'swing', kind:'authored_replay', metadata:{qualification:'unqualified_authored_replay', fit_id:'fit', model_id:'model', profile_id:'profile', hash}};
function summary() {
  return {replay_id:'replay-1', sample_count:21, dt_s:0.01, backend:'mujoco', metadata:{
    schema:'necromatcher/authored-replay/1', scientific_qualified:false,
    physical_source_time_qualified:false, independent_replay_executed:true,
    root_policy:'unactuated', initial_state_policy:'exact_saved_pose_and_authored_rates',
    fit_id:'fit', model_id:'model', profile_id:'profile', capture_id:'capture',
    fit_hash:hash, model_hash:hash, profile_hash:hash, capture_hash:hash,
  }};
}
beforeEach(() => {
  Object.values(api).forEach((mock) => mock.mockReset());
  api.fetchAssets.mockResolvedValue({assets:[asset, {...asset, dataset_id:'fit', kind:'kinematic_fit'}]});
  api.fetchReplaySummary.mockResolvedValue(summary());
  api.replayDataUrl.mockReturnValue('/verified.h5');
});
it('recalls only authored replay assets and exposes verified parents and clock qualification', async () => {
  render(<ReplayControls swing="swing" revision={0} />);
  await userEvent.click(await screen.findByRole('button', {name:'Recall replay-1'}));
  expect(await screen.findByText(/21 Samples/)).toHaveTextContent('mujoco');
  expect(screen.getByText(/Authored Seconds/)).toHaveTextContent('Physical source time: unqualified');
  expect(screen.getByText(/Fit: fit/)).toHaveTextContent('Profile: profile');
  expect(screen.getByRole('link', {name:'Download Verified Replay HDF5'})).toHaveAttribute('href','/verified.h5');
  expect(screen.queryByRole('button', {name:'Recall fit'})).not.toBeInTheDocument();
});
it.each(['scientific_qualified','physical_source_time_qualified','independent_replay_executed'])('rejects missing qualification %s without a download', async (key) => {
  const value = summary(); delete (value.metadata as Record<string, unknown>)[key];
  api.fetchReplaySummary.mockResolvedValue(value);
  render(<ReplayControls swing="swing" revision={0} />);
  await userEvent.click(await screen.findByRole('button', {name:'Recall replay-1'}));
  expect(await screen.findByRole('alert')).toHaveTextContent('Replay qualification');
  expect(screen.queryByRole('link')).not.toBeInTheDocument();
});
it('rejects a different parent and malformed parent hash', async () => {
  const value = summary(); value.metadata.fit_id='other'; value.metadata.model_hash='bad';
  api.fetchReplaySummary.mockResolvedValue(value);
  render(<ReplayControls swing="swing" revision={0} />);
  await userEvent.click(await screen.findByRole('button', {name:'Recall replay-1'}));
  expect(await screen.findByRole('alert')).toHaveTextContent('Replay parent');
  expect(screen.queryByRole('link')).not.toBeInTheDocument();
});
it('does not display a late summary from the previous swing', async () => {
  let finish!: (value: ReturnType<typeof summary>) => void;
  api.fetchReplaySummary.mockImplementation(() => new Promise((resolve) => {finish=resolve;}));
  const view=render(<ReplayControls swing="swing" revision={0} />);
  await userEvent.click(await screen.findByRole('button', {name:'Recall replay-1'}));
  api.fetchAssets.mockResolvedValue({assets:[]});
  view.rerender(<ReplayControls swing="other" revision={0} />);
  finish(summary());
  expect(await screen.findByText('No Registered Authored Replays.')).toBeInTheDocument();
  expect(screen.queryByRole('link')).not.toBeInTheDocument();
});
