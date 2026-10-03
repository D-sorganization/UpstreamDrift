import { beforeEach, expect, it, vi } from 'vitest';
import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { VideoExportControls } from './VideoExportControls';
const api = vi.hoisted(() => ({submitVideoExport: vi.fn(), fetchVideoExport: vi.fn(), cancelVideoExport: vi.fn(), videoExportDownloadUrl: (run: string) => `/export/${run}/download`}));
vi.mock('@/api/necromatcher', () => api);
const run = (values = {}) => ({run_id:'export-1',source_fit_id:'fit-1',status:'running',acceptance:'partial',qualification:'monocular_research_hypothesis',blockers:[],message:'Encoding',fraction:null,control_available:true,execution_started:true,execution_verified:false,download_available:false,...values});
beforeEach(() => {api.submitVideoExport.mockReset(); api.fetchVideoExport.mockReset(); api.cancelVideoExport.mockReset();});
it('displays the stored reviewed source window when recalling an export', async () => {
  api.fetchVideoExport.mockResolvedValue(run({source_fit_scope: {
    first_frame: 0, end_exclusive_frame: 191,
    review: {reason: 'Reviewed conservative boundary', uncertainty_policy: 'Release unmeasured'},
  }}));
  render(<VideoExportControls fit="fit-1" initialRunId="export-1" />);
  expect(await screen.findByText(/Reviewed Original Frames: 0 to 191/)).toHaveTextContent('Release unmeasured');
});
it('opts into a translucent model proxy and resets options for another fit', async () => {
  api.submitVideoExport.mockResolvedValue(run({status: 'failed', acceptance: 'rejected'}));
  const view = render(<VideoExportControls fit="fit-1" />);
  const user = userEvent.setup();
  expect(screen.getByLabelText('Model Proxy Opacity')).toBeDisabled();
  await user.click(screen.getByRole('checkbox', {name: 'Show Translucent Model Proxy'}));
  await user.clear(screen.getByLabelText('Model Proxy Opacity'));
  await user.type(screen.getByLabelText('Model Proxy Opacity'), '0.6');
  await user.click(screen.getByRole('button', {name: 'Export Research Overlay'}));
  expect(api.submitVideoExport).toHaveBeenCalledWith('fit-1', {shape_overlay: {opacity: 0.6}});
  expect(screen.getByText(/Model Proxy Retains the Skeleton/)).toBeInTheDocument();
  view.rerender(<VideoExportControls fit="fit-2" />);
  expect(screen.getByRole('checkbox', {name: 'Show Translucent Model Proxy'})).not.toBeChecked();
  await user.click(screen.getByRole('button', {name: 'Export Research Overlay'}));
  expect(api.submitVideoExport).toHaveBeenLastCalledWith('fit-2');
});
it('blocks malformed opacity before submitting and displays recalled proxy options', async () => {
  api.fetchVideoExport.mockResolvedValue(run({status: 'succeeded', acceptance: 'rejected', shape_overlay: {opacity: 0.4}}));
  render(<VideoExportControls fit="fit-1" initialRunId="export-1" />);
  expect(await screen.findByText(/Stored Model Proxy Opacity/)).toHaveTextContent('0.4');
  const user = userEvent.setup();
  await user.click(screen.getByRole('checkbox', {name: 'Show Translucent Model Proxy'}));
  await user.clear(screen.getByLabelText('Model Proxy Opacity'));
  expect(screen.getByRole('button', {name: 'Export Research Overlay'})).toBeDisabled();
  await user.type(screen.getByLabelText('Model Proxy Opacity'), '1.2');
  expect(screen.getByRole('button', {name: 'Export Research Overlay'})).toBeDisabled();
  expect(api.submitVideoExport).not.toHaveBeenCalled();
});
it('opts into reviewed shaft lines and clears the record for a different source', async () => {
  api.submitVideoExport.mockResolvedValue(run({status: 'failed', acceptance: 'rejected'}));
  const view = render(<VideoExportControls fit="fit-1" />);
  const record = {schema: 'necromatcher/shaft-axis-evidence/1', capture_id: 'capture', frames: [{frame_index: 0}]};
  const file = new File([JSON.stringify(record)], 'review.json', {type: 'application/json'});
  file.text = vi.fn().mockResolvedValue(JSON.stringify(record));
  const user = userEvent.setup();
  await user.upload(screen.getByLabelText('Reviewed Shaft Evidence'), file);
  await screen.findByText(/1 Reviewed Frame/);
  await user.click(screen.getByRole('button', {name: 'Export Research Overlay'}));
  expect(api.submitVideoExport).toHaveBeenCalledWith('fit-1', {shaft_evidence: record});
  view.rerender(<VideoExportControls fit="fit-2" />);
  expect(screen.queryByText(/1 Reviewed Frame/)).not.toBeInTheDocument();
  await user.click(screen.getByRole('button', {name: 'Export Research Overlay'}));
  expect(api.submitVideoExport).toHaveBeenLastCalledWith('fit-2');
});
it('submits once and downloads verified computation with rejected scientific acceptance', async () => {
  api.submitVideoExport.mockResolvedValue(run());
  api.fetchVideoExport.mockResolvedValue(run({status:'succeeded',acceptance:'rejected',execution_verified:true,download_available:true,blockers:['physical_clock_unknown']}));
  const onRun=vi.fn(); render(<VideoExportControls fit="fit-1" onRun={onRun} />);
  await userEvent.setup().click(screen.getByRole('button',{name:'Export Research Overlay'}));
  expect(api.submitVideoExport).toHaveBeenCalledExactlyOnceWith('fit-1');
  expect(onRun).toHaveBeenCalledWith('export-1');
  expect(await screen.findByRole('link',{name:'Download Overlay Package'})).toHaveAttribute('href','/export/export-1/download');
  expect(screen.getByRole('status')).toHaveTextContent('succeeded · rejected');
});
it('suppresses pending response after fit selection changes', async () => {
  let finish!: (value: unknown) => void;
  api.submitVideoExport.mockImplementation(() => new Promise((resolve) => {finish=resolve;}));
  const onRun=vi.fn(); const view=render(<VideoExportControls fit="fit-1" onRun={onRun} />);
  await userEvent.setup().click(screen.getByRole('button',{name:'Export Research Overlay'}));
  expect(screen.getByRole('button',{name:'Starting Export…'})).toBeDisabled();
  view.rerender(<VideoExportControls fit="fit-2" onRun={onRun} />);
  await act(async () => finish(run({status:'succeeded',download_available:true,execution_verified:true})));
  expect(onRun).not.toHaveBeenCalled(); expect(screen.queryByRole('link')).not.toBeInTheDocument();
});
it('cancels owned work without offering its artifact', async () => {
  api.fetchVideoExport.mockResolvedValue(run()); api.cancelVideoExport.mockResolvedValue(run({status:'cancelled',acceptance:'interrupted',message:'Cancelled'}));
  render(<VideoExportControls fit="fit-1" initialRunId="export-1" />);
  await userEvent.setup().click(await screen.findByRole('button',{name:'Cancel Overlay Export'}));
  await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('cancelled'));
  expect(api.cancelVideoExport).toHaveBeenCalledWith('export-1'); expect(screen.queryByRole('link')).not.toBeInTheDocument();
});
it('rejects recalled work belonging to another source fit', async () => {
  api.fetchVideoExport.mockResolvedValue(run({source_fit_id:'another',status:'succeeded',download_available:true,execution_verified:true}));
  render(<VideoExportControls fit="fit-1" initialRunId="export-1" />);
  expect(await screen.findByRole('alert')).toHaveTextContent('another source version'); expect(screen.queryByRole('link')).not.toBeInTheDocument();
});
it('shows backend failure and permits retry', async () => {
  api.submitVideoExport.mockRejectedValue(new Error('Nonuniform source PTS')); render(<VideoExportControls fit="fit-1" />);
  await userEvent.setup().click(screen.getByRole('button',{name:'Export Research Overlay'}));
  expect(await screen.findByRole('alert')).toHaveTextContent('Nonuniform source PTS');
  expect(screen.getByRole('button',{name:'Export Research Overlay'})).toBeEnabled(); expect(screen.queryByRole('link')).not.toBeInTheDocument();
});
it('does not advertise orphan cancellation or unverified download', async () => {
  api.fetchVideoExport.mockResolvedValue(run({control_available:false,download_available:true})); render(<VideoExportControls fit="fit-1" initialRunId="export-1" />);
  await screen.findByRole('status'); expect(screen.queryByRole('button',{name:'Cancel Overlay Export'})).not.toBeInTheDocument(); expect(screen.queryByRole('link')).not.toBeInTheDocument();
});
it('offers guarded verification for historically completed stored packages', async () => {
  api.fetchVideoExport.mockResolvedValue(run({status:'succeeded',acceptance:'rejected',execution_verified:true,download_available:false,control_available:false,producer_source_commit:'older-commit'}));
  render(<VideoExportControls fit="fit-1" initialRunId="export-1" />);
  expect(await screen.findByRole('link',{name:'Verify Stored Overlay Package'})).toHaveAttribute('href','/export/export-1/download');
  expect(screen.queryByRole('link',{name:'Download Overlay Package'})).not.toBeInTheDocument();
  expect(screen.getByText(/Download Readiness Unverified/)).toHaveTextContent('Guarded verification can reject changed files');
  expect(screen.getByText(/Producer Commit/)).toHaveTextContent('older-commit');
});
it.each(['failed','cancelled','running'])('never offers stored verification for %s runs', async (status) => {
  api.fetchVideoExport.mockResolvedValue(run({status,execution_verified:true,download_available:false,control_available:false}));
  render(<VideoExportControls fit="fit-1" initialRunId="export-1" />);
  await screen.findByRole('status');
  expect(screen.queryByRole('link')).not.toBeInTheDocument();
});
