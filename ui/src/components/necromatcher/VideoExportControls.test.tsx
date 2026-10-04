import { beforeEach, expect, it, vi } from 'vitest';
import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { VideoExportControls } from './VideoExportControls';
const api = vi.hoisted(() => ({submitVideoExport: vi.fn(), fetchVideoExport: vi.fn(), cancelVideoExport: vi.fn(), videoExportDownloadUrl: (run: string) => `/export/${run}/download`}));
vi.mock('@/api/necromatcher', () => api);
const run = (values = {}) => ({run_id:'export-1',source_fit_id:'fit-1',status:'running',acceptance:'partial',qualification:'monocular_research_hypothesis',blockers:[],message:'Encoding',fraction:null,control_available:true,execution_started:true,execution_verified:false,download_available:false,...values});
beforeEach(() => {api.submitVideoExport.mockReset(); api.fetchVideoExport.mockReset(); api.cancelVideoExport.mockReset();});
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
it('submits the opt-in force layer only after it is enabled', async () => {
  api.submitVideoExport.mockResolvedValue(run()); api.fetchVideoExport.mockResolvedValue(run());
  render(<VideoExportControls fit="fit-1" />); const user = userEvent.setup();
  expect(screen.queryByText(/not measured forces/)).not.toBeInTheDocument();
  await user.click(screen.getByRole('checkbox',{name:/Draw force and torque glyphs/}));
  await user.click(screen.getByRole('checkbox',{name:/contact/})); await user.click(screen.getByRole('checkbox',{name:/Shade body segments/}));
  expect(screen.getByText(/not measured forces/)).toBeInTheDocument();
  await user.click(screen.getByRole('button',{name:'Export Research Overlay'}));
  expect(api.submitVideoExport).toHaveBeenCalledExactlyOnceWith('fit-1',{enabled:true,kinds:['joint_reaction','contact'],scale:1,segment_shading:true});
});
