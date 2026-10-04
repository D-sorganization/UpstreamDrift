import { beforeEach, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ReplayImpactControls } from './ReplayImpactControls';
const api=vi.hoisted(()=>({submitReplayImpact:vi.fn(),fetchReplayImpact:vi.fn(),cancelReplayImpact:vi.fn(),replayImpactDownloadUrl:vi.fn()}));
vi.mock('@/api/necromatcher',()=>api);
function declaration() {return {
  geometry:{body:'club',local_head_point_m:[0,0,0.2],local_face_normal:[1,0,0],local_face_up:[0,0,1],mass_kg:0.2,moi_kg_m2:0.001,assumption_description:'Authored effective club assumptions'},
  selection:{recorded_sample_index:3,world_to_flight_rotation:[[1,0,0],[0,1,0],[0,0,1]],world_to_flight_translation_m:[0,0,0],selection_description:'Operator-selected sample, not detected contact'},
};}
function run(status='succeeded') {return {run_id:'a'.repeat(32),replay_id:'replay',status,acceptance:'rejected',message:'Research only',blockers:[],control_available:status==='running',fraction:null,scientific_qualified:false,physical_source_time_qualified:false,execution_verified:status==='succeeded',download_available:status==='succeeded',artifactsummary:status==='succeeded'?{files:['trajectory.json','impact-receipt.json','result.json','request.json'],...declaration(),clockpolicy:'authored_simulation_seconds',summary:{carry_m:150,max_height_m:20,flight_time_s:4,landing_angle_deg:30}}:null};}
async function upload(value: unknown) {
  await userEvent.upload(screen.getByLabelText('Impact Declaration JSON'),new File([JSON.stringify(value)],'impact.json',{type:'application/json'}));
}
beforeEach(()=>{Object.values(api).forEach((mock)=>mock.mockReset());api.submitReplayImpact.mockResolvedValue(run());api.replayImpactDownloadUrl.mockReturnValue('/impact.zip');});
it('submits the complete declaration and explicit budget without inferring impact or physical time',async()=>{
  render(<ReplayImpactControls replay="replay" sampleCount={21} />);
  expect(screen.getByRole('button',{name:'Preview Research Impact'})).toBeDisabled();
  await upload(declaration());
  await userEvent.type(screen.getByLabelText('Impact Budget (s)'),'120');
  expect(screen.getByText(/Selected Replay Sample: 3/)).toBeInTheDocument();
  await userEvent.click(screen.getByRole('button',{name:'Preview Research Impact'}));
  expect(api.submitReplayImpact).toHaveBeenCalledExactlyOnceWith('replay',{...declaration(),budget_wall_s:120});
  expect(await screen.findByRole('link',{name:'Download Research Trajectory and Receipt Bundle'})).toHaveAttribute('href','/impact.zip');
  expect(screen.getByText(/Authored Seconds/)).toHaveTextContent('Unqualified');
});
it.each(['boolean mass','out-of-range sample','nonunit axis','reflection','extra field'])('rejects %s before submission',async(kind)=>{
  const value=declaration() as unknown as {geometry:Record<string,unknown>;selection:Record<string,unknown>};
  if(kind==='boolean mass') value.geometry.mass_kg=true;
  if(kind==='out-of-range sample') value.selection.recorded_sample_index=21;
  if(kind==='nonunit axis') value.geometry.local_face_normal=[2,0,0];
  if(kind==='reflection') value.selection.world_to_flight_rotation=[[-1,0,0],[0,1,0],[0,0,1]];
  if(kind==='extra field') value.geometry.source_path='C:/wrong';
  render(<ReplayImpactControls replay="replay" sampleCount={21} />);
  await upload(value);
  expect(await screen.findByRole('alert')).toBeInTheDocument();
  expect(screen.getByRole('button',{name:'Preview Research Impact'})).toBeDisabled();
  expect(api.submitReplayImpact).not.toHaveBeenCalled();
});
it('polls and cancels an owned running request',async()=>{
  api.submitReplayImpact.mockResolvedValue(run('running'));api.fetchReplayImpact.mockResolvedValue(run('running'));api.cancelReplayImpact.mockResolvedValue({...run('cancelled'),acceptance:'interrupted'});
  render(<ReplayImpactControls replay="replay" sampleCount={21} />);await upload(declaration());
  await userEvent.type(screen.getByLabelText('Impact Budget (s)'),'120');
  await userEvent.click(screen.getByRole('button',{name:'Preview Research Impact'}));
  await waitFor(()=>expect(api.fetchReplayImpact).toHaveBeenCalled());
  await userEvent.click(screen.getByRole('button',{name:'Cancel Research Impact'}));
  expect(await screen.findByRole('status')).toHaveTextContent('cancelled');
  expect(screen.queryByRole('link')).not.toBeInTheDocument();
});
it('rejects foreign or qualified server results and never offers their download',async()=>{
  api.submitReplayImpact.mockResolvedValue({...run(),replay_id:'foreign',scientific_qualified:true});
  render(<ReplayImpactControls replay="replay" sampleCount={21} />);await upload(declaration());
  await userEvent.type(screen.getByLabelText('Impact Budget (s)'),'120');await userEvent.click(screen.getByRole('button',{name:'Preview Research Impact'}));
  expect(await screen.findByRole('alert')).toBeInTheDocument();expect(screen.queryByRole('link')).not.toBeInTheDocument();
});
it('clears declarations and suppresses late results when replay changes',async()=>{
  let finish!: (value:ReturnType<typeof run>)=>void;
  api.submitReplayImpact.mockImplementation(()=>new Promise((resolve)=>{finish=resolve;}));
  const view=render(<ReplayImpactControls replay="replay" sampleCount={21} />);await upload(declaration());
  await userEvent.type(screen.getByLabelText('Impact Budget (s)'),'120');await userEvent.click(screen.getByRole('button',{name:'Preview Research Impact'}));
  view.rerender(<ReplayImpactControls replay="other" sampleCount={21} />);finish(run());
  expect(screen.getByRole('button',{name:'Preview Research Impact'})).toBeDisabled();expect(screen.queryByRole('link')).not.toBeInTheDocument();
});
it('retains the actual run sample when a different declaration is imported afterward',async()=>{
  render(<ReplayImpactControls replay="replay" sampleCount={21} />);await upload(declaration());
  await userEvent.type(screen.getByLabelText('Impact Budget (s)'),'120');await userEvent.click(screen.getByRole('button',{name:'Preview Research Impact'}));
  await screen.findByRole('link',{name:'Download Research Trajectory and Receipt Bundle'});
  const changed=declaration();changed.selection.recorded_sample_index=4;await upload(changed);
  expect(await screen.findByText(/Run Selected Replay Sample: 3/)).toBeInTheDocument();
});
it('recalls a saved verified run by exact ID without resubmission or invented assumptions',async()=>{
  api.fetchReplayImpact.mockResolvedValue(run());
  render(<ReplayImpactControls replay="replay" sampleCount={21} />);
  const button=screen.getByRole('button',{name:'Recall Research Impact Run'});
  expect(button).toBeDisabled();
  await userEvent.type(screen.getByLabelText('Saved Impact Run ID'),'bad');
  expect(button).toBeDisabled();expect(api.fetchReplayImpact).not.toHaveBeenCalled();
  await userEvent.clear(screen.getByLabelText('Saved Impact Run ID'));
  await userEvent.type(screen.getByLabelText('Saved Impact Run ID'),'a'.repeat(32));
  await userEvent.click(button);
  expect(await screen.findByRole('link',{name:'Download Research Trajectory and Receipt Bundle'})).toBeInTheDocument();
  expect(api.fetchReplayImpact).toHaveBeenCalledExactlyOnceWith('replay','a'.repeat(32));
  expect(api.submitReplayImpact).not.toHaveBeenCalled();
  expect(screen.getByText(/Stored assumptions and selected sample/)).toBeInTheDocument();
  expect(screen.getByText(/Carry: 150 m/)).toHaveTextContent('Flight Time: 4 s');
  expect(screen.getByText(/Run Selected Replay Sample: 3/)).toHaveTextContent('authored_simulation_seconds');
  expect(screen.getByRole('link',{name:'Ball Flight Viewer'})).toHaveAttribute('href','/ball-flight');
});
it.each(['clock','numeric type','missing declaration','unverified artifact'])('blocks incomplete saved %s metadata',async(kind)=>{
  const value=run();
  if(kind==='clock') value.artifactsummary!.clockpolicy='physical_seconds';
  if(kind==='numeric type') (value.artifactsummary!.summary as unknown as Record<string,unknown>).carry_m=true;
  if(kind==='missing declaration') delete (value.artifactsummary as unknown as Record<string,unknown>).selection;
  if(kind==='unverified artifact') {value.execution_verified=false;value.download_available=false;}
  api.fetchReplayImpact.mockResolvedValue(value);
  render(<ReplayImpactControls replay="replay" sampleCount={21} />);
  await userEvent.type(screen.getByLabelText('Saved Impact Run ID'),'a'.repeat(32));await userEvent.click(screen.getByRole('button',{name:'Recall Research Impact Run'}));
  expect(await screen.findByRole('alert')).toBeInTheDocument();expect(screen.queryByRole('link')).not.toBeInTheDocument();
});
