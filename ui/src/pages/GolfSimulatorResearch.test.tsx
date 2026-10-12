import { beforeEach, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { GolfSimulatorPage } from './GolfSimulator';
import {ResearchGolfSimulator} from '@/components/necromatcher/ResearchGolfSimulator';
const fetch = vi.hoisted(()=>vi.fn());
vi.mock('@/api/fetch',()=>({apiFetch:fetch}));
vi.mock('@/components/visualization/BallFlightScene3D',()=>({BallFlightScene3D:({trajectories}:{trajectories:unknown})=><pre data-testid="retained-trajectory">{JSON.stringify(trajectories)}</pre>}));
const run='a'.repeat(32);
const research={scientific_qualified:false,physical_source_time_qualified:false,replay_clock_policy:'authored_simulation_seconds',replay_id:'replay',run_id:run,result_sha256:'sha256:'+ '1'.repeat(64),receipt_sha256:'sha256:'+ '2'.repeat(64),trajectory_sha256:'sha256:'+ '3'.repeat(64),recorded_time_s:0.25,assumptions:{geometry:'Authored club',selection:'Declared sample'},qualification:{contact:'unverified',numerical:'unverified',scientific:'unverified'}};
beforeEach(()=>{vi.spyOn(crypto,'randomUUID').mockReturnValue('11111111-1111-1111-1111-111111111111');fetch.mockReset();window.history.replaceState({},'',`/tools/golf-simulator?replay=replay&impactRun=${run}`);fetch.mockImplementation(async(path:string)=>{
 if(path.endsWith('/session')) return {state:'idle',session_id:'web-session'};
 if(path.endsWith('/prepare-research-impact')) return {prepared_shot_id:'prep',shot_id:'research-11111111-1111-1111-1111-111111111111',context_revision:1,is_armed:false,created_at_utc:'now',research};
 if(path.endsWith('/arm')) return {arm_token:'token'};
 if(path.endsWith('/submit')) return {state:'confirmed_accepted',shot_id:'research-11111111-1111-1111-1111-111111111111'};
 if(path.endsWith('/local-trajectory')) return {shot_id:'research-11111111-1111-1111-1111-111111111111',provenance:'local_reference',simulated_at_utc:'now',samples:[{time_s:0,position_m:[0,0,0],velocity_mps:[20,0,5]},{time_s:1,position_m:[20,0,4],velocity_mps:[19,0,3]}],research};
 return {destinations:[]};
});});
it('requires explicit local preparation and displays actual local samples without automatic submission',async()=>{
 render(<GolfSimulatorPage />);
 expect(screen.getByRole('button',{name:'Prepare Research Impact'})).toBeDisabled();
 expect(fetch.mock.calls.some(([path])=>String(path).includes('/shot/'))).toBe(false);
 fireEvent.click(screen.getByRole('button',{name:'Connect'}));
 await waitFor(()=>expect(screen.getByRole('button',{name:'Prepare Research Impact'})).toBeEnabled());
 fireEvent.click(screen.getByRole('button',{name:'Prepare Research Impact'}));
 await screen.findByText(/Authored Replay Time: 0.25 s/);
 const call=fetch.mock.calls.find(([path])=>String(path).endsWith('/prepare-research-impact'))!;
 expect(JSON.parse(call[1].body)).toMatchObject({replay_id:'replay',run_id:run,context_revision:1,session_id:'web-session'});
 expect(fetch.mock.calls.some(([path])=>String(path).endsWith('/submit'))).toBe(false);
 fireEvent.click(screen.getByRole('button',{name:'Arm for Impact'}));
 await waitFor(()=>expect(screen.getByRole('button',{name:'Trigger Impact Submit'})).toBeEnabled());
 fireEvent.click(screen.getByRole('button',{name:'Trigger Impact Submit'}));
 expect(await screen.findByTestId('retained-trajectory')).toHaveTextContent('[[0,0,0],[20,0,4]]');
 expect(screen.getByText(/This is a new local flight/)).toHaveTextContent('saved original');
});
it('malformed research queries never fall back to manual preparation',()=>{
 window.history.replaceState({},'','/tools/golf-simulator?replay=replay&impactRun=wrong');render(<GolfSimulatorPage />);
 expect(screen.queryByRole('button',{name:'Prepare Shot'})).not.toBeInTheDocument();
 expect(screen.getByRole('alert')).toHaveTextContent('Invalid research');
});
it.each(['foreign','qualified','clock','hash'])('rejects %s context and cancels the owned preparation',async(kind)=>{
 const bad={...research};
 if(kind==='foreign') bad.run_id='b'.repeat(32);
 if(kind==='qualified') bad.scientific_qualified=true;
 if(kind==='clock') bad.replay_clock_policy='physical_seconds';
 if(kind==='hash') bad.result_sha256='bad';
 fetch.mockImplementation(async(path:string)=>path.endsWith('/session')?{state:'idle',session_id:'web-session'}:path.endsWith('/prepare-research-impact')?{prepared_shot_id:'prep',shot_id:'research-11111111-1111-1111-1111-111111111111',context_revision:1,is_armed:false,created_at_utc:'now',research:bad}:{destinations:[]});
 render(<GolfSimulatorPage />);fireEvent.click(screen.getByRole('button',{name:'Connect'}));
 await waitFor(()=>expect(screen.getByRole('button',{name:'Prepare Research Impact'})).toBeEnabled());fireEvent.click(screen.getByRole('button',{name:'Prepare Research Impact'}));
 expect(await screen.findByRole('alert')).toHaveTextContent('research');
 expect(screen.getByRole('button',{name:'Arm for Impact'})).toBeDisabled();
 expect(fetch.mock.calls.some(([path,options])=>String(path).endsWith('/cancel') && JSON.parse(options.body).prepared_shot_id==='prep')).toBe(true);
});
it('cancels a late preparation for a replaced replay without exposing it',async()=>{
 let finish!:(value:unknown)=>void;
 fetch.mockImplementation(async(path:string)=>path.endsWith('/session')?{state:'idle',session_id:'web-session'}:path.endsWith('/prepare-research-impact')?new Promise(resolve=>{finish=resolve;}):{destinations:[]});
 const view=render(<ResearchGolfSimulator search={`?replay=replay&impactRun=${run}`} />);
 fireEvent.click(screen.getByRole('button',{name:'Connect'}));await waitFor(()=>expect(screen.getByRole('button',{name:'Prepare Research Impact'})).toBeEnabled());
 fireEvent.click(screen.getByRole('button',{name:'Prepare Research Impact'}));
 view.rerender(<ResearchGolfSimulator search={`?replay=other&impactRun=${run}`} />);
 finish({prepared_shot_id:'late',shot_id:'research-11111111-1111-1111-1111-111111111111',context_revision:1,is_armed:false,created_at_utc:'now',research});
 await waitFor(()=>expect(fetch.mock.calls.some(([path,options])=>String(path).endsWith('/cancel') && JSON.parse(options.body).prepared_shot_id==='late')).toBe(true));
 expect(screen.queryByLabelText('Verified Research Context')).not.toBeInTheDocument();expect(screen.getByRole('button',{name:'Arm for Impact'})).toBeDisabled();
});
it('cancels an existing preparation before replacing destination and disables external research',async()=>{
 render(<GolfSimulatorPage />);fireEvent.click(screen.getByRole('button',{name:'Connect'}));await waitFor(()=>expect(screen.getByRole('button',{name:'Prepare Research Impact'})).toBeEnabled());
 fireEvent.click(screen.getByRole('button',{name:'Prepare Research Impact'}));await screen.findByLabelText('Verified Research Context');
 fireEvent.change(screen.getByLabelText('Simulator Destination'),{target:{value:'gspro'}});
 await waitFor(()=>expect(screen.getByLabelText('Simulator Destination')).toHaveValue('gspro'));
 expect(screen.queryByLabelText('Verified Research Context')).not.toBeInTheDocument();
 expect(screen.getByRole('button',{name:'Prepare Research Impact'})).toBeDisabled();
 expect(fetch.mock.calls.some(([path])=>String(path).endsWith('/cancel'))).toBe(true);
});
it('failed research admission leaves no manual fallback or arm action',async()=>{
 fetch.mockImplementation(async(path:string)=>{if(path.endsWith('/session')) return {state:'idle',session_id:'web-session'};if(path.endsWith('/prepare-research-impact')) throw new Error('Saved evidence changed');return {destinations:[]};});
 render(<GolfSimulatorPage />);fireEvent.click(screen.getByRole('button',{name:'Connect'}));await waitFor(()=>expect(screen.getByRole('button',{name:'Prepare Research Impact'})).toBeEnabled());fireEvent.click(screen.getByRole('button',{name:'Prepare Research Impact'}));
 expect(await screen.findByRole('alert')).toHaveTextContent('Saved evidence changed');expect(screen.queryByRole('button',{name:'Prepare Shot'})).not.toBeInTheDocument();expect(screen.getByRole('button',{name:'Arm for Impact'})).toBeDisabled();
});
it('query replacement waits for owned cancellation before connecting a new identity',async()=>{
 let release!:(value:unknown)=>void;
 const normal=fetch.getMockImplementation()!;
 fetch.mockImplementation((path:string,...args:unknown[])=>path.endsWith('/cancel')?new Promise(resolve=>{release=resolve;}):normal(path,...args));
 const view=render(<ResearchGolfSimulator search={`?replay=replay&impactRun=${run}`} />);
 fireEvent.click(screen.getByRole('button',{name:'Connect'}));await waitFor(()=>expect(screen.getByRole('button',{name:'Prepare Research Impact'})).toBeEnabled());fireEvent.click(screen.getByRole('button',{name:'Prepare Research Impact'}));await screen.findByLabelText('Verified Research Context');
 view.rerender(<ResearchGolfSimulator search={`?replay=other&impactRun=${run}`} />);
 expect(screen.getByRole('button',{name:'Connect'})).toBeDisabled();
 await waitFor(()=>expect(release).toBeDefined());release({});
 await waitFor(()=>expect(screen.getByRole('button',{name:'Connect'})).toBeEnabled());
 expect(fetch.mock.calls.filter(([path])=>String(path).endsWith('/session'))).toHaveLength(1);
});
it('failed query turnover cancellation blocks a new connection and exposes the failure',async()=>{
 const normal=fetch.getMockImplementation()!;
 fetch.mockImplementation((path:string,...args:unknown[])=>path.endsWith('/cancel')?Promise.reject(new Error('Cancellation refused')):normal(path,...args));
 const view=render(<ResearchGolfSimulator search={`?replay=replay&impactRun=${run}`} />);
 fireEvent.click(screen.getByRole('button',{name:'Connect'}));await waitFor(()=>expect(screen.getByRole('button',{name:'Prepare Research Impact'})).toBeEnabled());fireEvent.click(screen.getByRole('button',{name:'Prepare Research Impact'}));await screen.findByLabelText('Verified Research Context');
 view.rerender(<ResearchGolfSimulator search={`?replay=other&impactRun=${run}`} />);
 expect(await screen.findByRole('alert')).toHaveTextContent('Cancellation refused');
 expect(screen.getByRole('button',{name:'Connect'})).toBeDisabled();expect(screen.getByRole('button',{name:'Prepare Research Impact'})).toBeDisabled();
 expect(fetch.mock.calls.filter(([path])=>String(path).endsWith('/session'))).toHaveLength(1);
});


