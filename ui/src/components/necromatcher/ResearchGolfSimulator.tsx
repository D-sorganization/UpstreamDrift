import {WorkspaceShell} from '@/components/layout/WorkspaceShell';
import {BallFlightScene3D} from '@/components/visualization/BallFlightScene3D';
import {researchIdentity, type ResearchContext} from './researchGolfContracts';
import {useResearchGolf} from './useResearchGolf';
function Context({context}:{context:ResearchContext}) {
  return <section aria-label="Verified Research Context"><p>Source: MODEL_CONTACT · Authored Replay Time: {context.recorded_time_s} s</p>
    <p>Replay: {context.replay_id} · Impact Run: {context.run_id}</p>
    <p>Contact, Numerical and Scientific Qualification: Unverified. Physical Source Time Unqualified.</p>
    <pre className="whitespace-pre-wrap break-all">{JSON.stringify(context.assumptions,null,2)}</pre>
    <details><summary>Verified Research Provenance</summary><pre className="whitespace-pre-wrap break-all">{JSON.stringify(context,null,2)}</pre></details></section>;
}
function ResearchConsole({replay,run,valid}:{replay:string;run:string;valid:boolean}) {
  const job=useResearchGolf(replay,run);
  const canPrepare=valid && job.destination==='local' && ['CONNECTED','ACCEPTED','REJECTED'].includes(job.status);
  return <div className="space-y-4 p-4"><h1>Local Research Simulation</h1>
    <p>Explicit authored-replay research only. No measured contact, physical source clock, avatar or course qualification.</p>
    <p>Selected Replay: {replay} · Saved Impact Run: {run}</p>
    {!valid && <p role="alert">Invalid research replay or impact run query. Manual preparation is unavailable for this link.</p>}
    <label>Simulator Destination<select aria-label="Simulator Destination" value={job.destination} disabled={job.busy || job.status==='ARMED'} onChange={event=>void job.replace(event.target.value)}><option value="local">Local Reference Simulator</option><option value="gspro">GSPro</option></select></label>
    {job.destination!=='local' && <p>Research model shots are available only in the local simulator.</p>}
    <button disabled={!valid || !job.canConnect || job.status==='ARMED'} onClick={()=>void job.connect()}>Connect</button>
    <p data-testid="status-badge">{job.status}</p>
    <button disabled={job.busy || !canPrepare} onClick={()=>void job.prepare()}>Prepare Research Impact</button>
    <button disabled={job.busy || job.status!=='PREPARED'} onClick={()=>void job.arm()}>Arm for Impact</button>
    <button disabled={job.busy || !['PREPARED','ARMED'].includes(job.status)} onClick={()=>void job.stop()}>Cancel</button>
    <button disabled={job.busy || job.status!=='ARMED'} onClick={()=>void job.submit()}>Trigger Impact Submit</button>
    {job.prepared && <Context context={job.prepared.research} />}
    {job.error && <p role="alert">{job.error}</p>}
    {job.trajectory && <section><h2>New Local Simulation</h2><p>This is a new local flight simulation, separate from the saved original research trajectory. Local Reference defaults and settings apply; the saved original environment is not automatically inherited. Delivery accepted does not mean scientifically qualified.</p>
      <p>{job.trajectory.samples.length} Retained Samples · {job.trajectory.provenance} · {job.trajectory.simulated_at_utc}</p>
      <div className="h-80"><BallFlightScene3D trajectories={[{modelKey:job.trajectory.shot_id,modelName:'Local Research Simulation',color:'#60a5fa',positions:job.trajectory.samples.map(sample=>sample.position_m)}]} /></div></section>}
  </div>;
}
export function ResearchGolfSimulator({search}:{search:string}) {
  const identity=researchIdentity(search);
  return <WorkspaceShell><ResearchConsole replay={identity?.replay ?? ''} run={identity?.run ?? ''} valid={identity!==null} /></WorkspaceShell>;
}
