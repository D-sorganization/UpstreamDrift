import { useEffect, useMemo, useRef, useState } from 'react';
import { cancelReplayImpact, fetchReplayImpact, replayImpactDownloadUrl, submitReplayImpact, type ReplayImpactDeclaration, type ReplayImpactRun } from '@/api/necromatcher';
import { useResearchJob } from './useResearchJob';

const fileNames=['trajectory.json','impact-receipt.json','result.json','request.json'];
function exactRecord(value: unknown, fields: string[]): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)
    || Object.keys(value).sort().join('|') !== [...fields].sort().join('|')) throw new Error('Declaration must contain exactly the documented fields.');
  return value as Record<string,unknown>;
}
function vector(value: unknown): number[] {
  if (!Array.isArray(value) || value.length !== 3 || !value.every((entry) => typeof entry === 'number' && Number.isFinite(entry))) throw new Error('Vectors must contain three finite numbers.');
  return value;
}
function trimmed(value: unknown) {
  if (typeof value !== 'string' || !value.trim() || value.trim() !== value) throw new Error('Authored descriptions and body must be nonempty trimmed text.');
}
const dot=(a: number[],b: number[]) => a.reduce((sum,x,index)=>sum+x*b[index],0);
function checkedDeclaration(value: unknown, sampleCount: number): ReplayImpactDeclaration {
  const record=exactRecord(value,['geometry','selection']);
  const geometry=exactRecord(record.geometry,['body','local_head_point_m','local_face_normal','local_face_up','mass_kg','moi_kg_m2','assumption_description']);
  const selection=exactRecord(record.selection,['recorded_sample_index','world_to_flight_rotation','world_to_flight_translation_m','selection_description']);
  trimmed(geometry.body); trimmed(geometry.assumption_description); trimmed(selection.selection_description);
  vector(geometry.local_head_point_m); vector(selection.world_to_flight_translation_m);
  const normal=vector(geometry.local_face_normal), up=vector(geometry.local_face_up);
  if(Math.abs(dot(normal,normal)-1)>1e-12 || Math.abs(dot(up,up)-1)>1e-12 || Math.abs(dot(normal,up))>1e-12) throw new Error('Face axes must be unit and perpendicular; no normalization is inferred.');
  for(const key of ['mass_kg','moi_kg_m2']) if(typeof geometry[key] !== 'number' || !Number.isFinite(geometry[key]) || geometry[key] <= 0) throw new Error('Effective mass and MOI must be finite positive numbers.');
  const index=selection.recorded_sample_index;
  if(typeof index !== 'number' || !Number.isInteger(index) || index<0 || index>=sampleCount) throw new Error('Choose an existing recorded replay sample.');
  const matrix=selection.world_to_flight_rotation;
  if(!Array.isArray(matrix) || matrix.length !== 3) throw new Error('Flight rotation must be a proper 3×3 rotation.');
  const rows=matrix.map(vector);
  for(let i=0;i<3;i++) for(let j=0;j<3;j++) if(Math.abs(dot(rows[i],rows[j])-(i===j?1:0))>1e-12) throw new Error('Flight rotation must be orthonormal.');
  const [a,b,c]=rows;
  const determinant=a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0]);
  if(Math.abs(determinant-1)>1e-12) throw new Error('Flight rotation must be proper, without reflection.');
  return record as unknown as ReplayImpactDeclaration;
}
function checkedRun(value: ReplayImpactRun, replay: string, sampleCount: number) {
  if(value.replay_id !== replay || !/^[a-f0-9]{32}$/.test(value.run_id)
    || value.scientific_qualified !== false || value.physical_source_time_qualified !== false
    || !['pending','running','succeeded','failed','cancelled'].includes(value.status)
    || !['partial','interrupted','rejected'].includes(value.acceptance)
    || typeof value.execution_verified !== 'boolean' || typeof value.download_available !== 'boolean'
    || typeof value.control_available !== 'boolean' || value.fraction !== null
    || typeof value.message !== 'string' || !Array.isArray(value.blockers) || !value.blockers.every((x)=>typeof x==='string')) throw new Error('Impact run identity or qualification is inconsistent.');
  if(value.download_available && (value.status !== 'succeeded' || !value.execution_verified
    || !value.artifactsummary || !Array.isArray(value.artifactsummary.files)
    || [...value.artifactsummary.files].sort().join('|') !== [...fileNames].sort().join('|'))) throw new Error('Impact bundle is not verified complete.');
  if(value.artifactsummary) {
    if(!value.execution_verified || value.status !== 'succeeded') throw new Error('Saved impact artifact is not verified.');
    checkedDeclaration({geometry:value.artifactsummary.geometry,selection:value.artifactsummary.selection},sampleCount);
    const summary=exactRecord(value.artifactsummary.summary,['carry_m','max_height_m','flight_time_s','landing_angle_deg']);
    if(value.artifactsummary.clockpolicy !== 'authored_simulation_seconds'
      || !Object.values(summary).every((x)=>typeof x==='number' && Number.isFinite(x))) throw new Error('Saved impact summary or authored clock is invalid.');
  }
  // The shared job hook names its identity source_fit_id; this adapter binds it to the replay, never a fit.
  return {...value,source_fit_id:value.replay_id};
}

function ImpactForm({replay,sampleCount}: {replay: string; sampleCount: number}) {
  const [declaration,setDeclaration]=useState<ReplayImpactDeclaration | null>(null);
  const [budget,setBudget]=useState('');
  const [error,setError]=useState('');
  const [savedRun,setSavedRun]=useState('');
  const [initialRunId,setInitialRunId]=useState<string>();
  const [submitted,setSubmitted]=useState<(ReplayImpactDeclaration & {budget_wall_s:number}) | null>(null);
  const pending=useRef<(ReplayImpactDeclaration & {budget_wall_s:number}) | null>(null);
  const generation=useRef({value:0});
  useEffect(()=>{const epoch=generation.current;return ()=>{epoch.value++;};},[]);
  const api=useMemo(()=>({
    submit:async(id: string,payload:ReplayImpactDeclaration & {budget_wall_s:number})=>checkedRun(await submitReplayImpact(id,payload),replay,sampleCount),
    view:async(run: string)=>checkedRun(await fetchReplayImpact(replay,run),replay,sampleCount),
    cancel:async(run: string)=>checkedRun(await cancelReplayImpact(replay,run),replay,sampleCount),
  }),[replay,sampleCount]);
  const job=useResearchJob({fit:replay,api,initialRunId,onRun:()=>setSubmitted(pending.current)});
  const active=Boolean(job.run && ['pending','running'].includes(job.run.status));
  const disabled=active || job.submitting;
  const invalidBudget=!budget.trim() || !Number.isFinite(Number(budget)) || Number(budget)<=0 || Number(budget)>600;
  async function load(file: File | undefined) {
    const token=++generation.current.value;
    setDeclaration(null);setError('');
    if(!file) return;
    try {
      if(file.size>1024*1024) throw new Error('Impact declaration exceeds 1 MiB.');
      const checked=checkedDeclaration(JSON.parse(await file.text()),sampleCount);
      if(token===generation.current.value) setDeclaration(checked);
    } catch(reason) {if(token===generation.current.value) setError(reason instanceof Error ? reason.message : 'Declaration could not be read.');}
  }
  return <section aria-label="Research Impact Preview" className="space-y-2 border-t border-gray-600 pt-3">
    <h4 className="font-semibold">Research Impact Preview</h4>
    <p className="text-xs text-orange-300">Authored Seconds · Scientific and Physical Source Time Unqualified. Contact is operator-selected; no impact event or club speed is inferred.</p>
    <label className="block">Saved Impact Run ID<input value={savedRun} disabled={disabled} onChange={(event)=>setSavedRun(event.target.value)} /></label>
    <button type="button" disabled={disabled || !/^[a-f0-9]{32}$/.test(savedRun)} onClick={()=>{setSubmitted(null);setInitialRunId(savedRun);}}>Recall Research Impact Run</button>
    <label className="block">Impact Declaration JSON<input type="file" accept=".json,application/json" disabled={disabled} onChange={(event)=>void load(event.target.files?.[0])} /></label>
    <p className="text-xs">Import exact geometry and selection records: body-local head point, perpendicular unit face axes, effective mass/MOI, proper world-to-flight transform, recorded sample and authored descriptions. Metres, kg and kg·m² are required.</p>
    <label className="block">Impact Budget (s)<input type="number" min="0" max="600" value={budget} disabled={disabled} onChange={(event)=>setBudget(event.target.value)} /></label>
    {declaration && <div className="text-xs"><p>Replay: {replay} · Selected Replay Sample: {declaration.selection.recorded_sample_index}</p><p>{declaration.geometry.assumption_description}</p><p>{declaration.selection.selection_description}</p><p>Body: {declaration.geometry.body} · Effective Mass: {declaration.geometry.mass_kg} kg · MOI: {declaration.geometry.moi_kg_m2} kg·m²</p></div>}
    <button type="button" disabled={disabled || !declaration || invalidBudget} onClick={()=>{if(declaration && !invalidBudget) {pending.current={...declaration,budget_wall_s:Number(budget)};void job.start(pending.current);}}}>Preview Research Impact</button>
    {job.run && <div><p role="status">{job.run.status} · {job.run.acceptance} · {job.run.message}</p><p>Run: {job.run.run_id} · Replay: {job.run.replay_id}</p>{job.run.blockers.map((item)=><p key={item}>{item}</p>)}</div>}
    {job.run && submitted && !job.run.artifactsummary && <div className="text-xs"><p>Run Selected Replay Sample: {submitted.selection.recorded_sample_index} · Requested Budget: {submitted.budget_wall_s} s</p><p>Run Geometry Assumptions: {submitted.geometry.assumption_description}</p><p>Run Selection Assumptions: {submitted.selection.selection_description}</p></div>}
    {job.run?.artifactsummary && <SavedImpactSummary artifact={job.run.artifactsummary} />}
    {job.run && !submitted && <p className="text-xs">Stored assumptions and selected sample are retained in the verified receipt bundle; the current declaration is not the saved request.</p>}
    {active && job.controlAvailable && <button type="button" onClick={()=>void job.cancel()}>Cancel Research Impact</button>}
    {job.run?.status==='succeeded' && job.run.execution_verified && job.run.download_available && <a href={replayImpactDownloadUrl(replay,job.run.run_id)}>Download Research Trajectory and Receipt Bundle</a>}
    {job.run?.status==='succeeded' && job.run.execution_verified && job.run.download_available && <a href={`/tools/golf-simulator?replay=${encodeURIComponent(replay)}&impactRun=${encodeURIComponent(job.run.run_id)}`}>Open Local Research Simulation</a>}
    {job.run?.status==='succeeded' && job.run.execution_verified && job.run.download_available && <p className="text-xs">Extract trajectory.json from the ZIP and use Import Trajectory in the <a href="/ball-flight">Ball Flight Viewer</a>. Retain impact-receipt.json for authored clock, assumptions and qualification context.</p>}
    {(error || job.error) && <p role="alert">{error || job.error}</p>}
  </section>;
}
function SavedImpactSummary({artifact}: {artifact: NonNullable<ReplayImpactRun['artifactsummary']>}) {
  return <div className="text-xs space-y-1">
    <p>Run Selected Replay Sample: {artifact.selection.recorded_sample_index} · Clock: {artifact.clockpolicy}</p>
    <p>Run Geometry Assumptions: {artifact.geometry.assumption_description}</p><p>Run Selection Assumptions: {artifact.selection.selection_description}</p>
    <p>Saved Body: {artifact.geometry.body} · Effective Mass: {artifact.geometry.mass_kg} kg · MOI: {artifact.geometry.moi_kg_m2} kg·m²</p>
    <p>Carry: {artifact.summary.carry_m} m · Maximum Height: {artifact.summary.max_height_m} m · Flight Time: {artifact.summary.flight_time_s} s · Landing Angle: {artifact.summary.landing_angle_deg}°</p>
  </div>;
}
export function ReplayImpactControls(props: {replay: string; sampleCount: number}) {
  return <ImpactForm key={`${props.replay}/${props.sampleCount}`} {...props} />;
}
