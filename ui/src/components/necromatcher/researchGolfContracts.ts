export interface ResearchContext {
  replay_id: string; run_id: string; result_sha256: string; receipt_sha256: string;
  trajectory_sha256: string; recorded_time_s: number; assumptions: Record<string, unknown>;
  qualification: {contact: 'unverified'; numerical: 'unverified'; scientific: 'unverified'};
  scientific_qualified: false; physical_source_time_qualified: false;
  replay_clock_policy: 'authored_simulation_seconds';
}
export interface ResearchPrepared {
  prepared_shot_id: string; shot_id: string; context_revision: number;
  is_armed: boolean; created_at_utc: string; research: ResearchContext;
}
export interface ResearchTrajectory {
  shot_id: string; provenance: string; simulated_at_utc: string; research: ResearchContext;
  samples: {time_s: number; position_m: number[]; velocity_mps: number[]}[];
}
export function researchIdentity(search: string) {
  const params = new URLSearchParams(search);
  const replay = params.get('replay') ?? '';
  const run = params.get('impactRun') ?? '';
  if (params.getAll('replay').length !== 1 || params.getAll('impactRun').length !== 1
    || !/^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/.test(replay) || !/^[a-f0-9]{32}$/.test(run)) return null;
  return {replay, run};
}
function record(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Invalid research record.');
  return value as Record<string, unknown>;
}
export function checkedResearch(value: unknown, replay: string, run: string): ResearchContext {
  const context = record(value), qualification = record(context.qualification);
  if (context.replay_id !== replay || context.run_id !== run
    || context.scientific_qualified !== false || context.physical_source_time_qualified !== false
    || context.replay_clock_policy !== 'authored_simulation_seconds'
    || ['contact','numerical','scientific'].some(key=>qualification[key] !== 'unverified')
    || Object.keys(qualification).length !== 3
    || typeof context.recorded_time_s !== 'number' || !Number.isFinite(context.recorded_time_s)
    || context.recorded_time_s < 0
    || ['result_sha256','receipt_sha256','trajectory_sha256'].some(key=>typeof context[key] !== 'string' || !/^sha256:[a-f0-9]{64}$/.test(context[key] as string))) throw new Error('Invalid or foreign research provenance.');
  record(context.assumptions);
  return JSON.parse(JSON.stringify(context)) as ResearchContext;
}
export function checkedPrepared(value: unknown, replay: string, run: string): ResearchPrepared {
  const response = record(value);
  if (typeof response.prepared_shot_id !== 'string' || !response.prepared_shot_id
    || typeof response.shot_id !== 'string' || !response.shot_id || response.is_armed !== false
    || response.context_revision !== 1 || typeof response.created_at_utc !== 'string') throw new Error('Invalid research preparation.');
  return {...response, research:checkedResearch(response.research,replay,run)} as unknown as ResearchPrepared;
}
export function checkedTrajectory(value: unknown, prepared: ResearchPrepared): ResearchTrajectory {
  const response = record(value);
  const context = checkedResearch(response.research,prepared.research.replay_id,prepared.research.run_id);
  if (response.shot_id !== prepared.shot_id || typeof response.provenance !== 'string' || !response.provenance
    || typeof response.simulated_at_utc !== 'string' || !response.simulated_at_utc
    || stable(context) !== stable(prepared.research)
    || !Array.isArray(response.samples) || !response.samples.length) throw new Error('Invalid local research trajectory.');
  let previous = -Infinity;
  for (const entry of response.samples) {
    const sample = record(entry);
    if (typeof sample.time_s !== 'number' || !Number.isFinite(sample.time_s) || sample.time_s < 0 || sample.time_s <= previous
      || ['position_m','velocity_mps'].some(key=>!Array.isArray(sample[key]) || (sample[key] as unknown[]).length !== 3 || !(sample[key] as unknown[]).every(x=>typeof x==='number' && Number.isFinite(x)))) throw new Error('Invalid local research samples.');
    previous = sample.time_s;
  }
  return JSON.parse(JSON.stringify(response)) as ResearchTrajectory;
}
function stable(value: unknown): string {
  if(Array.isArray(value)) return '['+value.map(stable).join(',')+']';
  if(value && typeof value==='object') return '{'+Object.entries(value).sort(([a],[b])=>a.localeCompare(b)).map(([key,item])=>JSON.stringify(key)+':'+stable(item)).join(',')+'}';
  return JSON.stringify(value);
}
