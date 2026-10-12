import { useResearchJob } from './useResearchJob';
import { ShaftEvidenceInput } from './ShaftEvidenceInput';
import { SourceScopeInput, SourceScopeSummary } from './SourceScopeInput';
import { useEffect, useState } from 'react';
import { fetchRefitPlan, submitRefit, fetchRefit, cancelRefit, type RefitPlan, type RefitRun, type RefitOptions, type ImageFitRecipe, type ScheduledFitConstraintRecipe, type SourceFitScope } from '@/api/necromatcher';

const refitApi = {submit: submitRefit, view: fetchRefit, cancel: cancelRefit};

type Props = {fit: string; onStored: () => void; initialRunId?: string; onRun?: (run: string) => void};
const errorText = (error: unknown) => error instanceof Error ? error.message : 'The research job could not be reached.';
const parseNumbers = (text: string) => text.trim() ? text.split(',').map((value) => value.trim() ? Number(value) : NaN) : [];
const defaultConfig: ImageFitRecipe = {max_iterations: 100, prior_weight: 0.1, smoothness_weight: 0.01, closure_weight: 100};
const isScheduled = (options: ImageFitRecipe['constraint_options']): options is ScheduledFitConstraintRecipe => Boolean(options && 'schedule' in options);
const sourceSeconds = ([numerator, denominator]: [number, number]) => numerator / denominator;
const field = 'block w-full rounded bg-gray-900 border border-gray-600 p-2 mt-1';

export function RefitControls(props: Props) {
  return <RefitForm key={props.fit} {...props} />;
}

function RefitForm({fit, onStored, initialRunId, onRun}: Props) {
  const [plan, setPlan] = useState<RefitPlan | null>(null);
  const [id, setId] = useState('');
  const [frames, setFrames] = useState('');
  const [scales, setScales] = useState('');
  const [options, setOptions] = useState({knot_count: 2, unknown_visibility_weight: 0.5, budget_wall_s: 600});
  const [config, setConfig] = useState<ImageFitRecipe>(defaultConfig);
  const [initialization, setInitialization] = useState<NonNullable<RefitOptions['initialization_source']>>('sampled_parent');
  const [planError, setPlanError] = useState('');
  const [shaftEvidence, setShaftEvidence] = useState<Record<string, unknown> | null>(null);
  const [shaftBlocked, setShaftBlocked] = useState(false);
  const [sourceScope, setSourceScope] = useState<SourceFitScope | null>(null);
  const [scopeBlocked, setScopeBlocked] = useState(false);
  const job = useResearchJob<RefitRun, RefitOptions & {new_fit_id: string}>({fit, onCompleted: onStored, initialRunId, onRun, api: refitApi});
  const {run, submitting, controlAvailable} = job;
  useEffect(() => {
    let active = true;
    fetchRefitPlan(fit).then((value) => {
      if (!active || value.source_fit_id !== fit) return;
      setPlan(value);
      const previous = value.recorded_options;
      const indices = previous?.frame_indices ?? value.frame_indices.filter((_, i, all) => i === 0 || i === all.length - 1 || i % Math.max(1, Math.ceil(all.length / 60)) === 0);
      setFrames(indices.join(', '));
      setScales(previous?.coordinate_scales.join(', ') ?? '');
      setConfig(structuredClone(value.baseline_config ?? previous?.config ?? defaultConfig));
      setOptions(previous ? {knot_count: previous.knot_count, unknown_visibility_weight: previous.unknown_visibility_weight, budget_wall_s: previous.budget_wall_s} : {knot_count: Math.min(12, indices.length), unknown_visibility_weight: 0.5, budget_wall_s: 600});
    }).catch((reason) => {if (active) setPlanError(errorText(reason));});
    return () => {active = false;};
  }, [fit]);
  const indices = parseNumbers(frames);
  const priorScales = parseNumbers(scales);
  const savedSpline = plan?.preserved_spline;
  const resume = initialization === 'preserved_spline';
  const restricted = initialization === 'restricted_spline';
  const knotCount = resume ? savedSpline?.knot_count ?? 0 : options.knot_count;
  const interval = savedSpline?.source_interval;
  const ranges = config.coordinate_bounds?.length ?? 0;
  const constraints = config.constraint_options;
  const schedule = isScheduled(constraints) ? constraints.schedule : null;
  const pins = isScheduled(constraints) ? [...new Set(constraints.schedule.phases.flatMap((phase) => phase.pinned_spheres))] : constraints?.pinned_spheres ?? [];
  const effectiveScope = sourceScope ?? plan?.source_scope;
  const scopeValid = (!restricted || Boolean(effectiveScope && indices[0] === effectiveScope.first_frame && indices[indices.length - 1] === effectiveScope.end_exclusive_frame - 1)) && (!effectiveScope || indices.every((index) => index >= effectiveScope.first_frame && index < effectiveScope.end_exclusive_frame));
  const valid = plan && id.trim() && indices.length >= 2 && indices.every((n, i) => Number.isInteger(n) && plan.frame_indices.includes(n) && (i === 0 || n > indices[i - 1])) && priorScales.length === plan.coordinate_order.length && priorScales.every((n) => Number.isFinite(n) && n > 0) && knotCount >= 2 && knotCount <= indices.length && (!resume || (savedSpline?.available && interval && indices[0] === plan.frame_indices[0] && indices[indices.length - 1] === plan.frame_indices[plan.frame_indices.length - 1]));
  const busy = submitting || Boolean(run && ['pending', 'running'].includes(run.status));
  const error = planError || job.error;
  return <section aria-label="Research Refit" className="space-y-3 border-t border-gray-600 pt-4">
    <h3 className="font-semibold">Research Refit</h3><p className="text-sm">Source Version: {fit}. A new version preserves the original. Physical Time, Camera and Dynamics Remain Unqualified.</p>
    {!plan && !error && <p>Loading Source Choices…</p>}
    {plan && <form onSubmit={(event) => {event.preventDefault(); if (!valid || !scopeValid || shaftBlocked || scopeBlocked) return; void job.start({...options, knot_count: knotCount, config: {...config, ...(resume || restricted ? {initialization_policy: 'strict' as const} : {})}, operation: restricted ? 'restrict_initialization' : 'fit', initialization_source: initialization, new_fit_id: id.trim(), frame_indices: indices, coordinate_scales: priorScales, ...(shaftEvidence ? {shaft_evidence: shaftEvidence} : {}), ...(sourceScope ? {source_scope: sourceScope} : {})});}}>
      <fieldset disabled={busy} className="space-y-2">
        <label className="block">New Fit Version<input className={field} required pattern="[A-Za-z0-9][A-Za-z0-9_.-]{0,127}" value={id} onChange={(event) => setId(event.target.value)} /></label>
        <label className="block">Initialization Source<select className={field} value={initialization} onChange={(event) => setInitialization(event.target.value as typeof initialization)}>
          <option value="sampled_parent">Sample Parent Poses</option>
          <option value="preserved_spline" disabled={!savedSpline?.available}>Resume Saved Spline</option>
          <option value="restricted_spline" disabled={!savedSpline?.available}>Lossless Restricted Seed</option>
        </select></label>
        {!savedSpline?.available && savedSpline?.reason && <p className="text-xs">Saved Spline Unavailable: {savedSpline.reason}</p>}
        <p className="text-xs">{ranges} Authored Range{ranges === 1 ? '' : 's'}; {pins.filter((name) => name === 'heel_r' || name === 'heel_l').length} Authored Heel Pins. Source Timing Remains Unknown. Saved Contact and Range Hypotheses Are Retained.</p>
        {schedule && <p className="text-xs">{schedule.phases.length} Authored Contact Phases; Review Source Interval: {sourceSeconds(schedule.phases[0].start_pts)} to {sourceSeconds(schedule.phases[schedule.phases.length - 1].end_pts)}. Authored Contact Hypothesis; Phase Pins: {pins.join(', ') || 'None'}. Contact Remains Unqualified.</p>}
        {resume && interval && <p className="text-xs">Full Saved Source Interval: {interval[0]} to {interval[1]}. Preserved Knot Clock and Strict Initialization; Include Both Endpoint Frames.</p>}
        {restricted && <p className="text-xs">Create an Unoptimized Research Seed with the Exact Saved Curve Inside the Reviewed Window. Include Both Reviewed Endpoints and Enter the Retained Knot Count. Contact and Shaft Evidence Must Already Match This Window; They Are Not Changed Automatically.</p>}
        <label className="block">Source Frame Indices<textarea className={field} value={frames} onChange={(event) => setFrames(event.target.value)} /></label>
        <p className="text-xs">Coordinate Order: {plan.coordinate_order.map((name, i) => `${name} (${plan.coordinate_units[i]})`).join(', ')}</p>
        <label className="block">Coordinate Prior Scales<textarea className={field} value={scales} onChange={(event) => setScales(event.target.value)} /></label>
        <p className="text-xs">Enter one positive scale per coordinate in the units above. Scales are fitting priors, not measured anatomy.</p>
        {([
          ['knot_count', 'Spline Knots', 2, 1],
          ['budget_wall_s', 'Wall Budget (Seconds)', 0.01, 'any'],
          ['unknown_visibility_weight', 'Unknown Visibility Weight', 0, 'any'],
        ] as const).map(([key, label, min, step]) => <label className="block" key={key}>{label}<input className={field} type="number" required min={min} step={step} max={key === 'unknown_visibility_weight' ? 1 : undefined} disabled={key === 'knot_count' && resume} value={key === 'knot_count' ? knotCount : options[key]} onChange={(event) => setOptions({...options, [key]: Number(event.target.value)})} /></label>)}
        {([
          ['max_iterations', 'Evaluation Budget', 1, 1], ['prior_weight', 'Pose Prior Weight', 0, 'any'],
          ['smoothness_weight', 'Smoothness Weight', 0, 'any'], ['closure_weight', 'Grip Closure Weight', 0, 'any'],
        ] as const).map(([key, label, min, step]) => <label className="block" key={key}>{label}<input className={field} type="number" required min={min} step={step} value={config[key]} onChange={(event) => setConfig({...config, [key]: Number(event.target.value)})} /></label>)}
        <ShaftEvidenceInput disabled={busy} onChange={(record, blocked) => {setShaftEvidence(record); setShaftBlocked(blocked);}} />
        <SourceScopeInput fit={fit} disabled={busy} inherited={plan.source_scope} binding={plan.source_scope_binding} onChange={(record, blocked) => {setSourceScope(record); setScopeBlocked(blocked);}} />
        {effectiveScope && <p className="text-xs">Selected Fit Frames: {indices[0]} to {indices[indices.length - 1]}. Every Selected Frame Must Be Inside the Reviewed Window.</p>}
        <button className="rounded bg-blue-700 px-3 py-2 disabled:opacity-50" disabled={!valid || !scopeValid || busy || shaftBlocked || scopeBlocked} type="submit">{restricted ? 'Create Lossless Restricted Seed' : 'Start Research Refit'}</button>
      </fieldset>
    </form>}
    {run && <div><p role="status">{run.status} · {run.acceptance} · {run.message}</p><p className="text-xs">Run: {run.run_id} · New Version: {run.new_fit_id}</p>{run.blockers.map((reason) => <p className="text-xs text-orange-300" key={reason}>{reason}</p>)}</div>}
    <SourceScopeSummary scope={run?.source_fit_scope} binding={run?.source_fit_scope_binding} />
    {run && controlAvailable && ['pending', 'running'].includes(run.status) && <button type="button" className="rounded border p-2" onClick={() => void job.cancel()}>Cancel Research Refit</button>}
    {error && <p role="alert">{error}</p>}
  </section>;
}
