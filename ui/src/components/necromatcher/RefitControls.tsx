import { useEffect, useRef, useState } from 'react';
import { fetchRefitPlan, submitRefit, fetchRefit, cancelRefit, type RefitPlan, type RefitRun, type RefitOptions } from '@/api/necromatcher';

type Props = {fit: string; onStored: () => void; initialRunId?: string; onRun?: (run: string) => void};
const errorText = (error: unknown) => error instanceof Error ? error.message : 'The research job could not be reached.';
const parseNumbers = (text: string) => text.trim() ? text.split(',').map((value) => value.trim() ? Number(value) : NaN) : [];
const field = 'block w-full rounded bg-gray-900 border border-gray-600 p-2 mt-1';

export function RefitControls(props: Props) {
  return <RefitForm key={props.fit} {...props} />;
}

function RefitForm({fit, onStored, initialRunId, onRun}: Props) {
  const [plan, setPlan] = useState<RefitPlan | null>(null);
  const [id, setId] = useState('');
  const [frames, setFrames] = useState('');
  const [scales, setScales] = useState('');
  const [options, setOptions] = useState({knot_count: 2, max_iterations: 100, prior_weight: 0.1, smoothness_weight: 0.01, closure_weight: 100, unknown_visibility_weight: 0.5, budget_wall_s: 600});
  const [run, setRun] = useState<RefitRun | null>(null);
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const mounted = useRef(true);
  const notified = useRef('');
  useEffect(() => {
    mounted.current = true;
    fetchRefitPlan(fit).then((value) => {
      if (!mounted.current || value.source_fit_id !== fit) return;
      setPlan(value);
      const previous = value.recorded_options;
      const indices = previous?.frame_indices ?? value.frame_indices.filter((_, i, all) => i === 0 || i === all.length - 1 || i % Math.max(1, Math.ceil(all.length / 60)) === 0);
      setFrames(indices.join(', '));
      setScales(previous?.coordinate_scales.join(', ') ?? '');
      setOptions(previous ? {knot_count: previous.knot_count, ...previous.config, unknown_visibility_weight: previous.unknown_visibility_weight, budget_wall_s: previous.budget_wall_s} : {knot_count: Math.min(12, indices.length), max_iterations: 100, prior_weight: 0.1, smoothness_weight: 0.01, closure_weight: 100, unknown_visibility_weight: 0.5, budget_wall_s: 600});
    }).catch((reason) => {if (mounted.current) setError(errorText(reason));});
    return () => {mounted.current = false;};
  }, [fit]);
  useEffect(() => {
    if (!initialRunId) return;
    let active = true;
    fetchRefit(initialRunId).then((value) => {
      if (!active) return;
      if (value.source_fit_id !== fit) {setError('This run belongs to another source version.'); return;}
      if (value.status === 'succeeded') notified.current = value.run_id;
      setRun(value);
    }).catch((reason) => {if (active) setError(errorText(reason));});
    return () => {active = false;};
  }, [initialRunId, fit]);
  useEffect(() => {
    if (run?.status === 'succeeded' && notified.current !== run.run_id) {
      notified.current = run.run_id;
      onStored();
    }
  }, [run, onStored]);
  const activeRunId = run?.run_id;
  const activeStatus = run?.status;
  const controlAvailable = run?.control_available !== false;
  useEffect(() => {
    if (!activeRunId || !activeStatus || !controlAvailable || !['pending', 'running'].includes(activeStatus)) return;
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const value = await fetchRefit(activeRunId!);
        if (!active || value.source_fit_id !== fit || value.run_id !== activeRunId) return;
        setRun(value);
        setError('');
        if (['pending', 'running'].includes(value.status)) timer = setTimeout(poll, 500);
      } catch (reason) {
        if (active) {setError(errorText(reason)); timer = setTimeout(poll, 1000);}
      }
    }
    timer = setTimeout(poll, 100);
    return () => {active = false; clearTimeout(timer);};
  }, [activeRunId, activeStatus, controlAvailable, fit]);
  const indices = parseNumbers(frames);
  const priorScales = parseNumbers(scales);
  const valid = plan && id.trim() && indices.length >= 2 && indices.every((n, i) => Number.isInteger(n) && plan.frame_indices.includes(n) && (i === 0 || n > indices[i - 1])) && priorScales.length === plan.coordinate_order.length && priorScales.every((n) => Number.isFinite(n) && n > 0) && options.knot_count <= indices.length;
  const busy = submitting || Boolean(run && ['pending', 'running'].includes(run.status));
  async function start() {
    setSubmitting(true); setError('');
    try {
      const payload: RefitOptions & {new_fit_id: string} = {...options, new_fit_id: id.trim(), frame_indices: indices, coordinate_scales: priorScales};
      const result = await submitRefit(fit, payload);
      if (mounted.current && result.source_fit_id === fit) {setRun(result); onRun?.(result.run_id);}
    } catch (reason) {if (mounted.current) setError(errorText(reason));}
    finally {if (mounted.current) setSubmitting(false);}
  }
  async function cancel() {
    if (!run) return;
    try {
      const result = await cancelRefit(run.run_id);
      if (mounted.current && result.source_fit_id === fit) setRun(result);
    } catch (reason) {if (mounted.current) setError(errorText(reason));}
  }
  return <section aria-label="Research Refit" className="space-y-3 border-t border-gray-600 pt-4">
    <h3 className="font-semibold">Research Refit</h3><p className="text-sm">Source Version: {fit}. A new version preserves the original. Physical Time, Camera and Dynamics Remain Unqualified.</p>
    {!plan && !error && <p>Loading Source Choices…</p>}
    {plan && <form onSubmit={(event) => {event.preventDefault(); void start();}}>
      <fieldset disabled={busy} className="space-y-2">
        <label className="block">New Fit Version<input className={field} required pattern="[A-Za-z0-9][A-Za-z0-9_.-]{0,127}" value={id} onChange={(event) => setId(event.target.value)} /></label>
        <label className="block">Source Frame Indices<textarea className={field} value={frames} onChange={(event) => setFrames(event.target.value)} /></label>
        <p className="text-xs">Coordinate Order: {plan.coordinate_order.map((name, i) => `${name} (${plan.coordinate_units[i]})`).join(', ')}</p>
        <label className="block">Coordinate Prior Scales<textarea className={field} value={scales} onChange={(event) => setScales(event.target.value)} /></label>
        <p className="text-xs">Enter one positive scale per coordinate in the units above. Scales are fitting priors, not measured anatomy.</p>
        {([
          ['knot_count', 'Spline Knots', 2, 1], ['max_iterations', 'Evaluation Budget', 1, 1],
          ['budget_wall_s', 'Wall Budget (Seconds)', 0.01, 0.01], ['prior_weight', 'Pose Prior Weight', 0, 0.01],
          ['smoothness_weight', 'Smoothness Weight', 0, 0.01], ['closure_weight', 'Grip Closure Weight', 0, 1],
          ['unknown_visibility_weight', 'Unknown Visibility Weight', 0, 0.01],
        ] as const).map(([key, label, min, step]) => <label className="block" key={key}>{label}<input className={field} type="number" required min={min} step={step} max={key === 'unknown_visibility_weight' ? 1 : undefined} value={options[key]} onChange={(event) => setOptions({...options, [key]: Number(event.target.value)})} /></label>)}
        <button className="rounded bg-blue-700 px-3 py-2 disabled:opacity-50" disabled={!valid || busy} type="submit">Start Research Refit</button>
      </fieldset>
    </form>}
    {run && <div><p role="status">{run.status} · {run.acceptance} · {run.message}</p><p className="text-xs">Run: {run.run_id} · New Version: {run.new_fit_id}</p>{run.blockers.map((reason) => <p className="text-xs text-orange-300" key={reason}>{reason}</p>)}</div>}
    {run && controlAvailable && ['pending', 'running'].includes(run.status) && <button type="button" className="rounded border p-2" onClick={() => void cancel()}>Cancel Research Refit</button>}
    {error && <p role="alert">{error}</p>}
  </section>;
}
