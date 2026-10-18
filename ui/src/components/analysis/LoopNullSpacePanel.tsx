/**
 * LoopNullSpacePanel - closed-loop null-space exploration (NSX-44, #12576).
 *
 * Mirrors the PyQt6 tabs (NSX-43): the Python plot model is the single source
 * of every panel, so this component submits a job (preset selector), shows
 * progress, offers Cancel, and renders the plot model the job carries. It does
 * no math and takes no display decision. When the job carries no plot model
 * (for example mid-run, before the display decisions exist) the panel says so
 * and shows nothing in its place.
 */

import { useState, type FormEvent } from 'react';
import {
  LOOP_NULLSPACE_PRESETS,
  isTerminalLoopNullspaceJob,
  plotModelOf,
  type LoopNullspaceRequest,
} from '@/api/loopNullSpace';
import { useLoopNullspaceJob } from '@/api/useLoopNullspaceJob';
import { parseFrameIndices } from './loopNullSpace/format';
import { PlotModelView } from './loopNullSpace/PlotModelView';

const FIELD = 'bg-gray-900 border border-gray-600 rounded px-2 py-1 text-sm text-gray-100';

export function LoopNullSpacePanel() {
  const { job, error, busy, start, cancel } = useLoopNullspaceJob();
  const [sourceId, setSourceId] = useState('');
  const [engine, setEngine] = useState('');
  const [preset, setPreset] = useState(LOOP_NULLSPACE_PRESETS[0].id);
  const [frames, setFrames] = useState('0');
  const [formError, setFormError] = useState<string | null>(null);

  const submit = (event: FormEvent) => {
    event.preventDefault();
    const indices = parseFrameIndices(frames);
    if (!sourceId.trim() || !engine.trim()) {
      setFormError('Source and engine are required.');
      return;
    }
    if (indices === null) {
      setFormError('Frame indices must be comma-separated whole numbers.');
      return;
    }
    setFormError(null);
    const request: LoopNullspaceRequest = {
      source_id: sourceId.trim(),
      engine: engine.trim(),
      preset,
      frame_indices: indices,
    };
    void start(request);
  };

  const model = job ? plotModelOf(job) : null;
  const active = job !== null && !isTerminalLoopNullspaceJob(job);

  return (
    <section
      data-testid="loop-nullspace-panel"
      aria-label="Loop Null-Space Exploration"
      className="bg-gray-800 rounded-lg border border-gray-700 p-4"
    >
      <h3 className="text-lg font-semibold text-gray-100 mb-3">Loop Null-Space Exploration</h3>
      <form onSubmit={submit} className="flex flex-wrap gap-3 items-end mb-3">
        <label className="text-xs text-gray-300 flex flex-col gap-1">
          Source
          <input className={FIELD} value={sourceId} onChange={(e) => setSourceId(e.target.value)} />
        </label>
        <label className="text-xs text-gray-300 flex flex-col gap-1">
          Engine
          <input className={FIELD} value={engine} onChange={(e) => setEngine(e.target.value)} />
        </label>
        <label className="text-xs text-gray-300 flex flex-col gap-1">
          Preset
          <select className={FIELD} value={preset} onChange={(e) => setPreset(e.target.value)}>
            {LOOP_NULLSPACE_PRESETS.map((p) => (
              <option key={p.id} value={p.id}>
                {p.label}
              </option>
            ))}
          </select>
        </label>
        <label className="text-xs text-gray-300 flex flex-col gap-1">
          Frame indices
          <input className={FIELD} value={frames} onChange={(e) => setFrames(e.target.value)} />
        </label>
        <button
          type="submit"
          disabled={busy || active}
          className="px-3 py-1 rounded bg-blue-600 text-sm text-white disabled:opacity-50"
        >
          Run
        </button>
        <button
          type="button"
          onClick={() => void cancel()}
          disabled={!active}
          className="px-3 py-1 rounded bg-gray-600 text-sm text-white disabled:opacity-50"
        >
          Cancel
        </button>
      </form>
      {formError && (
        <p role="alert" data-testid="nsx-form-error" className="text-sm text-red-300">
          {formError}
        </p>
      )}
      {error && (
        <p role="alert" data-testid="nsx-error" className="text-sm text-red-300">
          {error}
        </p>
      )}
      {job && (
        <p data-testid="nsx-job-status" role="status" className="text-xs text-gray-300 mb-2">
          Job status: {job.status}
          {job.stage ? `; stage: ${job.stage}` : ''}; progress: {Math.round(job.progress_percent)}%
          {job.cache_hit ? '; served from cache' : ''}
          {job.status === 'failed' && job.error ? `; error: ${job.error}` : ''}
        </p>
      )}
      {job && !model && (
        <p data-testid="nsx-no-model" className="text-xs text-amber-300">
          {job.status === 'completed'
            ? 'This job carries no plot model, so nothing is drawn.'
            : job.status === 'cancelled' || job.status === 'failed'
              ? 'No result to show.'
              : 'Partial results are held back until their display decisions are available.'}
        </p>
      )}
      {model && (
        <>
          {job?.status !== 'completed' && (
            <p data-testid="nsx-partial-banner" className="text-xs text-amber-300 mb-2">
              Partial result: later panels are still running or were not produced.
            </p>
          )}
          <PlotModelView model={model} />
        </>
      )}
    </section>
  );
}
