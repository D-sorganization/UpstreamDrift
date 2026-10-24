/**
 * LoopNullSpacePanel - closed-loop null-space exploration (NSX-44, #12576).
 *
 * Mirrors the PyQt6 tabs (NSX-43): the Python plot model is the single source
 * of every panel, so this component submits a job (preset selector), shows
 * progress, offers Cancel, and renders the plot model the job carries. It does
 * no math and takes no display decision. When the job carries no plot model
 * (for example mid-run, before the display decisions exist) the panel says so
 * and shows nothing in its place.
 *
 * NSX-100 (#12729): the Variability & Consistency section adds the sensitivity
 * request controls (their lists are the backend catalogue's, served by
 * `GET .../sensitivity-options`) and renders the plot model's `sensitivity`
 * panels. Without the catalogue the engine stays a plain text field and no
 * sensitivity block is sent.
 */

import { useCallback, useEffect, useState, type FormEvent } from 'react';
import {
  fetchLoopNullspacePresets,
  fetchLoopNullspaceSensitivityOptions,
  isTerminalLoopNullspaceJob,
  plotModelOf,
  type LoopNullspaceJobRequest,
  type LoopNullspacePreset,
  type SensitivityCatalogue,
} from '@/api/loopNullSpace';
import { useLoopNullspaceJob } from '@/api/useLoopNullspaceJob';
import { HelpButton } from '../ui/HelpPanel';
import { parseFrameIndices } from './loopNullSpace/format';
import { PlotModelView } from './loopNullSpace/PlotModelView';
import { VariabilityControls, type ControlsState } from './loopNullSpace/VariabilityControls';
import { VariabilityView } from './loopNullSpace/VariabilityView';
import { buildSensitivityOptions } from './loopNullSpace/variabilityRequest';

const DEFAULT_OPTIONS = {
  metric: 'identity',
  include_envelopes: true,
  include_allocation: false,
  effort_weight: 1,
  rate_weight: 0,
} as const;

const FIELD = 'bg-gray-900 border border-gray-600 rounded px-2 py-1 text-sm text-gray-100';

export function LoopNullSpacePanel() {
  const { job, error, busy, start, cancel } = useLoopNullspaceJob();
  const [sourceId, setSourceId] = useState('');
  const [engine, setEngine] = useState('');
  const [presets, setPresets] = useState<LoopNullspacePreset[]>([]);
  const [presetError, setPresetError] = useState<string | null>(null);
  const [preset, setPreset] = useState('');

  useEffect(() => {
    const controller = new AbortController();
    fetchLoopNullspacePresets(controller.signal)
      .then((list) => {
        setPresets(list);
        setPreset((current) => current || list[0]?.id || '');
      })
      .catch((e: unknown) => {
        if (!controller.signal.aborted) setPresetError(e instanceof Error ? e.message : String(e));
      });
    return () => controller.abort();
  }, []);
  const [catalogue, setCatalogue] = useState<SensitivityCatalogue | null>(null);
  const [catalogueError, setCatalogueError] = useState<string | null>(null);
  const [controls, setControls] = useState<ControlsState | null>(null);
  const onControls = useCallback((state: ControlsState) => setControls(state), []);

  useEffect(() => {
    const controller = new AbortController();
    fetchLoopNullspaceSensitivityOptions(controller.signal)
      .then(setCatalogue)
      .catch((e: unknown) => {
        if (!controller.signal.aborted) {
          setCatalogueError(e instanceof Error ? e.message : String(e));
        }
      });
    return () => controller.abort();
  }, []);
  const [frames, setFrames] = useState('0');
  const [formError, setFormError] = useState<string | null>(null);

  const submit = (event: FormEvent) => {
    event.preventDefault();
    const indices = parseFrameIndices(frames);
    if (!preset) {
      setFormError('No preset is available; the preset list has not loaded.');
      return;
    }
    const chosenEngine = catalogue && controls ? controls.engine : engine.trim();
    if (!sourceId.trim() || !chosenEngine) {
      setFormError('Source and engine are required.');
      return;
    }
    if (indices === null) {
      setFormError('Frame indices must be comma-separated whole numbers.');
      return;
    }
    let sensitivity;
    if (catalogue && controls?.enabled) {
      try {
        if (controls.error) throw new Error(controls.error);
        sensitivity = buildSensitivityOptions(controls.choices, catalogue);
      } catch (e) {
        setFormError(`Variability options: ${e instanceof Error ? e.message : String(e)}`);
        return;
      }
    }
    setFormError(null);
    const request: LoopNullspaceJobRequest = {
      source_id: sourceId.trim(),
      engine: chosenEngine,
      preset,
      frame_indices: indices,
      ...(sensitivity ? { options: { ...DEFAULT_OPTIONS, sensitivity } } : {}),
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
      <div className="flex items-center gap-2 mb-3">
        <h3 className="text-lg font-semibold text-gray-100">Loop Null-Space Exploration</h3>
        <HelpButton topicId="loop_nullspace" tooltip="Open the Loop Null-Space help page" />
      </div>
      <form onSubmit={submit} className="flex flex-wrap gap-3 items-end mb-3">
        <label className="text-xs text-gray-300 flex flex-col gap-1">
          Source
          <input className={FIELD} value={sourceId} onChange={(e) => setSourceId(e.target.value)} />
        </label>
        {!catalogue && (
          <label className="text-xs text-gray-300 flex flex-col gap-1">
            Engine
            <input className={FIELD} value={engine} onChange={(e) => setEngine(e.target.value)} />
          </label>
        )}
        <label className="text-xs text-gray-300 flex flex-col gap-1">
          Preset
          <select className={FIELD} value={preset} onChange={(e) => setPreset(e.target.value)}>
            {presets.map((p) => (
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
      <section data-testid="nsx-variability" aria-label="Variability & Consistency">
        <h4 className="text-sm font-semibold text-gray-200 mb-2">Variability &amp; Consistency</h4>
        {catalogue && <VariabilityControls catalogue={catalogue} onChange={onControls} />}
        {catalogueError && (
          <p role="alert" data-testid="nsx-catalogue-error" className="text-sm text-red-300">
            Sensitivity options unavailable: {catalogueError}
          </p>
        )}
        {model?.sensitivity && <VariabilityView sensitivity={model.sensitivity} />}
      </section>
      {presetError && (
        <p role="alert" data-testid="nsx-preset-error" className="text-sm text-red-300">
          Presets unavailable: {presetError}
        </p>
      )}
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
