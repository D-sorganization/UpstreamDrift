import { useState } from 'react';
import {
  submitVideoExport,
  fetchVideoExport,
  cancelVideoExport,
  videoExportDownloadUrl,
  type ForceLayer,
  type VideoExportRun,
  type VideoOverlayOptions,
} from '@/api/necromatcher';
import { useResearchJob } from './useResearchJob';
import { ShaftEvidenceInput } from './ShaftEvidenceInput';
import { SourceScopeSummary } from './SourceScopeInput';

type OverlayOptions = VideoOverlayOptions | ForceLayer | undefined;

const exportApi = {
  submit: (fit: string, payload?: OverlayOptions) => {
    if (!payload) return submitVideoExport(fit);
    return submitVideoExport(fit, payload);
  },
  view: fetchVideoExport,
  cancel: cancelVideoExport,
};

const KINDS: ForceLayer['kinds'] = ['joint_reaction', 'joint_actuator', 'contact', 'external'];
const OFF: ForceLayer = { enabled: false, kinds: ['joint_reaction'], scale: 1, segment_shading: false };

type Props = { fit: string; initialRunId?: string; onRun?: (run: string) => void };

export function VideoExportControls(props: Props) {
  return <VideoExportForm key={props.fit} {...props} />;
}

function VideoExportForm(props: Props) {
  const [shaftEvidence, setShaftEvidence] = useState<Record<string, unknown> | null>(null);
  const [shaftBlocked, setShaftBlocked] = useState(false);
  const [showShape, setShowShape] = useState(false);
  const [opacity, setOpacity] = useState('0.35');
  const invalidOpacity = showShape && (opacity.trim() === '' || !Number.isFinite(Number(opacity)) || Number(opacity) < 0 || Number(opacity) > 1);
  const job = useResearchJob<VideoExportRun, OverlayOptions>({ ...props, api: exportApi });
  const { run, error, submitting, controlAvailable } = job;
  const [layer, setLayer] = useState<ForceLayer>(OFF);
  const toggleKind = (kind: ForceLayer['kinds'][number]) =>
    setLayer((current) => {
      const kinds = current.kinds.includes(kind) ? current.kinds.filter((k) => k !== kind) : [...current.kinds, kind];
      return kinds.length ? { ...current, kinds } : current;
    });
  const active = Boolean(run && ['pending', 'running'].includes(run.status));
  const ready = run?.status === 'succeeded' && run.execution_verified && run.download_available;
  const stored = run?.status === 'succeeded' && run.execution_verified && !run.download_available;

  return (
    <section aria-label="Research Overlay Export" className="space-y-3 border-t border-gray-600 pt-4">
      <h3 className="font-semibold">Research Overlay Export</h3>
      <p className="text-sm">
        Export the original source frames with the native research model overlay, first/middle/last stills and an exact provenance manifest.
      </p>
      <p className="text-xs text-orange-300">Monocular Research Hypothesis · Camera, Anatomy and Physical Time Remain Unqualified.</p>
      <ShaftEvidenceInput disabled={submitting || active} onChange={(record, blocked) => { setShaftEvidence(record); setShaftBlocked(blocked); }} />
      <label className="block text-sm">
        <input type="checkbox" checked={showShape} disabled={submitting || active} onChange={(event) => setShowShape(event.target.checked)} /> Show Translucent Model Proxy
      </label>
      <label className="block text-sm">
        Model Proxy Opacity <input type="number" min="0" max="1" step="0.05" value={opacity} disabled={!showShape || submitting || active} onChange={(event) => setOpacity(event.target.value)} />
      </label>
      <p className="text-xs text-orange-300">Model Proxy Retains the Skeleton; Authored Geometry Is Uncalibrated and Does Not Establish Historical Anatomy.</p>
      <fieldset className="space-y-1 text-sm">
        <legend>Optional Force and Torque Layer (off by default)</legend>
        <label className="block">
          <input type="checkbox" checked={layer.enabled} onChange={(e) => setLayer({ ...layer, enabled: e.target.checked })} /> Draw force and torque glyphs
        </label>
        {layer.enabled && (
          <>
            {KINDS.map((kind) => (
              <label key={kind} className="mr-3">
                <input type="checkbox" checked={layer.kinds.includes(kind)} onChange={() => toggleKind(kind)} /> {kind.replace('_', ' ')}
              </label>
            ))}
            <label className="block">
              Glyph scale <input type="number" min={0.1} max={100} step={0.1} value={layer.scale} onChange={(e) => { const scale = Number(e.target.value); if (scale > 0 && scale <= 100) setLayer({ ...layer, scale }); }} />
            </label>
            <label className="block">
              <input type="checkbox" checked={layer.segment_shading} onChange={(e) => setLayer({ ...layer, segment_shading: e.target.checked })} /> Shade body segments
            </label>
            <p className="text-xs text-orange-300">Research fit: unqualified camera and dynamics; not measured forces. Needs a fit that preserves its spline derivatives.</p>
          </>
        )}
      </fieldset>
      <button
        type="button"
        className="rounded bg-blue-700 px-3 py-2 disabled:opacity-50"
        disabled={submitting || active || shaftBlocked || invalidOpacity}
        onClick={() => {
          if (shaftBlocked || invalidOpacity) return;
          const hasShaftOrShape = Boolean(shaftEvidence || showShape);
          if (hasShaftOrShape) {
            const options: VideoOverlayOptions = {
              ...(shaftEvidence ? { shaft_evidence: shaftEvidence } : {}),
              ...(showShape ? { shape_overlay: { opacity: Number(opacity) } } : {}),
              ...(layer.enabled ? { force_layer: layer } : {}),
            };
            void job.start(options);
          } else if (layer.enabled) {
            void job.start(layer);
          } else {
            void job.start(undefined);
          }
        }}
      >
        {submitting ? 'Starting Export…' : 'Export Research Overlay'}
      </button>
      {run && (
        <div>
          <p role="status">{run.status} · {run.acceptance} · {run.message}</p>
          <p className="text-xs">Run: {run.run_id} · Source Fit: {run.source_fit_id}</p>
          {run.blockers.map((reason) => (
            <p key={reason} className="text-xs text-orange-300">{reason}</p>
          ))}
        </div>
      )}
      {run?.shape_overlay && <p className="text-xs">Stored Model Proxy Opacity: {run.shape_overlay.opacity} · Uncalibrated Authored Geometry; Skeleton Retained.</p>}
      <SourceScopeSummary scope={run?.source_fit_scope} binding={run?.source_fit_scope_binding} />
      {active && controlAvailable && <button type="button" className="rounded border p-2" onClick={() => void job.cancel()}>Cancel Overlay Export</button>}
      {ready && <a className="block text-blue-300 hover:underline" href={videoExportDownloadUrl(run.run_id)}>Download Overlay Package</a>}
      {stored && (
        <div className="space-y-2">
          <p className="text-xs text-orange-300">Download Readiness Unverified. Guarded verification can reject changed files.</p>
          <a className="block text-blue-300 hover:underline" href={videoExportDownloadUrl(run.run_id)}>Verify Stored Overlay Package</a>
        </div>
      )}
      {run?.producer_source_commit && <p className="text-xs">Producer Commit: {run.producer_source_commit} · Historical Execution; Current Source Equality Unverified.</p>}
      {error && <p role="alert">{error}</p>}
    </section>
  );
}
