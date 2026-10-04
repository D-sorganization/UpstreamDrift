import { useState } from 'react';
import { submitVideoExport, fetchVideoExport, cancelVideoExport, videoExportDownloadUrl, type ForceLayer, type VideoExportRun } from '@/api/necromatcher';
import { useResearchJob } from './useResearchJob';
const exportApi = {submit: (fit: string, layer: ForceLayer) => layer.enabled ? submitVideoExport(fit, layer) : submitVideoExport(fit), view: fetchVideoExport, cancel: cancelVideoExport};
const KINDS: ForceLayer['kinds'] = ['joint_reaction', 'joint_actuator', 'contact', 'external'];
const OFF: ForceLayer = {enabled: false, kinds: ['joint_reaction'], scale: 1, segment_shading: false};
type Props = {fit: string; initialRunId?: string; onRun?: (run: string) => void};
export function VideoExportControls(props: Props) {
  return <VideoExportForm key={props.fit} {...props} />;
}
function VideoExportForm(props: Props) {
  const job = useResearchJob<VideoExportRun, ForceLayer>({...props, api: exportApi});
  const {run, error, submitting, controlAvailable} = job;
  const [layer, setLayer] = useState<ForceLayer>(OFF);
  const toggleKind = (kind: ForceLayer['kinds'][number]) => setLayer((current) => {
    const kinds = current.kinds.includes(kind) ? current.kinds.filter((k) => k !== kind) : [...current.kinds, kind];
    return kinds.length ? {...current, kinds} : current;
  });
  const active = Boolean(run && ['pending', 'running'].includes(run.status));
  const ready = run?.status === 'succeeded' && run.execution_verified && run.download_available;
  return <section aria-label="Research Overlay Export" className="space-y-3 border-t border-gray-600 pt-4">
    <h3 className="font-semibold">Research Overlay Export</h3>
    <p className="text-sm">Export the original source frames with the native research model overlay, first/middle/last stills and an exact provenance manifest.</p>
    <p className="text-xs text-orange-300">Monocular Research Hypothesis · Camera, Anatomy and Physical Time Remain Unqualified.</p>
    <fieldset className="space-y-1 text-sm">
      <legend>Optional Force and Torque Layer (off by default)</legend>
      <label className="block"><input type="checkbox" checked={layer.enabled} onChange={(e) => setLayer({...layer, enabled: e.target.checked})} /> Draw force and torque glyphs</label>
      {layer.enabled && <>
        {KINDS.map((kind) => <label key={kind} className="mr-3"><input type="checkbox" checked={layer.kinds.includes(kind)} onChange={() => toggleKind(kind)} /> {kind.replace('_', ' ')}</label>)}
        <label className="block">Glyph scale <input type="number" min={0.1} max={100} step={0.1} value={layer.scale} onChange={(e) => {const scale = Number(e.target.value); if (scale > 0 && scale <= 100) setLayer({...layer, scale});}} /></label>
        <label className="block"><input type="checkbox" checked={layer.segment_shading} onChange={(e) => setLayer({...layer, segment_shading: e.target.checked})} /> Shade body segments</label>
        <p className="text-xs text-orange-300">Research fit: unqualified camera and dynamics; not measured forces. Needs a fit that preserves its spline derivatives.</p>
      </>}
    </fieldset>
    <button type="button" className="rounded bg-blue-700 px-3 py-2 disabled:opacity-50" disabled={submitting || active} onClick={() => void job.start(layer)}>{submitting ? 'Starting Export…' : 'Export Research Overlay'}</button>
    {run && <div><p role="status">{run.status} · {run.acceptance} · {run.message}</p><p className="text-xs">Run: {run.run_id} · Source Fit: {run.source_fit_id}</p>{run.blockers.map((reason) => <p key={reason} className="text-xs text-orange-300">{reason}</p>)}</div>}
    {active && controlAvailable && <button type="button" className="rounded border p-2" onClick={() => void job.cancel()}>Cancel Overlay Export</button>}
    {ready && <a className="block text-blue-300 hover:underline" href={videoExportDownloadUrl(run.run_id)}>Download Overlay Package</a>}
    {error && <p role="alert">{error}</p>}
  </section>;
}
