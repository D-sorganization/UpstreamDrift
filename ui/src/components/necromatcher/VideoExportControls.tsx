import { submitVideoExport, fetchVideoExport, cancelVideoExport, videoExportDownloadUrl, type VideoExportRun } from '@/api/necromatcher';
import { useResearchJob } from './useResearchJob';
const exportApi = {submit: (fit: string) => submitVideoExport(fit), view: fetchVideoExport, cancel: cancelVideoExport};
type Props = {fit: string; initialRunId?: string; onRun?: (run: string) => void};
export function VideoExportControls(props: Props) {
  return <VideoExportForm key={props.fit} {...props} />;
}
function VideoExportForm(props: Props) {
  const job = useResearchJob<VideoExportRun, undefined>({...props, api: exportApi});
  const {run, error, submitting, controlAvailable} = job;
  const active = Boolean(run && ['pending', 'running'].includes(run.status));
  const ready = run?.status === 'succeeded' && run.execution_verified && run.download_available;
  return <section aria-label="Research Overlay Export" className="space-y-3 border-t border-gray-600 pt-4">
    <h3 className="font-semibold">Research Overlay Export</h3>
    <p className="text-sm">Export the original source frames with the native research model overlay, first/middle/last stills and an exact provenance manifest.</p>
    <p className="text-xs text-orange-300">Monocular Research Hypothesis · Camera, Anatomy and Physical Time Remain Unqualified.</p>
    <button type="button" className="rounded bg-blue-700 px-3 py-2 disabled:opacity-50" disabled={submitting || active} onClick={() => void job.start(undefined)}>{submitting ? 'Starting Export…' : 'Export Research Overlay'}</button>
    {run && <div><p role="status">{run.status} · {run.acceptance} · {run.message}</p><p className="text-xs">Run: {run.run_id} · Source Fit: {run.source_fit_id}</p>{run.blockers.map((reason) => <p key={reason} className="text-xs text-orange-300">{reason}</p>)}</div>}
    {active && controlAvailable && <button type="button" className="rounded border p-2" onClick={() => void job.cancel()}>Cancel Overlay Export</button>}
    {ready && <a className="block text-blue-300 hover:underline" href={videoExportDownloadUrl(run.run_id)}>Download Overlay Package</a>}
    {error && <p role="alert">{error}</p>}
  </section>;
}
