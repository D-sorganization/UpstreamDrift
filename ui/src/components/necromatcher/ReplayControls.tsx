import { useEffect, useRef, useState } from 'react';
import { fetchAssets, fetchReplaySummary, replayDataUrl, type HistoricalAsset, type ReplaySummary } from '@/api/necromatcher';

function verifiedSummary(asset: HistoricalAsset, summary: ReplaySummary): ReplaySummary {
  const meta = summary.metadata;
  if (!meta || meta.schema !== 'necromatcher/authored-replay/1'
    || meta.scientific_qualified !== false || meta.physical_source_time_qualified !== false
    || meta.independent_replay_executed !== true || meta.root_policy !== 'unactuated'
    || meta.initial_state_policy !== 'exact_saved_pose_and_authored_rates'
    || asset.metadata.qualification !== 'unqualified_authored_replay') {
    throw new Error('Replay qualification metadata is missing or inconsistent.');
  }
  for (const parent of ['fit', 'model', 'profile', 'capture'] as const) {
    const id=meta[`${parent}_id`];
    const saved=asset.metadata[`${parent}_id`];
    if (typeof id !== 'string' || !id.trim() || (saved !== undefined && saved !== id)
      || typeof meta[`${parent}_hash`] !== 'string'
      || !/^sha256:[a-f0-9]{64}$/.test(meta[`${parent}_hash`] as string)) {
      throw new Error('Replay parent identity or hash is missing or inconsistent.');
    }
  }
  if (summary.replay_id !== asset.dataset_id || !Number.isInteger(summary.sample_count)
    || summary.sample_count < 2 || !Number.isFinite(summary.dt_s) || summary.dt_s <= 0
    || typeof summary.backend !== 'string' || !summary.backend.trim()) {
    throw new Error('Replay sample, backend or identity metadata is invalid.');
  }
  return summary;
}

function ReplayLibrary({swing}: {swing: string}) {
  const [assets,setAssets]=useState<HistoricalAsset[] | null>(null);
  const [summary,setSummary]=useState<ReplaySummary | null>(null);
  const [error,setError]=useState('');
  const [busy,setBusy]=useState(false);
  const generation=useRef({value:0});
  useEffect(() => {
    const epoch=generation.current;
    let active=true;
    fetchAssets(swing).then((value) => {
      if (active) setAssets(value.assets.filter((asset) => asset.kind === 'authored_replay' && asset.session_id === swing));
    }).catch((reason: unknown) => {if(active) setError(reason instanceof Error ? reason.message : 'Replay recall failed.');});
    return () => {active=false; epoch.value++;};
  },[swing]);
  async function recall(asset: HistoricalAsset) {
    const token=++generation.current.value;
    setSummary(null); setError(''); setBusy(true);
    try {
      const value=verifiedSummary(asset,await fetchReplaySummary(asset.dataset_id));
      if (token === generation.current.value) setSummary(value);
    } catch(reason) {
      if(token === generation.current.value) setError(reason instanceof Error ? reason.message : 'Replay recall failed.');
    } finally {if(token === generation.current.value) setBusy(false);}
  }
  return <section className="space-y-2 border-t border-gray-700 pt-4 text-sm text-gray-300">
    <h3 className="font-semibold text-white">Authored Replay Recall</h3>
    <p>Registered replay data uses authored seconds. It does not qualify historical dynamics.</p>
    {assets === null && !error && <p>Loading Replays…</p>}
    {assets?.length === 0 && <p>No Registered Authored Replays.</p>}
    {assets?.map((asset) => <button key={asset.dataset_id} disabled={busy} onClick={() => void recall(asset)} className="block rounded border border-gray-600 p-2">Recall {asset.dataset_id}</button>)}
    {error && <p role="alert" className="text-red-300">{error}</p>}
    {summary && <div className="space-y-2 break-all">
      <p>{summary.replay_id}: {summary.sample_count} Samples | {summary.backend} | Step {summary.dt_s} s</p>
      <p>Authored Seconds | Scientific status: unqualified | Physical source time: unqualified</p>
      <p>Independent replay: executed | Root policy: unactuated | Initial state: exact saved pose and authored rates</p>
      <p>Fit: {String(summary.metadata.fit_id)} | Model: {String(summary.metadata.model_id)} | Profile: {String(summary.metadata.profile_id)} | Capture: {String(summary.metadata.capture_id)}</p>
      {['fit','model','profile','capture'].map((parent) => <p key={parent}>{parent} hash: {String(summary.metadata[`${parent}_hash`])}</p>)}
      <a className="text-blue-300 underline" href={replayDataUrl(summary.replay_id)}>Download Verified Replay HDF5</a>
    </div>}
  </section>;
}

export function ReplayControls({swing,revision}: {swing: string; revision: number}) {
  return <ReplayLibrary key={`${swing}/${revision}`} swing={swing} />;
}
