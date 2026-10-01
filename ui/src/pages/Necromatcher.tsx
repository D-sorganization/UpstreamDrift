import { KeypointOverlay } from '@/components/visualization/KeypointOverlay';
import { useCallback, useEffect, useState } from 'react';
import { Link, useSearchParams } from 'react-router';
import { WorkspaceShell } from '@/components/layout/WorkspaceShell';
import { LibraryActions } from '@/components/necromatcher/LibraryActions';
import { fetchPlayers, fetchSwings, fetchAssets, fetchCaptureFrame, captureFrameImageUrl, swingExportUrl,
  type HistoricalPlayer, type HistoricalSwing, type HistoricalAsset, type CaptureFrame } from '@/api/necromatcher';

const card = 'rounded-lg border border-gray-700 bg-gray-800 p-4 text-left hover:border-blue-400 focus-visible:outline-2 focus-visible:outline-blue-400';
const readableError = (error: unknown) => error instanceof Error ? error.message : 'The library could not be loaded. Retry after checking the backend.';

function useScopedLoad<T>(key: string, revision: number, load: () => Promise<T>, initial: T) {
  const [result, setResult] = useState({key: '', revision: -1, data: initial, error: '', loading: false});
  useEffect(() => {
    if (!key) return;
    let active = true;
    load().then((data) => { if (active) setResult({key, revision, data, error: '', loading: false}); })
      .catch((reason) => { if (active) setResult({key, revision, data: initial, error: readableError(reason), loading: false}); });
    return () => { active = false; };
  }, [key, revision, load, initial]);
  return result.key === key && result.revision === revision ? result : {data: initial, error: '', loading: Boolean(key)};
}
const emptyPlayers: HistoricalPlayer[] = [];
const emptySwings: HistoricalSwing[] = [];
const emptyAssets: HistoricalAsset[] = [];

export function NecromatcherPage() {
  const [query, setQuery] = useSearchParams();
  const player = query.get('player') ?? '';
  const swing = query.get('swing') ?? '';
  const capture = query.get('capture') ?? '';
  const frameIndex = Math.max(0, Math.trunc(Number(query.get('frame')) || 0));
  const [revision, setRevision] = useState(0);
  const playerLoad = useScopedLoad('players', revision, useCallback(async () => (await fetchPlayers()).players, []), emptyPlayers);
  const players = playerLoad.data;
  const swingLoad = useScopedLoad(player, revision, useCallback(async () => (await fetchSwings(player)).swings, [player]), emptySwings);
  const swings = swingLoad.data;
  const selectedSwing = swings.some((item) => item.subject_id === player && item.session_id === swing) ? swing : '';
  const assetLoad = useScopedLoad(selectedSwing ? `${player}/${selectedSwing}` : '', revision,
    useCallback(async () => (await fetchAssets(selectedSwing)).assets, [selectedSwing]), emptyAssets);
  const selectedAssets = assetLoad.data.filter((item) => item.session_id === selectedSwing);
  const selectedCapture = selectedAssets.some((item) => item.kind === 'image_capture' && item.dataset_id === capture) ? capture : '';
  const frameLoad = useScopedLoad<CaptureFrame | null>(selectedCapture ? `${selectedCapture}/${frameIndex}` : '', revision,
    useCallback(() => fetchCaptureFrame(selectedCapture, frameIndex), [selectedCapture, frameIndex]), null);
  const frame = frameLoad.data;
  const error = playerLoad.error || swingLoad.error || assetLoad.error || frameLoad.error;
  const loading = playerLoad.loading;

  function selectPlayer(id: string) {
    setQuery({ player: id });
  }
  function selectSwing(id: string) {
    setQuery({ player, swing: id });
  }
  const currentFrame = frame?.capture_id === capture && frame.frame_index === frameIndex ? frame : null;
  const frameCount = selectedAssets.find((asset) => asset.dataset_id === capture)?.metadata.frame_count
    ?? (frame?.capture_id === capture ? frame.frame_count : 1);
  const sidebar = <div className="p-4 space-y-4">
    <h2 className="text-lg font-semibold text-white">Saved Swings</h2>
    {!player && <p className="text-sm text-gray-400">Select a Historical Player.</p>}
    {player && swings.length === 0 && <p className="text-sm text-gray-400">No Swings Saved for This Player.</p>}
    {swings.map((item) => <button key={item.session_id} className={`${card} w-full text-white`} aria-pressed={swing === item.session_id} onClick={() => selectSwing(item.session_id)}>{item.name}</button>)}
    {selectedSwing && <a className="block text-blue-300 hover:underline" href={swingExportUrl(selectedSwing)}>Export Swing Package</a>}
    <Link className="block text-blue-300 hover:underline" to="/tools/matched-swings">Matched Swing Results</Link>
    <LibraryActions player={player} swing={selectedSwing} onChanged={() => setRevision((value) => value + 1)} />
  </div>;
  const details = <div className="p-4 space-y-4 text-gray-200"><h2 className="text-lg font-semibold">Models and Controls</h2>
    {selectedAssets.filter((x) => x.kind !== 'image_capture').map((item) => <div key={item.dataset_id} className={card}>
      <h3 className="font-medium">{item.dataset_id}</h3><p className="text-sm text-gray-400">{item.kind === 'native_model' ? 'Candidate Model' : item.kind === 'kinematic_fit' ? 'Kinematic Research Fit' : 'Authored Controls'}</p>
      <p className="text-sm text-gray-400">{item.metadata.qualification}</p>
      {item.metadata.engine && <p>{item.metadata.engine}</p>}{item.metadata.model_id && <p>Model: {item.metadata.model_id}</p>}
    </div>)}
    {!selectedAssets.some((x) => x.kind !== 'image_capture') && <p className="text-sm text-gray-400">No Models or Driving Profiles Saved.</p>}
  </div>;
  return <WorkspaceShell leftPanel={sidebar} rightPanel={details} leftPanelLabel="Saved Swings" rightPanelLabel="Models and Controls">
    <div className="p-6 space-y-6 overflow-y-auto h-full text-gray-100">
      <header><Link to="/" className="text-blue-300 hover:underline">Launcher</Link><h1 className="text-3xl font-bold mt-3">Necromatcher</h1><p className="text-gray-400 mt-2">Historical Swings, Models and Controls</p></header>
      {error && <div role="alert" className="rounded border border-red-500 p-3 text-red-200">{error}<button className="block mt-2 text-blue-300 hover:underline" onClick={() => setRevision((value) => value + 1)}>Retry Loading</button></div>}
      {loading && <p role="status">Loading Historical Players…</p>}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">{players.map((item) => <button key={item.subject_id} aria-label={item.display_name} aria-pressed={player === item.subject_id} className={card} onClick={() => selectPlayer(item.subject_id)}>
        <span aria-hidden="true" className="block text-blue-300 text-2xl mb-2">{item.display_name.split(' ').map((x) => x[0]).join('')}</span><span className="text-lg font-semibold">{item.display_name}</span>
      </button>)}</div>
      {!loading && players.length === 0 && <p>No Historical Players Saved. Add a Player to Begin.</p>}
      {selectedAssets.filter((x) => x.kind === 'image_capture').map((item) => <button key={item.dataset_id} className={card} aria-pressed={capture === item.dataset_id} onClick={() => { setQuery({player, swing, capture: item.dataset_id}); }}>
        <span className="block font-semibold">{item.dataset_id}</span><span className="text-sm text-gray-400">{item.metadata.frame_count} Source Frames · Image Observations</span>
      </button>)}
      {selectedCapture && <SourceFrameReview capture={selectedCapture} frame={currentFrame} failed={Boolean(frameLoad.error)} frameCount={frameCount} frameIndex={frameIndex} onChange={(index) => setQuery({player, swing, capture, frame: String(index)})} />}
    </div>
  </WorkspaceShell>;
}

function SourceFrameReview({capture, frame, failed, frameCount, frameIndex, onChange}: {capture: string; frame: CaptureFrame | null; failed: boolean; frameCount: number; frameIndex: number; onChange: (index: number) => void}) {
  const pts = frame ? frame.frame.pts_ticks * frame.frame.timebase_numerator / frame.frame.timebase_denominator : null;
  return <section className="space-y-3" aria-label="Source Frame Review">
    {frame ? <><div className="relative max-w-4xl"><img alt="Historical Source Frame" src={captureFrameImageUrl(capture, frameIndex)} className="w-full h-auto rounded" />
      <KeypointOverlay points={frame.observation.landmarks} coordinates="normalized_image_xy" width={frame.image_width} height={frame.image_height} showVisibility />
    </div>
    <p className="text-sm text-gray-300">Source PTS: {pts?.toFixed(3)} s · Physical Time: Unknown</p>
    <p className="text-sm text-gray-400">{frame.observation.status === 'missing' ? 'Detector Returned No Landmarks for This Frame.' : 'Landmarks Are Image Observations; Depth and Joint Torques Require a Fitted Model.'}</p></> : !failed && <p role="status">Loading Source Frame…</p>}
    <label className="block text-sm">Source Frame {frameIndex + 1} of {frameCount}<input aria-label="Source Frame" className="block w-full mt-2" type="range" min="0" max={frameCount - 1} value={frameIndex} onChange={(event) => onChange(Number(event.target.value))} /></label>
  </section>;
}
