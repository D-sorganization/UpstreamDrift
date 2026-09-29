/**
 * MotionCapture - Motion capture tool page with skeleton visualization.
 *
 * Provides capture source selection (C3D, OpenPose, MediaPipe),
 * 2D/3D skeleton visualization, and recording/playback controls.
 * Connects to the motion-capture REST API.
 *
 * See issue #1206
 */

import {
  lazy,
  Suspense,
  useState,
  useCallback,
  useEffect,
  useMemo,
  useRef,
} from 'react';
import { apiFetch, apiFetchForm } from '@/api/fetch';
import { WorkspaceShell } from '@/components/layout/WorkspaceShell';
import { useRealtimeChannel } from '@/hooks/useRealtimeChannel';
import { extractSkeletonJoints, frameHasDepth } from '@/utils/skeletonJoints';

// Lazy so three.js stays out of this route's chunk until the 3D view is
// actually rendered (issue #8406; same code-splitting rationale as #7433).
const MocapSkeleton3D = lazy(
  () => import('@/components/visualization/MocapSkeleton3D'),
);

/**
 * Load state for the capture-source catalogue (issue #8080).
 *
 * `ready` with an empty `sources` array is a real, renderable outcome — the
 * backend can legitimately report no configured sources — and must not be
 * confused with `loading`.
 */
export type SourcesStatus = 'loading' | 'ready' | 'error';

/** Capture source from the API. See issues #1206, #7454 */
export interface CaptureSource {
  id: string;
  name: string;
  type: string;
  available: boolean;
  /** Why the source is unavailable (null when available). See #7454 */
  reason?: string | null;
  description: string;
}

/** Metadata returned by the C3D upload endpoint. See issue #7454 */
export interface C3DUploadResult {
  recording_name: string;
  marker_names: string[];
  frame_rate: number;
  total_frames: number;
  duration_seconds: number;
  native_units: string;
  converted_units: string;
}

/** One frame of skeleton data from the API. See issue #7454 */
export interface SkeletonFrame {
  frame_index: number;
  timestamp: number;
  joints: JointData[];
}

/** Joint data for skeleton rendering. See issue #1206 */
export interface JointData {
  name: string;
  position: number[];
  confidence: number;
  parent: string | null;
}

/** Recording metadata. See issue #1206 */
export interface RecordingInfo {
  name: string;
  source_type: string;
  total_frames: number;
  duration_seconds: number;
  frame_rate: number;
  joint_names: string[];
}

/** Capture session state. See issue #1206 */
export interface CaptureSession {
  session_id: string;
  status: string;
  source_type: string;
  message: string;
}

/** Playback state. See issue #1206 */
export interface PlaybackState {
  recording_name: string;
  status: string;
  current_frame: number;
  total_frames: number;
}

/** Skeleton view mode: 2D SVG or 3D R3F canvas. See issue #8406 */
export type SkeletonViewMode = '2d' | '3d';

/**
 * SkeletonRenderer - 2D SVG skeleton visualization.
 */
function SkeletonRenderer({
  joints,
  width,
  height,
}: {
  joints: JointData[];
  width: number;
  height: number;
}) {
  // Build a lookup from name to joint
  const jointMap = useMemo(() => {
    const map = new Map<string, JointData>();
    for (const j of joints) {
      map.set(j.name, j);
    }
    return map;
  }, [joints]);

  // Scale positions to SVG coordinates
  const scale = (pos: number[], idx: number) => {
    if (idx === 0) return (pos[0] + 1) * (width / 2); // X: [-1,1] -> [0,width]
    return (1 - pos[1]) * (height / 2); // Y: [-1,1] -> [height,0] (flip)
  };

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="w-full h-full"
      data-testid="skeleton-renderer"
    >
      {/* Background */}
      <rect x={0} y={0} width={width} height={height} fill="#111827" rx={4} />

      {/* Grid */}
      {Array.from({ length: 5 }, (_, i) => {
        const x = (i / 4) * width;
        const y = (i / 4) * height;
        return (
          <g key={i}>
            <line
              x1={x}
              y1={0}
              x2={x}
              y2={height}
              stroke="rgba(255,255,255,0.05)"
            />
            <line
              x1={0}
              y1={y}
              x2={width}
              y2={y}
              stroke="rgba(255,255,255,0.05)"
            />
          </g>
        );
      })}

      {/* Bones (lines between parent-child joints) */}
      {joints.map((joint) => {
        if (!joint.parent) return null;
        const parent = jointMap.get(joint.parent);
        if (!parent) return null;

        const confidence = Math.min(joint.confidence, parent.confidence);
        const opacity = 0.3 + confidence * 0.7;

        return (
          <line
            key={`bone-${joint.name}`}
            x1={scale(parent.position, 0)}
            y1={scale(parent.position, 1)}
            x2={scale(joint.position, 0)}
            y2={scale(joint.position, 1)}
            stroke={`rgba(59, 130, 246, ${opacity})`}
            strokeWidth={2}
            strokeLinecap="round"
          />
        );
      })}

      {/* Joints (circles) */}
      {joints.map((joint) => {
        const x = scale(joint.position, 0);
        const y = scale(joint.position, 1);
        const opacity = 0.4 + joint.confidence * 0.6;
        const r = 3 + joint.confidence * 3;

        return (
          <g key={`joint-${joint.name}`}>
            <circle
              cx={x}
              cy={y}
              r={r}
              fill={`rgba(96, 165, 250, ${opacity})`}
              stroke="white"
              strokeWidth={0.5}
            />
            {/* Label (only for key joints) */}
            {joint.confidence > 0.8 && (
              <text
                x={x + r + 2}
                y={y + 3}
                fill="rgba(156, 163, 175, 0.7)"
                fontSize={8}
              >
                {joint.name.replace('_', ' ')}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

/**
 * MotionCapturePage - Full motion capture tool page.
 *
 * See issue #1206
 */
export function MotionCapturePage() {
  // Source list and selection are driven entirely by GET /capture/sources —
  // no hardcoded source list or default (issue #7454).
  const [sources, setSources] = useState<CaptureSource[]>([]);
  // #8080: an empty source list is no longer overloaded to mean "still
  // loading". These three states are rendered distinctly.
  const [sourcesStatus, setSourcesStatus] = useState<SourcesStatus>('loading');
  const [sourcesError, setSourcesError] = useState<string | null>(null);
  const [selectedSource, setSelectedSource] = useState<string>('');
  const [joints, setJoints] = useState<JointData[]>([]);
  const [recordings, setRecordings] = useState<RecordingInfo[]>([]);
  const [activeSession, setActiveSession] = useState<CaptureSession | null>(null);
  const [selectedRecording, setSelectedRecording] = useState<string | null>(null);
  const [playback, setPlayback] = useState<PlaybackState | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadResult, setUploadResult] = useState<C3DUploadResult | null>(null);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  // #8406: 2D/3D skeleton view. Null = auto (3D when frames carry depth).
  const [viewModeOverride, setViewModeOverride] =
    useState<SkeletonViewMode | null>(null);

  // #8406: live pose streaming. When `pose/canonical` messages carrying joint
  // positions arrive over the realtime WebSocket they drive the skeleton
  // views; otherwise the polled frame data below is used unchanged.
  const { message: livePoseMessage } = useRealtimeChannel<unknown>(
    'pose/canonical',
  );
  const liveJoints = useMemo(
    () => extractSkeletonJoints(livePoseMessage),
    [livePoseMessage],
  );
  const displayJoints = liveJoints ?? joints;

  const viewMode: SkeletonViewMode =
    viewModeOverride ?? (frameHasDepth(displayJoints) ? '3d' : '2d');

  // Fetch available sources; default-select the first available one.
  //
  // #8080: this used to swallow every failure into an empty `sources` array,
  // and the sidebar rendered "Loading sources..." whenever that array was
  // empty. A refused connection, a 404, a malformed body, and a genuinely
  // empty catalogue were therefore indistinguishable — all three showed a
  // spinner that never resolved and offered no retry. The request also had no
  // timeout, so a hung API left the promise pending forever.
  //
  // The load now drives an explicit state machine (loading | ready | error)
  // with a retry action, and each terminal state renders differently.
  const loadSources = useCallback(async () => {
    setSourcesStatus('loading');
    setSourcesError(null);
    try {
      const data = await apiFetch<CaptureSource[]>(
        '/api/tools/motion-capture/sources',
      );
      if (!Array.isArray(data)) {
        throw new Error('Capture-source response was not a list');
      }
      setSources(data);
      setSourcesStatus('ready');
      const firstAvailable = data.find((s) => s.available);
      if (firstAvailable) {
        setSelectedSource((prev) => prev || firstAvailable.type);
      }
    } catch (err) {
      setSources([]);
      setSourcesStatus('error');
      setSourcesError(
        err instanceof Error ? err.message : 'Failed to load capture sources',
      );
    }
  }, []);

  useEffect(() => {
    void loadSources();
  }, [loadSources]);

  // Fetch skeleton template when source changes (joint sets come from the
  // backend — never hardcoded in the UI; issue #7454)
  useEffect(() => {
    if (!selectedSource) return;
    async function fetchSkeleton() {
      try {
        const data = await apiFetch<JointData[]>(
          `/api/tools/motion-capture/skeleton/${selectedSource}`,
        );
        setJoints(data);
      } catch {
        // API may not be available
      }
    }
    fetchSkeleton();
  }, [selectedSource]);

  // Fetch recordings
  const fetchRecordings = useCallback(async () => {
    try {
      const data = await apiFetch<RecordingInfo[]>('/api/tools/motion-capture/recordings');
      setRecordings(data);
    } catch {
      // API may not be available
    }
  }, []);

  useEffect(() => {
    fetchRecordings();
  }, [fetchRecordings]);

  // Start capture session
  const handleStartCapture = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await apiFetch<CaptureSession>(
        '/api/tools/motion-capture/session/start',
        {
          method: 'POST',
          body: JSON.stringify({
            source_type: selectedSource,
            frame_rate: 30.0,
          }),
        },
      );
      setActiveSession(data);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Failed to start capture',
      );
    } finally {
      setLoading(false);
    }
  }, [selectedSource]);

  // Stop capture session
  const handleStopCapture = useCallback(async () => {
    if (!activeSession) return;
    setLoading(true);
    setError(null);

    try {
      await apiFetch<unknown>(
        `/api/tools/motion-capture/session/${activeSession.session_id}/stop`,
        { method: 'POST' },
      );

      setActiveSession(null);
      await fetchRecordings();
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Failed to stop capture',
      );
    } finally {
      setLoading(false);
    }
  }, [activeSession, fetchRecordings]);

  // Playback control
  const handlePlayback = useCallback(
    async (action: string, seekFrame?: number) => {
      if (!selectedRecording) return;
      setLoading(true);
      setError(null);

      try {
        const data = await apiFetch<PlaybackState>(
          '/api/tools/motion-capture/playback',
          {
            method: 'POST',
            body: JSON.stringify({
              recording_name: selectedRecording,
              action,
              seek_frame: seekFrame ?? null,
            }),
          },
        );
        setPlayback(data);
      } catch (err) {
        setError(
          err instanceof Error ? err.message : 'Playback control failed',
        );
      } finally {
        setLoading(false);
      }
    },
    [selectedRecording],
  );

  // Fetch a single frame of an uploaded/recorded clip into the visualizer
  const loadFrame = useCallback(
    async (recordingName: string, frameIndex: number) => {
      try {
        const frame = await apiFetch<SkeletonFrame>(
          `/api/tools/motion-capture/frame/${encodeURIComponent(recordingName)}/${frameIndex}`,
        );
        setJoints(frame.joints);
      } catch {
        // API may not be available
      }
    },
    [],
  );

  // C3D upload: multipart POST, then select the new recording (issue #7454)
  const handleC3DUpload = useCallback(
    async (file: File) => {
      setUploading(true);
      setError(null);
      try {
        const formData = new FormData();
        formData.append('file', file);
        const result = await apiFetchForm<C3DUploadResult>(
          '/api/tools/motion-capture/upload-c3d',
          formData,
        );
        setUploadResult(result);
        setSelectedRecording(result.recording_name);
        setPlayback(null);
        await fetchRecordings();
        await loadFrame(result.recording_name, 0);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'C3D upload failed');
      } finally {
        setUploading(false);
        if (fileInputRef.current) fileInputRef.current.value = '';
      }
    },
    [fetchRecordings, loadFrame],
  );

  // Simple playback loop: advance frames at the recording's rate while
  // status is "playing", rendering each frame through the visualizer.
  useEffect(() => {
    if (!playback || playback.status !== 'playing' || !selectedRecording) {
      return;
    }
    const rec = recordings.find((r) => r.name === selectedRecording);
    const fps = Math.min(Math.max(rec?.frame_rate ?? 30, 1), 30);
    let frame = playback.current_frame;
    const timer = window.setInterval(() => {
      frame = playback.total_frames > 0 ? (frame + 1) % playback.total_frames : 0;
      loadFrame(selectedRecording, frame);
    }, 1000 / fps);
    return () => window.clearInterval(timer);
  }, [playback, selectedRecording, recordings, loadFrame]);

  const selectedSourceInfo = sources.find((s) => s.type === selectedSource);

  const leftPanel = (
    <div className="flex flex-col flex-1 min-h-0">
        <div className="p-4 border-b border-gray-700">
          <h1 className="heading-page mb-1">Motion Capture</h1>
          <p className="text-xs text-gray-400">
            C3D, OpenPose, and MediaPipe analysis
          </p>
        </div>

        {/* Capture Source Selector */}
        <div className="p-4 border-b border-gray-700 space-y-3">
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
            Capture Source
          </h3>

          {sources.map((source) => (
            <button
              key={source.id}
              data-testid={`source-${source.id}`}
              onClick={() => setSelectedSource(source.type)}
              disabled={!source.available}
              className={`w-full text-left p-2.5 rounded transition-colors ${
                selectedSource === source.type
                  ? 'bg-blue-900/40 ring-1 ring-blue-500/50'
                  : source.available
                    ? 'hover:bg-gray-700/50'
                    : 'opacity-40 cursor-not-allowed'
              }`}
            >
              <div className="flex items-center gap-2">
                <div
                  className={`w-2 h-2 rounded-full ${
                    source.available ? 'bg-green-500' : 'bg-gray-500'
                  }`}
                />
                <span className="text-xs text-gray-200 font-medium">
                  {source.name}
                </span>
              </div>
              <div className="text-xs text-gray-400 mt-0.5 ml-4">
                {source.description}
              </div>
              {!source.available && (
                <div
                  className="text-xs text-amber-500/80 mt-0.5 ml-4"
                  data-testid={`source-reason-${source.id}`}
                >
                  {source.reason ?? 'Unavailable'}
                </div>
              )}
            </button>
          ))}

          {/* #8080: three distinct terminal states, never an endless spinner. */}
          {sourcesStatus === 'loading' && (
            <div
              className="text-xs text-gray-400 italic text-center py-2"
              data-testid="sources-loading"
            >
              Loading sources...
            </div>
          )}

          {sourcesStatus === 'error' && (
            <div
              className="space-y-2 rounded border border-red-500/40 bg-red-950/30 p-2.5"
              data-testid="sources-error"
              role="alert"
            >
              <div className="text-xs font-medium text-red-300">
                Capture sources unavailable
              </div>
              <div className="text-xs text-red-200/80 break-words">
                {sourcesError ?? 'The motion-capture service did not respond.'}
              </div>
              <div className="text-xs text-gray-400">
                Check that the API server is running, then retry.
              </div>
              <button
                type="button"
                data-testid="sources-retry"
                onClick={() => void loadSources()}
                className="w-full rounded bg-red-600 px-2 py-1 text-xs font-medium text-white transition-colors hover:bg-red-500"
              >
                Retry
              </button>
            </div>
          )}

          {sourcesStatus === 'ready' && sources.length === 0 && (
            <div
              className="space-y-1 rounded border border-gray-700 bg-gray-800/50 p-2.5"
              data-testid="sources-empty"
            >
              <div className="text-xs font-medium text-gray-300">
                No capture sources configured
              </div>
              <div className="text-xs text-gray-400">
                The service reported an empty catalogue. Install a capture
                backend (MediaPipe, OpenPose) or upload a C3D recording.
              </div>
            </div>
          )}
        </div>

        {/* C3D File Upload (only for the c3d source; issue #7454) */}
        {selectedSource === 'c3d' && selectedSourceInfo?.available && (
          <div className="p-4 border-b border-gray-700 space-y-2">
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
              C3D File
            </h3>
            <input
              ref={fileInputRef}
              type="file"
              accept=".c3d"
              data-testid="c3d-file-input"
              disabled={uploading}
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleC3DUpload(file);
              }}
              className="block w-full text-xs text-gray-400 file:mr-2 file:py-1.5 file:px-3 file:rounded file:border-0 file:bg-blue-600 file:text-white file:text-xs hover:file:bg-blue-500 file:cursor-pointer"
            />
            {uploading && (
              <div className="text-xs text-gray-400 italic">Uploading…</div>
            )}
            {uploadResult && (
              <div
                className="space-y-1 text-xs bg-gray-700/30 p-2 rounded"
                data-testid="c3d-upload-metadata"
              >
                <div className="flex justify-between">
                  <span className="text-gray-400">Markers</span>
                  <span className="text-gray-200 font-mono">
                    {uploadResult.marker_names.length}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Rate</span>
                  <span className="text-gray-200 font-mono">
                    {uploadResult.frame_rate} Hz
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Frames</span>
                  <span className="text-gray-200 font-mono">
                    {uploadResult.total_frames}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Duration</span>
                  <span className="text-gray-200 font-mono">
                    {uploadResult.duration_seconds.toFixed(2)}s
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Units</span>
                  <span className="text-gray-200 font-mono">
                    {uploadResult.native_units || '?'} → {uploadResult.converted_units}
                  </span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Session Controls */}
        <div className="p-4 border-b border-gray-700 space-y-2">
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
            Capture Session
          </h3>

          {activeSession ? (
            <>
              <div className="flex items-center gap-2 text-xs">
                <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                <span className="text-red-400">Recording...</span>
                <span className="text-gray-400 ml-auto">
                  {activeSession.session_id}
                </span>
              </div>
              <button
                onClick={handleStopCapture}
                disabled={loading}
                className="w-full py-2 px-4 bg-red-600 hover:bg-red-500 disabled:bg-gray-600 text-white text-sm font-medium rounded transition-colors"
                data-testid="stop-capture-btn"
              >
                Stop Recording
              </button>
            </>
          ) : (
            <button
              onClick={handleStartCapture}
              disabled={loading}
              className="w-full py-2 px-4 bg-green-600 hover:bg-green-500 disabled:bg-gray-600 text-white text-sm font-medium rounded transition-colors"
              data-testid="start-capture-btn"
            >
              Start Capture
            </button>
          )}
        </div>

        {/* Recordings */}
        <div className="p-4 border-b border-gray-700">
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
            Recordings ({recordings.length})
          </h3>

          {recordings.length === 0 && (
            <div className="text-xs text-gray-400 italic text-center py-2">
              No recordings yet
            </div>
          )}

          {recordings.map((rec) => (
            <button
              key={rec.name}
              onClick={() => {
                setSelectedRecording(rec.name);
                setPlayback(null);
                loadFrame(rec.name, 0);
              }}
              className={`w-full text-left p-2 rounded mb-1 transition-colors ${
                selectedRecording === rec.name
                  ? 'bg-blue-900/40 ring-1 ring-blue-500/50'
                  : 'hover:bg-gray-700/50'
              }`}
            >
              <div className="text-xs text-gray-200 truncate">{rec.name}</div>
              <div className="text-xs text-gray-400 flex gap-2">
                <span>{rec.source_type}</span>
                <span>{rec.total_frames} frames</span>
                <span>{rec.duration_seconds.toFixed(1)}s</span>
              </div>
            </button>
          ))}
        </div>

        {/* Playback Controls */}
        {selectedRecording && (
          <div className="p-4 space-y-2">
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
              Playback
            </h3>

            <div className="flex gap-1">
              <button
                onClick={() => handlePlayback('play')}
                className="flex-1 py-1.5 bg-green-600 hover:bg-green-500 text-white text-xs rounded"
              >
                Play
              </button>
              <button
                onClick={() => handlePlayback('pause')}
                className="flex-1 py-1.5 bg-yellow-600 hover:bg-yellow-500 text-white text-xs rounded"
              >
                Pause
              </button>
              <button
                onClick={() => handlePlayback('stop')}
                className="flex-1 py-1.5 bg-red-600 hover:bg-red-500 text-white text-xs rounded"
              >
                Stop
              </button>
            </div>

            {playback && playback.total_frames > 0 && (
              <input
                type="range"
                min={0}
                max={playback.total_frames - 1}
                value={playback.current_frame}
                data-testid="playback-seek"
                onChange={(e) => {
                  const frame = Number(e.target.value);
                  handlePlayback('seek', frame);
                  if (selectedRecording) loadFrame(selectedRecording, frame);
                }}
                className="w-full"
              />
            )}

            {playback && (
              <div className="text-xs text-gray-400 text-center">
                Frame {playback.current_frame}/{playback.total_frames} |{' '}
                {playback.status}
              </div>
            )}
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="mx-4 mb-4 text-xs text-red-400 bg-red-900/20 p-2 rounded">
            {error}
          </div>
        )}
    </div>
  );

  const mainContent = (
    <div className="flex-1 flex items-center justify-center bg-gray-950 relative min-w-0 min-h-0 p-2 sm:p-4">
        {/* #7417: constrain the square by width AND height so it never
            overflows narrow or short windows. */}
        <div className="w-full aspect-square max-w-[min(42rem,90vw,calc(100vh-8rem))]">
          {viewMode === '3d' ? (
            <Suspense
              fallback={
                <div
                  className="w-full h-full flex items-center justify-center text-xs text-gray-400"
                  data-testid="skeleton-3d-loading"
                >
                  Loading 3D view…
                </div>
              }
            >
              <MocapSkeleton3D joints={displayJoints} />
            </Suspense>
          ) : (
            <SkeletonRenderer joints={displayJoints} width={500} height={500} />
          )}
        </div>

        {/* Source type overlay */}
        <div className="absolute top-4 left-4 bg-black/70 backdrop-blur-sm px-3 py-1.5 rounded-lg border border-white/10">
          <span className="text-sm text-gray-200 font-mono">
            {selectedSource}
          </span>
          {liveJoints && (
            <span
              className="ml-2 text-xs text-green-400 font-semibold"
              data-testid="live-pose-indicator"
            >
              LIVE
            </span>
          )}
        </div>

        {/* 2D/3D view toggle (#8406) */}
        <div className="absolute top-4 right-4 flex items-center gap-1 bg-black/70 backdrop-blur-sm px-1.5 py-1 rounded-lg border border-white/10">
          <button
            data-testid="view-toggle-2d"
            onClick={() => setViewModeOverride('2d')}
            className={`px-2 py-0.5 text-xs rounded transition-colors ${
              viewMode === '2d'
                ? 'bg-blue-600 text-white font-semibold'
                : 'text-gray-300 hover:text-white hover:bg-white/10'
            }`}
          >
            2D
          </button>
          <button
            data-testid="view-toggle-3d"
            onClick={() => setViewModeOverride('3d')}
            className={`px-2 py-0.5 text-xs rounded transition-colors ${
              viewMode === '3d'
                ? 'bg-blue-600 text-white font-semibold'
                : 'text-gray-300 hover:text-white hover:bg-white/10'
            }`}
          >
            3D
          </button>
        </div>

        {/* No data overlay */}
        {displayJoints.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="text-center">
              <h3 className="text-lg font-semibold text-gray-400 mb-2">
                No Skeleton Data
              </h3>
              <p className="text-sm text-gray-400 max-w-xs">
                Select a capture source and start recording, or load a
                recording to visualize skeleton data.
              </p>
            </div>
          </div>
        )}
    </div>
  );

  const rightPanel = (
    <div className="flex flex-col flex-1 min-h-0">
        <div className="p-4 border-b border-gray-700">
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
            Joint Data
          </h3>

          <div className="text-xs text-gray-400 mb-2">
            {displayJoints.length} joints detected
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-1">
          {displayJoints.map((joint) => (
            <div
              key={joint.name}
              className="bg-gray-700/30 p-1.5 rounded flex items-center gap-2"
            >
              <div
                className={`w-1.5 h-1.5 rounded-full ${
                  joint.confidence > 0.8
                    ? 'bg-green-500'
                    : joint.confidence > 0.5
                      ? 'bg-yellow-500'
                      : 'bg-red-500'
                }`}
              />
              <span className="text-xs text-gray-300 truncate flex-1">
                {joint.name}
              </span>
              <span className="text-xs text-gray-400 font-mono">
                {(joint.confidence * 100).toFixed(0)}%
              </span>
            </div>
          ))}

          {displayJoints.length === 0 && (
            <div className="text-xs text-gray-400 italic text-center py-4">
              No joints loaded
            </div>
          )}
        </div>

        {/* Selected Recording Info */}
        {selectedRecording && (
          <div className="p-4 border-t border-gray-700">
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
              Recording Info
            </h3>
            {recordings
              .filter((r) => r.name === selectedRecording)
              .map((rec) => (
                <div key={rec.name} className="space-y-1 text-xs">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Source</span>
                    <span className="text-gray-200">{rec.source_type}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Frames</span>
                    <span className="text-gray-200 font-mono">
                      {rec.total_frames}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Duration</span>
                    <span className="text-gray-200 font-mono">
                      {rec.duration_seconds.toFixed(1)}s
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Frame Rate</span>
                    <span className="text-gray-200 font-mono">
                      {rec.frame_rate} fps
                    </span>
                  </div>
                </div>
              ))}
          </div>
        )}
    </div>
  );

  return (
    <WorkspaceShell
      leftPanel={leftPanel}
      rightPanel={rightPanel}
      leftPanelLabel="Capture Controls"
      rightPanelLabel="Joint Data"
    >
      {mainContent}
    </WorkspaceShell>
  );
}
