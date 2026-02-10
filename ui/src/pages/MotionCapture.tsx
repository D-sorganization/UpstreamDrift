/**
 * MotionCapture - Motion capture tool page with skeleton visualization.
 *
 * Provides capture source selection (C3D, OpenPose, MediaPipe),
 * 2D/3D skeleton visualization, and recording/playback controls.
 * Connects to the motion-capture REST API.
 *
 * See issue #1206
 */

import { useState, useCallback, useEffect, useMemo } from 'react';

/** Capture source from the API. See issue #1206 */
export interface CaptureSource {
  id: string;
  name: string;
  type: string;
  available: boolean;
  description: string;
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
  const [sources, setSources] = useState<CaptureSource[]>([]);
  const [selectedSource, setSelectedSource] = useState<string>('mediapipe');
  const [joints, setJoints] = useState<JointData[]>([]);
  const [recordings, setRecordings] = useState<RecordingInfo[]>([]);
  const [activeSession, setActiveSession] = useState<CaptureSession | null>(null);
  const [selectedRecording, setSelectedRecording] = useState<string | null>(null);
  const [playback, setPlayback] = useState<PlaybackState | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch available sources
  useEffect(() => {
    async function fetchSources() {
      try {
        const response = await fetch('/api/tools/motion-capture/sources');
        if (!response.ok) return;
        const data = await response.json();
        setSources(data);
      } catch {
        // API may not be available
      }
    }
    fetchSources();
  }, []);

  // Fetch skeleton template when source changes
  useEffect(() => {
    async function fetchSkeleton() {
      try {
        const response = await fetch(
          `/api/tools/motion-capture/skeleton/${selectedSource}`,
        );
        if (!response.ok) return;
        const data: JointData[] = await response.json();
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
      const response = await fetch('/api/tools/motion-capture/recordings');
      if (!response.ok) return;
      const data = await response.json();
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
      const response = await fetch(
        '/api/tools/motion-capture/session/start',
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            source_type: selectedSource,
            frame_rate: 30.0,
          }),
        },
      );

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${response.status}`);
      }

      const data: CaptureSession = await response.json();
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
      const response = await fetch(
        `/api/tools/motion-capture/session/${activeSession.session_id}/stop`,
        { method: 'POST' },
      );

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${response.status}`);
      }

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
        const response = await fetch(
          '/api/tools/motion-capture/playback',
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              recording_name: selectedRecording,
              action,
              seek_frame: seekFrame ?? null,
            }),
          },
        );

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          throw new Error(errData.detail || `HTTP ${response.status}`);
        }

        const data: PlaybackState = await response.json();
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

  return (
    <div className="flex h-screen bg-gray-900 overflow-hidden">
      {/* Left Panel: Source Selection + Controls */}
      <aside className="w-80 bg-gray-800 border-r border-gray-700 flex flex-col flex-shrink-0 overflow-y-auto">
        <div className="p-4 border-b border-gray-700">
          <h2 className="text-lg font-bold text-white mb-1">Motion Capture</h2>
          <p className="text-xs text-gray-500">
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
              <div className="text-xs text-gray-500 mt-0.5 ml-4">
                {source.description}
              </div>
            </button>
          ))}

          {sources.length === 0 && (
            <div className="text-xs text-gray-500 italic text-center py-2">
              Loading sources...
            </div>
          )}
        </div>

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
                <span className="text-gray-500 ml-auto">
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
            <div className="text-xs text-gray-500 italic text-center py-2">
              No recordings yet
            </div>
          )}

          {recordings.map((rec) => (
            <button
              key={rec.name}
              onClick={() => {
                setSelectedRecording(rec.name);
                setPlayback(null);
              }}
              className={`w-full text-left p-2 rounded mb-1 transition-colors ${
                selectedRecording === rec.name
                  ? 'bg-blue-900/40 ring-1 ring-blue-500/50'
                  : 'hover:bg-gray-700/50'
              }`}
            >
              <div className="text-xs text-gray-200 truncate">{rec.name}</div>
              <div className="text-xs text-gray-500 flex gap-2">
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
      </aside>

      {/* Center: Skeleton Visualization */}
      <main className="flex-1 flex items-center justify-center bg-gray-950 relative min-w-0">
        <div className="w-full max-w-2xl aspect-square p-4">
          <SkeletonRenderer joints={joints} width={500} height={500} />
        </div>

        {/* Source type overlay */}
        <div className="absolute top-4 left-4 bg-black/70 backdrop-blur-sm px-3 py-1.5 rounded-lg border border-white/10">
          <span className="text-sm text-gray-200 font-mono">
            {selectedSource}
          </span>
        </div>

        {/* No data overlay */}
        {joints.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="text-center">
              <h3 className="text-lg font-semibold text-gray-400 mb-2">
                No Skeleton Data
              </h3>
              <p className="text-sm text-gray-500 max-w-xs">
                Select a capture source and start recording, or load a
                recording to visualize skeleton data.
              </p>
            </div>
          </div>
        )}
      </main>

      {/* Right Panel: Joint Data */}
      <aside className="w-72 bg-gray-800 border-l border-gray-700 flex flex-col flex-shrink-0 overflow-y-auto">
        <div className="p-4 border-b border-gray-700">
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
            Joint Data
          </h3>

          <div className="text-xs text-gray-400 mb-2">
            {joints.length} joints detected
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-1">
          {joints.map((joint) => (
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
              <span className="text-xs text-gray-500 font-mono">
                {(joint.confidence * 100).toFixed(0)}%
              </span>
            </div>
          ))}

          {joints.length === 0 && (
            <div className="text-xs text-gray-500 italic text-center py-4">
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
      </aside>
    </div>
  );
}
