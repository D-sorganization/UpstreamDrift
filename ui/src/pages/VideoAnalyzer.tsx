/**
 * VideoAnalyzer - Video-based swing analysis tool page.
 *
 * Provides video upload/playback, frame-by-frame analysis with overlay
 * controls, and integration with existing REST video endpoints.
 *
 * See issue #1206
 */

import { useState, useCallback, useRef } from 'react';

/** Video analysis result from the API. See issue #1206 */
export interface VideoAnalysisResult {
  filename: string;
  total_frames: number;
  valid_frames: number;
  average_confidence: number;
  quality_metrics: Record<string, number>;
  pose_data: PoseFrame[];
}

/** A single pose frame. See issue #1206 */
export interface PoseFrame {
  timestamp: number;
  confidence: number;
  joint_angles: Record<string, number>;
  keypoints: Record<string, number[]>;
}

/** Async task status. See issue #1206 */
export interface TaskStatus {
  task_id: string;
  status: 'started' | 'processing' | 'completed' | 'failed';
  progress?: number;
  result?: Record<string, unknown>;
  error?: string;
}

/**
 * PoseOverlay - Renders joint keypoints over the video frame.
 */
function PoseOverlay({
  frame,
  width,
  height,
}: {
  frame: PoseFrame | null;
  width: number;
  height: number;
}) {
  if (!frame || !frame.keypoints) return null;

  const entries = Object.entries(frame.keypoints);
  if (entries.length === 0) return null;

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="absolute inset-0 w-full h-full pointer-events-none"
      data-testid="pose-overlay"
    >
      {entries.map(([name, coords]) => {
        if (!coords || coords.length < 2) return null;
        return (
          <circle
            key={name}
            cx={coords[0]}
            cy={coords[1]}
            r={4}
            fill="rgba(0, 255, 100, 0.8)"
            stroke="white"
            strokeWidth={1}
          />
        );
      })}
    </svg>
  );
}

/**
 * JointAngleChart - Simple bar chart of joint angles.
 */
function JointAngleChart({
  angles,
}: {
  angles: Record<string, number>;
}) {
  const entries = Object.entries(angles);
  if (entries.length === 0) {
    return (
      <div className="text-xs text-gray-500 italic text-center py-2">
        No joint angle data
      </div>
    );
  }

  const maxAngle = Math.max(...entries.map(([, v]) => Math.abs(v)), 1);

  return (
    <div className="space-y-1" data-testid="joint-angle-chart">
      {entries.slice(0, 12).map(([name, value]) => (
        <div key={name} className="flex items-center gap-2">
          <span className="text-xs text-gray-400 w-24 truncate">{name}</span>
          <div className="flex-1 h-2 bg-gray-700 rounded overflow-hidden">
            <div
              className="h-full bg-blue-500 rounded"
              style={{ width: `${(Math.abs(value) / maxAngle) * 100}%` }}
            />
          </div>
          <span className="text-xs text-gray-300 font-mono w-12 text-right">
            {value.toFixed(1)}
          </span>
        </div>
      ))}
    </div>
  );
}

/**
 * VideoAnalyzerPage - Full video analysis tool page.
 *
 * See issue #1206
 */
export function VideoAnalyzerPage() {
  const [file, setFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<VideoAnalysisResult | null>(null);
  const [currentFrameIdx, setCurrentFrameIdx] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [estimatorType, setEstimatorType] = useState('mediapipe');
  const [minConfidence, setMinConfidence] = useState(0.5);
  const [showOverlay, setShowOverlay] = useState(true);
  const videoRef = useRef<HTMLVideoElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Handle file selection
  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const selected = e.target.files?.[0];
      if (selected) {
        setFile(selected);
        setVideoUrl(URL.createObjectURL(selected));
        setAnalysis(null);
        setCurrentFrameIdx(0);
        setError(null);
      }
    },
    [],
  );

  // Upload and analyze
  const handleAnalyze = useCallback(async () => {
    if (!file) {
      setError('Please select a video file first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const params = new URLSearchParams({
        estimator_type: estimatorType,
        min_confidence: minConfidence.toString(),
        enable_smoothing: 'true',
      });

      const response = await fetch(
        `/api/analyze/video?${params.toString()}`,
        {
          method: 'POST',
          body: formData,
        },
      );

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${response.status}`);
      }

      const data: VideoAnalysisResult = await response.json();
      setAnalysis(data);
      setCurrentFrameIdx(0);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setLoading(false);
    }
  }, [file, estimatorType, minConfidence]);

  // Frame navigation
  const totalFrames = analysis?.pose_data.length ?? 0;
  const currentFrame =
    analysis && totalFrames > 0
      ? analysis.pose_data[Math.min(currentFrameIdx, totalFrames - 1)]
      : null;

  const handlePrevFrame = useCallback(() => {
    setCurrentFrameIdx((prev) => Math.max(0, prev - 1));
  }, []);

  const handleNextFrame = useCallback(() => {
    setCurrentFrameIdx((prev) => Math.min(totalFrames - 1, prev + 1));
  }, [totalFrames]);

  return (
    <div className="flex h-screen bg-gray-900 overflow-hidden">
      {/* Left Panel: Controls */}
      <aside className="w-80 bg-gray-800 border-r border-gray-700 flex flex-col flex-shrink-0 overflow-y-auto">
        <div className="p-4 border-b border-gray-700">
          <h2 className="text-lg font-bold text-white mb-1">Video Analyzer</h2>
          <p className="text-xs text-gray-500">
            Upload and analyze golf swing videos
          </p>
        </div>

        {/* Upload */}
        <div className="p-4 border-b border-gray-700 space-y-3">
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
            Video Upload
          </h3>
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            onChange={handleFileSelect}
            className="hidden"
            data-testid="video-file-input"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            className="w-full py-2 px-4 bg-gray-700 hover:bg-gray-600 text-gray-200 text-sm rounded border border-gray-600 transition-colors"
          >
            {file ? file.name : 'Choose Video File...'}
          </button>
          {file && (
            <div className="text-xs text-gray-400">
              Size: {(file.size / 1024 / 1024).toFixed(1)} MB
            </div>
          )}
        </div>

        {/* Analysis Settings */}
        <div className="p-4 border-b border-gray-700 space-y-3">
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
            Settings
          </h3>

          <div>
            <label className="text-xs text-gray-400 block mb-1">
              Pose Estimator
            </label>
            <select
              value={estimatorType}
              onChange={(e) => setEstimatorType(e.target.value)}
              className="w-full bg-gray-700 text-gray-200 rounded px-2 py-1.5 text-sm border-none focus:ring-1 focus:ring-blue-400"
            >
              <option value="mediapipe">MediaPipe</option>
              <option value="openpose">OpenPose</option>
              <option value="blazepose">BlazePose</option>
            </select>
          </div>

          <div>
            <label className="text-xs text-gray-400 block mb-1">
              Min Confidence: {minConfidence.toFixed(2)}
            </label>
            <input
              type="range"
              min={0.1}
              max={0.95}
              step={0.05}
              value={minConfidence}
              onChange={(e) =>
                setMinConfidence(parseFloat(e.target.value))
              }
              className="w-full h-1.5 bg-gray-600 rounded-lg appearance-none cursor-pointer"
              aria-label="Minimum confidence threshold"
            />
          </div>

          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={showOverlay}
              onChange={(e) => setShowOverlay(e.target.checked)}
              className="rounded border-gray-600 bg-gray-700"
            />
            <span className="text-xs text-gray-300">Show Pose Overlay</span>
          </label>
        </div>

        {/* Analyze Button */}
        <div className="p-4 border-b border-gray-700">
          <button
            onClick={handleAnalyze}
            disabled={loading || !file}
            className="w-full py-2 px-4 bg-green-600 hover:bg-green-500 disabled:bg-gray-600 text-white text-sm font-medium rounded transition-colors"
            data-testid="analyze-btn"
          >
            {loading ? 'Analyzing...' : 'Analyze Video'}
          </button>
        </div>

        {/* Frame Navigation */}
        {analysis && (
          <div className="p-4 border-b border-gray-700 space-y-3">
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
              Frame Navigation
            </h3>

            <div className="flex items-center gap-2">
              <button
                onClick={handlePrevFrame}
                disabled={currentFrameIdx <= 0}
                className="px-3 py-1 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-700/50 text-gray-200 text-xs rounded"
              >
                Prev
              </button>
              <span className="text-xs text-gray-300 font-mono flex-1 text-center">
                {currentFrameIdx + 1} / {totalFrames}
              </span>
              <button
                onClick={handleNextFrame}
                disabled={currentFrameIdx >= totalFrames - 1}
                className="px-3 py-1 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-700/50 text-gray-200 text-xs rounded"
              >
                Next
              </button>
            </div>

            <input
              type="range"
              min={0}
              max={Math.max(0, totalFrames - 1)}
              value={currentFrameIdx}
              onChange={(e) =>
                setCurrentFrameIdx(parseInt(e.target.value, 10))
              }
              className="w-full h-1.5 bg-gray-600 rounded-lg appearance-none cursor-pointer"
              aria-label="Frame scrubber"
            />
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="mx-4 mb-4 text-xs text-red-400 bg-red-900/20 p-2 rounded">
            {error}
          </div>
        )}
      </aside>

      {/* Center: Video Player */}
      <main className="flex-1 flex items-center justify-center bg-gray-950 relative min-w-0">
        {videoUrl ? (
          <div className="relative w-full max-w-3xl aspect-video">
            <video
              ref={videoRef}
              src={videoUrl}
              controls
              className="w-full h-full rounded-lg"
              data-testid="video-player"
            />
            {showOverlay && currentFrame && (
              <PoseOverlay
                frame={currentFrame}
                width={640}
                height={480}
              />
            )}
          </div>
        ) : (
          <div className="text-center">
            <div className="text-4xl mb-4 text-gray-600">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="w-16 h-16 mx-auto text-gray-600"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1}
                  d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                />
              </svg>
            </div>
            <h3 className="text-lg font-semibold text-gray-400 mb-2">
              Upload a Video
            </h3>
            <p className="text-sm text-gray-500 max-w-xs">
              Select a golf swing video to analyze pose estimation,
              joint angles, and swing metrics.
            </p>
          </div>
        )}
      </main>

      {/* Right Panel: Analysis Results */}
      <aside className="w-72 bg-gray-800 border-l border-gray-700 flex flex-col flex-shrink-0 overflow-y-auto">
        {/* Summary */}
        <div className="p-4 border-b border-gray-700">
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
            Analysis Summary
          </h3>

          {analysis ? (
            <div className="space-y-2">
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Filename</span>
                <span className="text-gray-200 truncate max-w-[120px]">
                  {analysis.filename}
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Total Frames</span>
                <span className="text-gray-200 font-mono">
                  {analysis.total_frames}
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Valid Frames</span>
                <span className="text-gray-200 font-mono">
                  {analysis.valid_frames}
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Avg Confidence</span>
                <span className="text-blue-400 font-mono">
                  {(analysis.average_confidence * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          ) : (
            <div className="text-xs text-gray-500 italic text-center py-4">
              No analysis yet
            </div>
          )}
        </div>

        {/* Quality Metrics */}
        {analysis && Object.keys(analysis.quality_metrics).length > 0 && (
          <div className="p-4 border-b border-gray-700">
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
              Quality Metrics
            </h3>
            <div className="space-y-1">
              {Object.entries(analysis.quality_metrics).map(([key, val]) => (
                <div key={key} className="flex justify-between text-xs">
                  <span className="text-gray-400">{key}</span>
                  <span className="text-gray-200 font-mono">
                    {typeof val === 'number' ? val.toFixed(3) : String(val)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Current Frame Data */}
        <div className="p-4 flex-1">
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
            Frame Joint Angles
          </h3>

          {currentFrame ? (
            <>
              <div className="text-xs text-gray-500 mb-2">
                t={currentFrame.timestamp.toFixed(3)}s | conf={' '}
                {(currentFrame.confidence * 100).toFixed(1)}%
              </div>
              <JointAngleChart angles={currentFrame.joint_angles} />
            </>
          ) : (
            <div className="text-xs text-gray-500 italic text-center py-4">
              Analyze a video to view joint data
            </div>
          )}
        </div>
      </aside>
    </div>
  );
}
