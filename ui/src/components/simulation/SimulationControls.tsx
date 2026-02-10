/**
 * SimulationControls - Comprehensive simulation control panel.
 *
 * Provides play/pause/stop, speed slider, single-step, camera presets,
 * trajectory recording, and real-time stats display.
 *
 * See issue #1202
 */

import { useCallback, useEffect, useState } from 'react';
import { Play, Pause, Square, SkipForward, Camera, Circle, Download, Gauge } from 'lucide-react';

/** Camera preset identifiers matching the backend. */
export type CameraPreset = 'side' | 'front' | 'top' | 'follow_ball' | 'follow_club';

/** Simulation runtime statistics from the backend. */
export interface SimulationStats {
  simTime: number;
  fps: number;
  realTimeFactor: number;
  frameCount: number;
}

interface Props {
  isRunning: boolean;
  isPaused?: boolean;
  onStart: () => void;
  onStop: () => void;
  onPause: () => void;
  onResume: () => void;
  disabled?: boolean;
  /** Callback for single-step advance (optional). */
  onStep?: () => void;
  /** Callback when speed changes (optional). */
  onSpeedChange?: (speed: number) => void;
  /** Callback when camera preset changes (optional). */
  onCameraChange?: (preset: CameraPreset) => void;
  /** Callback for recording toggle (optional). */
  onRecordingToggle?: (recording: boolean) => void;
  /** Callback for trajectory export (optional). */
  onExportTrajectory?: () => void;
  /** Current simulation stats (optional). */
  stats?: SimulationStats;
  /** Initial speed factor (default 1.0). */
  initialSpeed?: number;
}

// Extracted button styles to reduce duplication and ensure consistency
const buttonStyles = {
  base: 'flex items-center justify-center gap-2 font-semibold py-2 px-4 rounded transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-900',
  primary: 'flex-1 bg-green-600 hover:bg-green-700 text-white focus:ring-green-400',
  warning: 'flex-1 bg-yellow-600 hover:bg-yellow-700 text-white focus:ring-yellow-400',
  danger: 'bg-red-600 hover:bg-red-700 text-white focus:ring-red-400',
  disabled: 'flex-1 bg-gray-600 text-gray-400 cursor-not-allowed',
  secondary: 'bg-gray-700 hover:bg-gray-600 text-white focus:ring-gray-400',
  accent: 'bg-blue-600 hover:bg-blue-700 text-white focus:ring-blue-400',
  recording: 'bg-red-500 hover:bg-red-600 text-white focus:ring-red-400 animate-pulse',
  small: 'py-1 px-2 text-sm',
};

const CAMERA_PRESETS: { id: CameraPreset; label: string }[] = [
  { id: 'side', label: 'Side' },
  { id: 'front', label: 'Front' },
  { id: 'top', label: 'Top' },
  { id: 'follow_ball', label: 'Ball' },
  { id: 'follow_club', label: 'Club' },
];

const SPEED_PRESETS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0];

export function SimulationControls({
  isRunning,
  isPaused,
  onStart,
  onStop,
  onPause,
  onResume,
  disabled,
  onStep,
  onSpeedChange,
  onCameraChange,
  onRecordingToggle,
  onExportTrajectory,
  stats,
  initialSpeed = 1.0,
}: Props) {
  const [speedFactor, setSpeedFactor] = useState(initialSpeed);
  const [isRecording, setIsRecording] = useState(false);
  const [activeCamera, setActiveCamera] = useState<CameraPreset>('side');

  // Handle speed change
  const handleSpeedChange = useCallback(
    (value: number) => {
      setSpeedFactor(value);
      onSpeedChange?.(value);
    },
    [onSpeedChange],
  );

  // Handle camera preset change
  const handleCameraChange = useCallback(
    (preset: CameraPreset) => {
      setActiveCamera(preset);
      onCameraChange?.(preset);
    },
    [onCameraChange],
  );

  // Handle recording toggle
  const handleRecordingToggle = useCallback(() => {
    const newState = !isRecording;
    setIsRecording(newState);
    onRecordingToggle?.(newState);
  }, [isRecording, onRecordingToggle]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Only handle shortcuts when not typing in an input
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      switch (e.code) {
        case 'Space':
          e.preventDefault();
          if (!isRunning) {
            onStart();
          } else if (isPaused) {
            onResume();
          } else {
            // When running and not paused: single-step if available, else pause
            if (onStep) {
              onStep();
            } else {
              onPause();
            }
          }
          break;
        case 'Escape':
          if (isRunning) {
            onStop();
          }
          break;
        case 'Period':
          // Single step with '.' key
          if (isRunning && isPaused && onStep) {
            onStep();
          }
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isRunning, isPaused, onStart, onStop, onPause, onResume, onStep]);

  return (
    <div className="space-y-3">
      {/* Primary controls */}
      <div
        className="flex gap-2"
        role="toolbar"
        aria-label="Simulation controls"
      >
        {!isRunning ? (
          <button
            onClick={onStart}
            disabled={disabled}
            aria-label="Start simulation"
            className={`${buttonStyles.base} ${disabled ? buttonStyles.disabled : buttonStyles.primary}`}
          >
            <Play size={20} aria-hidden="true" />
            Start
          </button>
        ) : (
          <>
            {isPaused ? (
              <button
                onClick={onResume}
                aria-label="Resume simulation"
                className={`${buttonStyles.base} ${buttonStyles.primary}`}
              >
                <Play size={20} aria-hidden="true" />
                Resume
              </button>
            ) : (
              <button
                onClick={onPause}
                aria-label="Pause simulation"
                className={`${buttonStyles.base} ${buttonStyles.warning}`}
              >
                <Pause size={20} aria-hidden="true" />
                Pause
              </button>
            )}

            {onStep && (
              <button
                onClick={onStep}
                disabled={!isPaused}
                aria-label="Single step"
                title="Single step (. key)"
                className={`${buttonStyles.base} ${buttonStyles.small} ${
                  isPaused ? buttonStyles.secondary : buttonStyles.disabled
                }`}
              >
                <SkipForward size={16} aria-hidden="true" />
              </button>
            )}

            <button
              onClick={onStop}
              aria-label="Stop simulation"
              className={`${buttonStyles.base} ${buttonStyles.danger}`}
            >
              <Square size={20} aria-hidden="true" />
              Stop
            </button>
          </>
        )}
      </div>

      {/* Speed control */}
      {onSpeedChange && (
        <div className="bg-gray-800 rounded p-3" role="group" aria-label="Speed control">
          <div className="flex items-center justify-between mb-1">
            <label
              htmlFor="speed-slider"
              className="text-xs text-gray-400 flex items-center gap-1"
            >
              <Gauge size={14} aria-hidden="true" />
              Speed
            </label>
            <span className="text-xs text-green-400 font-mono">
              {speedFactor.toFixed(1)}x
            </span>
          </div>
          <input
            id="speed-slider"
            type="range"
            min={0.1}
            max={10.0}
            step={0.1}
            value={speedFactor}
            onChange={(e) => handleSpeedChange(parseFloat(e.target.value))}
            className="w-full h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer accent-green-500"
            aria-label="Simulation speed"
            aria-valuemin={0.1}
            aria-valuemax={10.0}
            aria-valuenow={speedFactor}
            aria-valuetext={`${speedFactor.toFixed(1)}x speed`}
          />
          <div className="flex justify-between mt-1 gap-1">
            {SPEED_PRESETS.map((preset) => (
              <button
                key={preset}
                onClick={() => handleSpeedChange(preset)}
                className={`text-xs px-1 py-0.5 rounded ${
                  Math.abs(speedFactor - preset) < 0.05
                    ? 'bg-green-600 text-white'
                    : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
                }`}
                aria-label={`Set speed to ${preset}x`}
              >
                {preset}x
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Camera presets */}
      {onCameraChange && (
        <div className="bg-gray-800 rounded p-3" role="group" aria-label="Camera presets">
          <div className="flex items-center gap-1 mb-2">
            <Camera size={14} className="text-gray-400" aria-hidden="true" />
            <span className="text-xs text-gray-400">Camera</span>
          </div>
          <div className="flex gap-1">
            {CAMERA_PRESETS.map(({ id, label }) => (
              <button
                key={id}
                onClick={() => handleCameraChange(id)}
                aria-label={`${label} camera view`}
                aria-pressed={activeCamera === id}
                className={`${buttonStyles.base} ${buttonStyles.small} ${
                  activeCamera === id ? buttonStyles.accent : buttonStyles.secondary
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Recording controls */}
      {onRecordingToggle && (
        <div className="flex gap-2" role="group" aria-label="Recording controls">
          <button
            onClick={handleRecordingToggle}
            aria-label={isRecording ? 'Stop recording' : 'Start recording'}
            aria-pressed={isRecording}
            className={`${buttonStyles.base} ${buttonStyles.small} flex-1 ${
              isRecording ? buttonStyles.recording : buttonStyles.secondary
            }`}
          >
            <Circle
              size={14}
              fill={isRecording ? 'currentColor' : 'none'}
              aria-hidden="true"
            />
            {isRecording ? 'Recording...' : 'Record'}
          </button>
          {onExportTrajectory && (
            <button
              onClick={onExportTrajectory}
              disabled={isRecording}
              aria-label="Export trajectory"
              className={`${buttonStyles.base} ${buttonStyles.small} ${
                isRecording ? buttonStyles.disabled : buttonStyles.secondary
              }`}
            >
              <Download size={14} aria-hidden="true" />
              Export
            </button>
          )}
        </div>
      )}

      {/* Real-time stats display */}
      {stats && (
        <div
          className="bg-gray-800 rounded p-3 grid grid-cols-2 gap-2 text-xs"
          role="status"
          aria-label="Simulation statistics"
        >
          <div>
            <span className="text-gray-400">Sim Time</span>
            <span className="block text-white font-mono">
              {stats.simTime.toFixed(3)}s
            </span>
          </div>
          <div>
            <span className="text-gray-400">FPS</span>
            <span className="block text-white font-mono">
              {stats.fps.toFixed(0)}
            </span>
          </div>
          <div>
            <span className="text-gray-400">RT Factor</span>
            <span className="block text-white font-mono">
              {stats.realTimeFactor.toFixed(2)}x
            </span>
          </div>
          <div>
            <span className="text-gray-400">Frames</span>
            <span className="block text-white font-mono">
              {stats.frameCount}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
