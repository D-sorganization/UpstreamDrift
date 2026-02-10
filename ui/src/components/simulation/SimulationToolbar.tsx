/**
 * SimulationToolbar - Interactive tools for model positioning,
 * measurement, and force/torque visualization.
 *
 * Provides buttons and displays for:
 * - Body positioning (translate/rotate)
 * - Distance measurement between bodies
 * - Joint angle display
 * - Force/torque overlay toggle
 *
 * See issue #1179
 */

import { useState, useCallback, useEffect, useRef } from 'react';
import {
  Move,
  Ruler,
  RotateCw,
  Eye,
  EyeOff,
  Crosshair,
  ArrowUpDown,
} from 'lucide-react';

/** Joint angle data from the measurement tools endpoint. */
export interface JointAngleDisplay {
  joint_name: string;
  angle_rad: number;
  angle_deg: number;
  velocity: number;
  torque: number;
}

/** Measurement between two bodies. */
export interface MeasurementResult {
  body_a: string;
  body_b: string;
  distance: number;
  position_a: [number, number, number];
  position_b: [number, number, number];
  delta: [number, number, number];
}

/** Active tool mode. */
export type ToolMode = 'select' | 'position' | 'measure' | 'rotate';

interface Props {
  /** Whether the simulation is running */
  isRunning: boolean;
  /** Callback when force overlay visibility changes */
  onForceOverlayToggle?: (visible: boolean) => void;
  /** Callback when tool mode changes */
  onToolModeChange?: (mode: ToolMode) => void;
  /** Polling interval for joint angle display (ms) */
  pollInterval?: number;
}

const buttonBase =
  'flex items-center justify-center gap-1 py-1.5 px-3 rounded text-xs font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-offset-gray-900';

export function SimulationToolbar({
  isRunning,
  onForceOverlayToggle,
  onToolModeChange,
  pollInterval = 1000,
}: Props) {
  const [activeMode, setActiveMode] = useState<ToolMode>('select');
  const [showForces, setShowForces] = useState(false);
  const [jointAngles, setJointAngles] = useState<JointAngleDisplay[]>([]);
  const [measurements, setMeasurements] = useState<MeasurementResult[]>([]);
  const [showJoints, setShowJoints] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<NodeJS.Timeout | null>(null);

  // Handle tool mode change
  const handleModeChange = useCallback(
    (mode: ToolMode) => {
      setActiveMode(mode);
      onToolModeChange?.(mode);
    },
    [onToolModeChange],
  );

  // Toggle force overlays
  const handleForceToggle = useCallback(() => {
    const newState = !showForces;
    setShowForces(newState);
    onForceOverlayToggle?.(newState);
  }, [showForces, onForceOverlayToggle]);

  // Fetch joint angles from the measurement tools endpoint
  const fetchMeasurements = useCallback(async () => {
    try {
      const response = await fetch('/api/simulation/measurements');
      if (!response.ok) {
        if (response.status === 400) {
          // No engine loaded -- not an error
          return;
        }
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      setJointAngles(data.joint_angles ?? []);
      setMeasurements(data.measurements ?? []);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Fetch failed');
    }
  }, []);

  // Poll for joint angles when showing and simulation is running
  useEffect(() => {
    if (showJoints && isRunning) {
      fetchMeasurements();
      pollRef.current = setInterval(fetchMeasurements, pollInterval);
    } else {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    }
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [showJoints, isRunning, pollInterval, fetchMeasurements]);

  return (
    <div className="space-y-2">
      {/* Tool mode buttons */}
      <div
        className="flex gap-1"
        role="toolbar"
        aria-label="Simulation tools"
      >
        <button
          onClick={() => handleModeChange('select')}
          aria-label="Select mode"
          aria-pressed={activeMode === 'select'}
          className={`${buttonBase} ${
            activeMode === 'select'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
          }`}
        >
          <Crosshair size={14} aria-hidden="true" />
          Select
        </button>
        <button
          onClick={() => handleModeChange('position')}
          aria-label="Position mode"
          aria-pressed={activeMode === 'position'}
          className={`${buttonBase} ${
            activeMode === 'position'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
          }`}
        >
          <Move size={14} aria-hidden="true" />
          Move
        </button>
        <button
          onClick={() => handleModeChange('rotate')}
          aria-label="Rotate mode"
          aria-pressed={activeMode === 'rotate'}
          className={`${buttonBase} ${
            activeMode === 'rotate'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
          }`}
        >
          <RotateCw size={14} aria-hidden="true" />
          Rotate
        </button>
        <button
          onClick={() => handleModeChange('measure')}
          aria-label="Measure mode"
          aria-pressed={activeMode === 'measure'}
          className={`${buttonBase} ${
            activeMode === 'measure'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
          }`}
        >
          <Ruler size={14} aria-hidden="true" />
          Measure
        </button>
      </div>

      {/* Toggle buttons */}
      <div className="flex gap-1">
        <button
          onClick={handleForceToggle}
          aria-label={showForces ? 'Hide force overlays' : 'Show force overlays'}
          aria-pressed={showForces}
          className={`${buttonBase} flex-1 ${
            showForces
              ? 'bg-yellow-600 text-white'
              : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
          }`}
        >
          <ArrowUpDown size={14} aria-hidden="true" />
          Forces
        </button>
        <button
          onClick={() => setShowJoints(!showJoints)}
          aria-label={showJoints ? 'Hide joint angles' : 'Show joint angles'}
          aria-pressed={showJoints}
          className={`${buttonBase} flex-1 ${
            showJoints
              ? 'bg-green-600 text-white'
              : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
          }`}
        >
          {showJoints ? (
            <Eye size={14} aria-hidden="true" />
          ) : (
            <EyeOff size={14} aria-hidden="true" />
          )}
          Joints
        </button>
      </div>

      {/* Error display */}
      {error && (
        <div className="text-xs text-red-400 px-2 py-1 bg-red-900/30 rounded">
          {error}
        </div>
      )}

      {/* Joint angles display */}
      {showJoints && jointAngles.length > 0 && (
        <div
          className="bg-gray-800 rounded p-2 max-h-48 overflow-y-auto"
          role="table"
          aria-label="Joint angles"
        >
          <div className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1">
            Joint Angles
          </div>
          <div className="space-y-0.5">
            {jointAngles.map((joint) => (
              <div
                key={joint.joint_name}
                className="flex justify-between text-xs"
              >
                <span className="text-gray-400 font-mono truncate max-w-[40%]">
                  {joint.joint_name}
                </span>
                <span className="text-white font-mono">
                  {joint.angle_deg.toFixed(1)}&deg;
                </span>
                <span className="text-gray-500 font-mono">
                  {joint.velocity.toFixed(2)} rad/s
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Active measurements */}
      {measurements.length > 0 && (
        <div
          className="bg-gray-800 rounded p-2"
          role="table"
          aria-label="Distance measurements"
        >
          <div className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1">
            Measurements
          </div>
          <div className="space-y-0.5">
            {measurements.map((m, idx) => (
              <div key={idx} className="flex justify-between text-xs">
                <span className="text-gray-400 font-mono">
                  {m.body_a} &harr; {m.body_b}
                </span>
                <span className="text-white font-mono">
                  {m.distance.toFixed(4)} m
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
