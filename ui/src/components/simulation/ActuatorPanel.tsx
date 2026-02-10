/**
 * ActuatorPanel - Per-actuator control sliders.
 *
 * Queries engine capabilities to dynamically generate sliders
 * for each actuator. Supports multiple control types: constant,
 * polynomial, PD gains, trajectory.
 *
 * See issue #1198
 */

import { useState, useCallback, useEffect, useRef } from 'react';

/** Actuator descriptor from the API. See issue #1198 */
export interface ActuatorInfo {
  index: number;
  name: string;
  control_type: string;
  value: number;
  min_value: number;
  max_value: number;
  units: string;
  joint_type: string;
}

/** Actuator panel state from the API. See issue #1198 */
export interface ActuatorPanelState {
  n_actuators: number;
  actuators: ActuatorInfo[];
  available_control_types: string[];
  engine_name: string;
}

interface ActuatorPanelProps {
  /** Whether the simulation is running */
  isRunning: boolean;
  /** Polling interval for state refresh (ms) */
  refreshInterval?: number;
}

const CONTROL_TYPE_LABELS: Record<string, string> = {
  constant: 'Constant',
  polynomial: 'Polynomial',
  pd_gains: 'PD Gains',
  trajectory: 'Trajectory',
};

/**
 * Single actuator slider row.
 */
function ActuatorSlider({
  actuator,
  onValueChange,
  onControlTypeChange,
  availableTypes,
}: {
  actuator: ActuatorInfo;
  onValueChange: (index: number, value: number) => void;
  onControlTypeChange: (index: number, type: string) => void;
  availableTypes: string[];
}) {
  // Track whether the user is actively dragging the slider.
  // While dragging, use the locally tracked value; otherwise
  // fall back to the externally-provided actuator value.
  const [isDragging, setIsDragging] = useState(false);
  const [dragValue, setDragValue] = useState(actuator.value);
  const localValue = isDragging ? dragValue : actuator.value;

  const debounceRef = useRef<NodeJS.Timeout | null>(null);

  const handleChange = useCallback(
    (newValue: number) => {
      setIsDragging(true);
      setDragValue(newValue);

      // Debounce API calls; stop dragging after debounce settles
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
      debounceRef.current = setTimeout(() => {
        onValueChange(actuator.index, newValue);
        setIsDragging(false);
      }, 50);
    },
    [actuator.index, onValueChange],
  );

  // Cleanup debounce on unmount
  useEffect(() => {
    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
    };
  }, []);

  const range = actuator.max_value - actuator.min_value;
  const percentage =
    range > 0
      ? ((localValue - actuator.min_value) / range) * 100
      : 50;

  return (
    <div className="bg-gray-700/30 p-2 rounded-md">
      {/* Header row: name + control type */}
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs font-mono text-gray-300 truncate max-w-[120px]">
          {actuator.name}
        </span>
        <select
          value={actuator.control_type}
          onChange={(e) =>
            onControlTypeChange(actuator.index, e.target.value)
          }
          className="text-xs bg-gray-600 text-gray-300 rounded px-1 py-0.5 border-none focus:ring-1 focus:ring-blue-400"
        >
          {availableTypes.map((type) => (
            <option key={type} value={type}>
              {CONTROL_TYPE_LABELS[type] || type}
            </option>
          ))}
        </select>
      </div>

      {/* Slider */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-gray-500 w-12 text-right font-mono">
          {actuator.min_value.toFixed(1)}
        </span>
        <div className="flex-1 relative">
          <input
            type="range"
            min={actuator.min_value}
            max={actuator.max_value}
            step={(range / 200) || 0.1}
            value={localValue}
            onChange={(e) => handleChange(parseFloat(e.target.value))}
            className="w-full h-1.5 bg-gray-600 rounded-lg appearance-none cursor-pointer"
            aria-label={`${actuator.name} control value`}
          />
          {/* Fill indicator */}
          <div
            className="absolute top-0 left-0 h-1.5 bg-blue-500 rounded-l-lg pointer-events-none"
            style={{ width: `${Math.max(0, Math.min(100, percentage))}%` }}
          />
        </div>
        <span className="text-xs text-gray-500 w-12 font-mono">
          {actuator.max_value.toFixed(1)}
        </span>
      </div>

      {/* Current value display */}
      <div className="flex items-center justify-between mt-1">
        <span className="text-xs text-gray-400">
          {actuator.joint_type}
        </span>
        <span className="text-xs font-mono text-blue-400">
          {localValue.toFixed(2)} {actuator.units}
        </span>
      </div>
    </div>
  );
}

/**
 * ActuatorPanel provides dynamic slider controls for each actuator
 * in the active simulation engine.
 *
 * See issue #1198
 */
export function ActuatorPanel({
  isRunning,
  refreshInterval = 1000,
}: ActuatorPanelProps) {
  const [panelState, setPanelState] = useState<ActuatorPanelState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [collapsed, setCollapsed] = useState(false);

  // Fetch actuator state from backend
  const fetchActuators = useCallback(async () => {
    try {
      const response = await fetch('/api/simulation/actuators');
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data: ActuatorPanelState = await response.json();
      setPanelState(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load actuators');
    }
  }, []);

  // Initial fetch and periodic refresh
  useEffect(() => {
    fetchActuators();

    if (isRunning && refreshInterval > 0) {
      const interval = setInterval(fetchActuators, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [fetchActuators, isRunning, refreshInterval]);

  // Send actuator command to backend
  const handleValueChange = useCallback(async (index: number, value: number) => {
    try {
      await fetch('/api/simulation/actuators', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          actuator_index: index,
          value,
          control_type: 'constant',
        }),
      });
    } catch {
      // Silently fail on network errors
    }
  }, []);

  const handleControlTypeChange = useCallback(
    async (index: number, controlType: string) => {
      try {
        await fetch('/api/simulation/actuators', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            actuator_index: index,
            value: 0,
            control_type: controlType,
          }),
        });
        // Refresh to get updated state
        fetchActuators();
      } catch {
        // Silently fail
      }
    },
    [fetchActuators],
  );

  return (
    <div className="bg-gray-700/50 p-3 rounded-md">
      {/* Header */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="flex items-center justify-between w-full text-left"
      >
        <h4 className="text-xs font-semibold text-gray-300 uppercase">
          Actuator Controls
        </h4>
        <span className="text-xs text-gray-500">
          {collapsed ? '+' : '-'}
          {panelState ? ` (${panelState.n_actuators})` : ''}
        </span>
      </button>

      {!collapsed && (
        <div className="mt-2 space-y-2">
          {error && (
            <div className="text-xs text-red-400 bg-red-900/20 p-2 rounded">
              {error}
            </div>
          )}

          {panelState && panelState.actuators.length > 0 ? (
            <>
              <div className="text-xs text-gray-500 mb-1">
                Engine: {panelState.engine_name}
              </div>
              {panelState.actuators.map((actuator) => (
                <ActuatorSlider
                  key={actuator.index}
                  actuator={actuator}
                  onValueChange={handleValueChange}
                  onControlTypeChange={handleControlTypeChange}
                  availableTypes={panelState.available_control_types}
                />
              ))}
            </>
          ) : (
            <div className="text-xs text-gray-500 italic text-center py-2">
              {panelState ? 'No actuators available' : 'Loading actuators...'}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
