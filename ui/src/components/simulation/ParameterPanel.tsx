import { useState, useEffect, useCallback } from 'react';

export interface SimulationParameters {
  duration: number;
  timestep: number;
  liveAnalysis: boolean;
  gpuAcceleration: boolean;
  model?: string;
}

interface Props {
  engine: string;
  disabled?: boolean;
  onChange: (params: SimulationParameters) => void;
}

// Engine-specific default configurations
const ENGINE_DEFAULTS: Record<string, Partial<SimulationParameters>> = {
  mujoco: {
    duration: 3.0,
    timestep: 0.002,
  },
  drake: {
    duration: 5.0,
    timestep: 0.001,
  },
  pinocchio: {
    duration: 3.0,
    timestep: 0.001,
  },
  opensim: {
    duration: 2.0,
    timestep: 0.005,
  },
  myosim: {
    duration: 3.0,
    timestep: 0.002,
  },
};

export function ParameterPanel({ engine, disabled, onChange }: Props) {
  const [duration, setDuration] = useState(3.0);
  const [timestep, setTimestep] = useState(0.002);
  const [liveAnalysis, setLiveAnalysis] = useState(true);
  const [gpuAcceleration, setGpuAcceleration] = useState(false);

  // Update defaults when engine changes
  useEffect(() => {
    const defaults = ENGINE_DEFAULTS[engine.toLowerCase()] || {};
    setDuration(defaults.duration ?? 3.0);
    setTimestep(defaults.timestep ?? 0.002);
  }, [engine]);

  // Notify parent of parameter changes
  const notifyChange = useCallback(() => {
    onChange({
      duration,
      timestep,
      liveAnalysis,
      gpuAcceleration,
    });
  }, [duration, timestep, liveAnalysis, gpuAcceleration, onChange]);

  useEffect(() => {
    notifyChange();
  }, [notifyChange]);

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">
        Simulation Parameters
      </h3>

      {/* Duration */}
      <div>
        <label
          htmlFor="duration-input"
          className="block text-sm font-medium text-gray-300 mb-1"
        >
          Duration (seconds)
        </label>
        <input
          id="duration-input"
          type="number"
          min="0.1"
          max="60"
          step="0.1"
          value={duration}
          onChange={(e) => setDuration(parseFloat(e.target.value) || 3.0)}
          disabled={disabled}
          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white
                     focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent
                     disabled:opacity-50 disabled:cursor-not-allowed"
          aria-describedby="duration-help"
        />
        <p id="duration-help" className="mt-1 text-xs text-gray-500">
          Simulation run time (0.1 - 60s)
        </p>
      </div>

      {/* Timestep */}
      <div>
        <label
          htmlFor="timestep-input"
          className="block text-sm font-medium text-gray-300 mb-1"
        >
          Timestep (seconds)
        </label>
        <select
          id="timestep-input"
          value={timestep}
          onChange={(e) => setTimestep(parseFloat(e.target.value))}
          disabled={disabled}
          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white
                     focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent
                     disabled:opacity-50 disabled:cursor-not-allowed"
          aria-describedby="timestep-help"
        >
          <option value="0.001">0.001s (High precision)</option>
          <option value="0.002">0.002s (Default)</option>
          <option value="0.005">0.005s (Fast)</option>
          <option value="0.01">0.01s (Very fast)</option>
        </select>
        <p id="timestep-help" className="mt-1 text-xs text-gray-500">
          Physics integration step size
        </p>
      </div>

      {/* Live Analysis Toggle */}
      <div className="flex items-center justify-between">
        <div>
          <label
            htmlFor="live-analysis-toggle"
            className="text-sm font-medium text-gray-300"
          >
            Live Analysis
          </label>
          <p className="text-xs text-gray-500">Stream joint angles & velocities</p>
        </div>
        <button
          id="live-analysis-toggle"
          role="switch"
          aria-checked={liveAnalysis}
          onClick={() => setLiveAnalysis(!liveAnalysis)}
          disabled={disabled}
          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors
                     focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-offset-2 focus:ring-offset-gray-800
                     disabled:opacity-50 disabled:cursor-not-allowed
                     ${liveAnalysis ? 'bg-blue-600' : 'bg-gray-600'}`}
        >
          <span
            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform
                       ${liveAnalysis ? 'translate-x-6' : 'translate-x-1'}`}
          />
        </button>
      </div>

      {/* GPU Acceleration Toggle */}
      <div className="flex items-center justify-between">
        <div>
          <label
            htmlFor="gpu-toggle"
            className="text-sm font-medium text-gray-300"
          >
            GPU Acceleration
          </label>
          <p className="text-xs text-gray-500">Use GPU for physics (if available)</p>
        </div>
        <button
          id="gpu-toggle"
          role="switch"
          aria-checked={gpuAcceleration}
          onClick={() => setGpuAcceleration(!gpuAcceleration)}
          disabled={disabled}
          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors
                     focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-offset-2 focus:ring-offset-gray-800
                     disabled:opacity-50 disabled:cursor-not-allowed
                     ${gpuAcceleration ? 'bg-green-600' : 'bg-gray-600'}`}
        >
          <span
            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform
                       ${gpuAcceleration ? 'translate-x-6' : 'translate-x-1'}`}
          />
        </button>
      </div>

      {/* Engine-specific info */}
      <div className="mt-4 p-3 bg-gray-700/50 rounded-md">
        <p className="text-xs text-gray-400">
          <span className="font-semibold text-gray-300">Engine:</span> {engine}
        </p>
        <p className="text-xs text-gray-500 mt-1">
          {engine.toLowerCase() === 'mujoco' && 'Full contact physics, muscle simulation'}
          {engine.toLowerCase() === 'drake' && 'Optimization & control focused'}
          {engine.toLowerCase() === 'pinocchio' && 'Fast rigid body dynamics'}
          {engine.toLowerCase() === 'opensim' && 'Musculoskeletal biomechanics'}
          {engine.toLowerCase() === 'myosim' && 'Muscle & tendon simulation'}
        </p>
      </div>
    </div>
  );
}
