import { useState, useEffect, useCallback, useMemo } from 'react';

export interface SimulationParameters {
  duration: number;
  timestep: number;
  liveAnalysis: boolean;
  gpuAcceleration: boolean;
  model?: string;
}

interface PresetEntry {
  name: string;
  params: {
    duration?: number;
    timestep?: number;
    liveAnalysis?: boolean;
    gpuAcceleration?: boolean;
  };
}

interface Props {
  engine: string;
  disabled?: boolean;
  onChange: (params: SimulationParameters) => void;
}

// Parameter bounds matching physics_parameters.py / presets.py PARAM_BOUNDS
const PARAM_BOUNDS = {
  duration: { min: 0.1, max: 60.0 },
  timestep: { min: 0.001, max: 0.01 },
};

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
  myosuite: {
    duration: 3.0,
    timestep: 0.002,
  },
};

function getEngineDefaults(engine: string): { duration: number; timestep: number } {
  const defaults = ENGINE_DEFAULTS[engine.toLowerCase()] || {};
  return {
    duration: defaults.duration ?? 3.0,
    timestep: defaults.timestep ?? 0.002,
  };
}

/** Client-side validation matching backend PARAM_BOUNDS */
function validateParams(duration: number, timestep: number): string[] {
  const errors: string[] = [];
  if (duration < PARAM_BOUNDS.duration.min || duration > PARAM_BOUNDS.duration.max) {
    errors.push(
      `Duration must be ${PARAM_BOUNDS.duration.min}–${PARAM_BOUNDS.duration.max}s (got ${duration})`
    );
  }
  if (timestep < PARAM_BOUNDS.timestep.min || timestep > PARAM_BOUNDS.timestep.max) {
    errors.push(
      `Timestep must be ${PARAM_BOUNDS.timestep.min}–${PARAM_BOUNDS.timestep.max}s (got ${timestep})`
    );
  }
  return errors;
}

export function ParameterPanel({ engine, disabled, onChange }: Props) {
  const engineDefaults = useMemo(() => getEngineDefaults(engine), [engine]);

  const [duration, setDuration] = useState(engineDefaults.duration);
  const [timestep, setTimestep] = useState(engineDefaults.timestep);
  const [liveAnalysis, setLiveAnalysis] = useState(true);
  const [gpuAcceleration, setGpuAcceleration] = useState(false);

  // Preset state
  const [presets, setPresets] = useState<PresetEntry[]>([]);
  const [presetName, setPresetName] = useState('');
  const [presetStatus, setPresetStatus] = useState<string | null>(null);
  const [selectedPreset, setSelectedPreset] = useState('');

  // Reset values when engine changes
  useEffect(() => {
    setDuration(engineDefaults.duration);
    setTimestep(engineDefaults.timestep);
  }, [engineDefaults]);

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

  // Load presets from backend on mount
  const fetchPresets = useCallback(async () => {
    try {
      const res = await fetch('/api/v1/presets');
      if (res.ok) {
        const data: { presets: PresetEntry[] } = await res.json();
        setPresets(data.presets ?? []);
      }
    } catch {
      // Network error — silently ignore, presets are optional
    }
  }, []);

  useEffect(() => {
    void fetchPresets();
  }, [fetchPresets]);

  // Save current params as a preset
  const handleSavePreset = useCallback(async () => {
    const name = presetName.trim();
    if (!name) {
      setPresetStatus('Enter a preset name first');
      return;
    }

    const validationErrors = validateParams(duration, timestep);
    if (validationErrors.length > 0) {
      setPresetStatus(validationErrors.join('; '));
      return;
    }

    try {
      const res = await fetch('/api/v1/presets', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name,
          params: { duration, timestep, liveAnalysis, gpuAcceleration },
        }),
      });
      if (res.ok) {
        setPresetStatus(`Saved "${name}"`);
        setPresetName('');
        await fetchPresets();
      } else {
        const err: { detail?: string | { message?: string } } = await res.json();
        const msg =
          typeof err.detail === 'string'
            ? err.detail
            : err.detail?.message ?? 'Failed to save preset';
        setPresetStatus(msg);
      }
    } catch {
      setPresetStatus('Network error saving preset');
    }
  }, [presetName, duration, timestep, liveAnalysis, gpuAcceleration, fetchPresets]);

  // Load a preset by name
  const handleLoadPreset = useCallback(
    (name: string) => {
      const entry = presets.find((p) => p.name === name);
      if (!entry) return;
      if (entry.params.duration !== undefined)
        setDuration(entry.params.duration);
      if (entry.params.timestep !== undefined)
        setTimestep(entry.params.timestep);
      if (entry.params.liveAnalysis !== undefined)
        setLiveAnalysis(entry.params.liveAnalysis);
      if (entry.params.gpuAcceleration !== undefined)
        setGpuAcceleration(entry.params.gpuAcceleration);
      setSelectedPreset(name);
      setPresetStatus(`Loaded "${name}"`);
    },
    [presets]
  );

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

      {/* Presets */}
      <div className="border-t border-gray-700 pt-4 space-y-2">
        <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
          Presets
        </h4>

        {/* Load Preset */}
        {presets.length > 0 && (
          <div>
            <label htmlFor="preset-load-select" className="sr-only">
              Load preset
            </label>
            <select
              id="preset-load-select"
              value={selectedPreset}
              onChange={(e) => {
                if (e.target.value) handleLoadPreset(e.target.value);
              }}
              disabled={disabled}
              className="w-full px-2 py-1.5 bg-gray-700 border border-gray-600 rounded-md
                         text-sm text-white focus:outline-none focus:ring-1 focus:ring-blue-400
                         disabled:opacity-50 disabled:cursor-not-allowed"
              aria-label="Load a saved preset"
            >
              <option value="">Load preset…</option>
              {presets.map((p) => (
                <option key={p.name} value={p.name}>
                  {p.name}
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Save Preset */}
        <div className="flex gap-2">
          <input
            type="text"
            value={presetName}
            onChange={(e) => setPresetName(e.target.value)}
            placeholder="Preset name…"
            disabled={disabled}
            maxLength={64}
            className="flex-1 px-2 py-1.5 bg-gray-700 border border-gray-600 rounded-md
                       text-sm text-white placeholder-gray-500
                       focus:outline-none focus:ring-1 focus:ring-blue-400
                       disabled:opacity-50 disabled:cursor-not-allowed"
            aria-label="New preset name"
          />
          <button
            onClick={() => void handleSavePreset()}
            disabled={disabled ?? !presetName.trim()}
            className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 disabled:opacity-50
                       disabled:cursor-not-allowed text-white text-sm rounded-md
                       focus:outline-none focus:ring-2 focus:ring-blue-400 whitespace-nowrap"
            aria-label="Save as preset"
          >
            Save
          </button>
        </div>

        {/* Status feedback */}
        {presetStatus && (
          <p className="text-xs text-blue-300" role="status" aria-live="polite">
            {presetStatus}
          </p>
        )}
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
