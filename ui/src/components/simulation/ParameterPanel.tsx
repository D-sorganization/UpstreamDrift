import { useState } from 'react';
import { Input, Select } from '@/components/ui';
import type { SimulationParameters } from '@/stores/useSimulationStore';

interface Props {
  engine: string;
  disabled?: boolean;
  /** Controlled parameter values (owned by useSimulationStore, #7424). */
  value: SimulationParameters;
  /** Apply a partial change to the store. */
  onChange: (params: Partial<SimulationParameters>) => void;
  /** Reset duration/timestep to the current engine's defaults. */
  onResetDefaults?: () => void;
}

/**
 * ParameterPanel is a controlled view over the simulation store (#7424).
 *
 * It holds NO mirror copy of duration/timestep/toggles — the store is the
 * single source of truth, so remounting (navigate away and back) or switching
 * engines can no longer clobber user-set values. The numeric inputs keep only a
 * transient raw-string buffer while editing so clearing the field does not snap
 * the committed value to a default mid-edit; the parsed value is committed on
 * change when valid and on blur otherwise.
 */
export function ParameterPanel({
  engine,
  disabled,
  value,
  onChange,
  onResetDefaults,
}: Props) {
  // Transient edit buffer for the duration field; null = mirror the store.
  const [durationDraft, setDurationDraft] = useState<string | null>(null);

  const commitDuration = (raw: string) => {
    const parsed = Number(raw);
    if (raw.trim() !== '' && !Number.isNaN(parsed)) {
      onChange({ duration: parsed });
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">
          Simulation Parameters
        </h3>
        {onResetDefaults && (
          <button
            type="button"
            onClick={onResetDefaults}
            disabled={disabled}
            className="text-xs text-blue-400 hover:text-blue-300 disabled:text-gray-600 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-blue-400 rounded px-1"
          >
            Reset to engine defaults
          </button>
        )}
      </div>

      {/* Duration */}
      <div>
        <label
          htmlFor="duration-input"
          className="block text-sm font-medium text-gray-300 mb-1"
        >
          Duration (seconds)
        </label>
        <Input
          id="duration-input"
          type="number"
          min="0.1"
          max="60"
          step="0.1"
          value={durationDraft ?? value.duration}
          onChange={(e) => {
            // Keep the raw string while editing so an empty field does not snap
            // to a default; commit the parsed value when it is valid.
            setDurationDraft(e.target.value);
            commitDuration(e.target.value);
          }}
          onBlur={(e) => {
            commitDuration(e.target.value);
            setDurationDraft(null);
          }}
          disabled={disabled}
          className="w-full"
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
        <Select
          id="timestep-input"
          value={value.timestep}
          onChange={(e) => onChange({ timestep: parseFloat(e.target.value) })}
          disabled={disabled}
          className="w-full"
          aria-describedby="timestep-help"
        >
          <option value="0.001">0.001s (High precision)</option>
          <option value="0.002">0.002s (Default)</option>
          <option value="0.005">0.005s (Fast)</option>
          <option value="0.01">0.01s (Very fast)</option>
        </Select>
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
          aria-checked={value.liveAnalysis}
          onClick={() => onChange({ liveAnalysis: !value.liveAnalysis })}
          disabled={disabled}
          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors
                     focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-offset-2 focus:ring-offset-gray-800
                     disabled:opacity-50 disabled:cursor-not-allowed
                     ${value.liveAnalysis ? 'bg-blue-600' : 'bg-gray-600'}`}
        >
          <span
            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform
                       ${value.liveAnalysis ? 'translate-x-6' : 'translate-x-1'}`}
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
          aria-checked={value.gpuAcceleration}
          onClick={() => onChange({ gpuAcceleration: !value.gpuAcceleration })}
          disabled={disabled}
          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors
                     focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-offset-2 focus:ring-offset-gray-800
                     disabled:opacity-50 disabled:cursor-not-allowed
                     ${value.gpuAcceleration ? 'bg-green-600' : 'bg-gray-600'}`}
        >
          <span
            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform
                       ${value.gpuAcceleration ? 'translate-x-6' : 'translate-x-1'}`}
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
