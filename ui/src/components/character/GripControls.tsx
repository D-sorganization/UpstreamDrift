import { useCallback, useState } from 'react';
import { apiFetch } from '@/api/fetch';
import { errorMessage, type CharacterSpecRequestBody } from './characterSpecTypes';
import {
  GRIP_PRESET_NAMES,
  GRIP_STYLES,
  gripRequestBody,
  hasPhysicalPick,
  type GripPicks,
  type GripPreset,
  type GripStyle,
} from './gripSelection';

interface GripRevision {
  spec_sha256: string;
  physics_spec_sha256: string;
  parent_physics_spec_sha256: string;
  refit_required: boolean;
}

interface GripControlsProps {
  picks: GripPicks;
  onChange: (next: GripPicks) => void;
  character?: CharacterSpecRequestBody | null;
}

/** [pick, label, min, max, step] of the physical sliders, in golf words. */
const SLIDERS: ReadonlyArray<readonly [keyof GripPicks, string, number, number, number]> = [
  ['lead_shaft_rotation', 'Lead Hand Rotation (deg, + Stronger)', -60, 60, 1],
  ['trail_shaft_rotation', 'Trail Hand Rotation (deg, + Stronger)', -60, 60, 1],
  ['lead_palm_diagonal', 'Lead Palm Diagonal (deg)', 5, 85, 1],
  ['trail_palm_diagonal', 'Trail Palm Diagonal (deg)', 5, 85, 1],
  ['hand_spacing', 'Hand Spacing (m)', 0.01, 0.2, 0.0025],
];
const VISUAL_SLIDERS: ReadonlyArray<readonly [keyof GripPicks, string, number, number, number]> = [
  ['grip_thumb_offset_deg', 'Lead Thumb Offset (deg, Picture Only)', 0, 45, 1],
  ['grip_strength_deg', 'Grip Strength Offset (deg, Picture Only)', -30, 30, 1],
  ['lead_roll_offset_deg', 'Lead Roll Offset (deg, Picture Only)', -30, 30, 1],
  ['trail_roll_offset_deg', 'Trail Roll Offset (deg, Picture Only)', -30, 30, 1],
];
const SLIDER_DEFAULT: Partial<Record<keyof GripPicks, number>> = {
  lead_shaft_rotation: 0,
  trail_shaft_rotation: 0,
  lead_palm_diagonal: 40,
  trail_palm_diagonal: 40,
  hand_spacing: 0.0762,
  grip_thumb_offset_deg: 10,
  grip_strength_deg: 0,
  lead_roll_offset_deg: 0,
  trail_roll_offset_deg: 0,
};

function Slider({
  name,
  label,
  min,
  max,
  step,
  value,
  onChange,
}: {
  name: string;
  label: string;
  min: number;
  max: number;
  step: number;
  value: number | undefined;
  onChange: (v: number | undefined) => void;
}) {
  const id = `grip-${name}`;
  return (
    <div>
      <label htmlFor={id} className="block text-xs font-semibold text-gray-300 mb-1">
        {label}: {value === undefined ? 'from the model' : value}
      </label>
      <div className="flex items-center gap-2">
        <input
          id={id}
          type="range"
          min={min}
          max={max}
          step={step}
          value={value ?? SLIDER_DEFAULT[name as keyof GripPicks] ?? 0}
          onChange={(e) => onChange(Number(e.target.value))}
          className="w-full"
        />
        <button
          type="button"
          aria-label={`Reset ${label}`}
          disabled={value === undefined}
          onClick={() => onChange(undefined)}
          className="text-xs text-gray-300 hover:text-white disabled:opacity-40"
        >
          Reset
        </button>
      </div>
    </div>
  );
}

/**
 * Grip presets and sliders (HRG-GR4, #13205, #13346).
 *
 * Strength, style, palm diagonal and spacing are physical: the Build button
 * asks the server for the new model revision (hashes), which needs a refit.
 * The lead thumb and roll offsets are picture-only and go in the appearance
 * export.
 */
export function GripControls({ picks, onChange, character }: GripControlsProps) {
  const [revision, setRevision] = useState<GripRevision | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const physical = hasPhysicalPick(picks);

  const set = (patch: GripPicks) => {
    setRevision(null);
    onChange({ ...picks, ...patch });
  };

  const build = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      const body = gripRequestBody(picks);
      const result = await apiFetch<GripRevision>('/api/character-builder/grip', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(character ? { ...body, character } : body),
      });
      setRevision(result);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  }, [picks, character]);

  return (
    <div className="space-y-3" data-testid="grip-controls">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Grip</h3>
      <div>
        <label htmlFor="grip-preset" className="block text-xs font-semibold text-gray-300 mb-1">
          Grip Strength
        </label>
        <select
          id="grip-preset"
          value={picks.preset ?? ''}
          onChange={(e) => set({ preset: (e.target.value || undefined) as GripPreset | undefined })}
          className="w-full bg-gray-700 border-none text-gray-200 rounded px-2 py-1.5 text-sm"
        >
          <option value="">From the model</option>
          {GRIP_PRESET_NAMES.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
      </div>
      {SLIDERS.map(([name, label, min, max, step]) => (
        <Slider
          key={name}
          name={name}
          label={label}
          min={min}
          max={max}
          step={step}
          value={picks[name] as number | undefined}
          onChange={(v) => set({ [name]: v } as GripPicks)}
        />
      ))}
      <div>
        <label htmlFor="grip-style" className="block text-xs font-semibold text-gray-300 mb-1">
          Grip Style
        </label>
        <select
          id="grip-style"
          value={picks.grip_style ?? ''}
          onChange={(e) => set({ grip_style: (e.target.value || undefined) as GripStyle | undefined })}
          className="w-full bg-gray-700 border-none text-gray-200 rounded px-2 py-1.5 text-sm"
        >
          <option value="">From the model</option>
          {GRIP_STYLES.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
      </div>
      {VISUAL_SLIDERS.map(([name, label, min, max, step]) => (
        <Slider
          key={name}
          name={name}
          label={label}
          min={min}
          max={max}
          step={step}
          value={picks[name] as number | undefined}
          onChange={(v) => set({ [name]: v } as GripPicks)}
        />
      ))}
      <p className="text-xs text-gray-400">
        Strength, style, palm diagonal and spacing change the physics: they make a new model
        revision that needs a refit. Thumb and roll offsets change the picture only.
      </p>
      <button
        type="button"
        onClick={() => void build()}
        disabled={busy || !physical}
        className="w-full bg-gray-600 hover:bg-gray-500 disabled:opacity-40 text-white rounded py-2 text-sm font-semibold"
      >
        {busy ? 'Building…' : 'Build Grip Revision'}
      </button>
      {revision && (
        <p data-testid="grip-revision" className="text-xs text-gray-300 break-all">
          New revision {revision.spec_sha256.slice(0, 12)} (physics{' '}
          {revision.physics_spec_sha256.slice(0, 12)}).
          {revision.refit_required ? ' Needs a refit.' : ''}
        </p>
      )}
      {error && (
        <div className="text-xs text-red-400 bg-red-950/30 p-2.5 rounded border border-red-900/50">
          {error}
        </div>
      )}
    </div>
  );
}
