/**
 * Types and constants for the spec-native Character Builder panel
 * (CMB-7a, #11658). Mirrors the shapes returned by
 * `src/shared/python/humanoid_character_builder/spec_export.py` and the
 * bounds in `CharacterSpecRequest` (`src/api/models/requests.py`).
 */

export interface SpecCharacterParameters {
  stature_m: number;
  mass_kg: number;
  trunk_scale: number;
  arm_scale: number;
  shoulder_scale: number;
  grip_roll_deg: number;
  club: string;
}

export interface CharacterPresetSummary {
  id: string;
  name: string;
  description: string;
  category: string;
  parameters: SpecCharacterParameters;
  provenance: string;
  limitations: string;
}

interface PreviewBody {
  name: string;
  mass_kg: number;
}

interface PreviewJoint {
  name: string;
  parent: string;
  child: string;
}

export interface CharacterPreviewResponse {
  spec_sha256: string;
  stature_m: number;
  bodies: PreviewBody[];
  joints: PreviewJoint[];
}

export interface CharacterBuildSummary {
  preset: string | null;
  spec_sha256: string;
  schema_version: string;
  qualification: string;
  bodies: number;
  joints: number;
  coordinates: number;
  total_mass_kg: number;
  club: string;
}

export type ExportFormat = 'spec' | 'urdf' | 'mjcf' | 'osim';

export interface SliderParams {
  stature_m: number;
  mass_kg: number;
  trunk_scale: number;
  arm_scale: number;
  shoulder_scale: number;
  club: string;
}

// Mirrors SpecCharacterParameters defaults in spec_params.py.
export const DEFAULT_PARAMS: SliderParams = {
  stature_m: 1.75,
  mass_kg: 75,
  trunk_scale: 1.0,
  arm_scale: 1.0,
  shoulder_scale: 1.0,
  club: 'driver',
};

// Matches EXPORT_FORMATS in spec_export.py — the fallback filename when the
// server's Content-Disposition header is missing or unparsable.
export const EXPORT_BUTTONS: { fmt: ExportFormat; label: string; ext: string }[] = [
  { fmt: 'spec', label: 'Spec JSON', ext: 'json' },
  { fmt: 'urdf', label: 'URDF', ext: 'urdf' },
  { fmt: 'mjcf', label: 'MJCF', ext: 'xml' },
  { fmt: 'osim', label: 'OpenSim', ext: 'osim' },
];

export function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
