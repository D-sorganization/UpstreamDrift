/**
 * Grip picks for the Appearance panel (HRG-GR4, #13205).
 *
 * Mirrors `src/shared/python/motion_matching/execution/grip_selection.py`
 * (`resolve_grip`): the same precedence (pick, then the document's grip, then
 * the defaults) and the same split between two kinds of pick.
 *
 * - Physical picks (strength preset, rotation about the shaft, palm diagonal,
 *   set angle, depth, hand spacing, grip style) change the model's physics.
 *   They are sent to `/api/character-builder/grip` and make a new model
 *   revision that needs a refit. They are never part of the appearance export.
 * - Visual picks (lead thumb, roll/strength offsets) are picture-only and go
 *   in the `.appearance.json` sidecar's `grip` block.
 *
 * `tests/fixtures/grip/grip_roundtrip.json` holds the resolved grips both
 * sides must produce. The Python resolver is the authority for ranges: the
 * checks here only stop an obviously wrong request.
 */

export const GRIP_PRESETS = { weak: -15, neutral: 0, strong: 15 } as const;
export type GripPreset = keyof typeof GRIP_PRESETS;
export const GRIP_PRESET_NAMES = Object.keys(GRIP_PRESETS) as GripPreset[];
export const GRIP_STYLES = ['overlap', 'interlock', 'ten_finger'] as const;
export type GripStyle = (typeof GRIP_STYLES)[number];

/** Published neutral palm diagonal (deg): the shaft crosses the palm at 40. */
export const NEUTRAL_PALM_DIAGONAL_DEG = 40;
export const DEFAULT_LEAD_BELOW_BUTT_M = 0.032;
export const DEFAULT_HAND_SPACING_M = 0.0762;

export interface HandAngles {
  alpha_deg: number;
  psi_deg: number;
  gamma_deg: number;
}

/** The physical grip, in the key order of the Python `grip_spec_to_physical`. */
export interface PhysicalGrip {
  lead: HandAngles;
  trail: HandAngles;
  lead_below_butt_m: number;
  hand_spacing_m: number;
  handedness: 'right' | 'left';
  style?: GripStyle;
}

export interface PhysicalGripPicks {
  preset?: GripPreset;
  lead_shaft_rotation?: number;
  trail_shaft_rotation?: number;
  lead_palm_diagonal?: number;
  trail_palm_diagonal?: number;
  lead_set_angle?: number;
  trail_set_angle?: number;
  lead_below_butt?: number;
  hand_spacing?: number;
  grip_style?: GripStyle;
}

export interface VisualGripPicks {
  grip_thumb_offset_deg?: number;
}

export type GripPicks = PhysicalGripPicks & VisualGripPicks;

export const DEFAULT_GRIP: PhysicalGrip = {
  lead: { alpha_deg: 0, psi_deg: NEUTRAL_PALM_DIAGONAL_DEG, gamma_deg: 0 },
  trail: { alpha_deg: 0, psi_deg: NEUTRAL_PALM_DIAGONAL_DEG, gamma_deg: 0 },
  lead_below_butt_m: DEFAULT_LEAD_BELOW_BUTT_M,
  hand_spacing_m: DEFAULT_HAND_SPACING_M,
  handedness: 'right',
};

const PHYSICAL_KEYS: ReadonlyArray<keyof PhysicalGripPicks> = [
  'preset',
  'lead_shaft_rotation',
  'trail_shaft_rotation',
  'lead_palm_diagonal',
  'trail_palm_diagonal',
  'lead_set_angle',
  'trail_set_angle',
  'lead_below_butt',
  'hand_spacing',
  'grip_style',
];

const RANGES = {
  alpha_deg: [-60, 60],
  psi_deg: [5, 85],
  gamma_deg: [-180, 180],
} as const;

/** True when any pick changes the physics (so a new revision and a refit). */
export function hasPhysicalPick(picks: GripPicks): boolean {
  return PHYSICAL_KEYS.some((k) => picks[k] !== undefined);
}

function checked(name: keyof HandAngles, value: number): number {
  const [lo, hi] = RANGES[name];
  if (!Number.isFinite(value) || value < lo || value > hi) {
    throw new Error(`${name} must be in [${lo}, ${hi}], got ${value}`);
  }
  return value;
}

function withPick(hand: HandAngles, name: keyof HandAngles, value: number | undefined): HandAngles {
  return value === undefined ? hand : { ...hand, [name]: checked(name, value) };
}

/**
 * The physical grip the picks resolve to: `base` (the document), then the
 * preset (both hands at that strength, palm diagonal 40, set angle 0), then
 * every explicit pick.
 */
export function resolveGrip(picks: GripPicks, base: PhysicalGrip = DEFAULT_GRIP): PhysicalGrip {
  let { lead, trail } = base;
  if (picks.preset !== undefined) {
    if (!(picks.preset in GRIP_PRESETS)) {
      throw new Error(`grip preset must be one of ${GRIP_PRESET_NAMES.join(', ')}, got ${picks.preset}`);
    }
    const hand = {
      alpha_deg: GRIP_PRESETS[picks.preset],
      psi_deg: NEUTRAL_PALM_DIAGONAL_DEG,
      gamma_deg: 0,
    };
    lead = { ...hand };
    trail = { ...hand };
  }
  lead = withPick(lead, 'alpha_deg', picks.lead_shaft_rotation);
  lead = withPick(lead, 'psi_deg', picks.lead_palm_diagonal);
  lead = withPick(lead, 'gamma_deg', picks.lead_set_angle);
  trail = withPick(trail, 'alpha_deg', picks.trail_shaft_rotation);
  trail = withPick(trail, 'psi_deg', picks.trail_palm_diagonal);
  trail = withPick(trail, 'gamma_deg', picks.trail_set_angle);
  const spacing = picks.hand_spacing ?? base.hand_spacing_m;
  if (!(spacing > 0)) throw new Error(`hand_spacing must be positive, got ${spacing}`);
  const out: PhysicalGrip = {
    lead,
    trail,
    lead_below_butt_m: picks.lead_below_butt ?? base.lead_below_butt_m,
    hand_spacing_m: spacing,
    handedness: base.handedness,
  };
  const style = picks.grip_style ?? base.style;
  if (style !== undefined && style !== 'overlap') {
    out.style = style;
  }
  return out;
}

/** The sidecar `grip` block of the visual picks (only the picked keys). */
export function visualGripBlock(picks: VisualGripPicks): Record<string, number | string> {
  const block: Record<string, number | string> = {};
  if (picks.grip_thumb_offset_deg !== undefined) {
    const t = picks.grip_thumb_offset_deg;
    if (!Number.isFinite(t) || t < 0 || t > 45) {
      throw new Error(`grip_thumb_offset_deg must be in [0, 45], got ${t}`);
    }
    block.lead_thumb_offset_deg = t;
  }
  return block;
}

/** The request body for `/api/character-builder/grip` (set picks only). */
export function gripRequestBody(picks: GripPicks): GripPicks {
  return Object.fromEntries(
    Object.entries(picks).filter(([, v]) => v !== undefined),
  ) as GripPicks;
}
