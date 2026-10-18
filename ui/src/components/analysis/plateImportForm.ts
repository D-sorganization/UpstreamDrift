/**
 * Text form behind the web plate declarations form (OF-07b, #13122).
 *
 * Mirrors `src/tools/ground_reaction_plots/plate_import_form.py` (the PyQt6
 * dialog is canonical): the same rows, units and initial values, plus `trial`,
 * the swing file relative to the server's plate-trial root. Physical facts
 * start empty: they are declared, never assumed. The server validates again
 * and stays authoritative; this only refuses what cannot be sent.
 */
import type { PlateAlignment, PlateOverlayRequest, PlateReading } from '@/api/groundReaction';

export type PlateFormSection = 'Trial' | 'Plate' | 'Stance' | 'Clock';

export interface PlateFormField {
  key: string;
  label: string;
  section: PlateFormSection;
  required: boolean;
  choices?: readonly string[];
  initial: string;
}

/**
 * Alignments the form offers. A `cross_correlation` offset was fitted against
 * a model, so the form also asks which one: it can never be evidence for it.
 */
export const ALIGNMENT_KINDS = [
  'manual',
  'shot_impact',
  'absolute_utc',
  'cross_correlation',
] as const;
export const HANDEDNESS_CHOICES = ['RH', 'LH'] as const;
/**
 * How the two CoP channels' timing skews relate, conservative first. Only
 * `independent` lets a skewed reading carry the CoP covariance.
 */
export const COP_SKEW_CHOICES = ['unknown', 'independent', 'common'] as const;

const field = (
  key: string,
  label: string,
  section: PlateFormSection,
  extra: Partial<Pick<PlateFormField, 'required' | 'choices' | 'initial'>> = {},
): PlateFormField => ({ key, label, section, required: true, initial: '', ...extra });

export const PLATE_FORM_FIELDS: readonly PlateFormField[] = [
  field('trial', 'Swing file (relative to the plate-trial root)', 'Trial'),
  field('device', 'Device label (optional)', 'Plate', { required: false }),
  field('uncertainty_source', 'Uncertainty source', 'Plate'),
  field('force_floor_n', 'Load uncertainty floor [N]', 'Plate'),
  field('force_fraction', 'Load uncertainty fraction [-]', 'Plate'),
  field('cop_m', 'CoP uncertainty [m]', 'Plate'),
  field('sensing_depth_m', 'Sensing plane depth [m]', 'Plate'),
  field('sensing_depth_sigma_m', 'Sensing depth uncertainty [m]', 'Plate'),
  field('capacity_n', 'Pair capacity [N]', 'Plate'),
  field('skew_s', 'Channel timing skew within a sample [s]', 'Plate'),
  field('cop_skew', 'CoP channel skews', 'Plate', {
    choices: COP_SKEW_CHOICES,
    initial: COP_SKEW_CHOICES[0],
  }),
  field('handedness', 'Handedness', 'Stance', { choices: HANDEDNESS_CHOICES, initial: 'RH' }),
  field('origin_m', 'Stance origin x, y, z [m]', 'Stance'),
  field('target_dir', 'Target direction x, y, z', 'Stance', { initial: '0, -1, 0' }),
  field('offset_s', 'Clock offset, model = trial + offset [s]', 'Clock'),
  field('sigma_offset_s', 'Clock offset uncertainty [s]', 'Clock'),
  field('alignment_kind', 'Alignment kind', 'Clock', {
    choices: ALIGNMENT_KINDS,
    initial: ALIGNMENT_KINDS[0],
  }),
  field(
    'derived_from_model',
    'Model the offset was fitted against (cross-correlation only)',
    'Clock',
    { required: false },
  ),
  field('alignment_note', 'How the offset was found', 'Clock'),
];

export interface PlateFormError {
  key: string;
  message: string;
}

export type PlateFormResult =
  | { request: PlateOverlayRequest; error: null }
  | { request: null; error: PlateFormError };

type Vector3 = [number, number, number];

class FormRefusal extends Error {
  constructor(
    readonly key: string,
    message: string,
  ) {
    super(message);
  }
}

/** Every row at its initial text. */
export function initialPlateFormTexts(): Record<string, string> {
  return Object.fromEntries(PLATE_FORM_FIELDS.map((f) => [f.key, f.initial]));
}

const BY_KEY = new Map(PLATE_FORM_FIELDS.map((f) => [f.key, f]));

function text(texts: Record<string, string>, key: string): string {
  return (texts[key] ?? BY_KEY.get(key)?.initial ?? '').trim();
}

function number(texts: Record<string, string>, key: string): number {
  const raw = text(texts, key);
  const value = Number(raw);
  if (raw === '' || !Number.isFinite(value)) {
    throw new FormRefusal(key, `expected a finite number, got '${raw}'`);
  }
  return value;
}

function vector(texts: Record<string, string>, key: string): Vector3 {
  const parts = text(texts, key).split(',').map((p) => p.trim());
  const values = parts.map(Number);
  if (parts.length !== 3 || parts.some((p) => p === '') || !values.every(Number.isFinite)) {
    throw new FormRefusal(key, 'expected three finite comma-separated numbers');
  }
  return values as Vector3;
}

function choice<T extends string>(texts: Record<string, string>, key: string): T {
  const value = text(texts, key);
  const choices = BY_KEY.get(key)?.choices ?? [];
  if (!choices.includes(value)) {
    throw new FormRefusal(key, `must be one of ${choices.join(', ')}`);
  }
  return value as T;
}

function trialPath(texts: Record<string, string>): string {
  const trial = text(texts, 'trial');
  const parts = trial.split('/');
  if (
    trial.startsWith('/') ||
    trial.includes('\\') ||
    trial.includes('\0') ||
    parts.some((p) => p === '..' || p === '.')
  ) {
    throw new FormRefusal('trial', 'must be a relative path inside the plate-trial root');
  }
  return trial;
}

/** Only a cross-correlated offset names the model it was fitted against. */
function derivedFromModel(
  texts: Record<string, string>,
  kind: PlateAlignment['kind'],
): string | null {
  const model = text(texts, 'derived_from_model');
  if (kind === 'cross_correlation' && model === '') {
    throw new FormRefusal('derived_from_model', 'required for a cross-correlation alignment');
  }
  if (kind !== 'cross_correlation' && model !== '') {
    throw new FormRefusal('derived_from_model', 'only a cross-correlation alignment names a model');
  }
  return model === '' ? null : model;
}

function requireRows(texts: Record<string, string>): void {
  for (const f of PLATE_FORM_FIELDS) {
    if (f.required && text(texts, f.key) === '') throw new FormRefusal(f.key, 'required');
  }
}

function build(texts: Record<string, string>): PlateOverlayRequest {
  requireRows(texts);
  const device = text(texts, 'device');
  const kind = choice<PlateAlignment['kind']>(texts, 'alignment_kind');
  return {
    trial: trialPath(texts),
    handedness: choice<'RH' | 'LH'>(texts, 'handedness'),
    origin_m: vector(texts, 'origin_m'),
    target_dir: vector(texts, 'target_dir'),
    declared: {
      force_floor_n: number(texts, 'force_floor_n'),
      force_fraction: number(texts, 'force_fraction'),
      cop_m: number(texts, 'cop_m'),
      sensing_depth_sigma_m: number(texts, 'sensing_depth_sigma_m'),
      uncertainty_source: text(texts, 'uncertainty_source'),
      sensing_depth_m: number(texts, 'sensing_depth_m'),
      capacity_n: number(texts, 'capacity_n'),
      device: device === '' ? null : device,
    },
    alignment: {
      offset_s: number(texts, 'offset_s'),
      sigma_offset_s: number(texts, 'sigma_offset_s'),
      kind,
      note: text(texts, 'alignment_note'),
      derived_from_model: derivedFromModel(texts, kind),
    },
    reading: {
      skew_s: number(texts, 'skew_s'),
      cop_skew: choice<PlateReading['cop_skew']>(texts, 'cop_skew'),
    },
  };
}

/**
 * The overlay request the form texts declare, or the topmost field at fault.
 * Numbers must be finite and vectors three finite numbers; ranges (positive
 * capacity, horizontal target line, and so on) are left to the server.
 */
export function requestFromPlateForm(texts: Record<string, string>): PlateFormResult {
  try {
    return { request: build(texts), error: null };
  } catch (e) {
    if (e instanceof FormRefusal) return { request: null, error: { key: e.key, message: e.message } };
    throw e;
  }
}
