/** Signed axial force in N. Wire fields match Python ForceColorScale. */
export interface ForceColorScale {
  enabled: boolean;
  tension_limit_n: number;
  compression_limit_n: number;
  deadband_n: number;
  tension_color: string;
  compression_color: string;
  neutral_color: string;
  /** Overlay strength in [0, 1]; 1 keeps the legacy fully opaque look. */
  opacity: number;
  /** replace: blend load color over the base; tint: shift the base toward the load color. */
  blend_mode: BlendMode;
}

export const BLEND_MODES = ['replace', 'tint'] as const;
export type BlendMode = (typeof BLEND_MODES)[number];

export const defaultForceColorScale: Readonly<ForceColorScale> = Object.freeze({
  enabled: false,
  tension_limit_n: 1000,
  compression_limit_n: 1000,
  deadband_n: 0,
  tension_color: '#0000ff',
  compression_color: '#ff0000',
  neutral_color: '#ffffff',
  opacity: 1,
  blend_mode: 'replace',
});

export function validateForceColorScale(input: Partial<ForceColorScale>): ForceColorScale {
  // Legacy persisted settings lack opacity/blend_mode; they default to the old look.
  const scale: ForceColorScale = { ...defaultForceColorScale, ...input };
  if (typeof scale.enabled !== 'boolean') throw new TypeError('enabled must be boolean');
  for (const key of ['tension_limit_n', 'compression_limit_n', 'deadband_n'] as const) {
    if (typeof scale[key] !== 'number' || !Number.isFinite(scale[key])) {
      throw new TypeError(`${key} must be finite`);
    }
  }
  if (scale.deadband_n < 0 || Math.min(scale.tension_limit_n, scale.compression_limit_n) <= scale.deadband_n) {
    throw new RangeError('Both limits must exceed the nonnegative neutral band');
  }
  if (typeof scale.opacity !== 'number' || !Number.isFinite(scale.opacity)) {
    throw new TypeError('opacity must be finite');
  }
  if (scale.opacity < 0 || scale.opacity > 1) throw new RangeError('opacity must be within [0, 1]');
  if (!(BLEND_MODES as readonly string[]).includes(scale.blend_mode)) {
    throw new RangeError(`blend_mode must be one of ${BLEND_MODES.join(', ')}`);
  }
  for (const key of ['tension_color', 'compression_color', 'neutral_color'] as const) {
    if (typeof scale[key] !== 'string' || !/^#[0-9a-fA-F]{6}$/.test(scale[key])) {
      throw new TypeError(`${key} must be an opaque #RRGGBB color`);
    }
  }
  const unknown = Object.keys(scale).filter(key => !(key in defaultForceColorScale));
  if (unknown.length) throw new TypeError(`Unknown force-color settings: ${unknown.join(', ')}`);
  return { ...scale };
}

const channels = (hex: string): number[] => [1, 3, 5].map(o => parseInt(hex.slice(o, o + 2), 16));

/** Neutral-to-endpoint endpoint and saturation fraction; the deadband yields fraction 0. */
function endpointFraction(value: number, scale: ForceColorScale): [string, number] {
  if (Math.abs(value) <= scale.deadband_n) return [scale.neutral_color.toLowerCase(), 0];
  const endpoint = value > 0 ? scale.tension_color : scale.compression_color;
  const limit = value > 0 ? scale.tension_limit_n : scale.compression_limit_n;
  return [endpoint.toLowerCase(), Math.min(1, (Math.abs(value) - scale.deadband_n) / (limit - scale.deadband_n))];
}

const toHex = (values: number[]): string =>
  '#' + values.map(v => Math.round(v).toString(16).padStart(2, '0')).join('');

/** The caller validates settings once at the UI/wire boundary, outside playback. */
export function forceColor(value: number | null | undefined, base: string, scale: ForceColorScale): string {
  if (value != null && typeof value !== 'number') throw new TypeError('Force must be numeric or missing');
  if (!scale.enabled || value == null || !Number.isFinite(value)) return base;
  const [endpoint, fraction] = endpointFraction(value, scale);
  const start = channels(scale.neutral_color);
  const end = channels(endpoint);
  return toHex(start.map((s, i) => s + fraction * (end[i] - s)));
}

/**
 * Translucent overlay: blend the load color over a #RRGGBB base by opacity and mode.
 * Returns null when disabled or the sample is unavailable (the caller keeps its base);
 * mirrors Python ForceColorScale.shade_hex.
 */
export function forceShade(value: number | null | undefined, base: string, scale: ForceColorScale): string | null {
  if (value != null && typeof value !== 'number') throw new TypeError('Force must be numeric or missing');
  if (!scale.enabled || value == null || !Number.isFinite(value)) return null;
  if (scale.opacity === 1 && scale.blend_mode === 'replace') return forceColor(value, base, scale);
  const [endpoint, fraction] = endpointFraction(value, scale);
  const start = channels(base);
  const tint = scale.blend_mode === 'tint';
  const target = tint ? channels(endpoint)
    : channels(forceColor(value, base, { ...scale, opacity: 1, blend_mode: 'replace' }));
  const weight = tint ? scale.opacity * fraction : scale.opacity;
  return toHex(start.map((s, i) => s + weight * (target[i] - s)));
}
