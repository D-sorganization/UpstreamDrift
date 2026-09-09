/** Signed axial force in N. Wire fields match Python ForceColorScale. */
export interface ForceColorScale {
  enabled: boolean;
  tension_limit_n: number;
  compression_limit_n: number;
  deadband_n: number;
  tension_color: string;
  compression_color: string;
  neutral_color: string;
}

export const defaultForceColorScale: Readonly<ForceColorScale> = Object.freeze({
  enabled: false,
  tension_limit_n: 1000,
  compression_limit_n: 1000,
  deadband_n: 0,
  tension_color: '#0000ff',
  compression_color: '#ff0000',
  neutral_color: '#ffffff',
});

export function validateForceColorScale(scale: ForceColorScale): ForceColorScale {
  if (typeof scale.enabled !== 'boolean') throw new TypeError('enabled must be boolean');
  for (const key of ['tension_limit_n', 'compression_limit_n', 'deadband_n'] as const) {
    if (typeof scale[key] !== 'number' || !Number.isFinite(scale[key])) {
      throw new TypeError(`${key} must be finite`);
    }
  }
  if (scale.deadband_n < 0 || Math.min(scale.tension_limit_n, scale.compression_limit_n) <= scale.deadband_n) {
    throw new RangeError('Both limits must exceed the nonnegative neutral band');
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

/** The caller validates settings once at the UI/wire boundary, outside playback. */
export function forceColor(value: number | null | undefined, base: string, scale: ForceColorScale): string {
  if (value != null && typeof value !== 'number') throw new TypeError('Force must be numeric or missing');
  if (!scale.enabled || value == null || !Number.isFinite(value)) return base;
  if (Math.abs(value) <= scale.deadband_n) return scale.neutral_color.toLowerCase();
  const endpoint = value > 0 ? scale.tension_color : scale.compression_color;
  const limit = value > 0 ? scale.tension_limit_n : scale.compression_limit_n;
  const fraction = Math.min(1, (Math.abs(value) - scale.deadband_n) / (limit - scale.deadband_n));
  return '#' + [1, 3, 5].map(offset => {
    const start = parseInt(scale.neutral_color.slice(offset, offset + 2), 16);
    const end = parseInt(endpoint.slice(offset, offset + 2), 16);
    return Math.round(start + fraction * (end - start)).toString(16).padStart(2, '0');
  }).join('');
}
