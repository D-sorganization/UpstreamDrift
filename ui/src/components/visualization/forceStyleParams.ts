/**
 * Maps the force-overlay panel's scale and group controls to API query
 * parameters (GCV-4, #11710). Mirrors the PyQt Visualization tab.
 */

import {
  STANDARD_GRAVITY_M_S2,
  type ForceStyleOptions,
  type GlyphGroup,
  type ScaleMode,
} from "@/types/glyphs";

export interface ForceScaleControls {
  scaleMode: ScaleMode;
  bodyMassKg: number;
  peakForceN: number;
  referenceLengthM: number;
  groups: GlyphGroup[];
}

/**
 * Convert panel controls to the API style keys.
 *
 * `body_weight` references `bodyMassKg * g`, `peak` references `peakForceN`,
 * and `fixed` sends no reference (the slider `scale_factor` applies).
 * Throws RangeError when a numeric control is not finite and positive.
 */
export function styleOptionsFromControls(
  c: ForceScaleControls,
): ForceStyleOptions {
  for (const [name, value] of [
    ["bodyMassKg", c.bodyMassKg],
    ["peakForceN", c.peakForceN],
    ["referenceLengthM", c.referenceLengthM],
  ] as const) {
    if (!Number.isFinite(value) || value <= 0) {
      throw new RangeError(`${name} must be finite and positive, got ${value}`);
    }
  }
  let reference: number | null = null;
  if (c.scaleMode === "body_weight") {
    reference = c.bodyMassKg * STANDARD_GRAVITY_M_S2;
  } else if (c.scaleMode === "peak") {
    reference = c.peakForceN;
  }
  return {
    scale_mode: c.scaleMode,
    reference_force_n: reference,
    reference_length_m: c.referenceLengthM,
    groups: [...c.groups],
  };
}

/** Append the style options to a query string builder. */
export function appendStyleParams(
  params: URLSearchParams,
  options: ForceStyleOptions,
): void {
  params.set("scale_mode", options.scale_mode);
  params.set("reference_length_m", String(options.reference_length_m));
  if (options.reference_force_n !== null) {
    params.set("reference_force_n", String(options.reference_force_n));
  }
  params.set("groups", options.groups.join(","));
}
