/**
 * Types and pure helpers for the Appearance panel (CMB-7 slice c, #11658).
 *
 * Mirrors the shapes returned by the companion
 * `/api/character-builder/appearance/library` and
 * `/api/character-builder/appearance/export` endpoints, which in turn mirror
 * `src/shared/python/model_appearance/schema.py` and `library.py`:
 * materials are named RGBA + roughness/metallic PBR definitions, skin tones
 * and club finishes are flat name lists, and clothing presets map an
 * anatomical *part* (torso, pelvis, upper_arm, shoulder, thigh, shin, foot,
 * hand) to a material name — a part absent from a preset shows skin.
 */

export interface AppearanceMaterial {
  base_color: number[];
  roughness: number;
  metallic: number;
  texture?: Record<string, unknown>;
}

export interface AppearanceLibrary {
  schema_version: string;
  skin_tones: string[];
  clothing: Record<string, Record<string, string>>;
  club_finishes: string[];
  headwear: string[];
  headwear_default_material: Record<string, string>;
  ground_materials: string[];
  materials: Record<string, AppearanceMaterial>;
}

/** The user's current choices; also the exact shape of the export request body. */
export interface AppearancePicks {
  skin_tone: string;
  clothing: string;
  club_finish: string;
  headwear: string;
  headwear_material?: string;
  ground_material: string;
}

/** Hex colours for the `CharacterPreview` mesh segments that take appearance. */
export interface AppearancePalette {
  head: string;
  trunk: string;
  upperArm: string;
  forearm: string;
  thigh: string;
  shank: string;
}

/**
 * Convert an `appearance-v1` RGBA base colour (0..1 components) to a hex
 * colour string, clamping each channel so an out-of-range value never
 * produces an invalid hex digit.
 */
export function rgbaToHex(baseColor: number[]): string {
  const toHex = (c: number) => {
    const byte = Math.round(Math.min(1, Math.max(0, c)) * 255);
    return byte.toString(16).padStart(2, '0');
  };
  const [r = 0, g = 0, b = 0] = baseColor;
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

/** Hex colour of a named material, or `fallback` when the name is unknown. */
function materialHex(
  library: AppearanceLibrary,
  name: string | undefined,
  fallback: string,
): string {
  if (!name) return fallback;
  const material = library.materials[name];
  return material ? rgbaToHex(material.base_color) : fallback;
}

/**
 * Derive preview colours from the current picks.
 *
 * Pure function: given the same library and picks it always returns the
 * same palette, so `CharacterPreview` can treat it as plain render input.
 * The forearm is never dressed by a clothing preset (no preset in the
 * library maps the `forearm` part), so it always shows skin.
 */
export function appearancePalette(
  library: AppearanceLibrary,
  picks: AppearancePicks,
): AppearancePalette {
  const skinHex = materialHex(library, picks.skin_tone, '#cc9966');
  const preset = library.clothing[picks.clothing] ?? {};
  const partHex = (part: string) => materialHex(library, preset[part], skinHex);

  return {
    head: skinHex,
    trunk: partHex('torso'),
    upperArm: partHex('upper_arm'),
    forearm: skinHex,
    thigh: partHex('thigh'),
    shank: partHex('shin'),
  };
}
