/**
 * Custom colour writer for the Appearance panel (HRG-B2, #13167).
 *
 * Mirrors `src/shared/python/model_appearance/appearance_selection.py`
 * (`apply_overrides`): a custom skin or hair colour becomes the document
 * material `custom_skin` / `custom_hair`; a shirt or trousers colour overrides
 * the library garment material *by the name the clothing preset uses*, keeping
 * its roughness, metallic and texture. Colours are non-linear sRGB in 0..1
 * (`byte / 255`, no rounding), the same contract as `appearanceDefaults`.
 * `tests/fixtures/appearance/colour_roundtrip.json` holds the values both
 * sides must produce.
 */

import type { AppearanceLibrary, AppearanceMaterial } from './appearanceTypes';

export const CUSTOM_SKIN = 'custom_skin';
export const CUSTOM_HAIR = 'custom_hair';
const SKIN_REFERENCE = 'skin_2';
const PANTS_PART = 'thigh';
const HEX = /^#[0-9a-fA-F]{6}$/;

/** Custom `#rrggbb` picks; `undefined` leaves the library default. */
export interface ColourPicks {
  skin_color?: string;
  hair_color?: string;
  top_color?: string;
  bottom_color?: string;
}

export interface CustomizableDocument {
  skin_tone: string;
  headwear_material?: string;
  materials: Record<string, AppearanceMaterial>;
}

export function isHex(text: string): boolean {
  return HEX.test(text);
}

/** `#rrggbb` to sRGB floats (`byte / 255`); throws on anything else. */
export function hexToRgb(hex: string): [number, number, number] {
  if (!isHex(hex)) {
    throw new Error(`colour must be a #rrggbb hex string, got ${JSON.stringify(hex)}`);
  }
  return [
    parseInt(hex.slice(1, 3), 16) / 255,
    parseInt(hex.slice(3, 5), 16) / 255,
    parseInt(hex.slice(5, 7), 16) / 255,
  ];
}

function recoloured(base: AppearanceMaterial, hex: string): AppearanceMaterial {
  return { ...base, base_color: [...hexToRgb(hex), 1] };
}

function libraryMaterial(library: AppearanceLibrary, name: string): AppearanceMaterial {
  const material = library.materials[name];
  if (!material) throw new Error(`unknown material ${JSON.stringify(name)}`);
  return material;
}

function garmentName(
  library: AppearanceLibrary,
  clothing: string,
  part: string,
  label: string,
): string {
  const name = library.clothing[clothing]?.[part];
  if (!name) {
    throw new Error(`clothing preset ${JSON.stringify(clothing)} has no ${label} to recolour`);
  }
  return name;
}

/**
 * The `skin_tone`, hair material name and `materials` entries the custom picks
 * write into the document, on top of `picks`' library choices.
 *
 * Precondition: `picks.headwear` is `hair` when `hair_color` is set (the Python
 * writer rejects a hair colour on a cap or bare head, so the panel disables it).
 */
export function customizeDocument(
  library: AppearanceLibrary,
  picks: {
    skin_tone: string;
    clothing: string;
    headwear: string;
    headwear_material?: string;
  } & ColourPicks,
): CustomizableDocument {
  const materials: Record<string, AppearanceMaterial> = {};
  let skin_tone = picks.skin_tone;
  let headwear_material = picks.headwear_material;
  if (picks.skin_color !== undefined) {
    const reference =
      library.materials[picks.skin_tone] ?? libraryMaterial(library, SKIN_REFERENCE);
    materials[CUSTOM_SKIN] = recoloured(reference, picks.skin_color);
    skin_tone = CUSTOM_SKIN;
  }
  if (picks.hair_color !== undefined) {
    if (picks.headwear !== 'hair') {
      throw new Error(
        `a hair colour needs headwear 'hair', but the choice is ${JSON.stringify(picks.headwear)}`,
      );
    }
    const defaultName = library.headwear_default_material['hair'] ?? '';
    materials[CUSTOM_HAIR] = recoloured(libraryMaterial(library, defaultName), picks.hair_color);
    headwear_material = CUSTOM_HAIR;
  }
  for (const [colour, part, label] of [
    [picks.top_color, 'torso', 'top'],
    [picks.bottom_color, PANTS_PART, 'bottom'],
  ] as const) {
    if (colour === undefined) continue;
    const name = garmentName(library, picks.clothing, part, label);
    materials[name] = recoloured(libraryMaterial(library, name), colour);
  }
  return { skin_tone, headwear_material, materials };
}
