/**
 * Green types on the shared core's green spec (#14464), labelled as the
 * desktop's Green Type selector labels them. Preset shapes come from
 * `/green-presets` at run time; `greenTypes.golden.json` pins the labels and
 * the request to the core.
 */
import type {
  GreenPresetName,
  GreenShapeRequest,
  Putt3DSimulationRequest,
} from "@/api/generated/types";

export type GreenPresets = Record<string, GreenShapeRequest>;

/** Display name of a green type (`null` is the cross slope). */
export function greenLabel(key: string | null): string {
  if (key === null) return "Cross Slope";
  return key
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

/** The request's green: none for the cross slope, else the preset by name. */
export function greenFromType(
  key: string | null,
  presets: GreenPresets,
): Putt3DSimulationRequest["green"] {
  if (key === null) return null;
  const spec = presets[key];
  if (!spec) throw new RangeError(`unknown green preset ${key}`);
  return { ...spec, preset: key as GreenPresetName };
}
