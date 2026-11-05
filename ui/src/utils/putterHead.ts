/**
 * Putter head controls pinned to the shared core (#14462): the typical head
 * and `/simulate-3d` mass/MOI bounds, held to the core by
 * `putterHead.golden.json`. Presets come from `/putter-presets` at run time.
 */
import type {
  BalancePresetResponse,
  Putt3DSimulationRequest,
} from "@/api/generated/types";

const G_CM2_PER_KG_M2 = 1e7;

export function kgM2ToGCm2(moiKgM2: number): number {
  return moiKgM2 * G_CM2_PER_KG_M2;
}

export function gCm2ToKgM2(moiGCm2: number): number {
  return moiGCm2 / G_CM2_PER_KG_M2;
}

export const TYPICAL_HEAD = { head_mass_kg: 0.35, head_moi_kg_m2: 4.5e-4 };

/** Head mass input in grams, within the request's 0.1–1.0 kg. */
export const HEAD_MASS_G = { min: 100, max: 1000, step: 5 };

/** Head MOI input in g·cm², within the request's 1e-5–1e-2 kg m². */
export const HEAD_MOI_GCM2 = { min: 100, max: 100_000, step: 50 };

export type PutterHead = Pick<
  Putt3DSimulationRequest,
  | "balance_type"
  | "head_mass_kg"
  | "head_moi_kg_m2"
  | "hosel_toe_m"
  | "hosel_forward_m"
>;

/** Request fields for a core preset (shown as simulated), or the typical head. */
export function headFromPreset(
  preset: BalancePresetResponse | null,
): PutterHead {
  if (preset === null) {
    return {
      balance_type: null,
      ...TYPICAL_HEAD,
      hosel_toe_m: 0,
      hosel_forward_m: 0,
    };
  }
  return {
    balance_type: preset.key,
    head_mass_kg: preset.head_mass_kg,
    head_moi_kg_m2: preset.moi_kg_m2,
    hosel_toe_m: preset.hosel_toe_m,
    hosel_forward_m: preset.hosel_forward_m,
  };
}
