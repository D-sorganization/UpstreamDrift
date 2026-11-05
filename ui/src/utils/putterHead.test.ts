import { describe, expect, it } from "vitest";

import golden from "./putterHead.golden.json";
import {
  HEAD_MASS_G,
  HEAD_MOI_GCM2,
  TYPICAL_HEAD,
  headFromPreset,
  kgM2ToGCm2,
} from "./putterHead";
import { DEFAULT_PUTT_3D_PARAMETERS } from "@/stores/usePuttingStore";

describe("putter head controls pinned to the shared core (#14462)", () => {
  it("uses the core's typical head as the store default", () => {
    expect(TYPICAL_HEAD.head_mass_kg).toBe(golden.typical.head_mass_kg);
    expect(TYPICAL_HEAD.head_moi_kg_m2).toBe(golden.typical.head_moi_kg_m2);
    expect(DEFAULT_PUTT_3D_PARAMETERS).toEqual(
      expect.objectContaining(TYPICAL_HEAD),
    );
  });

  it("bounds the mass and MOI inputs by the /simulate-3d request limits", () => {
    const [massLo, massHi] = golden.bounds.head_mass_kg;
    const [moiLo, moiHi] = golden.bounds.head_moi_kg_m2;
    expect([HEAD_MASS_G.min, HEAD_MASS_G.max]).toEqual([
      massLo * 1000,
      massHi * 1000,
    ]);
    expect(HEAD_MOI_GCM2.min).toBeCloseTo(kgM2ToGCm2(moiLo), 9);
    expect(HEAD_MOI_GCM2.max).toBeCloseTo(kgM2ToGCm2(moiHi), 9);
  });

  it.each(golden.presets)(
    "copies the $key preset's head into the request",
    (preset) => {
      expect(headFromPreset(preset)).toEqual({
        balance_type: preset.key,
        head_mass_kg: preset.head_mass_kg,
        head_moi_kg_m2: preset.moi_kg_m2,
        hosel_toe_m: preset.hosel_toe_m,
        hosel_forward_m: preset.hosel_forward_m,
      });
    },
  );

  it("returns to the typical head with no balance type", () => {
    expect(headFromPreset(null)).toEqual({
      balance_type: null,
      ...TYPICAL_HEAD,
      hosel_toe_m: 0,
      hosel_forward_m: 0,
    });
  });
});
