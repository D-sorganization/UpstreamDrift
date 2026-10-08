import { describe, it, expect } from "vitest";
import {
  ALL_GLYPH_GROUPS,
  DEFAULT_GLYPH_GROUPS,
  GLYPH_GROUP_LABELS,
} from "@/types/glyphs";
import {
  appendStyleParams,
  styleOptionsFromControls,
} from "./forceStyleParams";

const base = {
  scaleMode: "fixed" as const,
  bodyMassKg: 80,
  peakForceN: 2000,
  referenceLengthM: 0.5,
  groups: DEFAULT_GLYPH_GROUPS,
};

describe("styleOptionsFromControls (GCV-4)", () => {
  it("references body weight in body_weight mode", () => {
    const o = styleOptionsFromControls({ ...base, scaleMode: "body_weight" });
    expect(o.scale_mode).toBe("body_weight");
    expect(o.reference_force_n).toBeCloseTo(80 * 9.80665, 6);
    expect(o.reference_length_m).toBe(0.5);
  });

  it("references the peak force in peak mode", () => {
    const o = styleOptionsFromControls({ ...base, scaleMode: "peak" });
    expect(o.reference_force_n).toBe(2000);
  });

  it("sends no reference in fixed mode", () => {
    expect(styleOptionsFromControls(base).reference_force_n).toBeNull();
  });

  it("rejects a non-positive control", () => {
    expect(() => styleOptionsFromControls({ ...base, bodyMassKg: 0 })).toThrow(
      RangeError,
    );
  });

  it("serialises query parameters the API accepts", () => {
    const params = new URLSearchParams();
    appendStyleParams(
      params,
      styleOptionsFromControls({ ...base, scaleMode: "body_weight" }),
    );
    expect(params.get("scale_mode")).toBe("body_weight");
    expect(params.get("groups")).toBe(DEFAULT_GLYPH_GROUPS.join(","));
    expect(Number(params.get("reference_force_n"))).toBeGreaterThan(700);
  });
});

describe("glyph group contract", () => {
  it("labels every group and keeps raw contacts opt-in", () => {
    expect(ALL_GLYPH_GROUPS).toHaveLength(
      Object.keys(GLYPH_GROUP_LABELS).length,
    );
    expect(DEFAULT_GLYPH_GROUPS).not.toContain("contact_points");
    expect(DEFAULT_GLYPH_GROUPS).not.toContain("moment_about_com");
  });
});
