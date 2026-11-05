import { describe, expect, it } from "vitest";

import golden from "./counterTorque.golden.json";
import { COUNTER_TORQUE_CONTROLS, ctReadout, ctSummary } from "./counterTorque";

describe("counter-torque explorer (#14470)", () => {
  it("bounds every slider by the core's input range", () => {
    for (const control of COUNTER_TORQUE_CONTROLS) {
      expect([control.min, control.max]).toEqual(golden.ranges[control.key]);
    }
  });

  it("reads face, path, peak hand forces and gust wobble off the series", () => {
    const s = ctSummary(golden.response.samples);
    expect(s.faceDeg).toBe(golden.summary.face_deg);
    expect(s.pathDeg).toBe(golden.summary.path_deg);
    expect(s.peakForceLeftN).toBe(golden.summary.peak_force_left_n);
    expect(s.peakForceRightN).toBe(golden.summary.peak_force_right_n);
    expect(s.gustPreloadDeg).toBe(golden.summary.gust_preload_deg);
    expect(s.gustSlackDeg).toBe(golden.summary.gust_slack_deg);
  });

  it("names the readout lines", () => {
    expect(ctReadout(ctSummary(golden.response.samples))).toEqual([
      "Face at impact -0.12°, club path +0.11°",
      "Peak in-plane hand force: left 12.0 N, right 23.6 N",
      "Gust wobble: 0.08° with grip preload vs 0.46° slack",
    ]);
  });

  it("rejects a series without the core's fields", () => {
    expect(() => ctSummary({ t_s: [0] })).toThrow(/face_angle_impact_deg/);
  });
});
