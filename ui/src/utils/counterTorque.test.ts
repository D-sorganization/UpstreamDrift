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
    expect(ctSummary(golden.response.samples)).toEqual(golden.summary);
  });

  it("names the readout lines", () => {
    const summary = ctSummary(golden.response.samples);
    expect(ctReadout(summary, "imperial")[1]).toBe(
      "Peak in-plane hand force: left 2.7 lbf, right 5.3 lbf",
    );
    expect(ctReadout(summary, "metric")).toEqual([
      "Face at impact -0.12°, club path +0.11°",
      "Peak in-plane hand force: left 12.0 N, right 23.6 N",
      "Gust wobble: 0.08° with grip preload vs 0.46° slack",
    ]);
  });

  it("rejects a series without the core's fields", () => {
    expect(() => ctSummary({ t_s: [0] })).toThrow(/face_angle_impact_deg/);
  });
});
