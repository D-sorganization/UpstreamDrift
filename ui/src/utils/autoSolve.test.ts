import { describe, expect, it } from "vitest";

import golden from "./autoSolve.golden.json";
import { formatRead, solvedControls } from "./autoSolve";

describe("Auto-Solve read (#14466)", () => {
  it("writes the core's pace and aim into the controls", () => {
    expect(solvedControls(golden.solve)).toEqual({
      putter_speed_mps: golden.solve.putter_speed_mps,
      aim_heading_deg: golden.solve.aim_heading_deg,
    });
  });

  it("names the read like the desktop, in either unit system", () => {
    expect(formatRead(golden.solve, "metric")).toBe(golden.reads.metric);
    expect(formatRead(golden.solve, "imperial")).toBe(golden.reads.imperial);
  });

  it("says when no holing line was found and which side the aim breaks", () => {
    const miss = { ...golden.solve, break_m: 0.05, holed: false };
    expect(formatRead(miss, "metric")).toMatch(
      /aim 5\.0 cm left of the cup \(no holing line found\)$/,
    );
  });
});
