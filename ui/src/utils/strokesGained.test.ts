import { describe, expect, it } from "vitest";

import golden from "./strokesGained.golden.json";
import { SG_STUDY, puttLeaveFt, sgArgs, sgSummary } from "./strokesGained";
import { defaultNoise } from "./dispersion";
import { DEFAULT_PUTT_3D_PARAMETERS } from "@/stores/usePuttingStore";
import type { Putt3DSimulationResponse } from "@/api/generated/types";

const { case: sgCase } = golden;
const parameters = { ...DEFAULT_PUTT_3D_PARAMETERS, ...sgCase.parameters };

const resultAt = ([x_m, y_m]: number[], holed = false) =>
  ({
    holed,
    samples: [{ x_m, y_m }],
    surface: parameters,
  }) as unknown as Putt3DSimulationResponse;

describe("strokes-gained arguments pinned to the shared core (#14468)", () => {
  it("uses the core's study size, paces and distance grid", () => {
    expect(SG_STUDY).toEqual(golden.study);
  });

  it("takes the leave from the rest point to the hole, zero when holed", () => {
    expect(puttLeaveFt(resultAt(sgCase.rest_m))).toBeCloseTo(
      sgCase.args.leave_ft,
      12,
    );
    expect(puttLeaveFt(resultAt(sgCase.rest_m, true))).toBe(0);
  });

  it("maps the simulator putt to the core's /strokes-gained arguments", () => {
    const putt = { holed: false, leave_ft: sgCase.args.leave_ft };
    const args = sgArgs(parameters, defaultNoise(), putt);
    const expected = { ...sgCase.args, ...golden.study };
    expect(Object.keys(args).sort()).toEqual(Object.keys(expected).sort());
    for (const [key, value] of Object.entries(expected)) {
      const actual = args[key as keyof typeof args];
      if (typeof value === "number") expect(actual).toBeCloseTo(value, 9);
      else expect(actual).toEqual(value);
    }
  });

  it("summarises the core report in the user's units", () => {
    expect(sgSummary({ ...sgCase.report, sg_player: null }, "metric")).toBe(
      "From 3.0 m you average 1.78 putts.",
    );
  });
});
