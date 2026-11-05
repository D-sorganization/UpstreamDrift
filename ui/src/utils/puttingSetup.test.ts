/**
 * Saved scenarios on the React putting page (#14472). puttingSetup.golden.json
 * holds the shared core's setups (service_setup.setup_from_args, #14471); the
 * Python side checks the same file against the core.
 */
import { describe, expect, it } from "vitest";

import { DEFAULT_PUTT_3D_PARAMETERS } from "@/stores/usePuttingStore";
import { defaultNoise } from "./dispersion";
import { MAX_SCENARIOS, setupArgs } from "./puttingSetup";
import golden from "./puttingSetup.golden.json";

describe("putting setup parity with the shared core (#14472)", () => {
  it("uses the core's scenario cap", () => {
    expect(MAX_SCENARIOS).toBe(golden.max_scenarios);
  });

  it.each(golden.cases)(
    "sends the page's putt as the core setup ($setup.distance_ft ft)",
    ({ parameters, setup }) => {
      const page = { ...DEFAULT_PUTT_3D_PARAMETERS, ...parameters };
      const args = setupArgs(page, defaultNoise());
      expect(Object.keys(args).every((key) => key in setup)).toBe(true);
      for (const [key, value] of Object.entries(args)) {
        expect(value).toBeCloseTo(setup[key as keyof typeof setup], 10);
      }
    },
  );
});
