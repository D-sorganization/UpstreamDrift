/**
 * Saved scenarios and JSON setup files on the React putting page (#14472).
 * puttingSetup.golden.json holds the shared core's setups
 * (service_setup.setup_from_args / import_setup, #14471); the Python side
 * checks the same file against the core.
 */
import { describe, expect, it } from "vitest";

import { DEFAULT_PUTT_3D_PARAMETERS } from "@/stores/usePuttingStore";
import { defaultNoise } from "./dispersion";
import {
  MAX_SCENARIOS,
  SCHEMA_VERSION,
  applySetup,
  setupArgs,
  setupFileName,
} from "./puttingSetup";
import golden from "./puttingSetup.golden.json";

const start = { ...DEFAULT_PUTT_3D_PARAMETERS };

describe("putting setup parity with the shared core (#14472)", () => {
  it("uses the core's schema version and scenario cap", () => {
    expect(SCHEMA_VERSION).toBe(golden.schema_version);
    expect(MAX_SCENARIOS).toBe(golden.max_scenarios);
  });

  it.each(golden.cases)(
    "sends the page's putt as the core setup ($setup.distance_ft ft)",
    ({ parameters, setup }) => {
      const args = setupArgs({ ...start, ...parameters }, defaultNoise());
      expect(Object.keys(args).every((key) => key in setup)).toBe(true);
      for (const [key, value] of Object.entries(args)) {
        expect(value).toBeCloseTo(setup[key as keyof typeof setup], 10);
      }
    },
  );

  it.each([...golden.cases.map((c) => c.setup), golden.studio_import.setup])(
    "loads a core setup back onto the page ($distance_ft ft)",
    (setup) => {
      const loaded = applySetup(start, defaultNoise(), setup);
      const parameters = { ...start, ...loaded.parameters };
      const args = setupArgs(parameters, loaded.noise);
      for (const [key, value] of Object.entries(args)) {
        expect(value).toBeCloseTo(setup[key as keyof typeof setup], 10);
      }
    },
  );

  it("keeps the page's line direction when loading a setup", () => {
    const diagonal = { ...start, hole_x_m: 1, hole_y_m: 1 };
    const { parameters } = applySetup(
      diagonal,
      defaultNoise(),
      golden.cases[0].setup,
    );
    expect(parameters.hole_x_m).toBeCloseTo(parameters.hole_y_m!, 12);
  });

  it("names the export file with the schema", () => {
    expect(setupFileName()).toBe("putting-setup-v4.0.0.json");
  });
});
