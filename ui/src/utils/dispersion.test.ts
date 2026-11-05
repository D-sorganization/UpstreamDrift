import { describe, expect, it } from "vitest";

import { DEFAULT_PUTT_3D_PARAMETERS } from "@/stores/usePuttingStore";
import {
  BATCH_SIZES,
  NOISE_CHANNELS,
  SPREAD_N,
  defaultNoise,
  dispersionArgs,
} from "./dispersion";
import golden from "./dispersion.golden.json";

describe("dispersion parity with the shared core (#14460)", () => {
  it("shows every core noise channel with the core default and bounds", () => {
    expect(NOISE_CHANNELS.map((c) => c.key)).toEqual(
      Object.keys(golden.channels),
    );
    for (const channel of NOISE_CHANNELS) {
      const [initial, low, high] =
        golden.channels[channel.key as keyof typeof golden.channels];
      expect(defaultNoise()[channel.key]).toBe(initial);
      expect([channel.min, channel.max]).toEqual([low, high]);
    }
  });

  it("uses the core's batch sizes and spread sample count", () => {
    expect(BATCH_SIZES).toEqual(golden.batch_sizes);
    expect(SPREAD_N).toBe(golden.spread_n);
  });

  it.each(golden.cases)(
    "maps the simulator putt to core arguments ($args.distance_ft ft)",
    (golden_case) => {
      const parameters = {
        ...DEFAULT_PUTT_3D_PARAMETERS,
        ...golden_case.parameters,
      };
      const noise = defaultNoise();
      const args = dispersionArgs(parameters, noise, golden_case.n);
      const { distance_ft, ...rest } = golden_case.args;
      expect(args.distance_ft).toBeCloseTo(distance_ft, 10);
      expect(args).toMatchObject({ ...rest, ...noise, n: golden_case.n });
    },
  );
});
