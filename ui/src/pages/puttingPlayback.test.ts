import { describe, expect, it } from "vitest";

import type { Putt3DSampleResponse } from "@/api/generated/types";
import {
  IMPACT_LEAD_IN_S,
  ballRotationRad,
  sampleAtPlaybackTime,
} from "./puttingPlayback";

const samples: Putt3DSampleResponse[] = [
  {
    t_s: 0,
    x_m: 0,
    y_m: 0,
    z_m: 0,
    speed_mps: 2,
    spin_rad_s: -10,
    mode: "slide",
  },
  {
    t_s: 1,
    x_m: 1,
    y_m: 0.2,
    z_m: 0,
    speed_mps: 1,
    spin_rad_s: 20,
    mode: "roll",
  },
];

describe("putting playback model", () => {
  it("holds the ball at impact during the collision lead-in", () => {
    expect(sampleAtPlaybackTime(samples, IMPACT_LEAD_IN_S / 2)).toEqual(
      samples[0],
    );
  });

  it("interpolates the physical trajectory after impact", () => {
    const sample = sampleAtPlaybackTime(samples, IMPACT_LEAD_IN_S + 0.5);
    expect(sample.x_m).toBeCloseTo(0.5);
    expect(sample.y_m).toBeCloseTo(0.1);
    expect(sample.speed_mps).toBeCloseTo(1.5);
    expect(sample.mode).toBe("roll");
  });

  it("uses spin so the marker distinguishes skid from pure roll", () => {
    expect(ballRotationRad(samples[0], 0.1)).toBeCloseTo(-1);
    expect(ballRotationRad(samples[1], 0.1)).toBeCloseTo(2);
  });
});
