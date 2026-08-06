import { beforeEach, describe, expect, it } from "vitest";

import { DEFAULT_PUTT_3D_PARAMETERS, usePuttingStore } from "./usePuttingStore";

describe("usePuttingStore", () => {
  beforeEach(() => {
    usePuttingStore.getState().reset();
  });

  it("keeps reproducible physics defaults in one store", () => {
    expect(usePuttingStore.getState().parameters).toEqual(
      DEFAULT_PUTT_3D_PARAMETERS,
    );
    expect(usePuttingStore.getState().parameters.random_seed).toBe(8345);
  });

  it("resets playback when a physical parameter changes", () => {
    usePuttingStore.getState().setPlaybackTime(2);
    usePuttingStore.getState().setPlaying(true);
    usePuttingStore.getState().updateParameters({ loft_deg: 6 });

    expect(usePuttingStore.getState().playbackTimeS).toBe(0);
    expect(usePuttingStore.getState().playing).toBe(false);
    expect(usePuttingStore.getState().parameters.loft_deg).toBe(6);
  });
});
