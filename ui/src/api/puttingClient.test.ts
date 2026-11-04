import { beforeEach, describe, expect, it, vi } from "vitest";

import { apiFetch } from "./fetch";
import {
  STUDY_TIMEOUT_MS,
  fetchPutterPresets,
  runDispersion,
  runMakeSpread,
} from "./puttingClient";
import { DEFAULT_PUTT_3D_PARAMETERS } from "@/stores/usePuttingStore";
import { defaultNoise, dispersionArgs } from "@/utils/dispersion";

vi.mock("./fetch", () => ({ apiFetch: vi.fn() }));

const args = dispersionArgs(DEFAULT_PUTT_3D_PARAMETERS, defaultNoise(), 40);

describe("putting analytics client (#14460)", () => {
  beforeEach(() => vi.mocked(apiFetch).mockResolvedValue({}));

  it.each([
    [runDispersion, "dispersion"],
    [runMakeSpread, "make-spread"],
  ] as const)(
    "posts to the core %s route with the study timeout",
    async (run, route) => {
      await run(args);
      expect(apiFetch).toHaveBeenLastCalledWith(
        `/api/tools/putting-analytics/${route}`,
        {
          method: "POST",
          body: JSON.stringify(args),
          timeoutMs: STUDY_TIMEOUT_MS,
        },
      );
      expect(STUDY_TIMEOUT_MS).toBeGreaterThan(15_000);
    },
  );
});

describe("putter presets client (#14462)", () => {
  it("reads the balance presets from the core route", async () => {
    vi.mocked(apiFetch).mockResolvedValue({ presets: [] });
    await expect(fetchPutterPresets()).resolves.toEqual({ presets: [] });
    expect(apiFetch).toHaveBeenLastCalledWith(
      "/api/tools/putting-green/putter-presets",
    );
  });
});
