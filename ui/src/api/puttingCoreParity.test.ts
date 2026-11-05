/**
 * Cross-frontend golden parity (#14449): React, and Tauri (the React build in
 * Tauri mode), replay the core's frozen route JSON through the real client and
 * must read the PyQt6 values (scripts/putting_cross_frontend_golden.py).
 */
import { afterEach, describe, expect, it, vi } from "vitest";

import { DEFAULT_PUTT_3D_PARAMETERS } from "@/stores/usePuttingStore";
import { sampleAtPlaybackTime } from "@/pages/puttingPlayback";
import { puttLeaveFt } from "@/utils/strokesGained";
import { FT_TO_M } from "@/utils/units";
import { BACKEND_PORT } from "./backend";
import type { Putt3DSimulationResponse } from "./generated/types";
import golden from "./puttingCore.golden.json";
import { simulatePutt3D } from "./puttingClient";

type Case = (typeof golden.cases)[number];
const ROUTE = "/api/tools/putting-green/simulate-3d";
const { distance_m, launch_speed_rel } = golden.tolerance;

async function replay(putt: Case) {
  const fetch = vi.fn(
    async () =>
      new Response(JSON.stringify(putt.response), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
  );
  vi.stubGlobal("fetch", fetch);
  const result = await simulatePutt3D({
    ...DEFAULT_PUTT_3D_PARAMETERS,
    ...putt.request,
  });
  const [url, init] = fetch.mock.calls[0] as unknown as [string, RequestInit];
  return { url, body: JSON.parse(String(init.body)), result };
}

function expectNear(actual: number, expected: number, tolerance: number) {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(tolerance);
}

/** What the page shows for a putt against the PyQt6 scene's summary. */
function expectPyqt6Values(result: Putt3DSimulationResponse, putt: Case) {
  const { pyqt6 } = putt;
  const rest = sampleAtPlaybackTime(result.samples, Number.MAX_VALUE);
  expect(result.holed).toBe(pyqt6.holed);
  expectNear(result.total_distance_m, pyqt6.total_roll_m, distance_m);
  expectNear(puttLeaveFt(result) * FT_TO_M, pyqt6.leave_m, distance_m);
  expectNear(rest.y_m, pyqt6.rest_dy_m, distance_m);
  expectNear(
    result.collision.ball_speed_mps,
    pyqt6.launch_speed_mps,
    launch_speed_rel * pyqt6.launch_speed_mps,
  );
}

describe.each([
  ["React", false, ROUTE],
  ["Tauri", true, `http://localhost:${BACKEND_PORT}${ROUTE}`],
])("%s putting parity with the core (#14449)", (_surface, tauri, route) => {
  afterEach(() => {
    vi.unstubAllGlobals();
    delete (window as { __TAURI_INTERNALS__?: object }).__TAURI_INTERNALS__;
  });

  it.each(golden.cases)("$name", async (putt) => {
    if (tauri)
      (window as { __TAURI_INTERNALS__?: object }).__TAURI_INTERNALS__ = {};
    const { url, body, result } = await replay(putt);
    expect(url).toBe(route);
    expect(body).toMatchObject(putt.request);
    expectPyqt6Values(result, putt);
  });
});
