import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { Putt3DSimulationResponse } from "@/api/generated/types";
import { runStrokesGained } from "@/api/puttingClient";
import { usePuttingStore } from "@/stores/usePuttingStore";
import golden from "@/utils/strokesGained.golden.json";
import { StrokesGainedPanel } from "./StrokesGainedPanel";

vi.mock("@/api/puttingClient", () => ({ runStrokesGained: vi.fn() }));

const { case: sgCase } = golden;

function renderPanel() {
  const client = new QueryClient({
    defaultOptions: { mutations: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <StrokesGainedPanel />
    </QueryClientProvider>,
  );
}

describe("StrokesGainedPanel (#14468)", () => {
  beforeEach(() => {
    usePuttingStore.getState().reset();
    usePuttingStore.getState().setUnitSystem("imperial");
    usePuttingStore.getState().updateParameters(sgCase.parameters);
    vi.mocked(runStrokesGained).mockReset();
    vi.mocked(runStrokesGained).mockResolvedValue(sgCase.report);
  });

  it("enters the simulated putt and scores it on the core's E(d) curve", async () => {
    const [x_m, y_m] = sgCase.rest_m;
    usePuttingStore.getState().setResult({
      holed: false,
      samples: [{ x_m, y_m }],
      surface: { hole_x_m: 3.048, hole_y_m: 0 },
    } as unknown as Putt3DSimulationResponse);
    renderPanel();
    expect(screen.getByLabelText("Leave (ft)")).toHaveValue(1.1);
    fireEvent.click(screen.getByRole("button", { name: "Score Putt" }));

    await waitFor(() => expect(runStrokesGained).toHaveBeenCalledTimes(1));
    const args = vi.mocked(runStrokesGained).mock.calls[0][0];
    expect(args.holed).toBe(false);
    expect(args.leave_ft).toBeCloseTo(1.1, 9);
    expect(
      await screen.findByText(
        "From 10.0 ft you average 1.78 putts. This putt gained -0.22 strokes.",
      ),
    ).toBeInTheDocument();
    const chart = screen.getByRole("img", { name: /Your Expected Putts/ });
    expect(chart.querySelectorAll("circle")).toHaveLength(4);
  });

  it("sends a holed putt with no leave, in metric too", async () => {
    usePuttingStore.getState().setUnitSystem("metric");
    renderPanel();
    fireEvent.click(screen.getByLabelText("Holed"));
    expect(screen.getByLabelText("Leave (m)")).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Score Putt" }));
    await waitFor(() => expect(runStrokesGained).toHaveBeenCalledTimes(1));
    expect(vi.mocked(runStrokesGained).mock.calls[0][0]).toMatchObject({
      holed: true,
      leave_ft: 0,
    });
  });
});
