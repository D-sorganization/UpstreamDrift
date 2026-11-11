import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { runDispersion, runMakeSpread } from "@/api/puttingClient";
import { usePuttingStore } from "@/stores/usePuttingStore";
import golden from "@/utils/dispersion.golden.json";
import { DispersionPanel } from "./DispersionPanel";

vi.mock("@/api/puttingClient", () => ({
  simulatePutt3D: vi.fn(),
  runDispersion: vi.fn(),
  runMakeSpread: vi.fn(),
}));

const [goldenCase] = golden.cases;

function renderPanel() {
  const client = new QueryClient({
    defaultOptions: { mutations: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <DispersionPanel />
    </QueryClientProvider>,
  );
}

describe("DispersionPanel (#14460)", () => {
  beforeEach(() => {
    usePuttingStore.getState().reset();
    usePuttingStore.getState().setUnitSystem("imperial");
    usePuttingStore.getState().updateParameters(goldenCase.parameters);
    vi.mocked(runDispersion).mockResolvedValue({
      n: 3,
      distance_ft: 10,
      rest_x_ft: [0, 1.5, -2],
      rest_y_ft: [0, 0.5, -1],
      holed: [true, false, false],
      make_rate: goldenCase.make_rate!,
      mean_leave_ft: goldenCase.mean_leave_ft!,
    });
    vi.mocked(runMakeSpread).mockResolvedValue({
      distances_ft: [3, 6, 9],
      make_rate: [0.9, 0.5, 0.25],
      samples_per_distance: 60,
    });
  });

  it("sends the noise channels and batch size, then plots the core's rest positions", async () => {
    renderPanel();
    fireEvent.change(screen.getByLabelText("Face SD (°)"), {
      target: { value: "2.5" },
    });
    fireEvent.click(screen.getByRole("button", { name: "40 Putts" }));

    await waitFor(() => expect(runDispersion).toHaveBeenCalledTimes(1));
    const args = vi.mocked(runDispersion).mock.calls[0][0];
    expect(args).toMatchObject({
      ...goldenCase.args,
      face_sd_deg: 2.5,
      n: 40,
    });
    expect(await screen.findByText("17.5%")).toBeInTheDocument();
    expect(screen.getByText("1.99 ft")).toBeInTheDocument();
    // Lane 3 review: the batch is the core skill model, not the 3D putt.
    expect(screen.getByText(/not this 3D\s+putt/)).toBeInTheDocument();
    const scatter = screen.getByRole("img", { name: /Rest Positions/ });
    expect(scatter.querySelectorAll("circle[data-holed]")).toHaveLength(3);
  });

  it("shows percentage channels in percent and sends the core fraction", async () => {
    renderPanel();
    fireEvent.change(screen.getByLabelText("Speed CV (%)"), {
      target: { value: "12" },
    });
    fireEvent.click(screen.getByRole("button", { name: "100 Putts" }));
    await waitFor(() => expect(runDispersion).toHaveBeenCalled());
    const args = vi.mocked(runDispersion).mock.calls[0][0];
    expect(args.speed_cv).toBeCloseTo(0.12, 12);
    expect(args.n).toBe(100);
  });

  it("draws the 0-45 ft make-% spread from the core", async () => {
    renderPanel();
    fireEvent.click(
      screen.getByRole("button", { name: "0–45 ft Make-% Spread" }),
    );
    await waitFor(() => expect(runMakeSpread).toHaveBeenCalledTimes(1));
    expect(vi.mocked(runMakeSpread).mock.calls[0][0].n).toBe(golden.spread_n);
    const chart = await screen.findByRole("img", { name: /Make Rate by/ });
    expect(chart.querySelectorAll("circle")).toHaveLength(3);
  });
});
