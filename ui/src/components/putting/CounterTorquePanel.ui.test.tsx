import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { fetchCounterTorque, fetchPutterPresets } from "@/api/puttingClient";
import golden from "@/utils/counterTorque.golden.json";
import { CounterTorquePanel } from "./CounterTorquePanel";

vi.mock("@/api/puttingClient", () => ({
  fetchCounterTorque: vi.fn(),
  fetchPutterPresets: vi.fn(),
}));

function renderPanel() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <CounterTorquePanel />
    </QueryClientProvider>,
  );
}

describe("CounterTorquePanel (#14470)", () => {
  beforeEach(() => {
    vi.mocked(fetchCounterTorque).mockReset();
    vi.mocked(fetchCounterTorque).mockResolvedValue(golden.response);
    vi.mocked(fetchPutterPresets).mockResolvedValue({
      presets: [
        { key: "toe_hang", name: "Toe-Hang Blade" },
        { key: "face_balanced", name: "Face-Balanced Mallet" },
      ],
    } as Awaited<ReturnType<typeof fetchPutterPresets>>);
  });

  it("runs the core couple model and charts its hand couples", async () => {
    renderPanel();
    await screen.findByRole("option", { name: "Face-Balanced Mallet" });
    fireEvent.change(screen.getByLabelText("Putter Head"), {
      target: { value: "face_balanced" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Run Counter-Torque" }));

    await waitFor(() => expect(fetchCounterTorque).toHaveBeenCalledOnce());
    // The core's own grid (stroke_series default); the golden is a coarse one.
    expect(vi.mocked(fetchCounterTorque).mock.calls[0][0]).toEqual({
      ...golden.request,
      n_points: 210,
    });
    expect(
      await screen.findByText(
        "Peak in-plane hand force: left 12.0 N, right 23.6 N",
      ),
    ).toBeInTheDocument();
    const chart = screen.getByRole("img", { name: /Hand Couples/ });
    expect(chart.querySelectorAll("polyline")).toHaveLength(5);
  });

  it("sends a slider change and shows a core rejection", async () => {
    vi.mocked(fetchCounterTorque).mockRejectedValue(
      new Error("lie_deg must be in [55.0, 85.0]"),
    );
    renderPanel();
    fireEvent.change(screen.getByLabelText("Grip Preload"), {
      target: { value: "1.2" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Run Counter-Torque" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "lie_deg must be in [55.0, 85.0]",
    );
    expect(vi.mocked(fetchCounterTorque).mock.calls[0][0].preload_nm).toBe(1.2);
  });
});
