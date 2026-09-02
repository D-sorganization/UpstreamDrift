import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { Putt3DSimulationResponse } from "@/api/generated/types";
import { simulatePutt3D } from "@/api/puttingClient";
import { usePuttingStore } from "@/stores/usePuttingStore";
import { PuttingGreenPage } from "./PuttingGreen";

vi.mock("@/api/puttingClient", () => ({
  simulatePutt3D: vi.fn(),
}));

vi.mock("@/components/visualization/PuttingScene3D", () => ({
  PuttingScene3D: () => <div data-testid="putting-scene-mock" />,
}));

const response: Putt3DSimulationResponse = {
  roll_model: "usga-stimp-roll/1",
  samples: [
    {
      t_s: 0,
      x_m: 0,
      y_m: 0,
      z_m: 0,
      speed_mps: 1,
      spin_rad_s: 0,
      mode: "slide",
    },
    {
      t_s: 1,
      x_m: 1,
      y_m: 0,
      z_m: 0,
      speed_mps: 0,
      spin_rad_s: 0,
      mode: "rest",
    },
  ],
  collision: {
    ball_speed_mps: 1,
    putter_speed_before_mps: 1.8,
    putter_speed_after_mps: 1.5,
    launch_angle_deg: 2,
    spin_rad_s: 0,
    impulse_n_s: 0.05,
    contact_time_proxy_s: 0.0005,
    kinetic_energy_loss_j: 0.02,
    face_twist_rad_s: 0,
    twist_moment_n_m_s: 0,
  },
  surface: {
    width_m: 20,
    height_m: 20,
    grade_percent: 0,
    downhill_aspect_deg: 0,
    hole_x_m: 3,
    hole_y_m: 0,
  },
  holed: false,
  total_distance_m: 1,
  duration_s: 1,
  skid_distance_m: 0.1,
};

describe("PuttingGreenPage 3D workflow", () => {
  beforeEach(() => {
    usePuttingStore.getState().reset();
    vi.mocked(simulatePutt3D).mockResolvedValue(response);
    vi.stubGlobal(
      "requestAnimationFrame",
      vi.fn(() => 1),
    );
    vi.stubGlobal("cancelAnimationFrame", vi.fn());
  });

  it("submits store parameters and exposes playback plus impact readouts", async () => {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
    render(
      <QueryClientProvider client={queryClient}>
        <PuttingGreenPage />
      </QueryClientProvider>,
    );

    fireEvent.change(screen.getByLabelText("Dynamic Loft"), {
      target: { value: "5" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Simulate Putt" }));

    await waitFor(() => expect(simulatePutt3D).toHaveBeenCalledOnce());
    expect(vi.mocked(simulatePutt3D).mock.calls[0][0]).toEqual(
      expect.objectContaining({ loft_deg: 5, random_seed: 8345 }),
    );
    expect(await screen.findByTestId("putting-scene-mock")).toBeInTheDocument();
    expect(screen.getByLabelText("Putt playback position")).toBeInTheDocument();
    expect(screen.getByText("Putter After Impact")).toBeInTheDocument();
  });
});
