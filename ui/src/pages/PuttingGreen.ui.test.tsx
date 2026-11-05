import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { Putt3DSimulationResponse } from "@/api/generated/types";
import {
  fetchGreenPresets,
  fetchPutterPresets,
  simulatePutt3D,
  solveAim3D,
} from "@/api/puttingClient";
import { usePuttingStore } from "@/stores/usePuttingStore";
import autoSolve from "@/utils/autoSolve.golden.json";
import greens from "@/utils/greenTypes.golden.json";
import golden from "@/utils/putterHead.golden.json";
import { PuttingGreenPage } from "./PuttingGreen";

vi.mock("@/api/puttingClient", () => ({
  simulatePutt3D: vi.fn(),
  solveAim3D: vi.fn(),
  runDispersion: vi.fn(),
  runMakeSpread: vi.fn(),
  runStrokesGained: vi.fn(),
  fetchPutterPresets: vi.fn(),
  fetchGreenPresets: vi.fn(),
  fetchCounterTorque: vi.fn(),
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
  phases: [
    { mode: "slide", t_s: 0, index: 0 },
    { mode: "rest", t_s: 1, index: 1 },
  ],
};

describe("PuttingGreenPage 3D workflow", () => {
  beforeEach(() => {
    usePuttingStore.getState().reset();
    vi.mocked(simulatePutt3D).mockResolvedValue(response);
    vi.mocked(fetchPutterPresets).mockResolvedValue({ presets: [] });
    vi.mocked(fetchGreenPresets).mockResolvedValue(greens.presets);
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

  it("switches displayed units without changing the SI request (#14458)", async () => {
    localStorage.clear();
    usePuttingStore.getState().setUnitSystem("metric");
    render(
      <QueryClientProvider client={new QueryClient()}>
        <PuttingGreenPage />
      </QueryClientProvider>,
    );
    expect(screen.getByText("1.80 m/s")).toBeInTheDocument();
    expect(screen.getByText("3.00 m")).toBeInTheDocument();
    expect(screen.getByLabelText("Strike SD (mm)")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Units"), {
      target: { value: "imperial" },
    });
    expect(screen.getByText("4.03 mph")).toBeInTheDocument();
    expect(screen.getByText("9.8 ft")).toBeInTheDocument();
    // The batch-dispersion strike channel follows the units (#14460).
    expect(screen.getByLabelText("Strike SD (in)")).toBeInTheDocument();
    expect(localStorage.getItem("ud.unitSystem")).toBe("imperial");

    fireEvent.change(screen.getByLabelText("Hole Distance"), {
      target: { value: "20" },
    });
    expect(usePuttingStore.getState().parameters.hole_x_m).toBeCloseTo(
      6.096,
      9,
    );

    fireEvent.click(screen.getByRole("button", { name: "Simulate Putt" }));
    await waitFor(() => expect(simulatePutt3D).toHaveBeenCalled());
    expect(vi.mocked(simulatePutt3D).mock.lastCall?.[0]).toEqual(
      expect.objectContaining({ putter_speed_mps: 1.8 }),
    );
    expect(await screen.findByText("Total Distance")).toBeInTheDocument();
    expect(screen.getByText("3.28 ft")).toBeInTheDocument();
  });

  it("sends the chosen core balance preset to /simulate-3d (#14462)", async () => {
    vi.mocked(fetchPutterPresets).mockResolvedValue({
      presets: golden.presets,
    });
    const [preset] = golden.presets;
    render(
      <QueryClientProvider client={new QueryClient()}>
        <PuttingGreenPage />
      </QueryClientProvider>,
    );
    await screen.findByRole("option", { name: preset.name });
    fireEvent.change(screen.getByLabelText("Balance Type"), {
      target: { value: preset.key },
    });
    expect(screen.getByLabelText("Hosel Heel to Toe")).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: "Simulate Putt" }));
    await waitFor(() => expect(simulatePutt3D).toHaveBeenCalled());
    expect(vi.mocked(simulatePutt3D).mock.lastCall?.[0]).toEqual(
      expect.objectContaining({
        balance_type: preset.key,
        head_moi_kg_m2: preset.moi_kg_m2,
      }),
    );
  });

  it("jumps to the core phase marks and picks a camera view (#14474)", async () => {
    usePuttingStore.getState().setResult(response);
    render(
      <QueryClientProvider client={new QueryClient()}>
        <PuttingGreenPage />
      </QueryClientProvider>,
    );
    fireEvent.click(screen.getByRole("button", { name: "Rest 1.00 s" }));
    expect(usePuttingStore.getState().playbackTimeS).toBeCloseTo(1.18, 9);
    fireEvent.change(screen.getByLabelText("Camera view"), {
      target: { value: "face_on" },
    });
    expect(usePuttingStore.getState().cameraView).toBe("face_on");
  });

  it("sends the chosen core green preset to /simulate-3d (#14464)", async () => {
    render(
      <QueryClientProvider client={new QueryClient()}>
        <PuttingGreenPage />
      </QueryClientProvider>,
    );
    await screen.findByRole("option", { name: "Tiered" });
    fireEvent.change(screen.getByLabelText("Green Type"), {
      target: { value: "tiered" },
    });
    expect(screen.getByLabelText("Slope Grade")).toBeDisabled();
    expect(screen.getByLabelText("Bump Height")).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: "Simulate Putt" }));
    await waitFor(() => expect(simulatePutt3D).toHaveBeenCalled());
    expect(vi.mocked(simulatePutt3D).mock.lastCall?.[0].green).toEqual({
      ...greens.presets.tiered,
      preset: "tiered",
    });
  });

  it("writes the core's read into pace and aim, then simulates it (#14466)", async () => {
    usePuttingStore.getState().setUnitSystem("metric");
    vi.mocked(simulatePutt3D).mockClear();
    vi.mocked(solveAim3D).mockResolvedValue(autoSolve.solve);
    render(
      <QueryClientProvider client={new QueryClient()}>
        <PuttingGreenPage />
      </QueryClientProvider>,
    );
    fireEvent.click(screen.getByRole("button", { name: /Auto-Solve Read/ }));

    expect((await screen.findByRole("status")).textContent).toBe(
      autoSolve.reads.metric,
    );
    const solved = {
      putter_speed_mps: autoSolve.solve.putter_speed_mps,
      aim_heading_deg: autoSolve.solve.aim_heading_deg,
    };
    expect(vi.mocked(solveAim3D).mock.lastCall?.[0]).toEqual(
      expect.objectContaining({ hole_x_m: 3 }),
    );
    await waitFor(() =>
      expect(vi.mocked(simulatePutt3D).mock.lastCall?.[0]).toEqual(
        expect.objectContaining(solved),
      ),
    );
    expect(usePuttingStore.getState().parameters).toEqual(
      expect.objectContaining(solved),
    );
    expect(screen.getByLabelText("Aim Heading")).toHaveValue(
      String(autoSolve.solve.aim_heading_deg),
    );
  });

  it("shows why the core rejected the solve (#14466)", async () => {
    vi.mocked(simulatePutt3D).mockClear();
    vi.mocked(solveAim3D).mockRejectedValue(new Error("hole off the green"));
    render(
      <QueryClientProvider client={new QueryClient()}>
        <PuttingGreenPage />
      </QueryClientProvider>,
    );
    fireEvent.click(screen.getByRole("button", { name: /Auto-Solve Read/ }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Auto-Solve rejected: hole off the green",
    );
    expect(simulatePutt3D).not.toHaveBeenCalled();
  });
});
