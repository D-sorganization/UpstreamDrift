import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { Putt3DSimulationResponse } from "@/api/generated/types";
import { scenePose } from "@/utils/cameraPresets";

const camera = { position: { set: vi.fn() } };
const controls = { target: { set: vi.fn() }, update: vi.fn() };
vi.mock("@react-three/fiber", () => ({
  Canvas: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="putting-canvas">{children}</div>
  ),
  useThree: (pick: (state: object) => unknown) => pick({ camera, controls }),
}));

vi.mock("@react-three/drei", () => ({
  Line: () => <div data-testid="putt-trail" />,
  OrbitControls: () => <div data-testid="putting-orbit-controls" />,
}));

const result: Putt3DSimulationResponse = {
  roll_model: "usga-stimp-roll/1",
  samples: [
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
      y_m: 0,
      z_m: 0,
      speed_mps: 0,
      spin_rad_s: 0,
      mode: "rest",
    },
  ],
  collision: {
    ball_speed_mps: 2,
    putter_speed_before_mps: 1.8,
    putter_speed_after_mps: 1.55,
    launch_angle_deg: 2,
    spin_rad_s: -10,
    impulse_n_s: 0.09,
    contact_time_proxy_s: 0.0005,
    kinetic_energy_loss_j: 0.03,
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
  skid_distance_m: 0.2,
};

describe("PuttingScene3D", () => {
  it("renders an accessible orbitable green and trajectory", async () => {
    const { PuttingScene3D } = await import("./PuttingScene3D");
    render(
      <PuttingScene3D
        result={result}
        playbackTimeS={0.5}
        hoselToeM={0}
        hoselForwardM={0}
      />,
    );

    expect(screen.getByRole("img")).toHaveAttribute("tabIndex", "0");
    expect(screen.getByTestId("putting-canvas")).toBeInTheDocument();
    expect(screen.getByTestId("putt-trail")).toBeInTheDocument();
    expect(screen.getByTestId("putting-orbit-controls")).toBeInTheDocument();
  });

  it("places the camera at the chosen desktop view (#14474)", async () => {
    const { PuttingScene3D } = await import("./PuttingScene3D");
    render(
      <PuttingScene3D
        result={result}
        playbackTimeS={0}
        hoselToeM={0}
        hoselForwardM={0}
        view="overhead"
      />,
    );
    // Overhead looks down on the 3 m putt's tile middle from 9.42 m.
    const { eye, target } = scenePose(3, 0, 0, "overhead");
    expect(camera.position.set).toHaveBeenLastCalledWith(
      eye[0],
      eye[2],
      eye[1],
    );
    expect(eye[2]).toBeCloseTo(9.42 * Math.sin((89 * Math.PI) / 180), 9);
    expect(controls.target.set).toHaveBeenLastCalledWith(
      target[0],
      target[2],
      target[1],
    );
    expect(controls.update).toHaveBeenCalled();
  });
});
