import { describe, expect, it } from "vitest";

import golden from "./cameraPresets.golden.json";
import {
  CAMERA_PRESETS,
  PHASE_LABELS,
  VIEW_LABELS,
  cameraPose,
  cameraPosition,
  phaseJumps,
  puttTile,
  scenePose,
} from "./cameraPresets";
import { IMPACT_LEAD_IN_S } from "@/pages/puttingPlayback";

type Vec3 = [number, number, number];
const close = (got: number[], want: number[]) =>
  want.forEach((w, i) => expect(got[i]).toBeCloseTo(w, 9));

describe("putting camera presets pinned to the desktop tile (#14474)", () => {
  it("offers the desktop views and phase names", () => {
    expect(CAMERA_PRESETS).toEqual(golden.presets);
    expect(VIEW_LABELS).toEqual(golden.view_labels);
    expect(PHASE_LABELS).toEqual(golden.phase_labels);
  });

  it.each(golden.cases)("poses each camera as the desktop does", (sample) => {
    const scene = {
      greenSize: sample.green_size as [number, number],
      start: sample.start_xyz as Vec3,
      cup: sample.cup_xyz as Vec3,
    };
    for (const preset of CAMERA_PRESETS) {
      const pose = cameraPose(scene, preset);
      const want = sample.poses[preset];
      close(pose.center, want.center);
      expect(pose.distance).toBeCloseTo(want.distance, 9);
      expect(pose.elevationDeg).toBeCloseTo(want.elevation_deg, 9);
      expect(pose.azimuthDeg).toBeCloseTo(want.azimuth_deg, 9);
    }
  });

  it("puts the eye on the orbit sphere around the target", () => {
    const pose = {
      center: [1, 2, 0] as Vec3,
      distance: 3,
      elevationDeg: 90,
      azimuthDeg: 0,
    };
    close(cameraPosition(pose), [1, 2, 3]);
  });

  it("jumps to each core phase mark after the impact lead-in", () => {
    const jumps = phaseJumps([
      { mode: "slide", t_s: 0, index: 0 },
      { mode: "roll", t_s: 0.4321, index: 9 },
    ]);
    expect(jumps).toEqual([
      { label: "Skid 0.00 s", playbackTimeS: IMPACT_LEAD_IN_S },
      { label: "Roll 0.43 s", playbackTimeS: IMPACT_LEAD_IN_S + 0.4321 },
    ]);
  });

  it.each(golden.tiles)("frames the putt on the desktop's tile", (tile) => {
    const scene = puttTile(tile.distance_m, 0);
    close(scene.greenSize, tile.green_size);
    close(scene.start, tile.start_xy);
    close(scene.cup, tile.cup_xy);
  });

  it("turns the tile's views onto the putt's line", () => {
    const along = scenePose(3, 0, 0, "face_on");
    const turned = scenePose(0, 3, 0, "face_on");
    // A putt along +y is the +x putt turned 90° about the ball.
    close(turned.eye, [-along.eye[1], along.eye[0], along.eye[2]]);
    expect(along.target).toEqual([0, 0, 0]);
  });
});
