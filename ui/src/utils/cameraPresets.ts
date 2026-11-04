/**
 * Camera views and phase jumps for the React putting page (#14474), ported
 * from the desktop tile's _camera.py and _playback.py. cameraPresets.golden.json
 * pins them to the desktop; phase marks themselves come from the core.
 */

import type { PhaseMarkResponse } from "@/api/generated/types";
import { IMPACT_LEAD_IN_S } from "@/pages/puttingPlayback";

type Vec3 = [number, number, number];

export const CAMERA_PRESETS = [
  "orbit",
  "down_the_line",
  "face_on",
  "overhead",
] as const;
export type CameraPreset = (typeof CAMERA_PRESETS)[number];

export const VIEW_LABELS: Record<CameraPreset, string> = {
  orbit: "Orbit",
  down_the_line: "Down the line",
  face_on: "Face on",
  overhead: "Overhead",
};

export const PHASE_LABELS: Record<PhaseMarkResponse["mode"], string> = {
  airborne: "Hop",
  slide: "Skid",
  roll: "Roll",
  rest: "Rest",
};

/** An orbit camera: the point it looks at [m], its range [m] and angles [deg]. */
export interface CameraPose {
  center: Vec3;
  distance: number;
  elevationDeg: number;
  azimuthDeg: number;
}

/** The desktop scene's frame: green from (0, 0) to greenSize, z up. */
export interface CameraScene {
  greenSize: [number, number];
  start: Vec3;
  cup: Vec3;
}

const deg = (rad: number) => (rad * 180) / Math.PI;

/** The eye of ``pose``, as the desktop's orbit camera places it. */
export function cameraPosition(pose: CameraPose): Vec3 {
  const elev = (pose.elevationDeg * Math.PI) / 180;
  const azim = (pose.azimuthDeg * Math.PI) / 180;
  const [cx, cy, cz] = pose.center;
  return [
    cx + pose.distance * Math.cos(elev) * Math.cos(azim),
    cy + pose.distance * Math.cos(elev) * Math.sin(azim),
    cz + pose.distance * Math.sin(elev),
  ];
}

/** The pose of camera ``preset`` on ``scene`` (desktop camera_pose). */
export function cameraPose(
  scene: CameraScene,
  preset: CameraPreset,
): CameraPose {
  const [width, height] = scene.greenSize;
  const middle: Vec3 = [width / 2, height / 2, 0];
  const { start, cup } = scene;
  const length = Math.hypot(cup[0] - start[0], cup[1] - start[1]);
  if (length === 0) throw new RangeError("the ball and the cup must differ");
  const ux = (cup[0] - start[0]) / length;
  const uy = (cup[1] - start[1]) / length;
  switch (preset) {
    case "down_the_line":
      return {
        center: [0, 1, 2].map((i) => (start[i] + cup[i]) / 2) as Vec3,
        distance: length / 2 + 2.5,
        elevationDeg: 12,
        azimuthDeg: deg(Math.atan2(-uy, -ux)),
      };
    case "face_on":
      return {
        center: start,
        distance: 3,
        elevationDeg: 8,
        azimuthDeg: deg(Math.atan2(-ux, uy)),
      };
    case "overhead":
      return {
        center: middle,
        distance: Math.max(width, height) * 1.4 + 2,
        elevationDeg: 89,
        azimuthDeg: -90,
      };
    default:
      return {
        center: middle,
        distance: Math.max(width, height) * 1.5 + 4,
        elevationDeg: 32,
        azimuthDeg: -78,
      };
  }
}

/** One jump per core phase mark, timed on the playback clock. */
export function phaseJumps(phases: PhaseMarkResponse[]) {
  return phases.map((mark) => ({
    label: `${PHASE_LABELS[mark.mode]} ${mark.t_s.toFixed(2)} s`,
    playbackTimeS: IMPACT_LEAD_IN_S + mark.t_s,
  }));
}

/** The desktop tile around a putt: 0.9 m behind the ball, 1.4 m past the cup. */
export function puttTile(distanceM: number, cupZ: number): CameraScene {
  const height = Math.max(4, distanceM * 0.6);
  return {
    greenSize: [distanceM + 2.3, height],
    start: [0.9, height / 2, 0],
    cup: [0.9 + distanceM, height / 2, cupZ],
  };
}

/**
 * The eye and target of ``preset`` for the web putt from the origin to the
 * hole, z up: the desktop tile's view turned and moved onto the putt's line.
 */
export function scenePose(
  holeX: number,
  holeY: number,
  holeZ: number,
  preset: CameraPreset,
): { eye: Vec3; target: Vec3 } {
  const distance = Math.hypot(holeX, holeY);
  const tile = puttTile(distance, holeZ);
  const pose = cameraPose(tile, preset);
  const [ux, uy] = [holeX / distance, holeY / distance];
  const toPutt = ([x, y, z]: Vec3): Vec3 => {
    const [along, across] = [x - tile.start[0], y - tile.start[1]];
    return [along * ux - across * uy, along * uy + across * ux, z];
  };
  return { eye: toPutt(cameraPosition(pose)), target: toPutt(pose.center) };
}
