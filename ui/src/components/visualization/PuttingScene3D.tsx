/** R3F putting scene backed exclusively by the canonical Python trajectory. */
import { useMemo } from "react";
import { Line, OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";

import type { Putt3DSimulationResponse } from "@/api/generated/types";
import {
  SIDEKICK_FALLBACK_COLOR_TOKENS,
  sidekickTokenToCSSVariable,
} from "@/api/themeClient";
import {
  IMPACT_LEAD_IN_S,
  ballRotationRad,
  sampleAtPlaybackTime,
} from "@/pages/puttingPlayback";

// Deliberately enlarged from the physical 21.35 mm radius so the spin marker
// remains legible at a whole-putt camera scale. Physics positions remain SI.
const BALL_VISUAL_RADIUS_M = 0.04;

type ThemeToken = keyof typeof SIDEKICK_FALLBACK_COLOR_TOKENS;

function resolveThemeColor(token: ThemeToken): string {
  const fallback = SIDEKICK_FALLBACK_COLOR_TOKENS[token];
  if (typeof document === "undefined") return fallback;
  const value = getComputedStyle(document.documentElement)
    .getPropertyValue(sidekickTokenToCSSVariable(token))
    .trim();
  return value || fallback;
}

function surfaceElevation(
  xM: number,
  yM: number,
  gradePercent: number,
  aspectDeg: number,
): number {
  const aspectRad = (aspectDeg * Math.PI) / 180;
  return (
    -(gradePercent / 100) *
    (xM * Math.cos(aspectRad) + yM * Math.sin(aspectRad))
  );
}

interface Props {
  result: Putt3DSimulationResponse;
  playbackTimeS: number;
  hoselToeM: number;
  hoselForwardM: number;
}

function SurfaceMesh({
  widthM,
  heightM,
  gradePercent,
  aspectDeg,
  color,
}: {
  widthM: number;
  heightM: number;
  gradePercent: number;
  aspectDeg: number;
  color: string;
}) {
  const geometry = useMemo(() => {
    const halfWidth = widthM / 2;
    const halfHeight = heightM / 2;
    const corners: [number, number][] = [
      [-halfWidth, -halfHeight],
      [halfWidth, -halfHeight],
      [halfWidth, halfHeight],
      [-halfWidth, halfHeight],
    ];
    const positions = new Float32Array(
      corners.flatMap(([xM, yM]) => [
        xM,
        surfaceElevation(xM, yM, gradePercent, aspectDeg),
        yM,
      ]),
    );
    const aspectRad = (aspectDeg * Math.PI) / 180;
    const grade = gradePercent / 100;
    const normalLength = Math.hypot(grade, 1);
    const normal: [number, number, number] = [
      (grade * Math.cos(aspectRad)) / normalLength,
      1 / normalLength,
      (grade * Math.sin(aspectRad)) / normalLength,
    ];
    return {
      positions,
      normals: new Float32Array(Array.from({ length: 4 }, () => normal).flat()),
      indices: new Uint16Array([0, 2, 1, 0, 3, 2]),
    };
  }, [aspectDeg, gradePercent, heightM, widthM]);

  return (
    <mesh receiveShadow>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[geometry.positions, 3]}
        />
        <bufferAttribute
          attach="attributes-normal"
          args={[geometry.normals, 3]}
        />
        <bufferAttribute attach="index" args={[geometry.indices, 1]} />
      </bufferGeometry>
      <meshStandardMaterial color={color} roughness={0.92} />
    </mesh>
  );
}

export function PuttingScene3D({
  result,
  playbackTimeS,
  hoselToeM,
  hoselForwardM,
}: Props) {
  const sample = sampleAtPlaybackTime(result.samples, playbackTimeS);
  const physicsTimeS = Math.max(0, playbackTimeS - IMPACT_LEAD_IN_S);
  const rotation = ballRotationRad(sample, physicsTimeS);
  const putterX =
    playbackTimeS <= IMPACT_LEAD_IN_S
      ? -result.collision.putter_speed_before_mps *
        (IMPACT_LEAD_IN_S - playbackTimeS)
      : result.collision.putter_speed_after_mps *
        (playbackTimeS - IMPACT_LEAD_IN_S);
  const holeZ = surfaceElevation(
    result.surface.hole_x_m,
    result.surface.hole_y_m,
    result.surface.grade_percent,
    result.surface.downhill_aspect_deg,
  );
  const colors = useMemo(
    () => ({
      ball: resolveThemeColor("sidekick.color.selection.text"),
      canvas: resolveThemeColor("sidekick.color.canvas"),
      green: resolveThemeColor("sidekick.color.success"),
      hole: resolveThemeColor("sidekick.color.input"),
      marker: resolveThemeColor("sidekick.color.warning"),
      putter: resolveThemeColor("sidekick.color.accent"),
      shaft: resolveThemeColor("sidekick.color.text.muted"),
      trail: resolveThemeColor("sidekick.color.info"),
    }),
    [],
  );
  const trail = useMemo<[number, number, number][]>(
    () =>
      result.samples.map((point) => [point.x_m, point.z_m + 0.008, point.y_m]),
    [result.samples],
  );

  return (
    <div
      role="img"
      aria-label="3D putting simulation. Drag to orbit and scroll to zoom."
      tabIndex={0}
      className="h-full w-full focus:outline focus:outline-2 focus:outline-offset-2"
      data-testid="putting-scene-3d"
      style={{
        backgroundColor: "var(--sidekick-color-canvas)",
        outlineColor: "var(--sidekick-color-focus)",
      }}
    >
      <Canvas
        camera={{ position: [4.5, 3.2, 5.5], fov: 46, near: 0.01, far: 80 }}
      >
        <ambientLight color={colors.shaft} intensity={0.7} />
        <directionalLight
          color={colors.ball}
          position={[3, 8, 4]}
          intensity={1.1}
        />
        <OrbitControls
          enableDamping
          dampingFactor={0.06}
          minDistance={1}
          maxDistance={24}
          target={[result.surface.hole_x_m / 2, 0, 0]}
        />

        <SurfaceMesh
          widthM={result.surface.width_m}
          heightM={result.surface.height_m}
          gradePercent={result.surface.grade_percent}
          aspectDeg={result.surface.downhill_aspect_deg}
          color={colors.green}
        />

        <mesh
          position={[
            result.surface.hole_x_m,
            holeZ - 0.018,
            result.surface.hole_y_m,
          ]}
        >
          <cylinderGeometry args={[0.054, 0.054, 0.035, 32]} />
          <meshStandardMaterial color={colors.hole} />
        </mesh>
        <mesh
          position={[
            result.surface.hole_x_m,
            holeZ + 0.55,
            result.surface.hole_y_m,
          ]}
        >
          <cylinderGeometry args={[0.006, 0.006, 1.1, 12]} />
          <meshStandardMaterial color={colors.shaft} />
        </mesh>
        <mesh
          position={[
            result.surface.hole_x_m + 0.14,
            holeZ + 0.93,
            result.surface.hole_y_m,
          ]}
        >
          <boxGeometry args={[0.28, 0.18, 0.008]} />
          <meshStandardMaterial color={colors.marker} />
        </mesh>

        {trail.length >= 2 && (
          <Line points={trail} color={colors.trail} lineWidth={2} />
        )}

        <group
          position={[sample.x_m, sample.z_m + BALL_VISUAL_RADIUS_M, sample.y_m]}
          rotation={[0, 0, rotation]}
        >
          <mesh castShadow>
            <sphereGeometry args={[BALL_VISUAL_RADIUS_M, 24, 16]} />
            <meshStandardMaterial color={colors.ball} roughness={0.45} />
          </mesh>
          <mesh rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry
              args={[
                BALL_VISUAL_RADIUS_M * 0.72,
                BALL_VISUAL_RADIUS_M * 0.08,
                8,
                32,
              ]}
            />
            <meshStandardMaterial color={colors.marker} />
          </mesh>
        </group>

        {playbackTimeS <= IMPACT_LEAD_IN_S + 0.45 && (
          <group position={[putterX, 0.045, 0]}>
            <mesh position={[0, 0, 0]}>
              <boxGeometry args={[0.08, 0.065, 0.22]} />
              <meshStandardMaterial
                color={colors.putter}
                metalness={0.6}
                roughness={0.3}
              />
            </mesh>
            <mesh
              position={[hoselForwardM, 0.34, hoselToeM]}
              rotation={[0, 0, -0.16]}
            >
              <cylinderGeometry args={[0.007, 0.007, 0.68, 12]} />
              <meshStandardMaterial color={colors.shaft} metalness={0.7} />
            </mesh>
            <mesh position={[hoselForwardM, 0.075, hoselToeM]}>
              <sphereGeometry args={[0.014, 16, 12]} />
              <meshStandardMaterial color={colors.marker} />
            </mesh>
          </group>
        )}
      </Canvas>
    </div>
  );
}
