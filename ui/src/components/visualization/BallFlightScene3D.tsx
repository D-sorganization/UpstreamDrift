/**
 * BallFlightScene3D — 3D multi-model ball-flight trajectory overlay.
 *
 * Web counterpart of the desktop Shot Tracer overlay view. Reuses the
 * existing trail-rendering approach (drei `Line`, as in `GolferModel`'s
 * `ClubTrajectory`) rather than forking a new 3D stack.
 *
 * Coordinate convention: the physics API returns positions as
 * [downrange x (m), lateral y (m), height z (m)]; three.js is Y-up, so
 * samples map to [x, z, y].
 *
 * See issue #7456.
 */

import { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid, Line } from '@react-three/drei';

export interface ModelTrajectory3D {
  /** Stable model key (e.g. "waterloo_penner"). */
  modelKey: string;
  /** Human-readable model name for tooltips/legends. */
  modelName: string;
  /** CSS color used for this model's trail. */
  color: string;
  /** Trajectory samples in physics frame: [downrange, lateral, height] (m). */
  positions: number[][];
}

/** Map a physics-frame sample [x, y, z] to a three.js Y-up point. */
function toThreePoint(sample: number[]): [number, number, number] {
  return [sample[0], sample[2], sample[1]];
}

export function BallFlightScene3D({
  trajectories,
}: {
  trajectories: ModelTrajectory3D[];
}) {
  // Frame the camera around the longest trajectory.
  const maxRange = useMemo(() => {
    let max = 50;
    for (const t of trajectories) {
      for (const p of t.positions) {
        max = Math.max(max, Math.abs(p[0]), Math.abs(p[1]), Math.abs(p[2]));
      }
    }
    return max;
  }, [trajectories]);

  return (
    <div
      role="img"
      aria-label="3D ball-flight trajectory overlay. Use mouse to rotate view, scroll to zoom."
      tabIndex={0}
      className="relative w-full h-full focus:outline-none focus:ring-2 focus:ring-blue-400"
      data-testid="ball-flight-scene3d"
    >
      <Canvas
        camera={{
          position: [-maxRange * 0.4, maxRange * 0.5, maxRange * 0.9],
          fov: 50,
          near: 0.1,
          far: maxRange * 20,
        }}
        className="bg-gray-900 w-full h-full"
      >
        <ambientLight intensity={0.6} />
        <directionalLight position={[50, 100, 50]} intensity={1} />

        <OrbitControls
          enableDamping
          dampingFactor={0.05}
          minDistance={5}
          maxDistance={maxRange * 6}
          target={[maxRange * 0.5, 0, 0]}
        />

        <Grid
          infiniteGrid
          cellSize={10}
          cellThickness={0.5}
          sectionSize={50}
          sectionThickness={1}
          fadeDistance={maxRange * 4}
        />

        {trajectories.map(
          (t) =>
            t.positions.length >= 2 && (
              <Line
                key={t.modelKey}
                points={t.positions.map(toThreePoint)}
                color={t.color}
                lineWidth={2}
              />
            ),
        )}

        <axesHelper args={[10]} />
      </Canvas>
    </div>
  );
}
