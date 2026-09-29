/**
 * MocapSkeleton3D - R3F 3D visualization of a motion-capture skeleton frame.
 *
 * Renders joints as spheres and bones (parent-child connections) as line
 * segments inside an orbitable Canvas with a ground grid. Consumes the same
 * frame data shape (`JointData[]`) as the 2D SVG `SkeletonRenderer` in
 * `ui/src/pages/MotionCapture.tsx`, so the two views are interchangeable.
 *
 * Kept in its own module so route pages can lazy-load it and three.js stays
 * out of the initial chunk (same pattern as Scene3D).
 *
 * See issue #8406.
 */

import { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid, Line } from '@react-three/drei';

/**
 * Joint data consumed by the 3D skeleton. Structurally identical to
 * `JointData` in `ui/src/pages/MotionCapture.tsx` (declared here too so the
 * page can lazy-load this module without a value-level circular import).
 */
export interface MocapJoint {
  name: string;
  /** [x, y, z] position; z may be omitted for 2D sources. */
  position: number[];
  /** Detection confidence in [0, 1]. */
  confidence: number;
  parent: string | null;
}

interface MocapSkeleton3DProps {
  joints: MocapJoint[];
}

/** Normalize a joint position into a 3-tuple (missing z treated as 0). */
function toVec3(position: number[]): [number, number, number] {
  return [position[0] ?? 0, position[1] ?? 0, position[2] ?? 0];
}

interface BoneSegment {
  name: string;
  start: [number, number, number];
  end: [number, number, number];
  confidence: number;
}

/**
 * Joints + bones as three.js objects. Split from the Canvas wrapper so it can
 * be reused inside an existing scene if ever needed.
 */
export function MocapSkeletonJoints({ joints }: MocapSkeleton3DProps) {
  const bones = useMemo<BoneSegment[]>(() => {
    const jointMap = new Map<string, MocapJoint>();
    for (const joint of joints) {
      jointMap.set(joint.name, joint);
    }
    const segments: BoneSegment[] = [];
    for (const joint of joints) {
      if (!joint.parent) continue;
      const parent = jointMap.get(joint.parent);
      if (!parent) continue;
      segments.push({
        name: joint.name,
        start: toVec3(parent.position),
        end: toVec3(joint.position),
        confidence: Math.min(joint.confidence, parent.confidence),
      });
    }
    return segments;
  }, [joints]);

  return (
    <group>
      {/* Bones: parent → child segments, opacity scaled by confidence */}
      {bones.map((bone) => (
        <Line
          key={`bone-${bone.name}`}
          points={[bone.start, bone.end]}
          color="#3b82f6"
          lineWidth={2}
          transparent
          opacity={0.3 + bone.confidence * 0.7}
        />
      ))}

      {/* Joints: spheres sized and faded by confidence */}
      {joints.map((joint) => (
        <mesh
          key={`joint-${joint.name}`}
          position={toVec3(joint.position)}
          name={joint.name}
        >
          <sphereGeometry args={[0.02 + joint.confidence * 0.02, 12, 12]} />
          <meshStandardMaterial
            color="#60a5fa"
            transparent
            opacity={0.4 + joint.confidence * 0.6}
          />
        </mesh>
      ))}
    </group>
  );
}

/**
 * Full 3D skeleton view: Canvas + lights + orbit controls + grid + skeleton.
 */
export function MocapSkeleton3D({ joints }: MocapSkeleton3DProps) {
  return (
    <div
      role="img"
      aria-label="3D motion-capture skeleton. Use mouse to rotate view, scroll to zoom."
      className="w-full h-full"
      data-testid="mocap-skeleton-3d"
    >
      <Canvas
        camera={{ position: [1.6, 1.2, 1.6], fov: 50 }}
        className="bg-gray-900 w-full h-full"
      >
        <ambientLight intensity={0.6} />
        <directionalLight position={[5, 5, 5]} intensity={0.8} />

        <OrbitControls
          enableDamping
          dampingFactor={0.05}
          minDistance={0.5}
          maxDistance={10}
        />

        <Grid
          infiniteGrid
          cellSize={0.25}
          cellThickness={0.5}
          sectionSize={1}
          sectionThickness={1}
          fadeDistance={20}
        />

        <MocapSkeletonJoints joints={joints} />

        <axesHelper args={[0.5]} />
      </Canvas>
    </div>
  );
}

// Default export so route modules can `React.lazy(() => import(...))` it
// without a named-to-default mapping step.
export default MocapSkeleton3D;
