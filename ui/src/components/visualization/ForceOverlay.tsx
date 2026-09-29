/**
 * ForceOverlay - Force/torque vector overlay visualization using Three.js.
 *
 * Renders force and torque vectors as arrows in the 3D scene.
 * Supports color-coding by magnitude, per-body filtering, and
 * curved arrows for torque visualization.
 *
 * See issue #1199
 */

import { useMemo } from 'react';
import * as THREE from 'three';

/** Force vector data from the backend API. See issue #1199 */
export interface ForceVector3D {
  body_name: string;
  force_type: 'applied' | 'gravity' | 'contact' | 'bias';
  origin: [number, number, number];
  direction: [number, number, number];
  magnitude: number;
  color: [number, number, number, number];
  label?: string | null;
}

/** Overlay configuration. See issue #1199 */
export interface ForceOverlayConfig {
  enabled: boolean;
  forceTypes: string[];
  scaleFactor: number;
  colorByMagnitude: boolean;
  showLabels: boolean;
  bodyFilter: string[] | null;
}

interface ForceOverlayProps {
  /** Force vectors from the API */
  vectors: ForceVector3D[];
  /** Scale factor for arrow length */
  scaleFactor?: number;
  /** Maximum arrow length to prevent clipping */
  maxArrowLength?: number;
}

/**
 * Renders a single force arrow (shaft + cone head).
 */
function ForceArrowMesh({
  vector,
  scaleFactor,
  maxLength,
}: {
  vector: ForceVector3D;
  scaleFactor: number;
  maxLength: number;
}) {
  const color = useMemo(
    () => new THREE.Color(vector.color[0], vector.color[1], vector.color[2]),
    [vector.color],
  );

  const { shaftPosition, headPosition, quaternion, shaftLength } = useMemo(() => {
    const dir = new THREE.Vector3(...vector.direction).normalize();
    const length = Math.min(vector.magnitude * scaleFactor, maxLength);
    const sLen = Math.max(length - 0.08, 0.02);

    // Quaternion to rotate from Y-up to the direction
    const quat = new THREE.Quaternion().setFromUnitVectors(
      new THREE.Vector3(0, 1, 0),
      dir,
    );

    // Position shaft center along the direction
    const shaftPos = dir
      .clone()
      .multiplyScalar(sLen * 0.5)
      .toArray() as [number, number, number];

    // Position cone at end of shaft
    const headPos = dir
      .clone()
      .multiplyScalar(sLen + 0.04)
      .toArray() as [number, number, number];

    return {
      shaftPosition: shaftPos,
      headPosition: headPos,
      quaternion: quat,
      shaftLength: sLen,
    };
  }, [vector.direction, vector.magnitude, scaleFactor, maxLength]);

  return (
    <group position={vector.origin}>
      {/* Arrow shaft */}
      <mesh position={shaftPosition} quaternion={quaternion}>
        <cylinderGeometry args={[0.012, 0.012, shaftLength, 8]} />
        <meshStandardMaterial color={color} />
      </mesh>

      {/* Arrow head (cone) */}
      <mesh position={headPosition} quaternion={quaternion}>
        <coneGeometry args={[0.035, 0.08, 8]} />
        <meshStandardMaterial color={color} />
      </mesh>
    </group>
  );
}

/**
 * Renders a torque vector as a curved arrow (torus segment + cone).
 */
function TorqueArrowMesh({
  vector,
  scaleFactor,
}: {
  vector: ForceVector3D;
  scaleFactor: number;
}) {
  const color = useMemo(
    () => new THREE.Color(vector.color[0], vector.color[1], vector.color[2]),
    [vector.color],
  );

  const radius = useMemo(
    () => Math.min(vector.magnitude * scaleFactor * 0.5, 0.5),
    [vector.magnitude, scaleFactor],
  );

  const quaternion = useMemo(() => {
    const dir = new THREE.Vector3(...vector.direction).normalize();
    return new THREE.Quaternion().setFromUnitVectors(
      new THREE.Vector3(0, 1, 0),
      dir,
    );
  }, [vector.direction]);

  return (
    <group position={vector.origin} quaternion={quaternion}>
      {/* Curved arrow body (torus arc) */}
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[radius, 0.01, 8, 16, Math.PI * 1.5]} />
        <meshStandardMaterial color={color} />
      </mesh>

      {/* Arrow tip at end of arc */}
      <mesh
        position={[radius, 0, 0]}
        rotation={[0, 0, -Math.PI / 2]}
      >
        <coneGeometry args={[0.03, 0.06, 8]} />
        <meshStandardMaterial color={color} />
      </mesh>
    </group>
  );
}

/**
 * ForceOverlay renders all force/torque vectors in the 3D scene.
 *
 * Force vectors are rendered as straight arrows (shaft + cone).
 * Torque/applied vectors use the same representation but could
 * be toggled to curved arrows in the future.
 *
 * See issue #1199
 */
export function ForceOverlay({
  vectors,
  scaleFactor = 0.01,
  maxArrowLength = 2.0,
}: ForceOverlayProps) {
  if (!vectors || vectors.length === 0) {
    return null;
  }

  return (
    <group>
      {vectors.map((vector, idx) => {
        // Use torque visualization for applied torques
        if (vector.force_type === 'applied' && vector.magnitude > 0) {
          return (
            <TorqueArrowMesh
              key={`torque-${idx}`}
              vector={vector}
              scaleFactor={scaleFactor}
            />
          );
        }

        // Standard force arrows for everything else
        return (
          <ForceArrowMesh
            key={`force-${idx}`}
            vector={vector}
            scaleFactor={scaleFactor}
            maxLength={maxArrowLength}
          />
        );
      })}
    </group>
  );
}
