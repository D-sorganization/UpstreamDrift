/**
 * URDFViewer - Renders a URDF model using Three.js primitives.
 *
 * Parses URDF model data from the backend API and builds a kinematic
 * chain using Three.js groups and primitive geometries (box, cylinder,
 * sphere). Supports joint angle updates for real-time animation.
 *
 * See issue #1201
 */

import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

/** Geometry descriptor from the backend URDF parser. */
export interface URDFLinkGeometry {
  link_name: string;
  geometry_type: 'box' | 'cylinder' | 'sphere' | 'mesh';
  dimensions: Record<string, number>;
  origin: [number, number, number];
  rotation: [number, number, number];
  color: [number, number, number, number];
  mesh_path?: string | null;
}

/** Joint descriptor from the backend URDF parser. */
export interface URDFJointDescriptor {
  name: string;
  joint_type: 'revolute' | 'prismatic' | 'fixed' | 'continuous' | 'floating';
  parent_link: string;
  child_link: string;
  origin: [number, number, number];
  rotation: [number, number, number];
  axis: [number, number, number];
  lower_limit?: number | null;
  upper_limit?: number | null;
}

/** Parsed URDF model from the backend. */
export interface URDFModel {
  model_name: string;
  links: URDFLinkGeometry[];
  joints: URDFJointDescriptor[];
  root_link: string;
  urdf_raw?: string | null;
}

interface URDFViewerProps {
  /** URDF model data from the API */
  model: URDFModel | null;
  /** Current joint angles keyed by joint name or index */
  jointAngles?: Record<string, number> | number[];
  /** Opacity for the model (0-1) */
  opacity?: number;
  /** Whether to show joint axes */
  showAxes?: boolean;
}

/** Helper to create a Three.js Euler from RPY angles. */
function rpyToEuler(rpy: [number, number, number]): THREE.Euler {
  return new THREE.Euler(rpy[0], rpy[1], rpy[2], 'XYZ');
}

/**
 * Renders a single URDF link geometry as a Three.js mesh.
 */
function LinkMesh({ link }: { link: URDFLinkGeometry }) {
  const color = useMemo(
    () => new THREE.Color(link.color[0], link.color[1], link.color[2]),
    [link.color],
  );
  const opacity = link.color[3] ?? 1.0;
  const position = useMemo(
    () => new THREE.Vector3(...link.origin),
    [link.origin],
  );
  const rotation = useMemo(
    () => rpyToEuler(link.rotation),
    [link.rotation],
  );

  const geometry = useMemo(() => {
    const dims = link.dimensions;
    switch (link.geometry_type) {
      case 'box':
        return (
          <boxGeometry
            args={[
              dims.width ?? 0.1,
              dims.height ?? 0.1,
              dims.depth ?? 0.1,
            ]}
          />
        );
      case 'cylinder':
        return (
          <cylinderGeometry
            args={[
              dims.radius ?? 0.05,
              dims.radius ?? 0.05,
              dims.length ?? 0.3,
              16,
            ]}
          />
        );
      case 'sphere':
        return <sphereGeometry args={[dims.radius ?? 0.1, 16, 16]} />;
      default:
        // Fallback to a small sphere for unsupported geometry types (mesh, etc.)
        return <sphereGeometry args={[0.02, 8, 8]} />;
    }
  }, [link.geometry_type, link.dimensions]);

  return (
    <mesh position={position} rotation={rotation}>
      {geometry}
      <meshStandardMaterial
        color={color}
        opacity={opacity}
        transparent={opacity < 1.0}
      />
    </mesh>
  );
}

/**
 * Recursively builds the kinematic chain from URDF data.
 *
 * Each joint creates a nested Three.js group with the joint's
 * origin transform, and the child link geometry is placed inside.
 */
function KinematicChain({
  linkName,
  links,
  joints,
  jointAngles,
  showAxes,
}: {
  linkName: string;
  links: Map<string, URDFLinkGeometry>;
  joints: URDFJointDescriptor[];
  jointAngles: Map<string, number>;
  showAxes: boolean;
}) {
  const link = links.get(linkName);

  // Find child joints (where this link is the parent)
  const childJoints = useMemo(
    () => joints.filter((j) => j.parent_link === linkName),
    [joints, linkName],
  );

  // Create refs for animated joints
  const jointRefs = useRef<Map<string, THREE.Group>>(new Map());

  // Update joint angles on each frame
  useFrame(() => {
    for (const joint of childJoints) {
      const ref = jointRefs.current.get(joint.name);
      if (!ref) continue;

      const angle = jointAngles.get(joint.name) ?? 0;

      if (joint.joint_type === 'revolute' || joint.joint_type === 'continuous') {
        // Apply rotation around the joint axis
        const axis = new THREE.Vector3(...joint.axis).normalize();
        const quat = new THREE.Quaternion().setFromAxisAngle(axis, angle);
        // Combine with the joint's base rotation
        const baseEuler = rpyToEuler(joint.rotation);
        const baseQuat = new THREE.Quaternion().setFromEuler(baseEuler);
        ref.quaternion.copy(baseQuat).multiply(quat);
      } else if (joint.joint_type === 'prismatic') {
        // Apply translation along the joint axis
        const axis = new THREE.Vector3(...joint.axis).normalize();
        const basePos = new THREE.Vector3(...joint.origin);
        ref.position.copy(basePos.add(axis.multiplyScalar(angle)));
      }
    }
  });

  return (
    <group>
      {/* Render this link's geometry */}
      {link && <LinkMesh link={link} />}

      {/* Render child joints and their subtrees */}
      {childJoints.map((joint) => (
        <group
          key={joint.name}
          ref={(ref: THREE.Group | null) => {
            if (ref) {
              jointRefs.current.set(joint.name, ref);
            }
          }}
          position={joint.origin}
          rotation={rpyToEuler(joint.rotation)}
        >
          {showAxes && <axesHelper args={[0.1]} />}
          <KinematicChain
            linkName={joint.child_link}
            links={links}
            joints={joints}
            jointAngles={jointAngles}
            showAxes={showAxes}
          />
        </group>
      ))}
    </group>
  );
}

/**
 * URDFViewer component renders a URDF robot model in Three.js.
 *
 * Accepts parsed URDF data from the backend and builds a kinematic
 * chain with proper joint transforms. Joint angles can be updated
 * in real-time for simulation animation.
 */
export function URDFViewer({
  model,
  jointAngles,
  // opacity reserved for future group-level transparency. See issue #1201
  opacity: _opacity = 1.0, // eslint-disable-line @typescript-eslint/no-unused-vars
  showAxes = false,
}: URDFViewerProps) {
  // Build lookup maps
  const linkMap = useMemo(() => {
    const map = new Map<string, URDFLinkGeometry>();
    if (model) {
      for (const link of model.links) {
        map.set(link.link_name, link);
      }
    }
    return map;
  }, [model]);

  const jointAngleMap = useMemo(() => {
    const map = new Map<string, number>();
    if (!jointAngles || !model) return map;

    if (Array.isArray(jointAngles)) {
      // Map by joint index
      model.joints.forEach((joint, idx) => {
        if (idx < jointAngles.length) {
          map.set(joint.name, jointAngles[idx]);
        }
      });
    } else {
      // Map by joint name
      for (const [name, angle] of Object.entries(jointAngles)) {
        map.set(name, angle);
      }
    }
    return map;
  }, [jointAngles, model]);

  if (!model) {
    return null;
  }

  return (
    <group>
      <KinematicChain
        linkName={model.root_link}
        links={linkMap}
        joints={model.joints}
        jointAngles={jointAngleMap}
        showAxes={showAxes}
      />
    </group>
  );
}

// useURDFModel hook is exported from @/api/useURDFModel.ts
// to comply with react-refresh (components-only files).
