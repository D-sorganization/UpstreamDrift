import { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid, Environment, Line } from '@react-three/drei';
import * as THREE from 'three';
import type { SimulationFrame } from '@/api/client';

interface Props {
  engine: string;
  frame: SimulationFrame | null;
  frames?: SimulationFrame[];
}

// Store trajectory history for trail visualization
const MAX_TRAIL_POINTS = 100;

function GolferModel({ frame }: { frame: SimulationFrame | null }) {
  const groupRef = useRef<THREE.Group>(null);
  const torsoRef = useRef<THREE.Mesh>(null);
  const leftArmRef = useRef<THREE.Mesh>(null);
  const rightArmRef = useRef<THREE.Mesh>(null);
  const clubRef = useRef<THREE.Group>(null);

  // Update pose from simulation frame
  useFrame(() => {
    if (!groupRef.current) return;

    // Get animation data from frame
    const time = frame?.time ?? 0;
    const jointAngles = frame?.analysis?.joint_angles;

    if (jointAngles && jointAngles.length >= 4) {
      // Apply joint angles from simulation
      if (torsoRef.current) {
        torsoRef.current.rotation.y = jointAngles[0] || 0;
      }
      if (leftArmRef.current) {
        leftArmRef.current.rotation.z = jointAngles[1] || 0;
      }
      if (rightArmRef.current) {
        rightArmRef.current.rotation.z = -(jointAngles[2] || 0);
      }
      if (clubRef.current) {
        clubRef.current.rotation.x = jointAngles[3] || 0;
      }
    } else if (frame) {
      // Default animation based on time if no joint angles
      const swingPhase = Math.sin(time * 2) * 0.5;

      if (torsoRef.current) {
        torsoRef.current.rotation.y = swingPhase * 0.8;
      }
      if (leftArmRef.current) {
        leftArmRef.current.rotation.z = -0.3 + swingPhase * 0.5;
      }
      if (rightArmRef.current) {
        rightArmRef.current.rotation.z = 0.3 - swingPhase * 0.5;
      }
      if (clubRef.current) {
        clubRef.current.rotation.x = swingPhase * 1.5;
      }
    }
  });

  return (
    <group ref={groupRef}>
      {/* Torso */}
      <mesh ref={torsoRef} position={[0, 1, 0]}>
        <capsuleGeometry args={[0.15, 0.6, 8, 16]} />
        <meshStandardMaterial color="#4a90d9" />

        {/* Head (child of torso) */}
        <mesh position={[0, 0.52, 0]}>
          <sphereGeometry args={[0.12]} />
          <meshStandardMaterial color="#e5e7eb" />
        </mesh>

        {/* Left Arm */}
        <mesh ref={leftArmRef} position={[-0.25, 0.2, 0]} rotation={[0, 0, -0.3]}>
          <capsuleGeometry args={[0.05, 0.4, 4, 8]} />
          <meshStandardMaterial color="#4a90d9" />
        </mesh>

        {/* Right Arm with Club */}
        <mesh ref={rightArmRef} position={[0.25, 0.2, 0]} rotation={[0, 0, 0.3]}>
          <capsuleGeometry args={[0.05, 0.4, 4, 8]} />
          <meshStandardMaterial color="#4a90d9" />

          {/* Club Group */}
          <group ref={clubRef} position={[0, -0.3, 0]}>
            {/* Club Shaft */}
            <mesh position={[0, -0.4, 0]}>
              <cylinderGeometry args={[0.015, 0.015, 0.8, 8]} />
              <meshStandardMaterial color="#666666" />
            </mesh>
            {/* Club Head */}
            <mesh position={[0, -0.85, 0]} rotation={[0.3, 0, 0]}>
              <boxGeometry args={[0.1, 0.03, 0.08]} />
              <meshStandardMaterial color="#333333" />
            </mesh>
          </group>
        </mesh>
      </mesh>

      {/* Legs (static for now) */}
      <mesh position={[-0.1, 0.3, 0]}>
        <capsuleGeometry args={[0.06, 0.5, 4, 8]} />
        <meshStandardMaterial color="#2d3748" />
      </mesh>
      <mesh position={[0.1, 0.3, 0]}>
        <capsuleGeometry args={[0.06, 0.5, 4, 8]} />
        <meshStandardMaterial color="#2d3748" />
      </mesh>
    </group>
  );
}

function ClubTrajectory({ frames }: { frames?: SimulationFrame[] }) {
  // Build trajectory trail from frame history
  const points = useMemo(() => {
    if (!frames || frames.length < 2) return [];

    // Get club head positions from recent frames
    const trailPoints: [number, number, number][] = [];
    const recentFrames = frames.slice(-MAX_TRAIL_POINTS);

    for (const f of recentFrames) {
      // Calculate approximate club head position based on swing animation
      const time = f.time;
      const swingPhase = Math.sin(time * 2) * 0.5;

      // Club head traces an arc
      const x = 0.25 + Math.sin(swingPhase * 2) * 0.8;
      const y = 0.5 + Math.cos(swingPhase * 2) * 0.3;
      const z = Math.sin(swingPhase) * 0.3;

      trailPoints.push([x, y, z]);
    }

    return trailPoints;
  }, [frames]);

  if (points.length < 2) return null;

  return (
    <Line
      points={points}
      color="#ffcc00"
      lineWidth={2}
    />
  );
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
export function Scene3D({ engine: _engine, frame, frames }: Props) {
  // Note: _engine prop is currently unused but kept in props for API compatibility.
  // Using a stable key instead of key={engine} to avoid expensive Canvas recreation
  // on engine changes. The Canvas persists and scene content updates based on new data.
  // If a full reset is needed when switching engines, consider resetting frame/frames state
  // in the parent component instead of remounting the entire Canvas.
  return (
    <Canvas
      camera={{ position: [3, 2, 3], fov: 50 }}
      className="bg-gray-900 w-full h-full"
    >
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 10, 5]} intensity={1} />

      <OrbitControls
        enableDamping
        dampingFactor={0.05}
        minDistance={1}
        maxDistance={10}
      />

      <Grid
        infiniteGrid
        cellSize={0.5}
        cellThickness={0.5}
        sectionSize={2}
        sectionThickness={1}
        fadeDistance={30}
      />

      <GolferModel frame={frame} />
      <ClubTrajectory frames={frames} />

      {/* Add axes for reference */}
      <axesHelper args={[1]} />

      <Environment preset="studio" />
    </Canvas>
  );
}
