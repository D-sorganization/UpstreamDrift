import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Line } from '@react-three/drei';
import * as THREE from 'three';
import type { SimulationFrame } from '@/api/client';

const MAX_TRAIL_POINTS = 100;

interface GolferModelProps {
  frame: SimulationFrame | null;
  rootRef?: React.RefObject<THREE.Group | null>;
  selectedBodyName?: string | null;
  onSelectBody?: (name: string | null) => void;
}

export function GolferModel({
  frame,
  rootRef,
  selectedBodyName,
  onSelectBody,
}: GolferModelProps) {
  const torsoRef = useRef<THREE.Mesh>(null);
  const leftArmRef = useRef<THREE.Mesh>(null);
  const rightArmRef = useRef<THREE.Mesh>(null);
  const clubRef = useRef<THREE.Group>(null);

  // Update pose from simulation frame
  useFrame(() => {
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

  const getMaterial = (bodyName: string, defaultColor: string) => {
    const isSelected = selectedBodyName === bodyName;
    return (
      <meshStandardMaterial
        color={isSelected ? '#ffcc00' : defaultColor}
        emissive={isSelected ? '#332200' : '#000000'}
      />
    );
  };

  const handleMeshClick = (e: { stopPropagation: () => void }, bodyName: string) => {
    e.stopPropagation();
    if (onSelectBody) {
      onSelectBody(bodyName);
    }
  };

  return (
    <group ref={rootRef}>
      {/* Torso */}
      <mesh
        ref={torsoRef}
        position={[0, 1, 0]}
        name="torso"
        onClick={(e) => handleMeshClick(e, 'torso')}
      >
        <capsuleGeometry args={[0.15, 0.6, 8, 16]} />
        {getMaterial('torso', '#4a90d9')}

        {/* Head (child of torso) */}
        <mesh
          position={[0, 0.52, 0]}
          name="head"
          onClick={(e) => handleMeshClick(e, 'head')}
        >
          <sphereGeometry args={[0.12]} />
          {getMaterial('head', '#e5e7eb')}
        </mesh>

        {/* Left Arm */}
        <mesh
          ref={leftArmRef}
          position={[-0.25, 0.2, 0]}
          rotation={[0, 0, -0.3]}
          name="left_arm"
          onClick={(e) => handleMeshClick(e, 'left_arm')}
        >
          <capsuleGeometry args={[0.05, 0.4, 4, 8]} />
          {getMaterial('left_arm', '#4a90d9')}
        </mesh>

        {/* Right Arm with Club */}
        <mesh
          ref={rightArmRef}
          position={[0.25, 0.2, 0]}
          rotation={[0, 0, 0.3]}
          name="right_arm"
          onClick={(e) => handleMeshClick(e, 'right_arm')}
        >
          <capsuleGeometry args={[0.05, 0.4, 4, 8]} />
          {getMaterial('right_arm', '#4a90d9')}

          {/* Club Group */}
          <group ref={clubRef} position={[0, -0.3, 0]}>
            {/* Club Shaft */}
            <mesh
              position={[0, -0.4, 0]}
              name="club_shaft"
              onClick={(e) => handleMeshClick(e, 'club_shaft')}
            >
              <cylinderGeometry args={[0.015, 0.015, 0.8, 8]} />
              {getMaterial('club_shaft', '#666666')}
            </mesh>
            {/* Club Head */}
            <mesh
              position={[0, -0.85, 0]}
              rotation={[0.3, 0, 0]}
              name="club_head"
              onClick={(e) => handleMeshClick(e, 'club_head')}
            >
              <boxGeometry args={[0.1, 0.03, 0.08]} />
              {getMaterial('club_head', '#333333')}
            </mesh>
          </group>
        </mesh>
      </mesh>

      {/* Legs (static for now) */}
      <mesh
        position={[-0.1, 0.3, 0]}
        name="left_leg"
        onClick={(e) => handleMeshClick(e, 'left_leg')}
      >
        <capsuleGeometry args={[0.06, 0.5, 4, 8]} />
        {getMaterial('left_leg', '#2d3748')}
      </mesh>
      <mesh
        position={[0.1, 0.3, 0]}
        name="right_leg"
        onClick={(e) => handleMeshClick(e, 'right_leg')}
      >
        <capsuleGeometry args={[0.06, 0.5, 4, 8]} />
        {getMaterial('right_leg', '#2d3748')}
      </mesh>
    </group>
  );
}

export function ClubTrajectory({ frames }: { frames?: SimulationFrame[] }) {
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
