import { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid, Environment } from '@react-three/drei';
import * as THREE from 'three';
import { SimulationFrame } from '../../api/client';

interface Props {
  engine: string;
  frame: SimulationFrame | null;
}

function GolferModel({ frame }: { frame: SimulationFrame | null }) {
  const groupRef = useRef<THREE.Group>(null);

  // Update pose from simulation frame
  useFrame(() => {
    if (!frame?.state || !groupRef.current) return;

    // Apply joint angles to skeleton
    // This would map simulation state to Three.js bone rotations
    // Placeholder logic for now
  });

  return (
    <group ref={groupRef}>
      {/* Simplified golfer visualization */}
      <mesh position={[0, 1, 0]}>
        <capsuleGeometry args={[0.15, 0.6, 8, 16]} />
        <meshStandardMaterial color="#4a90d9" />
      </mesh>
      {/* Head */}
      <mesh position={[0, 1.4, 0]}>
         <sphereGeometry args={[0.12]} />
         <meshStandardMaterial color="#e5e7eb" />
      </mesh>
    </group>
  );
}

function ClubTrajectory({ frame }: { frame: SimulationFrame | null }) {
  // Render club head trajectory trail
  const points = useMemo(() => {
    if (!frame) return [];
    // TODO: Maintain history of points for trail
    return [[0,0,0], [1,1,1]]; 
  }, [frame]);

  if (points.length < 2) return null;

  const geometry = useMemo(() => {
      const geo = new THREE.BufferGeometry();
      geo.setFromPoints(points.map(p => new THREE.Vector3(...p)));
      return geo;
  }, [points]);

  return (
    <line geometry={geometry}>
      <lineBasicMaterial color="#ffcc00" linewidth={2} />
    </line>
  );
}

export function Scene3D({ engine, frame }: Props) {
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
      <ClubTrajectory frame={frame} />
      
      {/* Add axes for reference */}
      <axesHelper args={[1]} />

      <Environment preset="studio" />
    </Canvas>
  );
}
