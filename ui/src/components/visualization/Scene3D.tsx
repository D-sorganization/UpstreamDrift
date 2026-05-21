/**
 * Scene3D - 3D visualization of golf swing simulation.
 *
 * Renders either a URDF model (when available) or falls back to
 * hardcoded capsule/sphere geometry. Supports joint angle animation,
 * club trajectory trails, and force/torque overlays.
 *
 * See issue #1201, #1199
 */

import { useRef, useMemo, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid, Environment, TransformControls } from '@react-three/drei';
import * as THREE from 'three';
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib';
import type { SimulationFrame } from '@/api/client';
import { URDFViewer } from './URDFViewer';
import type { URDFModel } from './URDFViewer';
import { GolferModel, ClubTrajectory } from './GolferModel';
import { ForceOverlay as ForceOverlayComponent } from './ForceOverlay';
import type { ForceVector3D } from './ForceOverlay';

/** Force/torque overlay data for visualization. See issue #1179 */
export interface ForceOverlay {
  origin: [number, number, number];
  direction: [number, number, number];
  magnitude: number;
  color?: string;
  label?: string;
}

interface Props {
  engine: string;
  frame: SimulationFrame | null;
  frames?: SimulationFrame[];
  /** Optional URDF model to render instead of hardcoded geometry. See issue #1201 */
  urdfModel?: URDFModel | null;
  /** Whether to show joint axes on the URDF model */
  showJointAxes?: boolean;
  /** Force vectors to display as overlays. See issue #1179 */
  forceOverlays?: (ForceOverlay | ForceVector3D)[];
  /** Callback when gizmo is dragged to send position/rotation changes */
  onGizmoDrag?: (bodyName: string, position: number[], rotation: number[]) => void;
}

/**
 * CameraController updates OrbitControls target to follow the golfer's root position in Follow Mode.
 */
function CameraController({
  followMode,
  rootRef,
  orbitRef,
}: {
  followMode: boolean;
  rootRef: React.RefObject<THREE.Group | null>;
  orbitRef: React.RefObject<OrbitControlsImpl | null>;
}) {
  useFrame(() => {
    if (!followMode || !rootRef.current || !orbitRef.current) return;

    const controls = orbitRef.current;
    const worldPosition = new THREE.Vector3();
    rootRef.current.getWorldPosition(worldPosition);

    // Shift camera by the target translation delta to track target smoothly
    const targetDelta = worldPosition.clone().sub(controls.target);
    controls.object.position.add(targetDelta);

    controls.target.copy(worldPosition);
    controls.update();
  });

  return null;
}

export function Scene3D({
  engine: _engine, // eslint-disable-line @typescript-eslint/no-unused-vars
  frame,
  frames,
  urdfModel,
  showJointAxes = false,
  forceOverlays,
  onGizmoDrag,
}: Props) {
  const orbitRef = useRef<OrbitControlsImpl | null>(null);
  const rootRef = useRef<THREE.Group>(null);

  // Interaction State
  const [selectedBodyName, setSelectedBodyName] = useState<string | null>(null);
  const [selectedObject, setSelectedObject] = useState<THREE.Object3D | null>(null);
  const [transformMode, setTransformMode] = useState<'translate' | 'rotate'>('translate');
  const [followMode, setFollowMode] = useState<boolean>(false);

  // Map input force overlays to standard ForceVector3D format
  const mappedVectors = useMemo(() => {
    if (!forceOverlays) return [];
    return forceOverlays.map((fo) => {
      if ('force_type' in fo && Array.isArray(fo.color)) {
        return fo as ForceVector3D;
      }
      let parsedColor: [number, number, number, number] = [1, 0, 0, 1];
      if (typeof fo.color === 'string') {
        const temp = new THREE.Color(fo.color);
        parsedColor = [temp.r, temp.g, temp.b, 1];
      }
      return {
        body_name: 'unknown',
        force_type: 'contact',
        origin: fo.origin,
        direction: fo.direction,
        magnitude: fo.magnitude,
        color: parsedColor,
        label: fo.label || null,
      } as ForceVector3D;
    });
  }, [forceOverlays]);

  // Handle camera presets
  const handleCameraPreset = (preset: 'front' | 'side' | 'top') => {
    setFollowMode(false);
    if (!orbitRef.current) return;

    const controls = orbitRef.current;
    const camera = controls.object;
    const target = new THREE.Vector3(0, 1.0, 0);

    controls.target.copy(target);

    if (preset === 'front') {
      camera.position.set(0, 1.2, 3);
      camera.up.set(0, 1, 0);
    } else if (preset === 'side') {
      camera.position.set(3, 1.2, 0);
      camera.up.set(0, 1, 0);
    } else if (preset === 'top') {
      camera.position.set(0, 4, 0);
      camera.up.set(0, 0, -1);
    }

    controls.update();
  };

  const handleGizmoChange = () => {
    if (!selectedObject || !onGizmoDrag) return;
    const position = selectedObject.position.toArray();
    const rotation = selectedObject.rotation.toArray().slice(0, 3) as [number, number, number];
    onGizmoDrag(selectedBodyName || selectedObject.name || 'unknown', position, rotation);
  };

  // Determine whether to use URDF model or fallback
  const useURDF = urdfModel != null && urdfModel.links.length > 0;

  // Build joint angles from frame for URDF model
  const urdfJointAngles = useMemo(() => {
    if (!frame?.analysis?.joint_angles) return undefined;
    return frame.analysis.joint_angles;
  }, [frame]);

  return (
    <div
      role="img"
      aria-label="3D golf swing simulation visualization. Use mouse to rotate view, scroll to zoom."
      tabIndex={0}
      className="relative w-full h-full focus:outline-none focus:ring-2 focus:ring-blue-400"
    >
      <Canvas
        camera={{ position: [3, 2, 3], fov: 50 }}
        className="bg-gray-900 w-full h-full"
        onClick={() => {
          // Deselect on clicking empty background
          if (selectedObject) {
            setSelectedObject(null);
            setSelectedBodyName(null);
          }
        }}
      >
        <ambientLight intensity={0.5} />
        <directionalLight position={[10, 10, 5]} intensity={1} />

        <OrbitControls
          ref={orbitRef}
          enableDamping
          dampingFactor={0.05}
          minDistance={1}
          maxDistance={10}
        />

        <CameraController
          followMode={followMode}
          rootRef={rootRef}
          orbitRef={orbitRef}
        />

        <Grid
          infiniteGrid
          cellSize={0.5}
          cellThickness={0.5}
          sectionSize={2}
          sectionThickness={1}
          fadeDistance={30}
        />

        {/* Selected Link Gizmo Controls */}
        {selectedObject && (
          <TransformControls
            object={selectedObject}
            mode={transformMode}
            onObjectChange={handleGizmoChange}
          />
        )}

        {/* See issue #1201: Use URDF model when available, fallback to capsule geometry */}
        <group
          onClick={(e) => {
            e.stopPropagation();
            let current: THREE.Object3D | null = e.object;
            while (current && current.parent) {
              if (current.name) {
                setSelectedObject(current);
                setSelectedBodyName(current.name);
                break;
              }
              current = current.parent;
            }
          }}
        >
          {useURDF ? (
            <URDFViewer
              model={urdfModel}
              jointAngles={urdfJointAngles}
              showAxes={showJointAxes}
              selectedBodyName={selectedBodyName}
              onSelectBody={(name) => {
                setSelectedBodyName(name);
                // The parent group click event will capture the object ref
              }}
              rootRef={rootRef}
            />
          ) : (
            <GolferModel
              frame={frame}
              rootRef={rootRef}
              selectedBodyName={selectedBodyName}
              onSelectBody={(name) => {
                setSelectedBodyName(name);
              }}
            />
          )}
        </group>

        <ClubTrajectory frames={frames} />

        {/* See issue #1179, #1199: Force/torque overlays */}
        <ForceOverlayComponent vectors={mappedVectors} />

        <axesHelper args={[1]} />
        <Environment preset="studio" />
      </Canvas>

      {/* Floating Interactive Controls Panel */}
      <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 flex items-center gap-2 bg-black/75 backdrop-blur-md px-4 py-2 rounded-full border border-white/10 shadow-xl z-20">
        {/* Camera Views */}
        <div className="flex items-center gap-1 border-r border-white/10 pr-2">
          <span className="text-[10px] text-gray-500 uppercase font-bold tracking-wider mr-1">View</span>
          <button
            onClick={() => handleCameraPreset('front')}
            className="px-2 py-1 text-xs text-gray-300 hover:text-white rounded hover:bg-white/10 transition-colors"
          >
            Front
          </button>
          <button
            onClick={() => handleCameraPreset('side')}
            className="px-2 py-1 text-xs text-gray-300 hover:text-white rounded hover:bg-white/10 transition-colors"
          >
            Side
          </button>
          <button
            onClick={() => handleCameraPreset('top')}
            className="px-2 py-1 text-xs text-gray-300 hover:text-white rounded hover:bg-white/10 transition-colors"
          >
            Top
          </button>
          <button
            onClick={() => setFollowMode(!followMode)}
            className={`px-2 py-1 text-xs rounded transition-colors ${
              followMode ? 'bg-blue-600 text-white font-semibold' : 'text-gray-300 hover:text-white hover:bg-white/10'
            }`}
          >
            Follow
          </button>
        </div>

        {/* Transform Gizmo Controls */}
        <div className="flex items-center gap-1">
          <span className="text-[10px] text-gray-500 uppercase font-bold tracking-wider mr-1">Gizmo</span>
          <button
            onClick={() => setTransformMode('translate')}
            className={`px-2 py-1 text-xs rounded transition-colors ${
              transformMode === 'translate' ? 'bg-blue-600 text-white font-semibold' : 'text-gray-300 hover:text-white hover:bg-white/10'
            }`}
          >
            Translate
          </button>
          <button
            onClick={() => setTransformMode('rotate')}
            className={`px-2 py-1 text-xs rounded transition-colors ${
              transformMode === 'rotate' ? 'bg-blue-600 text-white font-semibold' : 'text-gray-300 hover:text-white hover:bg-white/10'
            }`}
          >
            Rotate
          </button>
          {selectedObject && (
            <button
              onClick={() => {
                setSelectedObject(null);
                setSelectedBodyName(null);
              }}
              className="px-2 py-1 text-xs text-red-400 hover:text-red-300 rounded hover:bg-white/10 transition-colors ml-1"
            >
              Clear
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
