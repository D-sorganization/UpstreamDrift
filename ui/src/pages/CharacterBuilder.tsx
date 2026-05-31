import { useState, useMemo, useCallback } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid, Environment } from '@react-three/drei';
import { getApiBase } from '@/api/backend';

interface SegmentBreakdown {
  name: string;
  lengthRatio: number;
  massRatio: number;
}

const SEGMENT_RATIOS: SegmentBreakdown[] = [
  { name: 'Head', lengthRatio: 0.1395, massRatio: 0.0694 },
  { name: 'Trunk', lengthRatio: 0.328, massRatio: 0.4346 },
  { name: 'Upper Arm', lengthRatio: 0.186, massRatio: 0.0271 },
  { name: 'Forearm', lengthRatio: 0.146, massRatio: 0.0162 },
  { name: 'Hand', lengthRatio: 0.108, massRatio: 0.0061 },
  { name: 'Thigh', lengthRatio: 0.245, massRatio: 0.1416 },
  { name: 'Shank', lengthRatio: 0.246, massRatio: 0.0433 },
  { name: 'Foot', lengthRatio: 0.152, massRatio: 0.0137 },
];

const BUILD_MULTIPLIERS: Record<
  string,
  { head: number; trunk: number; arms: number; legs: number }
> = {
  Athletic: { head: 1.0, trunk: 1.05, arms: 1.15, legs: 1.15 },
  Average: { head: 1.0, trunk: 1.0, arms: 1.0, legs: 1.0 },
  Heavy: { head: 1.1, trunk: 1.3, arms: 1.2, legs: 1.2 },
  Slim: { head: 0.95, trunk: 0.8, arms: 0.8, legs: 0.8 },
};

function CharacterPreview({
  height,
  weight,
  buildType,
}: {
  height: number;
  weight: number;
  buildType: string;
}) {
  const thicknessBase = Math.sqrt(weight / 80);
  const multiplier = BUILD_MULTIPLIERS[buildType] || BUILD_MULTIPLIERS.Average;

  const trunkRadius = 0.15 * thicknessBase * multiplier.trunk;
  const armRadius = 0.035 * thicknessBase * multiplier.arms;
  const legRadius = 0.05 * thicknessBase * multiplier.legs;

  const headLength = height * 0.1395;
  const trunkLength = height * 0.328;
  const upperArmLength = height * 0.186;
  const forearmLength = height * 0.146;
  const thighLength = height * 0.245;
  const shankLength = height * 0.246;

  const hipHeight = shankLength + thighLength;

  return (
    <group position={[0, -height / 2, 0]}>
      {/* Trunk */}
      <mesh position={[0, hipHeight + trunkLength / 2, 0]}>
        <cylinderGeometry args={[trunkRadius, trunkRadius * 0.9, trunkLength, 16]} />
        <meshStandardMaterial color="#4f46e5" roughness={0.4} metalness={0.1} />
      </mesh>

      {/* Head */}
      <mesh position={[0, hipHeight + trunkLength + headLength / 2, 0]}>
        <sphereGeometry args={[headLength / 2, 16, 16]} />
        <meshStandardMaterial color="#6366f1" roughness={0.4} metalness={0.1} />
      </mesh>

      {/* Left Thigh */}
      <mesh position={[-trunkRadius * 0.6, shankLength + thighLength / 2, 0]}>
        <cylinderGeometry args={[legRadius, legRadius * 0.9, thighLength, 16]} />
        <meshStandardMaterial color="#3b82f6" roughness={0.4} metalness={0.1} />
      </mesh>

      {/* Right Thigh */}
      <mesh position={[trunkRadius * 0.6, shankLength + thighLength / 2, 0]}>
        <cylinderGeometry args={[legRadius, legRadius * 0.9, thighLength, 16]} />
        <meshStandardMaterial color="#3b82f6" roughness={0.4} metalness={0.1} />
      </mesh>

      {/* Left Shank */}
      <mesh position={[-trunkRadius * 0.6, shankLength / 2, 0]}>
        <cylinderGeometry args={[legRadius * 0.9, legRadius * 0.8, shankLength, 16]} />
        <meshStandardMaterial color="#60a5fa" roughness={0.4} metalness={0.1} />
      </mesh>

      {/* Right Shank */}
      <mesh position={[trunkRadius * 0.6, shankLength / 2, 0]}>
        <cylinderGeometry args={[legRadius * 0.9, legRadius * 0.8, shankLength, 16]} />
        <meshStandardMaterial color="#60a5fa" roughness={0.4} metalness={0.1} />
      </mesh>

      {/* Left Upper Arm */}
      <mesh position={[-trunkRadius - armRadius, hipHeight + trunkLength - upperArmLength / 2, 0]}>
        <cylinderGeometry args={[armRadius, armRadius * 0.9, upperArmLength, 16]} />
        <meshStandardMaterial color="#a855f7" roughness={0.4} metalness={0.1} />
      </mesh>

      {/* Right Upper Arm */}
      <mesh position={[trunkRadius + armRadius, hipHeight + trunkLength - upperArmLength / 2, 0]}>
        <cylinderGeometry args={[armRadius, armRadius * 0.9, upperArmLength, 16]} />
        <meshStandardMaterial color="#a855f7" roughness={0.4} metalness={0.1} />
      </mesh>

      {/* Left Forearm */}
      <mesh position={[-trunkRadius - armRadius, hipHeight + trunkLength - upperArmLength - forearmLength / 2, 0]}>
        <cylinderGeometry args={[armRadius * 0.9, armRadius * 0.8, forearmLength, 16]} />
        <meshStandardMaterial color="#c084fc" roughness={0.4} metalness={0.1} />
      </mesh>

      {/* Right Forearm */}
      <mesh position={[trunkRadius + armRadius, hipHeight + trunkLength - upperArmLength - forearmLength / 2, 0]}>
        <cylinderGeometry args={[armRadius * 0.9, armRadius * 0.8, forearmLength, 16]} />
        <meshStandardMaterial color="#c084fc" roughness={0.4} metalness={0.1} />
      </mesh>
    </group>
  );
}

export function CharacterBuilderPage() {
  const [height, setHeight] = useState(1.8);
  const [weight, setWeight] = useState(80);
  const [buildType, setBuildType] = useState('Average');
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const breakdown = useMemo(() => {
    return SEGMENT_RATIOS.map((seg) => {
      const lengthVal = height * seg.lengthRatio;
      const massVal = weight * seg.massRatio;
      return {
        name: seg.name,
        length: `${(lengthVal * 100).toFixed(1)} cm`,
        mass: `${massVal.toFixed(2)} kg`,
      };
    });
  }, [height, weight]);

  const handleGenerate = useCallback(async () => {
    setGenerating(true);
    setError(null);
    try {
      // Response is URDF XML text (not JSON), so apiFetch is not suitable.
      // Build the URL via getApiBase() to stay Tauri-safe (#6897).
      const response = await fetch(`${getApiBase()}/api/character-builder/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          height_m: height,
          mass_kg: weight,
          build_type: buildType.toLowerCase(),
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to generate URDF: ${response.statusText}`);
      }

      const text = await response.text();
      const blob = new Blob([text], { type: 'text/xml' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${buildType.toLowerCase()}_humanoid.urdf`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
    } finally {
      setGenerating(false);
    }
  }, [height, weight, buildType]);

  return (
    <div className="flex h-screen bg-gray-900 overflow-hidden text-gray-200">
      {/* Left panel: controls and specs */}
      <aside className="w-96 bg-gray-800 border-r border-gray-700 flex flex-col flex-shrink-0 overflow-y-auto">
        <div className="p-6 border-b border-gray-700">
          <h1 className="text-xl font-bold text-white mb-1">Character Builder</h1>
          <p className="text-xs text-gray-400">
            Define humanoid properties and generate URDF models
          </p>
        </div>

        <div className="p-6 space-y-6">
          {/* Height Slider */}
          <div>
            <div className="flex justify-between items-center mb-1">
              <label htmlFor="height-slider" className="text-xs font-semibold text-gray-300">
                Height (m)
              </label>
              <span className="text-xs font-mono text-blue-400">{height.toFixed(2)} m</span>
            </div>
            <input
              id="height-slider"
              type="range"
              min={1.5}
              max={2.1}
              step={0.01}
              value={height}
              onChange={(e) => setHeight(parseFloat(e.target.value))}
              className="w-full h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer"
            />
          </div>

          {/* Weight Slider */}
          <div>
            <div className="flex justify-between items-center mb-1">
              <label htmlFor="weight-slider" className="text-xs font-semibold text-gray-300">
                Weight (kg)
              </label>
              <span className="text-xs font-mono text-blue-400">{weight} kg</span>
            </div>
            <input
              id="weight-slider"
              type="range"
              min={40}
              max={150}
              step={1}
              value={weight}
              onChange={(e) => setWeight(parseInt(e.target.value))}
              className="w-full h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer"
            />
          </div>

          {/* Build Type select */}
          <div>
            <label
              htmlFor="build-type-select"
              className="block text-xs font-semibold text-gray-300 mb-1"
            >
              Build Type
            </label>
            <select
              id="build-type-select"
              value={buildType}
              onChange={(e) => setBuildType(e.target.value)}
              className="w-full bg-gray-700 border-none text-gray-200 rounded px-2 py-1.5 text-sm focus:ring-1 focus:ring-blue-400"
            >
              <option value="Athletic">Athletic</option>
              <option value="Average">Average</option>
              <option value="Heavy">Heavy</option>
              <option value="Slim">Slim</option>
            </select>
          </div>

          {/* Segment breakdown */}
          <div className="border-t border-gray-700 pt-6">
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
              Segment Breakdown
            </h3>
            <div className="bg-gray-900/50 rounded-lg p-3 border border-gray-700/50">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="border-b border-gray-800 text-gray-500 font-semibold">
                    <th className="pb-1.5">Segment</th>
                    <th className="pb-1.5">Length</th>
                    <th className="pb-1.5">Mass</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-800/40">
                  {breakdown.map((row) => (
                    <tr key={row.name} className="text-gray-300">
                      <td className="py-1.5 font-medium">{row.name}</td>
                      <td className="py-1.5 font-mono">{row.length}</td>
                      <td className="py-1.5 font-mono">{row.mass}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Generate Trigger */}
        <div className="mt-auto p-6 border-t border-gray-700 bg-gray-800/50">
          {error && (
            <div className="mb-3 text-xs text-red-400 bg-red-950/30 p-2.5 rounded border border-red-900/50">
              {error}
            </div>
          )}
          <button
            onClick={handleGenerate}
            disabled={generating}
            className="w-full bg-blue-600 hover:bg-blue-500 disabled:bg-blue-800 text-white rounded py-2 text-sm font-semibold transition-colors shadow-lg shadow-blue-550/20"
          >
            {generating ? 'Generating...' : 'Generate URDF'}
          </button>
        </div>
      </aside>

      {/* Right panel: 3D Visualization */}
      <main className="flex-1 relative bg-gray-950">
        <Canvas camera={{ position: [0, 1.2, 2.5], fov: 45 }} className="w-full h-full">
          <ambientLight intensity={0.6} />
          <directionalLight position={[5, 10, 5]} intensity={1.2} castShadow />
          <directionalLight position={[-5, 5, -5]} intensity={0.4} />
          <OrbitControls
            enableDamping
            dampingFactor={0.05}
            target={[0, 0.9, 0]}
            maxPolarAngle={Math.PI / 2 + 0.1}
          />
          <Grid
            infiniteGrid
            cellSize={0.1}
            cellThickness={0.5}
            sectionSize={0.5}
            sectionThickness={1.0}
            fadeDistance={10}
            cellColor="#374151"
            sectionColor="#4b5563"
          />
          <CharacterPreview height={height} weight={weight} buildType={buildType} />
          <Environment preset="studio" />
        </Canvas>

        {/* Info Overlay */}
        <div className="absolute top-4 right-4 bg-gray-900/80 backdrop-blur border border-gray-700 p-3 rounded-lg text-xs space-y-1">
          <div className="text-gray-400 font-semibold mb-1">Preview Controls</div>
          <div>Left Click + Drag: Rotate</div>
          <div>Right Click + Drag: Pan</div>
          <div>Scroll: Zoom</div>
        </div>
      </main>
    </div>
  );
}
