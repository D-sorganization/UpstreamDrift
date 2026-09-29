import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid, Environment } from '@react-three/drei';
import { URDFViewer, URDFModel } from '@/components/visualization/URDFViewer';

interface ModelPreviewViewportProps {
  urdfModel: URDFModel | null;
  jointValues: Record<string, number>;
  activePreviewModel: string | null;
}

export function ModelPreviewViewport({
  urdfModel,
  jointValues,
  activePreviewModel,
}: ModelPreviewViewportProps) {
  return (
    <main className="flex-1 relative min-w-0 bg-gray-950">
      <Canvas camera={{ position: [3, 2, 3], fov: 50 }} className="w-full h-full">
        <ambientLight intensity={0.5} />
        <directionalLight position={[10, 10, 5]} intensity={1} />
        <OrbitControls enableDamping dampingFactor={0.05} />
        <Grid
          infiniteGrid
          cellSize={0.5}
          cellThickness={0.5}
          sectionSize={2}
          sectionThickness={1}
          fadeDistance={30}
        />
        {urdfModel && (
          <URDFViewer
            model={urdfModel}
            jointAngles={jointValues}
            showAxes={true}
          />
        )}
        <axesHelper args={[1]} />
        <Environment preset="studio" />
      </Canvas>

      {activePreviewModel && (
        <div className="absolute top-4 left-4 bg-gray-900/80 backdrop-blur-md px-3 py-1.5 rounded-lg border border-gray-850 shadow-xl">
          <span className="text-xs text-gray-200 font-mono">
            Preview: {activePreviewModel}
          </span>
        </div>
      )}
    </main>
  );
}
