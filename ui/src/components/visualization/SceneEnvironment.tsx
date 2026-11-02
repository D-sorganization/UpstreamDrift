import { Suspense } from 'react';
import { Environment } from '@react-three/drei';
import ErrorBoundary from '@/components/ui/ErrorBoundary';

const STUDIO_HDR = '/hdr/studio_small_03_1k.hdr';

function FallbackLighting() {
  return (
    <>
      <ambientLight intensity={0.65} name="hdr-fallback-ambient" />
      <directionalLight
        position={[5, 10, 5]}
        intensity={1.1}
        name="hdr-fallback-key"
      />
    </>
  );
}

export function SceneEnvironment() {
  return (
    <ErrorBoundary fallback={<FallbackLighting />}>
      <Suspense fallback={null}>
        <Environment files={STUDIO_HDR} />
      </Suspense>
    </ErrorBoundary>
  );
}
