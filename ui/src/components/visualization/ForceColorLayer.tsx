import { useEffect, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import type * as THREE from 'three';
import { forceColor } from './forceColors';
import type { ForceColorScale } from './forceColors';
import { bindSegmentMeshes, SegmentMaterialColors } from './segmentMaterialColors';

interface Props {
  rootRef: React.RefObject<THREE.Group | null>;
  segmentIds: readonly string[];
  forces: Readonly<Record<string, number | null>>;
  scale: ForceColorScale;
}

/** Shared scene attachment for primitive or asynchronously loaded mesh models. */
export function ForceColorLayer({ rootRef, segmentIds, forces, scale }: Props) {
  const state = useRef<{ adapter: SegmentMaterialColors; bindings: Record<string, THREE.Mesh[]> } | null>(null);
  useEffect(() => () => {
    state.current?.adapter.dispose();
    state.current = null;
  }, []);
  useFrame(() => {
    const root = rootRef.current;
    if (!scale.enabled || !root || typeof root.traverse !== 'function') {
      state.current?.adapter.dispose();
      state.current = null;
      return;
    }
    const bindings = bindSegmentMeshes(root, segmentIds);
    const previous = state.current;
    const changed = !previous || Object.keys(bindings).length !== Object.keys(previous.bindings).length
      || Object.entries(bindings).some(([id, meshes]) => {
        const old = previous.bindings[id];
        return !old || old.length !== meshes.length || meshes.some((mesh, index) => mesh !== old[index]);
      });
    if (changed) {
      previous?.adapter.dispose();
      state.current = { bindings, adapter: new SegmentMaterialColors(bindings) };
    }
    const suppliedForces = new Map(Object.entries(forces));
    const colors = Object.fromEntries(segmentIds.map(id => [id, forceColor(suppliedForces.get(id), '', scale) || null]));
    state.current?.adapter.apply(colors);
  });
  return null;
}
