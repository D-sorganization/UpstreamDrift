import { useEffect, useMemo, useState, type ReactNode } from 'react';
import * as THREE from 'three';
import { mergeVertices } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import type { ClubHeadData } from './clubHeadGeometry';
import { loadClubHead } from './clubHeadAssets';

interface ClubHeadProps {
  /** Club alias such as `driver`, `iron7` or `wedge56`. */
  club?: string;
  /** Material element shared by the head and hosel meshes. */
  material?: ReactNode;
  onClick?: (e: { stopPropagation: () => void }) => void;
  name?: string;
}

/** Smooth-shaded BufferGeometry for a placed head. */
function toGeometry(data: ClubHeadData): THREE.BufferGeometry {
  const raw = new THREE.BufferGeometry();
  raw.setAttribute('position', new THREE.BufferAttribute(data.positions, 3));
  const welded = mergeVertices(raw, 1e-6);
  welded.computeVertexNormals();
  raw.dispose();
  return welded;
}

/**
 * Realistic club head from the committed STL set (issue #11717).
 *
 * The origin is the sole point on the shaft axis; the shaft runs along +y and
 * the face looks along +x with the spec loft and lie. While the STL loads, or
 * if it cannot be loaded, the old box placeholder is drawn so a head is
 * always visible.
 */
export function ClubHead({ club = 'driver', material, onClick, name = 'club_head' }: ClubHeadProps) {
  const [loaded, setLoaded] = useState<{ club: string; data: ClubHeadData } | null>(null);

  useEffect(() => {
    let live = true;
    loadClubHead(club)
      .then((d) => live && setLoaded({ club, data: d }))
      .catch(() => undefined); // keep the placeholder when the STL is unavailable
    return () => {
      live = false;
    };
  }, [club]);

  const data = loaded && loaded.club === club ? loaded.data : null;

  const geometry = useMemo(() => (data ? toGeometry(data) : null), [data]);
  useEffect(() => () => geometry?.dispose(), [geometry]);

  if (!data || !geometry) {
    return (
      <mesh name={name} position={[0, 0.02, 0]} rotation={[0.3, 0, 0]} onClick={onClick}>
        <boxGeometry args={[0.1, 0.03, 0.08]} />
        {material}
      </mesh>
    );
  }

  const { bottomM, topM, radiusM } = data.hosel;
  return (
    <group name={name} onClick={onClick}>
      <mesh geometry={geometry} name={`${name}_body`}>
        {material}
      </mesh>
      <mesh position={[0, (bottomM + topM) / 2, 0]} name={`${name}_hosel`}>
        <cylinderGeometry args={[0.9 * radiusM, radiusM, topM - bottomM, 16]} />
        {material}
      </mesh>
    </group>
  );
}
