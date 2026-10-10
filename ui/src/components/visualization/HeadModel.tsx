import { useMemo } from 'react';
import {
  DEFAULT_HEAD_LENGTH_M,
  HEAD_COLORS,
  buildHeadParts,
  domeMesh,
  skullMesh,
  type GridMesh,
  type HeadPartSpec,
  type Headwear,
} from './headModelGeometry';

interface HeadModelProps {
  /** Head length (cervicale to vertex) in metres. */
  lengthM?: number;
  headwear?: Headwear;
  selected?: boolean;
  onClick?: (e: { stopPropagation: () => void }) => void;
  name?: string;
}

function GridGeometry({ mesh }: { mesh: GridMesh }) {
  return (
    <bufferGeometry onUpdate={(g) => g.computeVertexNormals()}>
      <bufferAttribute attach="attributes-position" args={[mesh.positions, 3]} />
      <bufferAttribute attach="index" args={[mesh.indices, 1]} />
    </bufferGeometry>
  );
}

function PartMaterial({ part, selected }: { part: HeadPartSpec; selected: boolean }) {
  const hit = selected && part.material === 'skin';
  return (
    <meshStandardMaterial
      color={hit ? '#ffcc00' : HEAD_COLORS[part.material]}
      emissive={hit ? '#332200' : '#000000'}
      roughness={part.material === 'eye_white' || part.material === 'iris_dark' ? 0.2 : 0.7}
    />
  );
}

/**
 * Visible head with a readable face (issue #11718): skull, eyes, brows, nose,
 * mouth, ears, neck and hair or cap. Mirrors `model_appearance/head.py`; the
 * origin is the neck point and the face looks along +x.
 */
export function HeadModel({
  lengthM = DEFAULT_HEAD_LENGTH_M,
  headwear = 'hair',
  selected = false,
  onClick,
  name = 'head',
}: HeadModelProps) {
  const parts = useMemo(() => buildHeadParts(lengthM, headwear), [lengthM, headwear]);
  const grids = useMemo(() => {
    const out = new Map<string, GridMesh>();
    out.set('skull', skullMesh(lengthM));
    for (const p of parts) {
      if (p.kind === 'dome' && p.dome) {
        out.set(p.name, domeMesh(lengthM, p.dome.front, p.dome.back, p.dome.scale));
      }
    }
    return out;
  }, [lengthM, parts]);

  // Canonical frame is x forward, y left, z up; the scene is y up.
  return (
    <group name={name} onClick={onClick} rotation={[-Math.PI / 2, 0, 0]}>
      {parts.map((part) => {
        const material = <PartMaterial part={part} selected={selected} />;
        if (part.kind === 'skull' || part.kind === 'dome') {
          return (
            <mesh key={part.name} name={`${name}_${part.name}`}>
              <GridGeometry mesh={grids.get(part.name) as GridMesh} />
              {material}
            </mesh>
          );
        }
        if (part.kind === 'cylinder') {
          return (
            <mesh
              key={part.name}
              name={`${name}_${part.name}`}
              position={part.centre}
              rotation={[Math.PI / 2, 0, 0]}
              scale={[part.half[0], 2 * part.half[2], part.half[1]]}
            >
              <cylinderGeometry args={[1, 1, 1, 24]} />
              {material}
            </mesh>
          );
        }
        return (
          <mesh
            key={part.name}
            name={`${name}_${part.name}`}
            position={part.centre}
            rotation={[0, part.tiltY, 0]}
            scale={part.half}
          >
            <sphereGeometry args={[1, 18, 10]} />
            {material}
          </mesh>
        );
      })}
    </group>
  );
}
