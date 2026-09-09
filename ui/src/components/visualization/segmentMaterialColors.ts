import * as THREE from 'three';

/** Match by declared model IDs, never by guessing which links have force data. */
export function bindSegmentMeshes(root: THREE.Object3D, segmentIds: readonly string[]): Record<string, THREE.Mesh[]> {
  const ids = new Set(segmentIds);
  const result: Record<string, THREE.Mesh[]> = Object.create(null);
  root.traverse(object => {
    if (!(object instanceof THREE.Mesh)) return;
    let owner: THREE.Object3D | null = object;
    while (owner) {
      if (ids.has(owner.name)) {
        (result[owner.name] ??= []).push(object);
        break;
      }
      if (owner === root) break;
      owner = owner.parent;
    }
  });
  return result;
}

interface Entry {
  mesh: THREE.Mesh;
  original: THREE.Material | THREE.Material[];
  override: THREE.Material | THREE.Material[] | null;
}

/** Thin Three.js adapter: hosts bind stable segment IDs to their own meshes.
 * Clones materials once while active; never mutates cached glTF asset materials.
 */
export class SegmentMaterialColors {
  private readonly entries = new Map<string, Entry[]>();

  constructor(segments: Readonly<Record<string, readonly THREE.Mesh[]>>) {
    const seen = new Set<THREE.Mesh>();
    for (const [segment, meshes] of Object.entries(segments)) {
      if (!segment) throw new TypeError('Segment identifiers must be nonempty');
      this.entries.set(segment, meshes.map(mesh => {
        if (!(mesh instanceof THREE.Mesh) || seen.has(mesh)) {
          throw new TypeError('Each mesh must belong to exactly one segment');
        }
        seen.add(mesh);
        return { mesh, original: mesh.material, override: null };
      }));
    }
  }

  apply(colors: Readonly<Record<string, string | null>>): void {
    for (const color of Object.values(colors)) {
      if (color !== null && !/^#[0-9a-fA-F]{6}$/.test(color)) {
        throw new TypeError('Overrides must be opaque #RRGGBB colors or null');
      }
    }
    for (const [segment, entries] of this.entries) {
      const color = Object.hasOwn(colors, segment) ? colors[segment] : null;
      for (const entry of entries) {
        if (!color) {
          this.restore(entry);
          continue;
        }
        if (!entry.override) {
          entry.override = Array.isArray(entry.original)
            ? entry.original.map(material => material.clone()) : entry.original.clone();
        }
        const materials = Array.isArray(entry.override) ? entry.override : [entry.override];
        for (const material of materials) {
          if ('color' in material && material.color instanceof THREE.Color) {
            material.color.set(color);
          }
        }
        entry.mesh.material = entry.override;
      }
    }
  }

  /** Restore and dispose only owned clones; repeated cleanup is safe. */
  dispose(): void {
    for (const entries of this.entries.values()) {
      for (const entry of entries) this.restore(entry);
    }
  }

  private restore(entry: Entry): void {
    if (!entry.override) return;
    entry.mesh.material = entry.original;
    const materials = Array.isArray(entry.override) ? entry.override : [entry.override];
    materials.forEach(material => material.dispose());
    entry.override = null;
  }
}
