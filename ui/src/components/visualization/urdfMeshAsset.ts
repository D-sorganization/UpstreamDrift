/**
 * URDF mesh-asset helpers for the glTF loading path (issue #8406).
 *
 * Kept outside `URDFViewer.tsx` so that component file exports only
 * components (react-refresh constraint).
 */

import { getApiBase } from '@/api/backend';

/**
 * True when a URDF `mesh_path` points at a glTF asset we can load in the
 * browser (.glb / .gltf). Other mesh formats (STL, DAE, OBJ) keep the
 * primitive fallback.
 */
export function isGltfMeshPath(meshPath: string | null | undefined): boolean {
  if (!meshPath) return false;
  const clean = meshPath.split('?')[0].split('#')[0].toLowerCase();
  return clean.endsWith('.glb') || clean.endsWith('.gltf');
}

/**
 * Build the backend URL that serves a URDF-referenced mesh asset.
 *
 * Points at `GET /api/models/mesh-asset` (`src/api/routes/models.py`), which
 * resolves the relative path against the allowed model directories and
 * rejects anything escaping them.
 *
 * @param meshPath - Relative mesh path from the URDF (e.g. `meshes/club.glb`).
 * @returns Full URL for the mesh asset endpoint.
 * @throws Error when `meshPath` is empty.
 */
export function meshAssetUrl(meshPath: string): string {
  if (!meshPath || meshPath.trim().length === 0) {
    throw new Error('meshPath must be a non-empty string');
  }
  return `${getApiBase()}/api/models/mesh-asset?path=${encodeURIComponent(meshPath)}`;
}
