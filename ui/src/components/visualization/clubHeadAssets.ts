/**
 * Loads the committed club-head STLs (GCV-11, issue #11717).
 *
 * The files and their `provenance.json` live at the repository root under
 * `assets/club_heads/` and are shared with the desktop adapter; Vite emits them
 * as URL assets and fetches happen lazily, one head at a time.
 */

import provenance from '../../../../assets/club_heads/provenance.json';
import {
  buildClubHeadData,
  libraryNameFor,
  type ClubHeadData,
  type ClubHeadSpec,
} from './clubHeadGeometry';

interface ProvenanceEntry {
  path: string;
  loft_deg: number;
  lie_deg: number;
  club_type: string;
}

const STL_URLS = import.meta.glob(
  [
    '../../../../assets/club_heads/*.stl',
    '../../../../src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/exploratory_gs3dx/models/gs3dx_driver_head.stl',
  ],
  { query: '?url', import: 'default' },
) as Record<string, () => Promise<string>>;

const cache = new Map<string, Promise<ClubHeadData>>();

/** Spec and repo-relative STL path for a club alias such as `iron7`. */
export function clubHeadEntry(club: string): { spec: ClubHeadSpec; path: string } {
  const libraryName = libraryNameFor(club);
  const entry = (provenance.heads as Record<string, ProvenanceEntry>)[libraryName];
  if (!entry) throw new Error(`no committed head STL for ${libraryName}`);
  return {
    path: entry.path,
    spec: {
      libraryName,
      loftDeg: entry.loft_deg,
      lieDeg: entry.lie_deg,
      clubType: entry.club_type,
    },
  };
}

/** Fetch, parse and place the head for `club`; results are cached per club. */
export function loadClubHead(club: string): Promise<ClubHeadData> {
  const { spec, path } = clubHeadEntry(club);
  let pending = cache.get(spec.libraryName);
  if (!pending) {
    const key = Object.keys(STL_URLS).find((k) => k.endsWith(path));
    if (!key) return Promise.reject(new Error(`STL asset not bundled: ${path}`));
    pending = STL_URLS[key]()
      .then((url) => fetch(url))
      .then((res) => {
        if (!res.ok) throw new Error(`club head fetch failed: ${res.status}`);
        return res.arrayBuffer();
      })
      .then((buf) => buildClubHeadData(buf, spec));
    cache.set(spec.libraryName, pending);
    pending.catch(() => cache.delete(spec.libraryName));
  }
  return pending;
}
