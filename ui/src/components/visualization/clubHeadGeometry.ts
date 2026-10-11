/**
 * Club-head geometry for the web golfer model (GCV-11, issue #11717).
 *
 * Geometry source: the committed STL heads under `assets/club_heads/` (with
 * `provenance.json`), the same files the desktop adapter
 * `src/shared/python/model_appearance/club_head_mesh.py` falls back to. The
 * STLs were generated once by the Tools parametric builder, which cannot run in
 * a browser, and re-deriving the shape in TypeScript would only duplicate a
 * shape the repository already ships (and could drift from it). This module
 * mirrors the adapter's placement maths instead, with the same constants.
 *
 * Frames. The STL *head frame* is x toward the target (face normal at zero
 * loft), y up, z toward the toe. The *web club frame* produced here has its
 * origin at the sole point on the shaft axis, +y along the shaft toward the
 * grip, +x the face direction and +z = x cross y. It is the desktop club-body
 * frame (shaft toward the grip along -y) turned 180 degrees about x, which is
 * the orientation the scene's club group already uses (grip up, head down).
 * Pure data only; no three.js import so the maths is testable anywhere.
 */

export type Vec3 = [number, number, number];

export const HOSEL_RADIUS_M = 0.0072;
export const HOSEL_ABOVE_CROWN_M: Record<string, number> = {
  Driver: 0.012,
  Wood: 0.014,
  Hybrid: 0.022,
};
export const HOSEL_ABOVE_CROWN_DEFAULT_M = 0.03; // irons and wedges
const HEEL_INSET_FRACTION = 0.12;

const IRONS: Record<number, string> = {
  3: '3-Iron',
  5: '5-Iron',
  7: '7-Iron',
  9: '9-Iron',
};
const WEDGES: Record<number, string> = {
  46: 'Pitching Wedge',
  52: 'Gap Wedge',
  56: 'Sand Wedge',
  60: 'Lob Wedge',
};
const WOODS: Record<number, string> = { 3: '3-Wood', 5: '5-Wood' };

/** Mirrors `library_name_for` in club_head_mesh.py. Throws on bad input. */
export function libraryNameFor(name: string): string {
  if (typeof name !== 'string') throw new TypeError('club name must be a string');
  const key = name.trim().toLowerCase();
  if (!key) throw new Error('club name must be non-empty');
  let kind: string;
  let num: number | null;
  const iron = /^(\d{1,2})[\s_-]*iron$/.exec(key);
  if (iron) {
    kind = 'iron';
    num = Number(iron[1]);
  } else {
    const m = /^(driver|fairway|wood|hybrid|iron|wedge)[\s_\-(]*(\d{1,2})?\)?$/.exec(key);
    if (!m) throw new Error(`unsupported club name ${JSON.stringify(name)}`);
    kind = m[1];
    num = m[2] === undefined ? null : Number(m[2]);
  }
  if (kind === 'driver') return 'Driver 10.5°';
  if (kind === 'hybrid') return '3-Hybrid';
  if (kind === 'fairway' || kind === 'wood') return lookup(WOODS, num ?? 3, 'fairway wood', name);
  if (kind === 'iron') return lookup(IRONS, num ?? 7, 'iron', name);
  return lookup(WEDGES, num ?? 56, 'wedge loft', name);
}

function nearest(table: Record<number, string>, value: number): number {
  const keys = Object.keys(table).map(Number);
  return keys.reduce((a, b) => {
    const da = Math.abs(a - value);
    const db = Math.abs(b - value);
    return db < da || (db === da && b < a) ? b : a;
  });
}

function lookup(table: Record<number, string>, value: number, label: string, raw: string): string {
  if (table[value]) return table[value];
  if (label === 'iron' && value >= 3 && value <= 9) return table[nearest(table, value)];
  if (label === 'wedge loft') {
    const near = nearest(table, value);
    if (Math.abs(near - value) <= 2) return table[near];
  }
  throw new Error(`unsupported ${label} ${value} in club name ${JSON.stringify(raw)}`);
}

/** Triangles as a flat `[x0,y0,z0, x1,y1,z1, x2,y2,z2, ...]` array in metres. */
export function parseBinaryStlMetres(buffer: ArrayBuffer): Float32Array {
  if (buffer.byteLength < 84) throw new Error('STL buffer is too short');
  const view = new DataView(buffer);
  const count = view.getUint32(80, true);
  if (buffer.byteLength !== 84 + count * 50) {
    throw new Error(`buffer is not a binary STL of ${count} triangles`);
  }
  const out = new Float32Array(count * 9);
  for (let t = 0; t < count; t++) {
    const base = 84 + t * 50 + 12; // skip the stored normal
    for (let k = 0; k < 9; k++) out[t * 9 + k] = view.getFloat32(base + 4 * k, true) * 1e-3;
  }
  return out;
}

function signedVolume(tri: Float32Array): number {
  let vol = 0;
  for (let i = 0; i < tri.length; i += 9) {
    const [ax, ay, az, bx, by, bz, cx, cy, cz] = Array.from(tri.subarray(i, i + 9));
    vol += ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx);
  }
  return vol / 6;
}

/** Reverse the winding of every triangle (swap vertices b and c). */
function flipped(tri: Float32Array): Float32Array {
  const out = new Float32Array(tri.length);
  for (let i = 0; i < tri.length; i += 9) {
    out.set(tri.subarray(i, i + 3), i);
    out.set(tri.subarray(i + 6, i + 9), i + 3);
    out.set(tri.subarray(i + 3, i + 6), i + 6);
  }
  return out;
}

/** Sole-to-grip unit vector in the head frame (leans toward the heel). */
export function shaftDirectionHead(lieDeg: number): Vec3 {
  const tau = ((90 - lieDeg) * Math.PI) / 180;
  return [0, Math.cos(tau), -Math.sin(tau)];
}

function bounds(tri: Float32Array, axis: number): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = axis; i < tri.length; i += 3) {
    lo = Math.min(lo, tri[i]);
    hi = Math.max(hi, tri[i]);
  }
  return [lo, hi];
}

/** Where the shaft axis meets the sole plane (mirrors `_sole_point`). */
export function solePoint(tri: Float32Array): Vec3 {
  const [ymin, ymax] = bounds(tri, 1);
  const [zmin, zmax] = bounds(tri, 2);
  const half = 0.5 * (ymax - ymin);
  let faceX = -Infinity;
  for (let i = 0; i < tri.length; i += 3) {
    if (Math.abs(tri[i + 2]) < 0.003 && Math.abs(tri[i + 1] - ymin - half) < 0.003) {
      faceX = Math.max(faceX, tri[i]);
    }
  }
  if (!Number.isFinite(faceX)) throw new Error('head has no face-centre vertices');
  return [faceX, ymin, zmin + HEEL_INSET_FRACTION * (zmax - zmin)];
}

/** Area-weighted face-patch normal in the head frame (mirrors `_face_patch`). */
export function measuredFaceNormal(tri: Float32Array): Vec3 {
  const [xmin, xmax] = bounds(tri, 0);
  const [ymin, ymax] = bounds(tri, 1);
  const xMid = 0.5 * (xmin + xmax);
  const yMid = 0.5 * (ymin + ymax);
  const sum: Vec3 = [0, 0, 0];
  for (let i = 0; i < tri.length; i += 9) {
    const a = Array.from(tri.subarray(i, i + 9));
    const u = [a[3] - a[0], a[4] - a[1], a[5] - a[2]];
    const v = [a[6] - a[0], a[7] - a[1], a[8] - a[2]];
    const c = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
    const len = Math.hypot(c[0], c[1], c[2]);
    if (len < 1e-18) continue;
    const cx = (a[0] + a[3] + a[6]) / 3;
    const cy = (a[1] + a[4] + a[7]) / 3;
    const cz = (a[2] + a[5] + a[8]) / 3;
    if (c[0] / len > 0.3 && cx > xMid && Math.hypot(cy - yMid, cz) < 0.015) {
      for (let k = 0; k < 3; k++) sum[k] += c[k] * 0.5; // |c|/2 = area, c/|c| * area
    }
  }
  const norm = Math.hypot(sum[0], sum[1], sum[2]);
  if (norm === 0) throw new Error('mesh has no face patch toward +x');
  return [sum[0] / norm, sum[1] / norm, sum[2] / norm];
}

/** Rotation rows taking head-frame vectors to the web club frame. */
export function headToWebClubRows(lieDeg: number): [Vec3, Vec3, Vec3] {
  const tau = ((90 - lieDeg) * Math.PI) / 180;
  // Rows are (x_h, s_h, x_h cross s_h): R maps those basis vectors to x, y, z.
  return [
    [1, 0, 0],
    [0, Math.cos(tau), -Math.sin(tau)],
    [0, Math.sin(tau), Math.cos(tau)],
  ];
}

function apply(rows: [Vec3, Vec3, Vec3], v: Vec3): Vec3 {
  return rows.map((r) => r[0] * v[0] + r[1] * v[1] + r[2] * v[2]) as Vec3;
}

/** Unit face normal (loft included) in the web club frame. */
export function webFaceNormal(loftDeg: number, lieDeg: number): Vec3 {
  const loft = (loftDeg * Math.PI) / 180;
  return apply(headToWebClubRows(lieDeg), [Math.cos(loft), Math.sin(loft), 0]);
}

/** Unit vector pointing up in the world, expressed in the web club frame. */
export function worldUpInClubFrame(lieDeg: number): Vec3 {
  return apply(headToWebClubRows(lieDeg), [0, 1, 0]);
}

export interface ClubHeadSpec {
  libraryName: string;
  loftDeg: number;
  lieDeg: number;
  clubType: string;
}

export interface ClubHeadData {
  spec: ClubHeadSpec;
  /** Head triangles, flat xyz, metres, outward winding, web club frame. */
  positions: Float32Array;
  /** Measured face normal (loft included), web club frame. */
  faceNormal: Vec3;
  /** Hosel tube along +y: bottom and top distances from the origin. */
  hosel: { bottomM: number; topM: number; radiusM: number };
}

/**
 * Build the placed head from STL bytes and the spec's loft/lie.
 *
 * Preconditions: a binary STL in millimetres and finite loft/lie angles.
 * Postconditions: non-empty triangles with outward winding, the shaft axis
 * through the origin along +y, and `faceNormal` measured from the mesh.
 */
export function buildClubHeadData(buffer: ArrayBuffer, spec: ClubHeadSpec): ClubHeadData {
  if (!Number.isFinite(spec.loftDeg) || !Number.isFinite(spec.lieDeg)) {
    throw new Error('loft and lie must be finite');
  }
  let tri = parseBinaryStlMetres(buffer);
  if (tri.length === 0) throw new Error('STL has no triangles');
  if (signedVolume(tri) < 0) tri = flipped(tri);
  const sole = solePoint(tri);
  const direction = shaftDirectionHead(spec.lieDeg);
  const rows = headToWebClubRows(spec.lieDeg);
  const out = new Float32Array(tri.length);
  for (let i = 0; i < tri.length; i += 3) {
    const p = apply(rows, [tri[i] - sole[0], tri[i + 1] - sole[1], tri[i + 2] - sole[2]]);
    out.set(p, i);
  }
  const [, ymax] = bounds(tri, 1);
  const above = HOSEL_ABOVE_CROWN_M[spec.clubType] ?? HOSEL_ABOVE_CROWN_DEFAULT_M;
  const topM = (ymax - sole[1] + above) / direction[1];
  return {
    spec,
    positions: out,
    faceNormal: apply(rows, measuredFaceNormal(tri)),
    hosel: { bottomM: 0.35 * topM, topM, radiusM: HOSEL_RADIUS_M },
  };
}
