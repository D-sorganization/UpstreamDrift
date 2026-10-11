/**
 * Visible golfer head for the web model (GCV-12, issue #11718).
 *
 * A TypeScript mirror of `src/shared/python/model_appearance/head.py` with the
 * same constants and proportions, so the browser and every engine render draw
 * one shape: a deformed-ellipsoid skull with a tapering jaw, eyes with irises,
 * brows, a nose, a mouth, ears, a neck and optional hair or cap with a visor.
 *
 * Canonical frame (as in head.py): origin at the cervicale, x forward, y left,
 * z up. {@link canonicalToScene} maps it to the three.js scene (y up, forward
 * +x, the golfer's left on -z). Visual only: no mass, inertia or collision.
 */

export type Vec3 = [number, number, number];

export const DEFAULT_HEAD_LENGTH_M = 0.2429; // de Leva (1996) male head length
export const SKULL_CENTRE_Z = 0.58; // fractions of head length
export const SKULL_HALF: Vec3 = [0.4, 0.31, 0.42];
export const SKULL_RINGS = 26;
export const SKULL_SIDES = 40;

export type Headwear = 'none' | 'hair' | 'cap';

export type HeadPartKind = 'skull' | 'dome' | 'ellipsoid' | 'cylinder';

export interface HeadPartSpec {
  name: string;
  kind: HeadPartKind;
  /** Library material role, see {@link HEAD_COLORS}. */
  material: string;
  /** Canonical-frame centre (ellipsoid and cylinder parts). */
  centre: Vec3;
  /** Half sizes along canonical x, y, z (cylinder: radius, radius, half height). */
  half: Vec3;
  /** Rotation about canonical y in radians (positive turns z toward x). */
  tiltY: number;
  /** Dome cut angles (radians) and scale, dome parts only. */
  dome?: { front: number; back: number; scale: number };
}

/** sRGB values of the matching `model_appearance/library.py` materials. */
export const HEAD_COLORS: Record<string, string> = {
  skin: '#cc9975', // skin_medium
  eye_white: '#f5f5f0',
  iris_dark: '#1a120d',
  brow_dark: '#240f06',
  lip_pink: '#9e4d4d',
  hair: '#332114', // hair_brown
  cap: '#1f3373', // cap_navy
};

export function validateHeadLength(lengthM: number): void {
  if (!Number.isFinite(lengthM) || lengthM <= 0) {
    throw new Error('Head length must be a positive finite number');
  }
}

/** Point on the deformed skull for a unit direction (canonical frame). */
export function skullPoint(direction: Vec3, length: number): Vec3 {
  const [hx, hy, hz] = SKULL_HALF.map((f) => f * length);
  const [dx, dy, dz] = direction;
  const low = Math.max(0, -dz); // 0 at/above the equator, 1 at the chin
  const width = 1 - 0.3 * low ** 1.4; // jaw narrows toward the chin
  const depth = 1 + (dx < 0 ? 0.06 : 0) * (1 - low); // rounder back of skull
  const chin = 0.1 * hx * low ** 2 * Math.max(0, dx); // chin carried forward
  return [hx * dx * depth + chin, hy * dy * width, SKULL_CENTRE_Z * length + hz * dz];
}

/** Unit direction for polar angle `theta` from +z and azimuth `phi` from +x. */
export function unitDirection(theta: number, phi: number): Vec3 {
  return [Math.sin(theta) * Math.cos(phi), Math.sin(theta) * Math.sin(phi), Math.cos(theta)];
}

/** Skull-surface point at a facial direction, pulled in by `inset`. */
export function facePoint(length: number, azimuth: number, elevation: number, inset: number): Vec3 {
  const p = skullPoint(
    [
      Math.cos(elevation) * Math.cos(azimuth),
      Math.cos(elevation) * Math.sin(azimuth),
      Math.sin(elevation),
    ],
    length,
  );
  const cz = SKULL_CENTRE_Z * length;
  return [p[0] * (1 - inset), p[1] * (1 - inset), cz + (p[2] - cz) * (1 - inset)];
}

function ellipsoid(
  name: string,
  material: string,
  centre: Vec3,
  half: Vec3,
  tiltY = 0,
): HeadPartSpec {
  return { name, kind: 'ellipsoid', material, centre, half, tiltY };
}

function facePartSpecs(L: number): HeadPartSpec[] {
  const parts: HeadPartSpec[] = [];
  for (const [side, sign] of [['l', 1], ['r', -1]] as const) {
    const eye = facePoint(L, sign * 0.4, 0.14, 0.04);
    parts.push(ellipsoid(`eye_${side}`, 'eye_white', eye, [0.05 * L, 0.062 * L, 0.05 * L]));
    const iris: Vec3 = [eye[0] + 0.04 * L, eye[1], eye[2]];
    parts.push(ellipsoid(`iris_${side}`, 'iris_dark', iris, [0.02 * L, 0.03 * L, 0.03 * L]));
    const brow = facePoint(L, sign * 0.4, 0.36, 0);
    parts.push(ellipsoid(`brow_${side}`, 'brow_dark', brow, [0.026 * L, 0.085 * L, 0.016 * L]));
  }
  const nose = facePoint(L, 0, -0.1, -0.12);
  parts.push(
    ellipsoid('nose', 'skin', nose, [0.075 * L, 0.045 * L, 0.115 * L], Math.atan(0.3)),
  );
  const mouth = facePoint(L, 0, -0.46, -0.01);
  parts.push(ellipsoid('mouth', 'lip_pink', mouth, [0.014 * L, 0.095 * L, 0.016 * L]));
  return parts;
}

function earSpecs(L: number): HeadPartSpec[] {
  return (
    [
      ['l', 1],
      ['r', -1],
    ] as const
  ).map(([side, sign]) =>
    ellipsoid(
      `ear_${side}`,
      'skin',
      [-0.02 * L, sign * 0.285 * L, (SKULL_CENTRE_Z - 0.03) * L],
      [0.07 * L, 0.03 * L, 0.12 * L],
    ),
  );
}

function neckSpec(L: number): HeadPartSpec {
  const radius = 0.21 * L;
  const bottom = 0.5 * radius - 0.004;
  const top = 0.4 * L;
  return {
    name: 'neck',
    kind: 'cylinder',
    material: 'skin',
    centre: [0, 0, 0.5 * (bottom + top)],
    half: [radius, radius * 1.08, 0.5 * (top - bottom)],
    tiltY: 0,
  };
}

function headwearSpecs(L: number, kind: Exclude<Headwear, 'none'>): HeadPartSpec[] {
  const dome = (name: string, front: number, back: number, scale: number): HeadPartSpec => ({
    name,
    kind: 'dome',
    material: kind,
    centre: [0, 0, SKULL_CENTRE_Z * L],
    half: [1, 1, 1],
    tiltY: 0,
    dome: { front, back, scale },
  });
  if (kind === 'hair') return [dome('hair', 0.95, 1.9, 1.05)];
  return [
    dome('cap', 1.0, 1.45, 1.06),
    ellipsoid(
      'visor',
      'cap',
      [0.43 * L, 0, (SKULL_CENTRE_Z + 0.16) * L],
      [0.2 * L, 0.26 * L, 0.018 * L],
    ),
  ];
}

/**
 * Head, face, ears, neck and optional hair or cap in the canonical frame.
 *
 * Preconditions: positive finite length. Postconditions: left and right parts
 * mirror about the mid-sagittal plane (y = 0), the skull vertex is `length`
 * above the neck point and the face features sit forward (+x) of the skull
 * centre.
 */
export function buildHeadParts(
  lengthM: number = DEFAULT_HEAD_LENGTH_M,
  headwear: Headwear = 'hair',
): HeadPartSpec[] {
  validateHeadLength(lengthM);
  const skull: HeadPartSpec = {
    name: 'skull',
    kind: 'skull',
    material: 'skin',
    centre: [0, 0, 0],
    half: [1, 1, 1],
    tiltY: 0,
  };
  const parts = [neckSpec(lengthM), skull, ...facePartSpecs(lengthM), ...earSpecs(lengthM)];
  if (headwear !== 'none') parts.push(...headwearSpecs(lengthM, headwear));
  return parts;
}

/** Map a canonical-frame point (x fwd, y left, z up) to the scene (y up). */
export function canonicalToScene(p: Vec3): Vec3 {
  return [p[0], p[2], 0 - p[1]]; // 0 - y avoids a negative zero
}

export interface GridMesh {
  positions: Float32Array;
  indices: Uint32Array;
}

/**
 * Closed polar grid mesh: apex, `rows` rings of `sides` points, bottom centre.
 * `point(theta, phi)` evaluates the surface; theta runs 0..1 over the rings in
 * units of the caller's choice. Winding is outward for surfaces that are
 * star-shaped about the +z axis.
 */
export function polarGridMesh(
  apex: Vec3,
  bottom: Vec3,
  rings: Vec3[][],
): GridMesh {
  const sides = rings[0].length;
  const pts: Vec3[] = [apex, ...rings.flat(), bottom];
  const idx: number[] = [];
  const ring = (j: number, i: number) => 1 + j * sides + (i % sides);
  const last = pts.length - 1;
  for (let i = 0; i < sides; i++) {
    idx.push(0, ring(0, i), ring(0, i + 1));
    for (let j = 0; j < rings.length - 1; j++) {
      idx.push(ring(j, i), ring(j + 1, i), ring(j + 1, i + 1));
      idx.push(ring(j, i), ring(j + 1, i + 1), ring(j, i + 1));
    }
    const k = rings.length - 1;
    idx.push(last, ring(k, i + 1), ring(k, i));
  }
  return { positions: new Float32Array(pts.flat()), indices: new Uint32Array(idx) };
}

/** Skull surface mesh (canonical frame, closed, outward winding). */
export function skullMesh(length: number): GridMesh {
  validateHeadLength(length);
  const rings: Vec3[][] = [];
  for (let r = 1; r <= SKULL_RINGS; r++) {
    const theta = (r / (SKULL_RINGS + 1)) * Math.PI;
    const ring: Vec3[] = [];
    for (let s = 0; s < SKULL_SIDES; s++) {
      ring.push(skullPoint(unitDirection(theta, (2 * Math.PI * s) / SKULL_SIDES), length));
    }
    rings.push(ring);
  }
  return polarGridMesh(skullPoint([0, 0, 1], length), skullPoint([0, 0, -1], length), rings);
}

/** Hair or cap shell over the upper skull (canonical frame, closed). */
export function domeMesh(
  length: number,
  front: number,
  back: number,
  scale: number,
): GridMesh {
  validateHeadLength(length);
  const cz = SKULL_CENTRE_Z * length;
  const scaled = (p: Vec3): Vec3 => [p[0] * scale, p[1] * scale, cz + (p[2] - cz) * scale];
  const rings: Vec3[][] = [];
  for (let r = 1; r <= 15; r++) {
    const ring: Vec3[] = [];
    for (let s = 0; s < SKULL_SIDES; s++) {
      const phi = (2 * Math.PI * s) / SKULL_SIDES;
      const cut = front + ((back - front) * (1 - Math.cos(phi))) / 2;
      ring.push(scaled(skullPoint(unitDirection((r / 15) * cut, phi), length)));
    }
    rings.push(ring);
  }
  return polarGridMesh(scaled(skullPoint([0, 0, 1], length)), [0, 0, cz], rings);
}

/** Signed volume of an indexed triangle mesh (positive for outward winding). */
export function meshVolume(mesh: GridMesh): number {
  const p = mesh.positions;
  let vol = 0;
  for (let t = 0; t < mesh.indices.length; t += 3) {
    const [a, b, c] = [0, 1, 2].map((k) => mesh.indices[t + k] * 3);
    vol +=
      p[a] * (p[b + 1] * p[c + 2] - p[b + 2] * p[c + 1]) -
      p[a + 1] * (p[b] * p[c + 2] - p[b + 2] * p[c]) +
      p[a + 2] * (p[b] * p[c + 1] - p[b + 1] * p[c]);
  }
  return vol / 6;
}
