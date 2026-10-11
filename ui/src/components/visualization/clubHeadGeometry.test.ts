import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  buildClubHeadData,
  libraryNameFor,
  measuredFaceNormal,
  parseBinaryStlMetres,
  shaftDirectionHead,
  webFaceNormal,
  worldUpInClubFrame,
  type ClubHeadSpec,
} from './clubHeadGeometry';

const REPO_ROOT = resolve(__dirname, '../../../..');
const MANIFEST = JSON.parse(
  readFileSync(resolve(REPO_ROOT, 'assets/club_heads/provenance.json'), 'utf8'),
) as {
  heads: Record<
    string,
    { path: string; loft_deg: number; lie_deg: number; club_type: string; sha256: string }
  >;
};

function load(club: string) {
  const library = libraryNameFor(club);
  const entry = MANIFEST.heads[library];
  const bytes = readFileSync(resolve(REPO_ROOT, entry.path));
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const spec: ClubHeadSpec = {
    libraryName: library,
    loftDeg: entry.loft_deg,
    lieDeg: entry.lie_deg,
    clubType: entry.club_type,
  };
  return { buffer, spec, data: buildClubHeadData(buffer, spec) };
}

const deg = (rad: number) => (rad * 180) / Math.PI;
const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

describe('libraryNameFor', () => {
  it('mirrors the desktop alias table', () => {
    expect(libraryNameFor('driver')).toBe('Driver 10.5°');
    expect(libraryNameFor('iron7')).toBe('7-Iron');
    expect(libraryNameFor('7-iron')).toBe('7-Iron');
    expect(libraryNameFor('wedge56')).toBe('Sand Wedge');
    expect(libraryNameFor('hybrid')).toBe('3-Hybrid');
    expect(libraryNameFor('iron(6)')).toBe('5-Iron');
  });

  it('rejects bad input', () => {
    expect(() => libraryNameFor('')).toThrow();
    expect(() => libraryNameFor('putter')).toThrow();
    expect(() => libraryNameFor('wedge30')).toThrow();
    expect(() => libraryNameFor(7 as unknown as string)).toThrow(TypeError);
  });
});

describe.each(['driver', 'iron7'])('club head geometry: %s', (club) => {
  it('is non-empty with finite vertices', () => {
    const { data } = load(club);
    expect(data.positions.length).toBeGreaterThan(0);
    expect(data.positions.length % 9).toBe(0);
    expect(Array.from(data.positions).every(Number.isFinite)).toBe(true);
  });

  it('measured face normal is unit and points down the target line (+x)', () => {
    const { data } = load(club);
    expect(Math.hypot(...data.faceNormal)).toBeCloseTo(1, 6);
    expect(data.faceNormal[0]).toBeGreaterThan(0.5);
  });

  it('loft angle matches the club spec within 0.5 degrees', () => {
    const { buffer, spec, data } = load(club);
    // In the head frame (y up) the face normal elevation is the loft.
    const headNormal = measuredFaceNormal(parseBinaryStlMetres(buffer));
    expect(Math.abs(deg(Math.asin(headNormal[1])) - spec.loftDeg)).toBeLessThan(0.5);
    // The placed head keeps it relative to the ground plane at address.
    const up = worldUpInClubFrame(spec.lieDeg);
    expect(Math.abs(deg(Math.asin(dot(data.faceNormal, up))) - spec.loftDeg)).toBeLessThan(0.5);
  });

  it('agrees with the analytic spec normal (desktop club_face_normal)', () => {
    const { spec, data } = load(club);
    const expected = webFaceNormal(spec.loftDeg, spec.lieDeg);
    const angle = deg(Math.acos(Math.min(1, dot(data.faceNormal, expected))));
    expect(angle).toBeLessThan(0.5);
  });

  it('puts the sole on the origin with the shaft axis along +y', () => {
    const { spec, data } = load(club);
    const p = data.positions;
    let toe = 0;
    for (let i = 0; i < p.length; i += 3) toe = Math.max(toe, Math.hypot(p[i], p[i + 2]));
    expect(toe).toBeGreaterThan(0.03);
    const s = shaftDirectionHead(spec.lieDeg);
    expect(Math.hypot(...s)).toBeCloseTo(1, 9);
    expect(data.hosel.topM).toBeGreaterThan(data.hosel.bottomM);
    expect(data.hosel.bottomM).toBeGreaterThan(0);
  });

  it('has outward winding (positive volume)', () => {
    const { data } = load(club);
    const p = data.positions;
    let vol = 0;
    for (let i = 0; i < p.length; i += 9) {
      vol +=
        p[i] * (p[i + 4] * p[i + 8] - p[i + 5] * p[i + 7]) -
        p[i + 1] * (p[i + 3] * p[i + 8] - p[i + 5] * p[i + 6]) +
        p[i + 2] * (p[i + 3] * p[i + 7] - p[i + 4] * p[i + 6]);
    }
    expect(vol / 6).toBeGreaterThan(0);
  });
});

describe('club head sizes', () => {
  function toeHeelLength(club: string): number {
    const { data } = load(club);
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 2; i < data.positions.length; i += 3) {
      lo = Math.min(lo, data.positions[i]);
      hi = Math.max(hi, data.positions[i]);
    }
    return hi - lo;
  }

  it('driver head is longer than a 7-iron head', () => {
    expect(toeHeelLength('driver')).toBeGreaterThan(toeHeelLength('iron7'));
  });
});

describe('parseBinaryStlMetres', () => {
  it('rejects a truncated buffer', () => {
    expect(() => parseBinaryStlMetres(new ArrayBuffer(10))).toThrow();
    expect(() => parseBinaryStlMetres(new ArrayBuffer(100))).toThrow();
  });

  it('converts millimetres to metres', () => {
    const buf = new ArrayBuffer(84 + 50);
    const view = new DataView(buf);
    view.setUint32(80, 1, true);
    view.setFloat32(84 + 12, 1000, true);
    expect(parseBinaryStlMetres(buf)[0]).toBeCloseTo(1, 6);
  });
});
