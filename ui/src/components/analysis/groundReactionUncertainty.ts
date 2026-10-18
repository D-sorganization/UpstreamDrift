/**
 * Measured-plate uncertainty shown on the ground-reaction charts.
 *
 * Mirrors the PyQt renderer (plotting/renderers/ground_reaction.py): the
 * measured load is shaded by its random 1-sigma and sampled measured CoP
 * points are ringed by theirs: the 1-sigma ellipse of the CoP x-y covariance,
 * tilted with the plate, or, for a payload without the covariance, an
 * axis-aligned ellipse of the per-world-axis sigmas or a circle of the
 * worse-axis sigma.  The covariance geometry mirrors
 * plotting/renderers/_cop_ellipse.py.
 */

/** Legend suffix of a 1-sigma band or ring (the PyQt `SIGMA_BAND_SUFFIX`). */
export const SIGMA_BAND_SUFFIX = '\u00b11\u03c3';
/** The PyQt `_PER_AXIS`: axis-aligned ellipses of the per-axis CoP sigmas. */
export const PER_AXIS = 'per axis';
/** The PyQt `_COVARIANCE`: ellipses of the CoP x-y covariance, tilted. */
export const COVARIANCE = 'covariance';
/** What the CoP ellipses show. */
export type RingKind = typeof PER_AXIS | typeof COVARIANCE;
/** Relative slack on `cov_xy^2 <= var_x var_y` (the PyQt `_PSD_REL_SLACK`). */
const PSD_REL_SLACK = 1e-9;
/** SVG rotations are kept within (-45, 45] degrees by swapping the semi-axes. */
const MAX_SVG_TILT_DEG = 45;
export const BAND_OPACITY = 0.2;
/** CoP rings drawn per foot at most (the PyQt `MAX_COP_MARKERS`). */
export const MAX_COP_MARKERS = 12;

const finite = (v: number | null | undefined): v is number =>
  v !== null && v !== undefined && Number.isFinite(v);

/** Evenly spaced frames, at most `maxMarkers`, with both a CoP and a sigma. */
export function copMarkerIndices(
  copX: Array<number | null>,
  sigma: Array<number | null>,
  maxMarkers: number,
): number[] {
  const usable = copX.flatMap((x, i) => (finite(x) && finite(sigma[i]) ? [i] : []));
  const stride = Math.max(1, Math.ceil(usable.length / maxMarkers));
  return usable.filter((_, k) => k % stride === 0);
}

/** What the bands and rings show and where the declared uncertainty came from. */
export function uncertaintyCaption(
  banded: boolean,
  ringed: RingKind | null,
  source: string | null | undefined,
): string | null {
  const parts = [
    ...(banded ? [`shaded: measured load ${SIGMA_BAND_SUFFIX}`] : []),
    ...(ringed !== null ? [`ellipses: measured CoP ${SIGMA_BAND_SUFFIX} ${ringed}`] : []),
  ];
  if (parts.length === 0) return null;
  const text = parts.join('; ');
  return `${text[0].toUpperCase()}${text.slice(1)} (declared uncertainty: ${source ?? 'not stated'})`;
}

/** A CoP sigma trace: `x`/`y` per world axis, `z` the worse axis. */
export interface CopSigmaTrace {
  x?: Array<number | null>;
  y?: Array<number | null>;
  z?: Array<number | null>;
}

/** A CoP covariance trace: `x` = var_x, `y` = var_y, `z` = cov_xy [m^2]. */
export type CopCovarianceTrace = CopSigmaTrace;

/** `[var_x, var_y, cov_xy]` of one 2x2 covariance. */
export type Covariance = [number, number, number];

/**
 * Semi-axes and tilt of the 1-sigma ellipse of a covariance: `angleDeg` is the
 * major axis from +x towards +y in (-90, 90] (0 for a circle).  Throws unless
 * the input is a finite positive semi-definite covariance (within rounding).
 */
export function covarianceEllipse(
  varX: number,
  varY: number,
  covXY: number,
): { major: number; minor: number; angleDeg: number } {
  if (![varX, varY, covXY].every(Number.isFinite) || varX < 0 || varY < 0) {
    throw new Error('covariance terms must be finite with non-negative variances');
  }
  if (covXY ** 2 > varX * varY + PSD_REL_SLACK * Math.max(varX, varY) ** 2) {
    throw new Error('not a covariance: |cov_xy| > sqrt(var_x * var_y)');
  }
  const mean = 0.5 * (varX + varY);
  const radius = Math.hypot(0.5 * (varX - varY), covXY);
  const angleDeg = (0.5 * Math.atan2(2 * covXY, varX - varY) * 180) / Math.PI;
  return {
    major: Math.sqrt(mean + radius),
    minor: Math.sqrt(Math.max(mean - radius, 0)),
    angleDeg,
  };
}

/**
 * The frame-`i` marker as a data-unit covariance and what it shows: the
 * covariance trace when every term is present, else the per-axis sigmas
 * (uncorrelated), else a circle of the worse axis; `null` without that.
 */
export function copMarkerCovariance(
  sigma: CopSigmaTrace,
  cov: CopCovarianceTrace | undefined,
  i: number,
): { terms: Covariance; kind: RingKind } | null {
  const xx = cov?.x?.[i];
  const yy = cov?.y?.[i];
  const xy = cov?.z?.[i];
  if (finite(xx) && finite(yy) && finite(xy)) return { terms: [xx, yy, xy], kind: COVARIANCE };
  const worse = sigma.z?.[i];
  if (!finite(worse)) return null;
  const sx = sigma.x?.[i];
  const sy = sigma.y?.[i];
  const [ax, ay] = finite(sx) && finite(sy) ? [sx, sy] : [worse, worse];
  return { terms: [ax ** 2, ay ** 2, 0], kind: PER_AXIS };
}

/**
 * An SVG `<ellipse>` for a data-unit covariance drawn with x and y scales
 * `kx`, `ky` (SVG units per data unit; SVG y points down): the covariance is
 * carried to SVG units as `J C J^T`, `J = diag(kx, -ky)`, then `rx`, `ry` and
 * `rotate(rotateDeg)` about the centre.  An axis-aligned ellipse stays
 * unrotated with `rx`, `ry` along x and y.
 */
export function svgEllipse(
  terms: Covariance,
  kx: number,
  ky: number,
): { rx: number; ry: number; rotateDeg: number } {
  const [xx, yy, xy] = terms;
  const e = covarianceEllipse(kx * kx * xx, ky * ky * yy, -kx * ky * xy);
  if (Math.abs(e.angleDeg) <= MAX_SVG_TILT_DEG) {
    return { rx: e.major, ry: e.minor, rotateDeg: e.angleDeg };
  }
  const quarter = e.angleDeg > 0 ? 90 : -90;
  return { rx: e.minor, ry: e.major, rotateDeg: e.angleDeg - quarter };
}
