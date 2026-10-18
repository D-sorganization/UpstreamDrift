/**
 * Measured-plate uncertainty shown on the ground-reaction charts.
 *
 * Mirrors the PyQt renderer (plotting/renderers/ground_reaction.py): the
 * measured load is shaded by its random 1-sigma and sampled measured CoP
 * points are ringed by theirs. The CoP sigma is one sigma per horizontal axis
 * (no 2-D covariance), so it is a circle in data units; the web panel scales x
 * and y separately, which draws that circle as an SVG ellipse.
 */

/** Legend suffix of a 1-sigma band or ring (the PyQt `SIGMA_BAND_SUFFIX`). */
export const SIGMA_BAND_SUFFIX = '\u00b11\u03c3';
/** The PyQt `_PER_AXIS`: the CoP sigma is per horizontal axis. */
export const PER_AXIS = 'per axis';
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
  circled: boolean,
  source: string | null | undefined,
): string | null {
  const parts = [
    ...(banded ? [`shaded: measured load ${SIGMA_BAND_SUFFIX}`] : []),
    ...(circled ? [`circles: measured CoP ${SIGMA_BAND_SUFFIX} ${PER_AXIS}`] : []),
  ];
  if (parts.length === 0) return null;
  const text = parts.join('; ');
  return `${text[0].toUpperCase()}${text.slice(1)} (declared uncertainty: ${source ?? 'not stated'})`;
}
