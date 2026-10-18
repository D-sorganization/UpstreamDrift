/**
 * Shared SVG trace geometry helpers.
 *
 * Used by both the grip-wrench charts (GCV-10, #11716) and the
 * ground-reaction charts (GCV-5, #11711): turning a time series with
 * possibly-unavailable (`null`) samples into SVG path data, and finding the
 * value range across a set of series for axis scaling. A `null` sample is
 * unavailable and must be drawn as a gap, never as zero.
 */

/** Data-to-SVG maps for the x and y axes (y grows downward on screen). */
function scalers(
  x0: number,
  x1: number,
  y0: number,
  y1: number,
  xRange: [number, number],
  yRange: [number, number],
): [(v: number) => number, (v: number) => number] {
  return [
    (v) => x0 + ((v - xRange[0]) / (xRange[1] - xRange[0] || 1)) * (x1 - x0),
    (v) => y1 - ((v - yRange[0]) / (yRange[1] - yRange[0] || 1)) * (y1 - y0),
  ];
}

/** Runs of consecutive available samples as SVG path data; null breaks the line. */
export function pathSegments(
  t: number[],
  y: Array<number | null>,
  x0: number,
  x1: number,
  y0: number,
  y1: number,
  xRange: [number, number],
  yRange: [number, number],
): string[] {
  const [sx, sy] = scalers(x0, x1, y0, y1, xRange, yRange);
  const out: string[] = [];
  let cur = '';
  y.forEach((v, i) => {
    if (v === null || v === undefined || !Number.isFinite(v)) {
      if (cur) out.push(cur);
      cur = '';
      return;
    }
    cur += `${cur ? 'L' : 'M'}${sx(t[i]).toFixed(2)},${sy(v).toFixed(2)}`;
  });
  if (cur) out.push(cur);
  return out;
}

/** Range of the available samples across series; `null` when none are available. */
export function availableRange(series: Array<Array<number | null>>): [number, number] | null {
  let lo = Infinity;
  let hi = -Infinity;
  series.forEach((s) =>
    s.forEach((v) => {
      if (v !== null && v !== undefined && Number.isFinite(v)) {
        lo = Math.min(lo, v);
        hi = Math.max(hi, v);
      }
    }),
  );
  return Number.isFinite(lo) ? [lo, hi] : null;
}

const finite = (v: number | null | undefined): v is number =>
  v !== null && v !== undefined && Number.isFinite(v);

/**
 * Closed SVG polygons shading `lower..upper`, one per run of two or more
 * samples where both bounds are available; an unavailable bound breaks the
 * band rather than bridging it.
 */
export function bandSegments(
  t: number[],
  lower: Array<number | null>,
  upper: Array<number | null>,
  x0: number,
  x1: number,
  y0: number,
  y1: number,
  xRange: [number, number],
  yRange: [number, number],
): string[] {
  const [sx, sy] = scalers(x0, x1, y0, y1, xRange, yRange);
  const point = (i: number, v: number) => `${sx(t[i]).toFixed(2)},${sy(v).toFixed(2)}`;
  const close = (run: number[]) => {
    const top = run.map((i, k) => `${k ? 'L' : 'M'}${point(i, upper[i] as number)}`);
    const bottom = [...run].reverse().map((i) => `L${point(i, lower[i] as number)}`);
    return `${top.join('')}${bottom.join('')}Z`;
  };
  const out: string[] = [];
  let run: number[] = [];
  const flush = () => {
    if (run.length > 1) out.push(close(run));
    run = [];
  };
  t.forEach((_, i) => (finite(lower[i]) && finite(upper[i]) ? run.push(i) : flush()));
  flush();
  return out;
}
