import { availableRange, bandSegments, pathSegments } from '@/api/traceGeometry';
import type { Band, DisplayClass, Overlay } from '@/api/loopNullSpace';
import { DecisionBadges } from './DecisionBadges';
import { DISPLAY_CLASS_TEXT, fmt } from './format';

const W = 360;
const H = 130;
const PAD = { l: 44, r: 8, t: 8, b: 16 };

const CLASS_FILL: Record<DisplayClass, string> = {
  DETERMINED_SIGN: '#0e7490',
  INDETERMINATE: '#6b7280',
  UNBOUNDED: 'url(#nsx-hatch)',
};

/**
 * Only a bound the model marks BOUNDED is drawn; any other status is a gap.
 * (The status is a model field; nothing is decided here.)
 */
function drawn(values: Array<number | null>, status: string[]): Array<number | null> {
  return values.map((v, i) => (status[i] === 'BOUNDED' ? v : null));
}

/** One band-first envelope chart; its representative, if any, is drawn ON it. */
export function BandChart({ band, overlay }: { band: Band; overlay?: Overlay }) {
  const t = band.frame_indices;
  const lower = drawn(band.lower, band.status_lower);
  const upper = drawn(band.upper, band.status_upper);
  const rep = overlay?.values ?? [];
  const yr = availableRange([lower, upper, rep]);
  const xr: [number, number] = [t[0] ?? 0, t[t.length - 1] ?? 1];
  const yRange: [number, number] = yr ?? [0, 1];
  const x0 = PAD.l;
  const x1 = W - PAD.r;
  const y0 = PAD.t;
  const y1 = H - PAD.b;
  const xOf = (f: number) => x0 + ((f - xr[0]) / (xr[1] - xr[0] || 1)) * (x1 - x0);
  const id = band.quantity_id;
  return (
    <figure data-testid={`nsx-band-${id}`} className="mb-3">
      <figcaption className="text-xs text-gray-300 mb-1">
        {id} ({band.units})
      </figcaption>
      <DecisionBadges decision={band.decision} testId={`nsx-band-${id}-decision`} />
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${id} band`} className="w-full">
        <defs>
          <pattern id="nsx-hatch" width="6" height="6" patternUnits="userSpaceOnUse">
            <path d="M0,6 L6,0" stroke="#f59e0b" strokeWidth="1" />
          </pattern>
        </defs>
        <rect x={x0} y={y0} width={x1 - x0} height={y1 - y0} fill="none" stroke="#4b5563" />
        {band.segments.map((s, i) => (
          <rect
            key={`seg-${i}`}
            data-testid={`nsx-band-${id}-strip-${i}`}
            x={xOf(s.first_frame_index)}
            y={y1}
            width={Math.max(xOf(s.last_frame_index) - xOf(s.first_frame_index), 2)}
            height={4}
            fill={CLASS_FILL[s.display_class]}
          />
        ))}
        {bandSegments(t, lower, upper, x0, x1, y0, y1, xr, yRange).map((d, i) => (
          <path
            key={`band-${i}`}
            data-testid={`nsx-band-${id}-area`}
            d={d}
            fill="#56B4E9"
            opacity={0.35}
          />
        ))}
        {overlay &&
          pathSegments(t, rep, x0, x1, y0, y1, xr, yRange).map((d, i) => (
            <path
              key={`rep-${i}`}
              data-testid={`nsx-band-${id}-representative`}
              d={d}
              fill="none"
              stroke="#E69F00"
              strokeWidth={1.5}
              strokeDasharray="4 2"
            />
          ))}
        <text x={2} y={y0 + 8} fontSize="8" fill="#9ca3af" data-testid={`nsx-band-${id}-ymax`}>
          {yr ? fmt(yr[1]) : ''}
        </text>
        <text x={2} y={y1} fontSize="8" fill="#9ca3af" data-testid={`nsx-band-${id}-ymin`}>
          {yr ? fmt(yr[0]) : ''}
        </text>
      </svg>
      {overlay && (
        <div
          data-testid={`nsx-band-${id}-representative-note`}
          className="text-[10px] text-gray-300"
        >
          Representative (dashed) on this band: {overlay.decision.label}
          <DecisionBadges
            decision={overlay.decision}
            testId={`nsx-band-${id}-representative-decision`}
          />
        </div>
      )}
      {band.infeasible_frame_indices.length > 0 && (
        <div data-testid={`nsx-band-${id}-infeasible`} className="text-[10px] text-red-300">
          Infeasible frames (no band drawn): {band.infeasible_frame_indices.join(', ')}
        </div>
      )}
      <table className="text-[10px] text-gray-300 w-full mt-1">
        <caption className="sr-only">{id} segments</caption>
        <thead>
          <tr className="text-left text-gray-400">
            <th scope="col">Frames</th>
            <th scope="col">Class</th>
            <th scope="col">Sign</th>
            <th scope="col">Binding hypotheses</th>
            <th scope="col">Active constraints</th>
          </tr>
        </thead>
        <tbody>
          {band.segments.map((s, i) => (
            <tr key={`row-${i}`} data-testid={`nsx-band-${id}-segment-${i}`}>
              <td>
                {s.first_frame_index}-{s.last_frame_index}
              </td>
              <td>{DISPLAY_CLASS_TEXT[s.display_class] ?? `Unknown class ${s.display_class}`}</td>
              <td>{s.sign_class ?? 'unavailable'}</td>
              <td>{s.binding_hypotheses.join(', ') || 'none'}</td>
              <td>{s.active_constraints.join(', ') || 'none'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </figure>
  );
}
