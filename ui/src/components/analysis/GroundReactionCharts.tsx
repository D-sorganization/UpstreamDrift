/**
 * GroundReactionCharts - ground-reaction loading on the body (GCV-5, #11711).
 *
 * Mirrors the PyQt6 sheet (plotting/renderers/ground_reaction.py): vertical
 * force per foot/net (BW when known), net force components, vertical load
 * share, the centre-of-pressure path, free moment and net moment about CoM.
 * Unavailable samples (`null`) break the line; never drawn as zero.
 */

import { useEffect, useMemo, useState, type ReactNode } from 'react';
import {
  availableRange,
  fetchGroundReaction,
  pathSegments,
  type GroundReactionResponse,
} from '@/api/groundReaction';

const W = 360;
const H = 120;
const PAD = { l: 40, r: 8, t: 8, b: 16 };
const BOUNDS = { x0: PAD.l, x1: W - PAD.r, y0: PAD.t, y1: H - PAD.b };

/** Same per-foot/net palette as the PyQt renderer's `_KEY_COLORS`/`_NET_COLOR`. */
const FOOT_COLORS = ['#0072B2', '#E69F00', '#CC79A7', '#56B4E9'];
const NET_COLOR = '#009E73';

/** Same axis-component palette as the PyQt renderer's `_AXIS_ROWS`. */
const AXIS_ROWS: Array<{ field: string; label: string; color: string }> = [
  { field: 'x', label: 'x', color: '#D55E00' },
  { field: 'y', label: 'y', color: '#009E73' },
  { field: 'z', label: 'z', color: '#0072B2' },
  { field: 'magnitude', label: '|F|', color: '#000000' },
];

const PANEL_TITLES = [
  'Vertical Ground Reaction Force',
  'Net Force Components',
  'Vertical Load Share',
  'Centre of Pressure Path',
  'Free Moment About the Vertical',
  'Net Moment About CoM',
] as const;

/** `x` is set only for path (non-time) panels, e.g. the CoP panel. */
type Row = { label: string; color: string; y: Array<number | null>; x?: number[] };
type Entry = { key: string; label: string; color: string };

function titleCase(key: string): string {
  return key
    .split('_')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
    .join(' ');
}

/** Per-foot entries, plus a trailing `net` entry when `includeNet`. */
function footKeys(data: GroundReactionResponse, includeNet: boolean): Entry[] {
  const feet = (data.feet ?? []).map((key, i) => ({
    key,
    label: titleCase(key),
    color: FOOT_COLORS[i % FOOT_COLORS.length],
  }));
  return includeNet ? [...feet, { key: 'net', label: 'Net', color: NET_COLOR }] : feet;
}

/** One row per key for `<key>_<quantity>`'s `field` component. */
function quantityRows(
  data: GroundReactionResponse,
  entries: Entry[],
  quantity: string,
  field: string,
): Row[] {
  return entries.map((e) => ({
    label: e.label,
    color: e.color,
    y: (data.traces?.[`${e.key}_${quantity}`]?.[field] ?? []) as Array<number | null>,
  }));
}

/** One row per key, read straight from `load_share` (feet only, never net). */
function loadShareRows(data: GroundReactionResponse): Row[] {
  return footKeys(data, false).map((e) => ({
    label: e.label,
    color: e.color,
    y: (data.load_share?.[e.key] ?? []) as Array<number | null>,
  }));
}

/** One row per axis component of a single trace, e.g. x/y/z/|F| of net force. */
function axisRows(
  data: GroundReactionResponse,
  trace: string,
  fields: Array<{ field: string; label: string; color: string }>,
): Row[] {
  return fields.map((f) => ({
    label: f.label,
    color: f.color,
    y: (data.traces?.[trace]?.[f.field] ?? []) as Array<number | null>,
  }));
}

/** Pair up x/y so a `null`/non-finite sample in either breaks the path, never the other alone. */
function sanitizedXY(
  x: Array<number | null>,
  y: Array<number | null>,
): { x: number[]; y: Array<number | null> } {
  const xs: number[] = [];
  const ys: Array<number | null> = [];
  x.forEach((xv, i) => {
    const yv = y[i] ?? null;
    const validX = xv !== null && xv !== undefined && Number.isFinite(xv);
    const validY = yv !== null && yv !== undefined && Number.isFinite(yv);
    xs.push(validX ? xv : NaN);
    ys.push(validX && validY ? yv : null);
  });
  return { x: xs, y: ys };
}

function copRows(data: GroundReactionResponse): Row[] {
  return footKeys(data, true).map((e) => {
    const trace = data.traces?.[`${e.key}_cop_m`];
    const { x, y } = sanitizedXY(
      (trace?.x ?? []) as Array<number | null>,
      (trace?.y ?? []) as Array<number | null>,
    );
    return { label: e.label, color: e.color, x, y };
  });
}

/** Shared figure/svg/legend chrome for a chart panel. */
function ChartFrame({
  id,
  title,
  suffix,
  legend,
  children,
}: {
  id: string;
  title: string;
  suffix: string;
  legend: Array<{ label: string; color: string }>;
  children: ReactNode;
}) {
  const { x0, x1, y0, y1 } = BOUNDS;
  return (
    <figure data-testid={`ground-reaction-chart-${id}`} className="mb-3">
      <figcaption className="text-xs text-gray-300 mb-1">
        {title} ({suffix})
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title} className="w-full">
        <rect x={x0} y={y0} width={x1 - x0} height={y1 - y0} fill="none" stroke="#4b5563" />
        {children}
      </svg>
      <div className="flex gap-3 text-[10px] text-gray-400">
        {legend.map((r) => (
          <span key={r.label}>
            <span style={{ color: r.color }}>&#9632;</span> {r.label}
          </span>
        ))}
      </div>
    </figure>
  );
}

function TimePanel({
  id,
  title,
  unit,
  t,
  rows,
  events,
}: {
  id: string;
  title: string;
  unit: string;
  t: number[];
  rows: Row[];
  events: Record<string, number>;
}) {
  const { x0, x1, y0, y1 } = BOUNDS;
  const yr = availableRange(rows.map((r) => r.y));
  const xr: [number, number] = [t[0] ?? 0, t[t.length - 1] ?? 1];
  const yRange: [number, number] = yr ?? [0, 1];
  const sx = (v: number) => x0 + ((v - xr[0]) / (xr[1] - xr[0] || 1)) * (x1 - x0);
  const eventEntries = Object.entries(events).filter(
    ([, time]) => time >= xr[0] && time <= xr[1],
  );

  return (
    <ChartFrame id={id} title={title} suffix={unit} legend={rows}>
      <text x={2} y={y0 + 8} fontSize="8" fill="#9ca3af">
        {yr ? yr[1].toFixed(2) : ''}
      </text>
      <text x={2} y={y1} fontSize="8" fill="#9ca3af">
        {yr ? yr[0].toFixed(2) : ''}
      </text>
      {eventEntries.map(([name, time]) => {
        const ex = sx(time);
        return (
          <g key={name} data-testid={`ground-reaction-event-${id}-${name}`}>
            <line x1={ex} x2={ex} y1={y0} y2={y1} stroke="#f87171" strokeDasharray="3" />
            <text x={ex + 1} y={y0 + 8} fontSize="7" fill="#f87171">
              {name}
            </text>
          </g>
        );
      })}
      {rows.map((r) => {
        const segs = pathSegments(t, r.y, x0, x1, y0, y1, xr, yRange);
        return segs.length === 0 ? null : (
          <path
            key={r.label}
            data-testid={`ground-reaction-line-${id}-${r.label}`}
            d={segs.join('')}
            fill="none"
            stroke={r.color}
            strokeWidth={1.5}
          />
        );
      })}
    </ChartFrame>
  );
}

function PathPanel({
  id,
  title,
  rows,
  reason,
}: {
  id: string;
  title: string;
  rows: Row[];
  reason: string | null | undefined;
}) {
  const { x0, x1, y0, y1 } = BOUNDS;
  const xr = availableRange(rows.map((r) => r.x ?? []));
  const yr = availableRange(rows.map((r) => r.y));
  const hasData = xr !== null && yr !== null;
  const xRange: [number, number] = xr ?? [0, 1];
  const yRange: [number, number] = yr ?? [0, 1];

  return (
    <ChartFrame id={id} title={title} suffix="m" legend={rows}>
      {hasData ? (
        rows.map((r) => {
          const segs = pathSegments(r.x ?? [], r.y, x0, x1, y0, y1, xRange, yRange);
          return segs.length === 0 ? null : (
            <path
              key={r.label}
              data-testid={`ground-reaction-line-${id}-${r.label}`}
              d={segs.join('')}
              fill="none"
              stroke={r.color}
              strokeWidth={1.5}
            />
          );
        })
      ) : (
        <text
          data-testid="ground-reaction-cop-unavailable"
          x={x0 + 4}
          y={(y0 + y1) / 2}
          fontSize="8"
          fill="#9ca3af"
        >
          unavailable{reason ? `: ${reason}` : ''}
        </text>
      )}
    </ChartFrame>
  );
}

export function GroundReactionCharts({ runId = null }: { runId?: string | null }) {
  const [data, setData] = useState<GroundReactionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    fetchGroundReaction({ runId }, controller.signal)
      .then((d) => {
        setError(null);
        setData(d);
      })
      .catch((e: unknown) => {
        if (controller.signal.aborted) return;
        setData(null);
        setError(e instanceof Error ? e.message : String(e));
      });
    return () => controller.abort();
  }, [runId]);

  const panels = useMemo(() => {
    if (!data || !data.available) return null;
    const inBw = Boolean(data.traces?.net_force_bw);
    const forceQuantity = inBw ? 'force_bw' : 'force_n';
    return {
      t: data.time_s ?? [],
      events: data.events ?? {},
      time: [
        {
          id: 'vertical_force',
          title: PANEL_TITLES[0],
          unit: inBw ? 'BW' : 'N',
          rows: quantityRows(data, footKeys(data, true), forceQuantity, 'z'),
        },
        {
          id: 'net_force',
          title: PANEL_TITLES[1],
          unit: 'N',
          rows: axisRows(data, 'net_force_n', AXIS_ROWS),
        },
        {
          id: 'load_share',
          title: PANEL_TITLES[2],
          unit: 'fraction',
          rows: loadShareRows(data),
        },
        {
          id: 'free_moment',
          title: PANEL_TITLES[4],
          unit: 'N*m',
          rows: quantityRows(data, footKeys(data, true), 'free_moment_nm', 'z'),
        },
        {
          id: 'net_moment',
          title: PANEL_TITLES[5],
          unit: 'N*m',
          rows: axisRows(data, 'net_moment_com_nm', AXIS_ROWS.slice(0, 3)),
        },
      ],
      cop: copRows(data),
    };
  }, [data]);

  return (
    <section
      className="bg-gray-800 rounded-lg border border-gray-700 p-4"
      data-testid="ground-reaction-charts"
    >
      <h3 className="text-sm font-semibold text-gray-300 mb-2">Ground Reaction</h3>
      {error && (
        <div role="alert" className="text-xs text-red-400">
          Ground reaction unavailable: {error}
        </div>
      )}
      {data && !data.available && (
        <div className="text-xs text-gray-400 italic" data-testid="ground-reaction-unavailable">
          Unavailable: {data.reason}
        </div>
      )}
      {data && data.available && panels && (
        <>
          {panels.time.slice(0, 3).map((p) => (
            <TimePanel key={p.id} {...p} t={panels.t} events={panels.events} />
          ))}
          <PathPanel
            id="cop_path"
            title={PANEL_TITLES[3]}
            rows={panels.cop}
            reason={data.reason}
          />
          {panels.time.slice(3).map((p) => (
            <TimePanel key={p.id} {...p} t={panels.t} events={panels.events} />
          ))}
        </>
      )}
    </section>
  );
}

export default GroundReactionCharts;
