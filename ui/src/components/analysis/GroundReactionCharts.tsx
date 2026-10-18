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
  bandSegments,
  fetchGroundReaction,
  fetchGroundReactionOverlay,
  pathSegments,
  type GroundReactionResponse,
  type PlateOverlayRequest,
} from '@/api/groundReaction';
import { PlateImportForm } from './PlateImportForm';
import {
  BAND_OPACITY,
  MAX_COP_MARKERS,
  PER_AXIS,
  SIGMA_BAND_SUFFIX,
  copMarkerIndices,
  uncertaintyCaption,
} from './groundReactionUncertainty';

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

/** Banner shown while the overlaid plate trial has no device qualification. */
const UNQUALIFIED_BANNER = 'Unqualified measurement';

/**
 * `x` is set only for path (non-time) panels, e.g. the CoP panel; `dashed`
 * marks a measured (plate) trace so it never reads as the model's own.
 */
type Row = {
  label: string;
  color: string;
  y: Array<number | null>;
  x?: number[];
  dashed?: boolean;
};
type Entry = { key: string; label: string; color: string };
/** A shaded `lower..upper` region, e.g. a measured load +/- its 1-sigma. */
type Band = {
  label: string;
  color: string;
  lower: Array<number | null>;
  upper: Array<number | null>;
};
/** Sampled measured CoP points ringed by their 1-sigma per axis (data units). */
type Ring = {
  label: string;
  color: string;
  points: Array<{ frame: number; x: number; y: number; r: number }>;
};

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

/** Dashed rows for the measured plate traces of `quantity` (feet only). */
function measuredRows(data: GroundReactionResponse, quantity: string, field: string): Row[] {
  return footKeys(data, false)
    .filter((e) => data.traces?.[`measured_${e.key}_${quantity}`])
    .map((e) => ({
      label: `Measured ${e.label}`,
      color: e.color,
      dashed: true,
      y: (data.traces?.[`measured_${e.key}_${quantity}`]?.[field] ?? []) as Array<number | null>,
    }));
}

/** Measured load +/- its 1-sigma, per foot whose overlay carries a sigma trace. */
function measuredLoadBands(data: GroundReactionResponse): Band[] {
  return footKeys(data, false).flatMap((e) => {
    const load = data.traces?.[`measured_${e.key}_force_n`]?.z;
    const sigma = data.traces?.[`measured_${e.key}_force_sigma_n`]?.z;
    if (!load || !sigma) return [];
    // A bound is available only where both the load and its sigma are.
    const bound = (sign: 1 | -1) =>
      load.map((f, i) => {
        const s = sigma[i];
        return f === null || s === null || s === undefined ? null : f + sign * s;
      });
    return [
      {
        label: `Measured ${e.label} ${SIGMA_BAND_SUFFIX}`,
        color: e.color,
        lower: bound(-1),
        upper: bound(1),
      },
    ];
  });
}

/** 1-sigma rings at sampled measured CoP points, per foot with a CoP sigma trace. */
function measuredCopRings(data: GroundReactionResponse): Ring[] {
  return footKeys(data, false).flatMap((e) => {
    const cop = data.traces?.[`measured_${e.key}_cop_m`];
    const sigma = data.traces?.[`measured_${e.key}_cop_sigma_m`]?.z;
    if (!cop || !sigma) return [];
    const x = (cop.x ?? []) as Array<number | null>;
    const y = (cop.y ?? []) as Array<number | null>;
    const points = copMarkerIndices(x, sigma, MAX_COP_MARKERS)
      .filter((i) => y[i] !== null && y[i] !== undefined)
      .map((i) => ({ frame: i, x: x[i] as number, y: y[i] as number, r: sigma[i] as number }));
    return points.length === 0
      ? []
      : [{ label: `Measured ${e.label} ${SIGMA_BAND_SUFFIX} ${PER_AXIS}`, color: e.color, points }];
  });
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

function copRow(data: GroundReactionResponse, trace: string, entry: Entry, dashed: boolean): Row {
  const source = data.traces?.[trace];
  const { x, y } = sanitizedXY(
    (source?.x ?? []) as Array<number | null>,
    (source?.y ?? []) as Array<number | null>,
  );
  return { label: entry.label, color: entry.color, x, y, dashed };
}

function copRows(data: GroundReactionResponse): Row[] {
  const modelled = footKeys(data, true).map((e) => copRow(data, `${e.key}_cop_m`, e, false));
  const measured = footKeys(data, false)
    .filter((e) => data.traces?.[`measured_${e.key}_cop_m`])
    .map((e) =>
      copRow(data, `measured_${e.key}_cop_m`, { ...e, label: `Measured ${e.label}` }, true),
    );
  return [...modelled, ...measured];
}

/** The unqualified-measurement banner, shown only when the server says so. */
function MeasurementBanner({ data }: { data: GroundReactionResponse }) {
  if (data.measured_qualified !== false) return null;
  return (
    <div
      role="status"
      data-testid="ground-reaction-unqualified-banner"
      className="text-xs font-semibold text-amber-500 mb-2"
    >
      {UNQUALIFIED_BANNER}
    </div>
  );
}

/** Says what the shaded bands are and where their declared uncertainty came from. */
function UncertaintyCaption({
  data,
  banded,
  circled,
}: {
  data: GroundReactionResponse;
  banded: boolean;
  circled: boolean;
}) {
  const caption = uncertaintyCaption(banded, circled, data.measured_uncertainty_source);
  if (caption === null) return null;
  return (
    <div data-testid="ground-reaction-uncertainty-caption" className="text-xs text-gray-400 mb-2">
      {caption}
    </div>
  );
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
  bands = [],
  events,
}: {
  id: string;
  title: string;
  unit: string;
  t: number[];
  rows: Row[];
  bands?: Band[];
  events: Record<string, number>;
}) {
  const { x0, x1, y0, y1 } = BOUNDS;
  const yr = availableRange([
    ...rows.map((r) => r.y),
    ...bands.flatMap((b) => [b.lower, b.upper]),
  ]);
  const xr: [number, number] = [t[0] ?? 0, t[t.length - 1] ?? 1];
  const yRange: [number, number] = yr ?? [0, 1];
  const sx = (v: number) => x0 + ((v - xr[0]) / (xr[1] - xr[0] || 1)) * (x1 - x0);
  const eventEntries = Object.entries(events).filter(
    ([, time]) => time >= xr[0] && time <= xr[1],
  );

  return (
    <ChartFrame id={id} title={title} suffix={unit} legend={[...rows, ...bands]}>
      {bands.map((b) => {
        const segs = bandSegments(t, b.lower, b.upper, x0, x1, y0, y1, xr, yRange);
        return segs.length === 0 ? null : (
          <path
            key={b.label}
            data-testid={`ground-reaction-band-${id}-${b.label.replace(` ${SIGMA_BAND_SUFFIX}`, '')}`}
            d={segs.join('')}
            fill={b.color}
            fillOpacity={BAND_OPACITY}
            stroke="none"
          />
        );
      })}
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
            strokeDasharray={r.dashed ? '4 2' : undefined}
          />
        );
      })}
    </ChartFrame>
  );
}

/** Each ring's data-unit extremes, so the axis ranges never clip a ring. */
function ringExtents(rings: Ring[], axis: 'x' | 'y'): number[][] {
  return rings.map((g) => g.points.flatMap((p) => [p[axis] - p.r, p[axis] + p.r]));
}

function PathPanel({
  id,
  title,
  rows,
  rings = [],
  reason,
}: {
  id: string;
  title: string;
  rows: Row[];
  rings?: Ring[];
  reason: string | null | undefined;
}) {
  const { x0, x1, y0, y1 } = BOUNDS;
  const xr = availableRange([...rows.map((r) => r.x ?? []), ...ringExtents(rings, 'x')]);
  const yr = availableRange([...rows.map((r) => r.y), ...ringExtents(rings, 'y')]);
  const hasData = xr !== null && yr !== null;
  const xRange: [number, number] = xr ?? [0, 1];
  const yRange: [number, number] = yr ?? [0, 1];
  // A circle in data units is an ellipse here: x and y are scaled separately.
  const kx = (x1 - x0) / (xRange[1] - xRange[0] || 1);
  const ky = (y1 - y0) / (yRange[1] - yRange[0] || 1);

  return (
    <ChartFrame id={id} title={title} suffix="m" legend={[...rows, ...rings]}>
      {hasData &&
        rings.flatMap((g) =>
          g.points.map((p) => (
            <ellipse
              key={`${g.label}-${p.frame}`}
              data-testid={`ground-reaction-cop-sigma-${g.label.replace(` ${SIGMA_BAND_SUFFIX} ${PER_AXIS}`, '')}-${p.frame}`}
              cx={x0 + (p.x - xRange[0]) * kx}
              cy={y1 - (p.y - yRange[0]) * ky}
              rx={p.r * kx}
              ry={p.r * ky}
              fill={g.color}
              fillOpacity={BAND_OPACITY}
              stroke={g.color}
              strokeOpacity={BAND_OPACITY}
            />
          )),
        )}
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
              strokeDasharray={r.dashed ? '4 2' : undefined}
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

export function GroundReactionCharts({
  runId = null,
  overlay = null,
}: {
  runId?: string | null;
  /** When set, the measured plate trial is overlaid via the plate-overlay route. */
  overlay?: PlateOverlayRequest | null;
}) {
  const [data, setData] = useState<GroundReactionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [formOverlay, setFormOverlay] = useState<PlateOverlayRequest | null>(null);
  const [plateError, setPlateError] = useState<string | null>(null);
  const activeOverlay = overlay ?? formOverlay;
  const fromForm = overlay === null && formOverlay !== null;

  useEffect(() => {
    const controller = new AbortController();
    const request = activeOverlay
      ? fetchGroundReactionOverlay(
          { ...activeOverlay, run_id: activeOverlay.run_id ?? runId },
          controller.signal,
        )
      : fetchGroundReaction({ runId }, controller.signal);
    request
      .then((d) => {
        setError(null);
        setData(d);
      })
      .catch((e: unknown) => {
        if (controller.signal.aborted) return;
        const message = e instanceof Error ? e.message : String(e);
        if (fromForm) {
          // A refused declaration belongs next to the form; show the model again.
          setPlateError(message);
          setFormOverlay(null);
          return;
        }
        setData(null);
        setError(message);
      });
    return () => controller.abort();
  }, [runId, activeOverlay, fromForm]);

  const submitPlateForm = (request: PlateOverlayRequest) => {
    setPlateError(null);
    setFormOverlay(request);
  };

  const panels = useMemo(() => {
    if (!data || !data.available) return null;
    const inBw = Boolean(data.traces?.net_force_bw);
    const forceQuantity = inBw ? 'force_bw' : 'force_n';
    const loadBands = inBw ? [] : measuredLoadBands(data);
    const copRings = measuredCopRings(data);
    return {
      banded: loadBands.length > 0,
      rings: copRings,
      t: data.time_s ?? [],
      events: data.events ?? {},
      time: [
        {
          id: 'vertical_force',
          title: PANEL_TITLES[0],
          unit: inBw ? 'BW' : 'N',
          rows: [
            ...quantityRows(data, footKeys(data, true), forceQuantity, 'z'),
            ...(inBw ? [] : measuredRows(data, 'force_n', 'z')),
          ],
          bands: loadBands,
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
      {overlay === null && (
        <PlateImportForm
          onSubmit={submitPlateForm}
          onClear={() => setFormOverlay(null)}
          serverError={plateError}
          active={formOverlay !== null}
        />
      )}
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
          <MeasurementBanner data={data} />
          <UncertaintyCaption data={data} banded={panels.banded}
            circled={panels.rings.length > 0}
          />
          {panels.time.slice(0, 3).map((p) => (
            <TimePanel key={p.id} {...p} t={panels.t} events={panels.events} />
          ))}
          <PathPanel
            id="cop_path"
            title={PANEL_TITLES[3]}
            rows={panels.cop}
            rings={panels.rings}
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
