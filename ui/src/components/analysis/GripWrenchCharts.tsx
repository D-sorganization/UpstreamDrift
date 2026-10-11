/**
 * GripWrenchCharts - hands' loading on the club over time (GCV-10, #11716).
 *
 * Mirrors the PyQt6 grip wrench plot: force per hand, net force at the grip
 * midpoint, the equivalent couple about the midpoint (world or club frame) and
 * the contact-force-moment vs applied-free-torque split, with the impact marker
 * and the `split_method`. Unavailable samples (`null`) break the line; they are
 * never drawn as zero.
 */

import { useEffect, useMemo, useState } from 'react';
import {
  availableRange,
  fetchGripWrench,
  pathSegments,
  type GripWrenchResponse,
} from '@/api/gripWrench';

type Frame = 'world' | 'club';
type Row = { trace: string; label: string; color: string };

const W = 360;
const H = 120;
const PAD = { l: 40, r: 8, t: 8, b: 16 };

/** GRIP #56B4E9 is the net colour; left lighter, right darker (ADR-0052). */
const COLORS = {
  left: '#9AD0F2',
  right: '#1B6C99',
  net: '#56B4E9',
  couple: '#E69F00',
  contact: '#009E73',
  free: '#CC79A7',
};

interface ChartSpec {
  id: string;
  title: string;
  unit: string;
  rows: Row[];
}

function Chart({
  spec,
  data,
  impact,
  field,
}: {
  spec: ChartSpec;
  data: GripWrenchResponse;
  impact: number | undefined;
  field: string;
}) {
  const t = data.time_s ?? [];
  const series = spec.rows.map((r) => ({
    ...r,
    y: (data.traces?.[r.trace]?.[field] ?? []) as Array<number | null>,
  }));
  const yr = availableRange(series.map((s) => s.y));
  const xr: [number, number] = [t[0] ?? 0, t[t.length - 1] ?? 1];
  const yRange: [number, number] = yr ?? [0, 1];
  const x0 = PAD.l;
  const x1 = W - PAD.r;
  const y0 = PAD.t;
  const y1 = H - PAD.b;
  const impactX =
    impact !== undefined && impact >= xr[0] && impact <= xr[1]
      ? x0 + ((impact - xr[0]) / (xr[1] - xr[0] || 1)) * (x1 - x0)
      : null;
  return (
    <figure data-testid={`grip-chart-${spec.id}`} className="mb-3">
      <figcaption className="text-xs text-gray-300 mb-1">
        {spec.title} ({spec.unit})
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={spec.title} className="w-full">
        <rect x={x0} y={y0} width={x1 - x0} height={y1 - y0} fill="none" stroke="#4b5563" />
        <text x={2} y={y0 + 8} fontSize="8" fill="#9ca3af">
          {yr ? yr[1].toFixed(1) : ''}
        </text>
        <text x={2} y={y1} fontSize="8" fill="#9ca3af">
          {yr ? yr[0].toFixed(1) : ''}
        </text>
        {impactX !== null && (
          <line
            data-testid={`grip-impact-marker-${spec.id}`}
            x1={impactX}
            x2={impactX}
            y1={y0}
            y2={y1}
            stroke="#f87171"
            strokeDasharray="3"
          />
        )}
        {series.map((s) =>
          pathSegments(t, s.y, x0, x1, y0, y1, xr, yRange).length === 0 ? null : (
            <path
              key={s.trace}
              data-testid={`grip-line-${spec.id}-${s.trace}`}
              d={pathSegments(t, s.y, x0, x1, y0, y1, xr, yRange).join('')}
              fill="none"
              stroke={s.color}
              strokeWidth={1.5}
            />
          ),
        )}
      </svg>
      <div className="flex gap-3 text-[10px] text-gray-400">
        {series.map((s) => (
          <span key={s.trace}>
            <span style={{ color: s.color }}>&#9632;</span> {s.label}
          </span>
        ))}
      </div>
    </figure>
  );
}

export function GripWrenchCharts({ runId = null }: { runId?: string | null }) {
  const [data, setData] = useState<GripWrenchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [frame, setFrame] = useState<Frame>('world');

  useEffect(() => {
    const controller = new AbortController();
    fetchGripWrench({ runId }, controller.signal)
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

  const coupleTrace = frame === 'world' ? 'couple_nm' : 'couple_local_nm';
  const charts = useMemo<ChartSpec[]>(
    () => [
      {
        id: 'hand_forces',
        title: 'Hand Force Magnitude',
        unit: 'N',
        rows: [
          { trace: 'left_force_n', label: 'Left', color: COLORS.left },
          { trace: 'right_force_n', label: 'Right', color: COLORS.right },
        ],
      },
      {
        id: 'net_force',
        title: 'Net Force at Grip Midpoint',
        unit: 'N',
        rows: [{ trace: 'net_force_n', label: 'Net', color: COLORS.net }],
      },
      {
        id: 'couple',
        title: `Equivalent Couple at Midpoint (${frame === 'world' ? 'World' : 'Club'})`,
        unit: 'N*m',
        rows: [{ trace: coupleTrace, label: 'Couple', color: COLORS.couple }],
      },
      {
        id: 'couple_split',
        title: 'Contact Force Moment vs Applied Free Torque',
        unit: 'N*m',
        rows: [
          { trace: 'contact_force_moment_nm', label: 'Contact force moment', color: COLORS.contact },
          { trace: 'applied_free_torque_nm', label: 'Free torque', color: COLORS.free },
        ],
      },
    ],
    [frame, coupleTrace],
  );

  const missing =
    data?.available &&
    Object.values(data.traces ?? {}).some((t) => (t.magnitude ?? []).some((v) => v === null));

  return (
    <section
      className="bg-gray-800 rounded-lg border border-gray-700 p-4"
      data-testid="grip-wrench-charts"
    >
      <h3 className="text-sm font-semibold text-gray-300 mb-2">Club Grip Force and Torque</h3>
      {error && (
        <div role="alert" className="text-xs text-red-400">
          Grip wrench unavailable: {error}
        </div>
      )}
      {data && !data.available && (
        <div className="text-xs text-gray-400 italic" data-testid="grip-unavailable">
          unavailable: {data.reason}
        </div>
      )}
      {data && data.available && (
        <>
          <div className="flex flex-wrap gap-3 items-center mb-2 text-xs text-gray-300">
            <span>
              Split method:{' '}
              <span className="font-mono text-sky-300" data-testid="grip-split-method">
                {data.split_method}
              </span>
            </span>
            <label>
              Couple frame
              <select
                aria-label="Couple frame"
                value={frame}
                onChange={(e) => setFrame(e.target.value as Frame)}
                className="ml-1 bg-gray-700 rounded px-1 py-0.5"
              >
                <option value="world">World</option>
                <option value="club">Club</option>
              </select>
            </label>
          </div>
          {missing && (
            <div className="text-[11px] text-amber-400/80 mb-2" data-testid="grip-unavailable-note">
              Gaps are unavailable samples (split or free torque not provided), not zero.
            </div>
          )}
          {charts.map((c) => (
            <Chart key={c.id} spec={c} data={data} impact={data.events?.impact} field="magnitude" />
          ))}
        </>
      )}
    </section>
  );
}

export default GripWrenchCharts;
