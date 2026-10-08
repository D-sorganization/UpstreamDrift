/**
 * ImpactParametersPanel — launch-monitor-style impact card (GCV-17, #11723).
 *
 * Mirrors the PyQt6 dock: speed, AoA, path, face, face-to-path, dynamic loft,
 * spin loft, low point, impact location and smash, with a target-line input,
 * units toggle, D-plane / path diagrams and explicit "unavailable" reasons.
 */

import { useEffect, useState } from 'react';
import { Link } from 'react-router';
import {
  fetchImpactParameters,
  formatImpactValue,
  impactExplorerHref,
  targetDirFromHeading,
  type ImpactParametersResponse,
} from '@/api/impactParameters';

type Units = 'mph' | 'm/s';

function arrow(deg: number | null | undefined, color: string, sign: 1 | -1) {
  if (deg === null || deg === undefined) return null;
  const rad = (sign * deg * Math.PI) / 180;
  return (
    <line
      x1={20}
      y1={50}
      x2={20 + 70 * Math.cos(rad)}
      y2={50 - 70 * Math.sin(rad)}
      stroke={color}
      strokeWidth={3}
    />
  );
}

export function ImpactDiagram({ dPlane }: { dPlane: Record<string, number | null> }) {
  return (
    <div className="flex gap-2" data-testid="impact-diagram">
      <svg viewBox="0 0 110 100" className="w-1/2" role="img" aria-label="Top view: club path and face angle">
        <line x1={20} y1={50} x2={90} y2={50} stroke="#9ca3af" strokeDasharray="4" />
        {arrow(dPlane.club_path_deg, '#60a5fa', -1)}
        {arrow(dPlane.face_angle_deg, '#fb923c', -1)}
      </svg>
      <svg viewBox="0 0 110 100" className="w-1/2" role="img" aria-label="Side view: attack angle and dynamic loft">
        <line x1={20} y1={50} x2={90} y2={50} stroke="#9ca3af" strokeDasharray="4" />
        {arrow(dPlane.attack_angle_deg, '#60a5fa', 1)}
        {arrow(dPlane.dynamic_loft_deg, '#fb923c', 1)}
      </svg>
    </div>
  );
}

export function ImpactParametersPanel({ runId = null }: { runId?: string | null }) {
  const [units, setUnits] = useState<Units>('mph');
  const [heading, setHeading] = useState(0);
  const [leftHanded, setLeftHanded] = useState(false);
  const [data, setData] = useState<ImpactParametersResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    fetchImpactParameters(
      {
        runId,
        targetDir: targetDirFromHeading(heading),
        handedness: leftHanded ? 'left' : 'right',
        units,
      },
      controller.signal,
    )
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
  }, [runId, units, heading, leftHanded]);

  return (
    <section className="bg-gray-800 rounded-lg border border-gray-700 p-4" data-testid="impact-parameters-panel">
      <h3 className="text-sm font-semibold text-gray-300 mb-3">Impact Parameters</h3>
      <div className="flex flex-wrap gap-3 items-end mb-3 text-xs text-gray-300">
        <label>
          Units
          <select
            aria-label="Units"
            value={units}
            onChange={(e) => setUnits(e.target.value as Units)}
            className="ml-1 bg-gray-700 rounded px-1 py-0.5"
          >
            <option value="mph">mph</option>
            <option value="m/s">m/s</option>
          </select>
        </label>
        <label>
          Target heading (deg)
          <input
            type="number"
            aria-label="Target heading"
            value={heading}
            min={-180}
            max={180}
            onChange={(e) => setHeading(Number(e.target.value) || 0)}
            className="ml-1 w-20 bg-gray-700 rounded px-1 py-0.5"
          />
        </label>
        <label>
          <input
            type="checkbox"
            checked={leftHanded}
            onChange={(e) => setLeftHanded(e.target.checked)}
          />{' '}
          Left-handed
        </label>
      </div>
      {error && <div role="alert" className="text-xs text-red-400">Impact parameters unavailable: {error}</div>}
      {data && !data.available && (
        <div className="text-xs text-gray-400 italic" data-testid="impact-unavailable">
          unavailable: {data.reason}
        </div>
      )}
      {data && data.available && (
        <>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs mb-3">
            {(data.rows ?? []).map((row) => (
              <div key={row.key} className="contents">
                <dt className="text-gray-400">{row.label}</dt>
                <dd
                  className="text-gray-100 font-mono"
                  data-testid={`impact-${row.key}`}
                  title={row.value == null ? (row.reason ?? undefined) : undefined}
                >
                  {formatImpactValue(row)}
                </dd>
              </div>
            ))}
          </dl>
          <ImpactDiagram dPlane={data.d_plane ?? {}} />
          <Link to={impactExplorerHref(data.rows ?? [])} className="text-xs text-blue-400 underline">
            Open in Impact Explorer
          </Link>
        </>
      )}
    </section>
  );
}

export default ImpactParametersPanel;
