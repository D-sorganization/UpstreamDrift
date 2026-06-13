/**
 * BallFlight — multi-model shot tracer / ball-flight comparison page.
 *
 * Web counterpart of the desktop Shot Tracer (`src/launchers/_shot_tracer_gui.py`):
 * a unit-labelled launch-conditions form (units are EXPLICIT on every field —
 * the deg-vs-rad / RPM-vs-rad/s convention bugs of issue #7246 make this
 * mandatory), a flight-model multi-select fed by `GET /tools/ball-flight/models`
 * (the same registry the desktop tracer iterates), a 3D trajectory overlay,
 * 2D side/top profiles, and a per-model metrics table.
 *
 * See issue #7456.
 */

import { useState, useEffect, useCallback, useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { apiFetch } from '@/api/fetch';
import { HelpfulField } from '@/components/ux/HelpfulField';
import { getFieldMetadata } from '@/ux/fieldMetadata';
import { BallFlightScene3D } from '@/components/visualization/BallFlightScene3D';
import { WorkspaceShell } from '@/components/layout/WorkspaceShell';
import {
  LAUNCH_FIELD_IDS,
  invalidLaunchFields,
  modelColor,
  type LaunchFieldId,
} from './ballFlightModel';

/** Flight-model metadata from GET /tools/ball-flight/models. See issue #7456 */
export interface FlightModelInfo {
  key: string;
  name: string;
  description: string;
  reference: string;
}

/** Single trajectory sample from the API. */
export interface TrajectorySample {
  time_s: number;
  position_m: number[];
  velocity_mps: number[];
}

/** Scalar metrics per model. */
export interface BallFlightSummary {
  carry_m: number;
  apex_m: number;
  flight_time_s: number;
  landing_angle_deg: number;
  lateral_deviation_m: number;
}

/** Per-model simulation result. */
export interface BallFlightModelResult {
  model_name: string;
  model_key: string;
  trajectory: TrajectorySample[];
  summary: BallFlightSummary;
}

/** Full simulate response (top-level mirrors first model; results has all). */
export interface BallFlightSimulationResponse extends BallFlightModelResult {
  results: BallFlightModelResult[];
}

/** Map registry field ids to API request keys (units stay explicit). */
const FIELD_TO_API_KEY: Record<LaunchFieldId, string> = {
  'ball_flight.ball_speed': 'ball_speed_mps',
  'ball_flight.launch_angle': 'launch_angle_deg',
  'ball_flight.azimuth_angle': 'azimuth_angle_deg',
  'ball_flight.spin_rate': 'spin_rate_rpm',
  'ball_flight.spin_axis_tilt': 'spin_axis_tilt_deg',
  'ball_flight.wind_speed': 'wind_speed_mps',
  'ball_flight.wind_direction': 'wind_direction_deg',
};

function defaultLaunchValues(): Record<LaunchFieldId, string> {
  return Object.fromEntries(
    LAUNCH_FIELD_IDS.map((fieldId) => [
      fieldId,
      String(getFieldMetadata(fieldId).default),
    ]),
  ) as Record<LaunchFieldId, string>;
}

/** 2D profile chart shared by the side (height) and top (lateral) views. */
function ProfileChart({
  results,
  colorByKey,
  yIndex,
  yLabel,
  testId,
}: {
  results: BallFlightModelResult[];
  colorByKey: Record<string, string>;
  yIndex: 1 | 2;
  yLabel: string;
  testId: string;
}) {
  return (
    <div className="h-56" data-testid={testId}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart margin={{ top: 8, right: 16, bottom: 16, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis
            dataKey="x"
            type="number"
            stroke="#9ca3af"
            label={{
              value: 'Downrange (m)',
              position: 'insideBottom',
              offset: -10,
              fill: '#9ca3af',
            }}
          />
          <YAxis
            dataKey="y"
            type="number"
            stroke="#9ca3af"
            label={{
              value: yLabel,
              angle: -90,
              position: 'insideLeft',
              fill: '#9ca3af',
            }}
          />
          <Tooltip
            contentStyle={{ backgroundColor: '#1f2937', border: 'none' }}
          />
          {results.map((r) => (
            <Line
              key={r.model_key}
              name={r.model_name}
              data={r.trajectory.map((s) => ({
                x: s.position_m[0],
                y: s.position_m[yIndex],
              }))}
              dataKey="y"
              dot={false}
              stroke={colorByKey[r.model_key]}
              strokeWidth={2}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/**
 * BallFlightPage — full shot-tracer tool page.
 *
 * See issue #7456.
 */
export function BallFlightPage() {
  const [values, setValues] = useState<Record<LaunchFieldId, string>>(
    defaultLaunchValues,
  );
  const [models, setModels] = useState<FlightModelInfo[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [results, setResults] = useState<BallFlightModelResult[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load the shared flight-model registry once.
  useEffect(() => {
    let cancelled = false;
    apiFetch<{ models: FlightModelInfo[] }>('/api/tools/ball-flight/models')
      .then((data) => {
        if (cancelled) return;
        setModels(data.models);
        // Default: first two models pre-selected, like the desktop tracer.
        setSelected(data.models.slice(0, 2).map((m) => m.key));
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(
          err instanceof Error ? err.message : 'Failed to load flight models',
        );
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const invalidFields = useMemo(() => invalidLaunchFields(values), [values]);

  const toggleModel = useCallback((key: string) => {
    setSelected((prev) =>
      prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key],
    );
  }, []);

  const handleSimulate = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const body: Record<string, unknown> = { models: selected };
      for (const fieldId of LAUNCH_FIELD_IDS) {
        body[FIELD_TO_API_KEY[fieldId]] = Number(values[fieldId]);
      }
      const data = await apiFetch<BallFlightSimulationResponse>(
        '/api/tools/ball-flight/simulate',
        { method: 'POST', body: JSON.stringify(body) },
      );
      setResults(data.results);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Simulation failed');
    } finally {
      setLoading(false);
    }
  }, [selected, values]);

  // Stable color per model key, following selection order.
  const colorByKey = useMemo(() => {
    const map: Record<string, string> = {};
    selected.forEach((key, i) => {
      map[key] = modelColor(i);
    });
    (results ?? []).forEach((r, i) => {
      map[r.model_key] = map[r.model_key] ?? modelColor(i);
    });
    return map;
  }, [selected, results]);

  const trajectories3d = useMemo(
    () =>
      (results ?? []).map((r) => ({
        modelKey: r.model_key,
        modelName: r.model_name,
        color: colorByKey[r.model_key],
        positions: r.trajectory.map((s) => s.position_m),
      })),
    [results, colorByKey],
  );

  const canSimulate =
    !loading && selected.length > 0 && invalidFields.length === 0;

  const leftPanel = (
    <div className="flex flex-col flex-1 min-h-0">
        <div className="p-4 border-b border-gray-700">
          <h1 className="heading-page mb-1">Shot Tracer</h1>
          <p className="text-xs text-gray-400">
            Multi-model ball-flight comparison
          </p>
        </div>

        <div className="p-4 border-b border-gray-700 space-y-3">
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
            Launch Conditions
          </h3>
          {LAUNCH_FIELD_IDS.map((fieldId) => (
            <HelpfulField
              key={fieldId}
              fieldId={fieldId}
              value={values[fieldId]}
              onChange={(raw) =>
                setValues((prev) => ({ ...prev, [fieldId]: raw }))
              }
              disabled={loading}
            />
          ))}
          {invalidFields.length > 0 && (
            <div
              className="text-xs text-amber-400 bg-amber-900/20 p-2 rounded"
              data-testid="validation-warning"
              role="alert"
            >
              Out-of-range or invalid:{' '}
              {invalidFields
                .map((id) => getFieldMetadata(id).label)
                .join(', ')}
            </div>
          )}
        </div>

        <div className="p-4 border-b border-gray-700 space-y-2">
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
            Flight Models
          </h3>
          {models.length === 0 && !error && (
            <p className="text-xs text-gray-400 italic">Loading models…</p>
          )}
          {models.map((model) => (
            <label
              key={model.key}
              className="flex items-center gap-2 text-sm text-gray-300 cursor-pointer"
              title={`${model.description}\nRef: ${model.reference}`}
            >
              <input
                type="checkbox"
                checked={selected.includes(model.key)}
                onChange={() => toggleModel(model.key)}
                aria-label={`Toggle ${model.name} model`}
              />
              <span
                className="w-2 h-2 rounded-full inline-block"
                style={{
                  backgroundColor: colorByKey[model.key] ?? '#6b7280',
                }}
              />
              {model.name}
            </label>
          ))}
        </div>

        <div className="p-4 space-y-2">
          <button
            onClick={handleSimulate}
            disabled={!canSimulate}
            className="w-full py-2 px-4 bg-green-600 hover:bg-green-500 disabled:bg-gray-600 text-white text-sm font-medium rounded transition-colors"
            data-testid="simulate-btn"
          >
            {loading ? 'Simulating…' : 'Simulate'}
          </button>
          {error && (
            <div
              className="text-xs text-red-400 bg-red-900/20 p-2 rounded"
              data-testid="error-message"
              role="alert"
            >
              {error}
            </div>
          )}
        </div>
    </div>
  );

  const mainContent = (
    <div className="flex-1 flex flex-col bg-gray-950 min-w-0 min-h-0 overflow-y-auto">
        <div className="h-64 sm:h-80 flex-shrink-0">
          <BallFlightScene3D trajectories={trajectories3d} />
        </div>
        {results && results.length > 0 ? (
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-4 p-4">
            <div>
              <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
                Side Profile — height (m) vs downrange (m)
              </h3>
              <ProfileChart
                results={results}
                colorByKey={colorByKey}
                yIndex={2}
                yLabel="Height (m)"
                testId="side-profile-chart"
              />
            </div>
            <div>
              <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
                Top View — lateral (m) vs downrange (m)
              </h3>
              <ProfileChart
                results={results}
                colorByKey={colorByKey}
                yIndex={1}
                yLabel="Lateral (m)"
                testId="top-profile-chart"
              />
            </div>
          </div>
        ) : (
          <div className="p-8 text-sm text-gray-400 italic text-center">
            Set launch conditions, pick at least one flight model, and click
            Simulate to compare trajectories.
          </div>
        )}
    </div>
  );

  const rightPanel = (
    <div className="flex flex-col flex-1 min-h-0">
        <div className="p-4">
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
            Metrics by Model
          </h3>
          {results && results.length > 0 ? (
            <table className="w-full text-xs" data-testid="metrics-table">
              <thead>
                <tr className="text-gray-400 text-left">
                  <th className="py-1 pr-2">Model</th>
                  <th className="py-1 pr-2">Carry (m)</th>
                  <th className="py-1 pr-2">Apex (m)</th>
                  <th className="py-1 pr-2">Time (s)</th>
                  <th className="py-1">Offline (m)</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r) => (
                  <tr
                    key={r.model_key}
                    className="text-gray-200 border-t border-gray-700"
                    data-testid={`metrics-row-${r.model_key}`}
                  >
                    <td className="py-1.5 pr-2">
                      <span
                        className="w-2 h-2 rounded-full inline-block mr-1.5"
                        style={{ backgroundColor: colorByKey[r.model_key] }}
                      />
                      {r.model_name}
                    </td>
                    <td className="py-1.5 pr-2 font-mono">
                      {r.summary.carry_m.toFixed(1)}
                    </td>
                    <td className="py-1.5 pr-2 font-mono">
                      {r.summary.apex_m.toFixed(1)}
                    </td>
                    <td className="py-1.5 pr-2 font-mono">
                      {r.summary.flight_time_s.toFixed(2)}
                    </td>
                    <td className="py-1.5 font-mono">
                      {r.summary.lateral_deviation_m.toFixed(1)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="text-xs text-gray-400 italic text-center py-4">
              Run a simulation to compare carry, apex, flight time, and
              offline distance across models.
            </p>
          )}
        </div>
    </div>
  );

  return (
    <WorkspaceShell
      leftPanel={leftPanel}
      rightPanel={rightPanel}
      leftPanelLabel="Launch Conditions"
      rightPanelLabel="Metrics by Model"
    >
      {mainContent}
    </WorkspaceShell>
  );
}
