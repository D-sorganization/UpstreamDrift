/**
 * SwingObjectiveLab — React / Tauri page for comparing competing downswing objectives.
 *
 * Web counterpart of PyQt6 Swing Objective Lab (`src/launchers/adapters/swing_objective_lab_embed.py`):
 * allows configuring golfer preset and shared effort budget (duration, hub/wrist torque limits, node count),
 * running the direct collocation comparison via POST /tools/swing-objectives/compare,
 * and inspecting both the per-objective metrics table and cross-evaluation matrix.
 *
 * Accessibility & Safety:
 * - Every matrix cell is explicitly labelled with text (color is never the only encoding).
 * - Surfaces a plain-language degeneracy alert when `is_degenerate` is true.
 *
 * See issue #9128.
 */

import React, { useState, useEffect, useCallback } from 'react';
import { apiFetch } from '@/api/fetch';
import { WorkspaceShell } from '@/components/layout/WorkspaceShell';

export interface GolferPresetInfo {
  name: string;
  arm_mass_kg: number;
  shaft_mass_kg: number;
  clubhead_mass_kg: number;
  arm_length_m: number;
  club_length_m: number;
  top_arm_angle_rad: number;
  top_wrist_cock_rad: number;
  duration_s: number;
  hub_torque_nm: number;
  wrist_torque_nm: number;
  node_count: number;
}

export interface SwingComparisonPayload {
  schema_version: string;
  objective_keys: string[];
  units: Record<string, string>;
  raw_values: Record<string, Record<string, number>>;
  matrix: number[][];
  torque_saturation: Record<string, number[]>;
  swing_distance: number[][];
  is_degenerate: boolean;
  diagnostics: Record<string, Record<string, number | boolean | string>>;
}

const OBJECTIVE_LABELS: Record<string, string> = {
  clubhead_speed: 'Clubhead Speed',
  centrifugal: 'Centrifugal Release',
  coriolis: 'Coriolis Transfer',
  energy_transfer: 'Energy Transfer',
  impulse_transfer: 'Grip Impulse',
};

export function formatObjectiveName(key: string): string {
  return OBJECTIVE_LABELS[key] || key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

export function SwingObjectiveLabPage(): React.ReactElement {
  const [presets, setPresets] = useState<GolferPresetInfo[]>([]);
  const [selectedPresetIndex, setSelectedPresetIndex] = useState<number>(0);

  // Budget parameters
  const [duration, setDuration] = useState<number>(0.28);
  const [hubTorque, setHubTorque] = useState<number>(250.0);
  const [wristTorque, setWristTorque] = useState<number>(20.0);
  const [nodeCount, setNodeCount] = useState<number>(21);

  // Golfer parameters
  const [armMass, setArmMass] = useState<number>(5.0);
  const [armLength, setArmLength] = useState<number>(0.65);
  const [clubLength, setClubLength] = useState<number>(1.10);

  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [comparison, setComparison] = useState<SwingComparisonPayload | null>(null);

  // Load presets on mount
  useEffect(() => {
    let active = true;
    async function loadPresets() {
      try {
        const response = await apiFetch<{ presets: GolferPresetInfo[] }>(
          '/tools/swing-objectives/presets'
        );
        if (active && response.presets && response.presets.length > 0) {
          setPresets(response.presets);
          const first = response.presets[0];
          setDuration(first.duration_s);
          setHubTorque(first.hub_torque_nm);
          setWristTorque(first.wrist_torque_nm);
          setNodeCount(first.node_count);
          setArmMass(first.arm_mass_kg);
          setArmLength(first.arm_length_m);
          setClubLength(first.club_length_m);
        }
      } catch (err: any) {
        if (active) {
          // Fallback initial values if API preset fetch fails
          console.warn('Could not fetch presets from API:', err);
        }
      }
    }
    loadPresets();
    return () => {
      active = false;
    };
  }, []);

  const handlePresetChange = (index: number) => {
    setSelectedPresetIndex(index);
    const p = presets[index];
    if (p) {
      setDuration(p.duration_s);
      setHubTorque(p.hub_torque_nm);
      setWristTorque(p.wrist_torque_nm);
      setNodeCount(p.node_count);
      setArmMass(p.arm_mass_kg);
      setArmLength(p.arm_length_m);
      setClubLength(p.club_length_m);
    }
  };

  const handleRunComparison = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const payload = {
        arm_mass_kg: armMass,
        arm_length_m: armLength,
        club_length_m: clubLength,
        duration_s: duration,
        hub_torque_nm: hubTorque,
        wrist_torque_nm: wristTorque,
        node_count: nodeCount,
      };
      const result = await apiFetch<SwingComparisonPayload>('/tools/swing-objectives/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      setComparison(result);
    } catch (err: any) {
      setError(err?.message || 'Failed to compute swing objective comparison.');
    } finally {
      setLoading(false);
    }
  }, [armMass, armLength, clubLength, duration, hubTorque, wristTorque, nodeCount]);

  const handleExportJson = () => {
    if (!comparison) return;
    const blob = new Blob([JSON.stringify(comparison, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `swing_objective_comparison_${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <WorkspaceShell
      title="Swing Objective Lab"
      description="Evaluate competing downswing optimization objectives under a shared effort budget."
      actions={
        <div className="flex items-center gap-2">
          {comparison && (
            <button
              type="button"
              onClick={handleExportJson}
              className="px-3 py-1.5 text-sm font-medium bg-gray-700 hover:bg-gray-600 text-gray-100 rounded border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
              data-testid="export-json-btn"
            >
              Export JSON
            </button>
          )}
          <button
            type="button"
            onClick={handleRunComparison}
            disabled={loading}
            className="px-4 py-1.5 text-sm font-semibold bg-blue-600 hover:bg-blue-500 disabled:bg-blue-800 disabled:text-gray-400 text-white rounded shadow focus:outline-none focus:ring-2 focus:ring-blue-400 flex items-center gap-2"
            data-testid="run-comparison-btn"
          >
            {loading ? (
              <>
                <span className="inline-block w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                <span>Optimizing…</span>
              </>
            ) : (
              <span>Run Comparison</span>
            )}
          </button>
        </div>
      }
    >
      <div className="space-y-6 max-w-7xl mx-auto p-4 text-gray-200">
        {/* Controls Card */}
        <section className="bg-gray-800/80 border border-gray-700 rounded-lg p-5 shadow-sm space-y-4">
          <h2 className="text-lg font-semibold text-gray-100 border-b border-gray-700 pb-2">
            Golfer Configuration & Shared Effort Budget
          </h2>

          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <label htmlFor="preset-select" className="block text-xs font-medium text-gray-300 mb-1">
                Golfer Preset
              </label>
              <select
                id="preset-select"
                value={selectedPresetIndex}
                onChange={(e) => handlePresetChange(Number(e.target.value))}
                className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-100 focus:ring-2 focus:ring-blue-500"
                data-testid="preset-select"
              >
                {presets.length > 0 ? (
                  presets.map((p, i) => (
                    <option key={p.name} value={i}>
                      {p.name}
                    </option>
                  ))
                ) : (
                  <option value={0}>Tour Driver (Default)</option>
                )}
              </select>
            </div>

            <div>
              <label htmlFor="duration-input" className="block text-xs font-medium text-gray-300 mb-1">
                Downswing Duration (s)
              </label>
              <input
                id="duration-input"
                type="number"
                step="0.01"
                min="0.10"
                max="1.50"
                value={duration}
                onChange={(e) => setDuration(parseFloat(e.target.value) || 0)}
                className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-100 focus:ring-2 focus:ring-blue-500"
                data-testid="duration-input"
              />
            </div>

            <div>
              <label htmlFor="hub-torque-input" className="block text-xs font-medium text-gray-300 mb-1">
                Hub Torque Limit (N·m)
              </label>
              <input
                id="hub-torque-input"
                type="number"
                step="10"
                min="10"
                max="1000"
                value={hubTorque}
                onChange={(e) => setHubTorque(parseFloat(e.target.value) || 0)}
                className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-100 focus:ring-2 focus:ring-blue-500"
                data-testid="hub-torque-input"
              />
            </div>

            <div>
              <label htmlFor="wrist-torque-input" className="block text-xs font-medium text-gray-300 mb-1">
                Wrist Torque Limit (N·m)
              </label>
              <input
                id="wrist-torque-input"
                type="number"
                step="1"
                min="1"
                max="200"
                value={wristTorque}
                onChange={(e) => setWristTorque(parseFloat(e.target.value) || 0)}
                className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-100 focus:ring-2 focus:ring-blue-500"
                data-testid="wrist-torque-input"
              />
            </div>

            <div>
              <label htmlFor="node-count-input" className="block text-xs font-medium text-gray-300 mb-1">
                Collocation Nodes
              </label>
              <input
                id="node-count-input"
                type="number"
                step="2"
                min="9"
                max="51"
                value={nodeCount}
                onChange={(e) => setNodeCount(parseInt(e.target.value, 10) || 21)}
                className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-100 focus:ring-2 focus:ring-blue-500"
                data-testid="node-count-input"
              />
            </div>

            <div>
              <label htmlFor="arm-mass-input" className="block text-xs font-medium text-gray-300 mb-1">
                Lumped Arm Mass (kg)
              </label>
              <input
                id="arm-mass-input"
                type="number"
                step="0.5"
                min="1.0"
                max="20.0"
                value={armMass}
                onChange={(e) => setArmMass(parseFloat(e.target.value) || 5.0)}
                className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-100 focus:ring-2 focus:ring-blue-500"
                data-testid="arm-mass-input"
              />
            </div>

            <div>
              <label htmlFor="arm-length-input" className="block text-xs font-medium text-gray-300 mb-1">
                Arm Length (m)
              </label>
              <input
                id="arm-length-input"
                type="number"
                step="0.05"
                min="0.30"
                max="1.20"
                value={armLength}
                onChange={(e) => setArmLength(parseFloat(e.target.value) || 0.65)}
                className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-100 focus:ring-2 focus:ring-blue-500"
                data-testid="arm-length-input"
              />
            </div>

            <div>
              <label htmlFor="club-length-input" className="block text-xs font-medium text-gray-300 mb-1">
                Club Length (m)
              </label>
              <input
                id="club-length-input"
                type="number"
                step="0.05"
                min="0.50"
                max="1.50"
                value={clubLength}
                onChange={(e) => setClubLength(parseFloat(e.target.value) || 1.10)}
                className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-100 focus:ring-2 focus:ring-blue-500"
                data-testid="club-length-input"
              />
            </div>
          </div>
        </section>

        {/* Error Alert */}
        {error && (
          <div
            role="alert"
            className="p-4 bg-red-950/80 border border-red-800 text-red-200 rounded-lg text-sm"
            data-testid="error-alert"
          >
            <strong>Optimization Error: </strong>
            {error}
          </div>
        )}

        {/* Degeneracy Warning */}
        {comparison?.is_degenerate && (
          <div
            role="alert"
            className="p-4 bg-amber-950/80 border border-amber-600 text-amber-100 rounded-lg text-sm space-y-1"
            data-testid="degeneracy-warning"
          >
            <div className="font-bold flex items-center gap-2 text-amber-300">
              <span aria-hidden="true">⚠️</span> Configuration Degeneracy Detected
            </div>
            <p>
              Near the golfer&apos;s minimum downswing duration the constraints pin the trajectory and every
              objective returns the same swing, filling the matrix with 100% entries that read as unanimous mechanism
              agreement but are an artifact of the configuration.
            </p>
          </div>
        )}

        {/* Results section */}
        {comparison && (
          <div className="space-y-6" data-testid="comparison-results">
            {/* Cross-Evaluation Matrix */}
            <section className="bg-gray-800/80 border border-gray-700 rounded-lg p-5 shadow-sm space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-gray-700 pb-2">
                <h2 className="text-lg font-semibold text-gray-100">
                  Cross-Evaluation Matrix (Relative Efficiency %)
                </h2>
                <span className="text-xs text-gray-400">
                  Columns normalized so each objective&apos;s best is 100%
                </span>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm border-collapse" data-testid="cross-eval-matrix">
                  <thead>
                    <tr className="bg-gray-900/90 text-gray-300 text-xs uppercase tracking-wider">
                      <th scope="col" className="p-3 border border-gray-700">
                        Optimized For \ Evaluated Against
                      </th>
                      {comparison.objective_keys.map((colKey) => (
                        <th key={colKey} scope="col" className="p-3 border border-gray-700 text-center">
                          {formatObjectiveName(colKey)}
                          <div className="text-gray-400 font-normal normal-case">
                            ({comparison.units[colKey] || '—'})
                          </div>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {comparison.objective_keys.map((rowKey, rowIndex) => (
                      <tr key={rowKey} className="hover:bg-gray-750/50">
                        <th
                          scope="row"
                          className="p-3 border border-gray-700 font-semibold text-gray-200 bg-gray-900/50"
                        >
                          {formatObjectiveName(rowKey)}
                        </th>
                        {comparison.matrix[rowIndex]?.map((score, colIndex) => {
                          const colKey = comparison.objective_keys[colIndex];
                          const isDiagonal = rowIndex === colIndex;
                          const scoreText = score.toFixed(1);
                          return (
                            <td
                              key={colKey}
                              className={`p-3 border border-gray-700 text-center font-mono ${
                                isDiagonal
                                  ? 'bg-blue-950/40 text-blue-200 font-bold'
                                  : score >= 98.0
                                  ? 'text-emerald-300'
                                  : score >= 90.0
                                  ? 'text-amber-200'
                                  : 'text-gray-300'
                              }`}
                              data-testid={`matrix-cell-${rowKey}-${colKey}`}
                              aria-label={`Optimized for ${formatObjectiveName(
                                rowKey
                              )}, evaluated on ${formatObjectiveName(colKey)}: ${scoreText}%`}
                            >
                              <div className="text-sm">{scoreText}%</div>
                              <div className="text-[10px] text-gray-400 font-sans">
                                {isDiagonal ? 'Self (100%)' : `${scoreText}% of max`}
                              </div>
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            {/* Per-Objective Metric & Torque Saturation Table */}
            <section className="bg-gray-800/80 border border-gray-700 rounded-lg p-5 shadow-sm space-y-4">
              <h2 className="text-lg font-semibold text-gray-100 border-b border-gray-700 pb-2">
                Per-Objective Optimization Diagnostics & Torque Saturation
              </h2>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm border-collapse" data-testid="metrics-table">
                  <thead>
                    <tr className="bg-gray-900/90 text-gray-300 text-xs uppercase tracking-wider">
                      <th scope="col" className="p-3 border border-gray-700">
                        Objective
                      </th>
                      <th scope="col" className="p-3 border border-gray-700 text-right">
                        Attained Value
                      </th>
                      <th scope="col" className="p-3 border border-gray-700 text-right">
                        Units
                      </th>
                      <th scope="col" className="p-3 border border-gray-700 text-center">
                        Hub Torque Saturation
                      </th>
                      <th scope="col" className="p-3 border border-gray-700 text-center">
                        Wrist Torque Saturation
                      </th>
                      <th scope="col" className="p-3 border border-gray-700 text-center">
                        Collocation Status
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {comparison.objective_keys.map((key) => {
                      const raw = comparison.raw_values[key]?.[key];
                      const unit = comparison.units[key] || '';
                      const sat = comparison.torque_saturation[key] || [0, 0];
                      const diag = comparison.diagnostics[key] || {};
                      const hubSatPct = ((sat[0] || 0) * 100).toFixed(1);
                      const wristSatPct = ((sat[1] || 0) * 100).toFixed(1);

                      return (
                        <tr key={key} className="hover:bg-gray-750/50">
                          <td className="p-3 border border-gray-700 font-semibold text-gray-200">
                            {formatObjectiveName(key)}
                          </td>
                          <td className="p-3 border border-gray-700 text-right font-mono text-gray-100">
                            {typeof raw === 'number' ? raw.toFixed(3) : '—'}
                          </td>
                          <td className="p-3 border border-gray-700 text-right text-gray-400 font-mono">
                            {unit}
                          </td>
                          <td className="p-3 border border-gray-700 text-center">
                            <span className="font-mono text-gray-200">{hubSatPct}%</span>
                            <span className="text-xs text-gray-400 block">at bound</span>
                          </td>
                          <td className="p-3 border border-gray-700 text-center">
                            <span className="font-mono text-gray-200">{wristSatPct}%</span>
                            <span className="text-xs text-gray-400 block">at bound</span>
                          </td>
                          <td className="p-3 border border-gray-700 text-center">
                            {diag.success ? (
                              <span className="px-2 py-0.5 rounded text-xs bg-emerald-950 text-emerald-300 border border-emerald-800">
                                Converged
                              </span>
                            ) : (
                              <span className="px-2 py-0.5 rounded text-xs bg-red-950 text-red-300 border border-red-800">
                                Infeasible / Unconverged
                              </span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </section>
          </div>
        )}
      </div>
    </WorkspaceShell>
  );
}
