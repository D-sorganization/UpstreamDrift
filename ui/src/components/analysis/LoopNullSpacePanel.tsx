/**
 * LoopNullSpacePanel — React surface for Closed-Loop Null-Space Exploration
 * (NSX-44, issue #12576; parity with PyQt6 `src/tools/grip_wrench_plots/gui.py`, NSX-43 #12575).
 *
 * Consumes `/analysis/loop-nullspace/jobs` (`POST` create, `GET` poll with
 * partial invariants first, `DELETE` cancel) and renders the 4 NSX panels
 * (`Invariants`, `Envelopes`, `Hand Split`, `Diagnostics`, with allocation
 * representatives overlaid on their envelope bands) directly from the
 * framework-free plot-model JSON with zero client-side math.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

export type SegmentDisplayClass = 'DETERMINED_SIGN' | 'INDETERMINATE' | 'UNBOUNDED';

export interface EnvelopeSegment {
  start_index: number;
  end_index: number;
  display_class: SegmentDisplayClass;
  binding_hypotheses: string[];
}

export interface PlotLineSeries {
  quantity_id: string;
  label: string;
  units: string;
  values: Array<number | null>;
  badge_text: string;
  render_mode: string;
  envelope_ref_id?: string | null;
}

export interface EnvelopeBandSeries {
  quantity_id: string;
  label: string;
  units: string;
  lower: Array<number | null>;
  upper: Array<number | null>;
  chebyshev_center: Array<number | null>;
  chebyshev_radius: Array<number | null>;
  segments: EnvelopeSegment[];
  badge_text: string;
  render_mode: string;
  representative?: Array<number | null> | null;
}

export interface PlotPanelModel {
  panel_id: string;
  title: string;
  line_series: PlotLineSeries[];
  envelope_series: EnvelopeBandSeries[];
  metrics: Record<string, number[]>;
}

export interface NSXPlotModel {
  schema_version: string;
  report_id: string;
  preset_name: string;
  time_s: number[];
  panels: {
    invariants: PlotPanelModel;
    envelopes: PlotPanelModel;
    hand_split: PlotPanelModel;
    allocation_representative?: PlotPanelModel;
    diagnostics: PlotPanelModel;
  };
}

export interface NSXJobStatusResponse {
  job_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress_percent?: number;
  compute_cache_key?: string;
  cache_hit?: boolean;
  partial_invariants?: Record<string, { label?: string; units?: string; values?: Array<number | null>; allocation_dependence?: string }>;
  plot_model?: NSXPlotModel | null;
  report?: Record<string, unknown> | null;
  error?: string | null;
}

export const NSX_PRESETS = [
  {
    name: 'capacity_min_effort',
    label: 'Capacity-Normalized Minimum Quadratic Effort Representative',
  },
  {
    name: 'envelope_only',
    label: 'Static Feasible Envelopes Only (No Representative Selection)',
  },
  {
    name: 'smooth_capacity',
    label: 'Rate-Smoothed Capacity-Normalized Effort Representative',
  },
] as const;

export const NSX_HELP_GUIDE_PATH = 'docs/user_guide/loop_nullspace_exploration.md';

export type NSXTabKey = 'invariants' | 'envelopes' | 'hand_split' | 'diagnostics';

interface LoopNullSpacePanelProps {
  /** Optional pre-loaded plot model (used by cross-UI golden fixture tests). */
  initialPlotModel?: NSXPlotModel | null;
  /** Polling interval in milliseconds for running jobs. */
  pollIntervalMs?: number;
}

function adaptModelFromReport(report: Record<string, unknown>, presetFallback: string): NSXPlotModel {
  if ('panels' in report && 'time_s' in report) {
    return report as unknown as NSXPlotModel;
  }
  const timeS = (report.time_s as number[]) ?? [];
  const invariantsMap = (report.invariants as Record<string, Record<string, unknown>>) ?? {};
  const envelopesMap = (report.envelopes as Record<string, Record<string, unknown>>) ?? {};
  const allocSec = (report.allocation as Record<string, unknown> | null) ?? null;
  const repsMap =
    allocSec && typeof allocSec === 'object'
      ? ((allocSec.representatives as Record<string, Record<string, unknown>>) ?? {})
      : {};
  const diagSec = (report.diagnostics as Record<string, number[]>) ?? {};
  const admSec = (report.admission as Record<string, number[]>) ?? {};

  const invLines: PlotLineSeries[] = Object.entries(invariantsMap).map(([qid, rec]) => ({
    quantity_id: qid,
    label: String(rec.label ?? qid),
    units: String(rec.units ?? ''),
    values: (rec.values as Array<number | null>) ?? [],
    badge_text: 'MODEL_CONDITIONED_INVARIANT',
    render_mode: 'INVARIANT_LINE_WITH_BADGE',
    envelope_ref_id: null,
  }));

  const torqueBands: EnvelopeBandSeries[] = [];
  const splitBands: EnvelopeBandSeries[] = [];
  Object.entries(envelopesMap).forEach(([qid, env]) => {
    const repRec = repsMap[qid];
    const signClasses = (env.sign_class as string[]) ?? [];
    const duals = (env.active_constraint_duals as Array<Record<string, number>>) ?? [];
    const segments: EnvelopeSegment[] = signClasses.map((sc, idx) => ({
      start_index: idx,
      end_index: idx,
      display_class:
        sc === 'UNBOUNDED'
          ? 'UNBOUNDED'
          : sc === 'INDETERMINATE'
            ? 'INDETERMINATE'
            : 'DETERMINED_SIGN',
      binding_hypotheses: Object.keys(duals[idx] ?? {}).sort(),
    }));
    const band: EnvelopeBandSeries = {
      quantity_id: qid,
      label: String(env.label ?? qid),
      units: String(env.units ?? ''),
      lower: (env.lower as Array<number | null>) ?? [],
      upper: (env.upper as Array<number | null>) ?? [],
      chebyshev_center: (env.chebyshev_center as Array<number | null>) ?? [],
      chebyshev_radius: (env.chebyshev_radius as Array<number | null>) ?? [],
      segments,
      badge_text: `PRIOR_CONDITIONED [${String(report.preset ?? presetFallback)}]`,
      render_mode: repRec ? 'BAND_WITH_REPRESENTATIVE' : 'ENVELOPE_BAND',
      representative: repRec ? ((repRec.values as Array<number | null>) ?? null) : null,
    };
    if (qid.startsWith('s1_') || qid.startsWith('s2_') || qid.startsWith('s3_') || qid.startsWith('s4_') || qid.startsWith('s5_') || qid.startsWith('s6_')) {
      splitBands.push(band);
    } else {
      torqueBands.push(band);
    }
  });

  return {
    schema_version: String(report.schema_version ?? '1.0.0'),
    report_id: String(report.report_id ?? 'nsx-report'),
    preset_name: String(report.preset ?? presetFallback),
    time_s: timeS,
    panels: {
      invariants: {
        panel_id: 'invariants',
        title: 'Allocation-Invariant Quantities',
        line_series: invLines,
        envelope_series: [],
        metrics: {},
      },
      envelopes: {
        panel_id: 'envelopes',
        title: 'Feasible Loop-Torque Envelopes',
        line_series: [],
        envelope_series: torqueBands,
        metrics: {},
      },
      hand_split: {
        panel_id: 'hand_split',
        title: 'Grip-Frame Hand-Split Coordinates (s1..s6)',
        line_series: [],
        envelope_series: splitBands,
        metrics: {},
      },
      diagnostics: {
        panel_id: 'diagnostics',
        title: 'Closure Rank, Conditioning, and Admission Residuals',
        line_series: [],
        envelope_series: [],
        metrics: {
          rank_jc: diagSec.rank_jc ?? [],
          condition_number_jc: diagSec.condition_number_jc ?? [],
          equality_nullity: diagSec.equality_nullity ?? [],
          chebyshev_radius: diagSec.chebyshev_radius ?? [],
          phi_scaled: admSec.phi_scaled ?? [],
          phidot_scaled: admSec.phidot_scaled ?? [],
          phiddot_scaled: admSec.phiddot_scaled ?? [],
        },
      },
    },
  };
}

function EnvelopeSeriesCard({ band }: { band: EnvelopeBandSeries }) {
  const showRepresentative = Array.isArray(band.representative) && band.representative.length > 0;
  return (
    <div
      className="bg-gray-900/70 rounded border border-gray-700 p-3 mb-2"
      data-testid={`nsx-band-${band.quantity_id}`}
    >
      <div className="flex flex-wrap items-center justify-between gap-2 mb-1">
        <span className="text-xs font-semibold text-gray-200">
          {band.label} ({band.units})
        </span>
        <span
          className="text-[10px] font-mono bg-sky-900/50 text-sky-300 px-2 py-0.5 rounded"
          data-testid={`nsx-badge-${band.quantity_id}`}
        >
          {band.badge_text}
        </span>
      </div>
      <div className="text-[11px] text-gray-400 font-mono mb-1">
        Band: [{band.lower.map((v) => (v === null ? 'null' : v)).join(', ')}] .. [
        {band.upper.map((v) => (v === null ? 'null' : v)).join(', ')}]
      </div>
      {showRepresentative && (
        <div
          className="text-[11px] text-emerald-300 font-mono mb-1"
          data-testid={`nsx-rep-${band.quantity_id}`}
        >
          Overlaid representative ({band.render_mode}): [
          {band.representative?.map((v) => (v === null ? 'null' : v)).join(', ')}]
        </div>
      )}
      <div className="flex flex-wrap gap-1.5 mt-1">
        {band.segments.map((seg, idx) => (
          <span
            key={`${band.quantity_id}-seg-${idx}`}
            data-testid={`nsx-segment-${band.quantity_id}-${seg.start_index}`}
            className={`text-[10px] px-1.5 py-0.5 rounded border ${
              seg.display_class === 'INDETERMINATE'
                ? 'bg-amber-900/40 border-amber-600 text-amber-200'
                : seg.display_class === 'UNBOUNDED'
                  ? 'bg-red-900/40 border-red-600 text-red-200'
                  : 'bg-gray-800 border-gray-600 text-gray-300'
            }`}
          >
            [{seg.start_index}..{seg.end_index}] {seg.display_class}
            {seg.binding_hypotheses.length > 0
              ? ` (${seg.binding_hypotheses.join(', ')})`
              : ''}
          </span>
        ))}
      </div>
    </div>
  );
}

export function LoopNullSpacePanel({
  initialPlotModel = null,
  pollIntervalMs = 50,
}: LoopNullSpacePanelProps) {
  const [preset, setPreset] = useState<string>(
    initialPlotModel?.preset_name ?? 'capacity_min_effort',
  );
  const [activeTab, setActiveTab] = useState<NSXTabKey>('invariants');
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<string>(
    initialPlotModel ? 'completed' : 'idle',
  );
  const [partialInvariants, setPartialInvariants] = useState<
    NSXJobStatusResponse['partial_invariants'] | null
  >(null);
  const [plotModel, setPlotModel] = useState<NSXPlotModel | null>(initialPlotModel);
  const [error, setError] = useState<string | null>(null);
  const cancelledRef = useRef<boolean>(false);

  useEffect(() => {
    if (initialPlotModel) {
      setPlotModel(initialPlotModel);
      setPreset(initialPlotModel.preset_name);
      setJobStatus('completed');
    }
  }, [initialPlotModel]);

  const startJob = useCallback(
    async (selectedPreset?: string) => {
      const targetPreset = selectedPreset ?? preset;
      cancelledRef.current = false;
      setError(null);
      setPartialInvariants(null);
      setJobStatus('running');
      try {
        const postResp = await fetch('/analysis/loop-nullspace/jobs', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            engine: 'pinocchio',
            preset: targetPreset,
            options: {},
          }),
        });
        if (!postResp.ok) {
          throw new Error(`Failed to start NSX job (${postResp.status})`);
        }
        const created = (await postResp.json()) as NSXJobStatusResponse;
        setJobId(created.job_id);
        if (created.partial_invariants) {
          setPartialInvariants(created.partial_invariants);
        }
        if (created.status === 'completed') {
          if (created.plot_model) {
            setPlotModel(created.plot_model);
          } else if (created.report) {
            setPlotModel(adaptModelFromReport(created.report, targetPreset));
          }
          setJobStatus('completed');
          return;
        }

        for (let attempt = 0; attempt < 20; attempt++) {
          if (cancelledRef.current) return;
          const pollResp = await fetch(
            `/analysis/loop-nullspace/jobs/${created.job_id}`,
            { method: 'GET' },
          );
          if (!pollResp.ok) {
            throw new Error(`Failed to poll NSX job (${pollResp.status})`);
          }
          const polled = (await pollResp.json()) as NSXJobStatusResponse;
          if (cancelledRef.current) return;
          if (polled.partial_invariants) {
            setPartialInvariants(polled.partial_invariants);
          }
          if (polled.status === 'completed') {
            if (polled.plot_model) {
              setPlotModel(polled.plot_model);
            } else if (polled.report) {
              setPlotModel(adaptModelFromReport(polled.report, targetPreset));
            }
            setJobStatus('completed');
            return;
          }
          if (polled.status === 'cancelled') {
            setJobStatus('cancelled');
            return;
          }
          if (polled.status === 'failed') {
            setError(polled.error ?? 'NSX job failed');
            setJobStatus('failed');
            return;
          }
          await new Promise((resolve) => setTimeout(resolve, pollIntervalMs));
        }
      } catch (err) {
        if (!cancelledRef.current) {
          setError(err instanceof Error ? err.message : String(err));
          setJobStatus('failed');
        }
      }
    },
    [preset, pollIntervalMs],
  );

  const cancelJob = useCallback(async () => {
    cancelledRef.current = true;
    setJobStatus('cancelled');
    if (!jobId) return;
    try {
      await fetch(`/analysis/loop-nullspace/jobs/${jobId}`, {
        method: 'DELETE',
      });
    } catch {
      // Cancellation state is already updated locally
    }
  }, [jobId]);

  const handlePresetChange = (nextPreset: string) => {
    setPreset(nextPreset);
    if (plotModel) {
      setPlotModel({
        ...plotModel,
        preset_name: nextPreset,
      });
    }
  };

  return (
    <section
      className="bg-gray-800 rounded-lg border border-gray-700 p-4"
      data-testid="loop-nullspace-panel"
    >
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
        <div>
          <h3 className="text-sm font-semibold text-gray-200">
            Closed-Loop Null-Space Exploration (NSX)
          </h3>
          <p className="text-xs text-gray-400">
            Band-first feasible torque and 6-DOF grip-split envelopes with model-conditioned invariants
          </p>
        </div>
        <a
          href={NSX_HELP_GUIDE_PATH}
          data-testid="nsx-help-link"
          className="text-xs text-sky-400 hover:underline"
        >
          Interpretation Guide ({NSX_HELP_GUIDE_PATH})
        </a>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3 mb-3 text-xs">
        <label className="text-gray-300">
          NSX Preset
          <select
            aria-label="NSX Preset"
            data-testid="nsx-preset-select"
            value={preset}
            onChange={(e) => handlePresetChange(e.target.value)}
            className="ml-2 bg-gray-700 text-gray-100 rounded px-2 py-1 border border-gray-600"
          >
            {NSX_PRESETS.map((p) => (
              <option key={p.name} value={p.name}>
                {p.name} — {p.label}
              </option>
            ))}
          </select>
        </label>

        <button
          type="button"
          data-testid="nsx-run-button"
          onClick={() => void startJob(preset)}
          className="px-3 py-1 bg-blue-600 hover:bg-blue-500 text-white rounded"
        >
          Run NSX
        </button>

        <button
          type="button"
          data-testid="nsx-cancel-button"
          disabled={jobStatus !== 'running'}
          onClick={() => void cancelJob()}
          className="px-3 py-1 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 text-gray-200 rounded"
        >
          Cancel NSX
        </button>

        <span className="text-gray-400 font-mono" data-testid="nsx-job-status">
          status: {jobStatus}
        </span>
      </div>

      {error && (
        <div role="alert" data-testid="nsx-error" className="text-xs text-red-400 mb-2">
          {error}
        </div>
      )}

      {/* Partial invariants shown first while job is running */}
      {partialInvariants && jobStatus === 'running' && (
        <div
          className="bg-sky-950/40 border border-sky-700 rounded p-2 mb-3 text-xs text-sky-200"
          data-testid="nsx-partial-invariants"
        >
          <div className="font-semibold mb-1">Partial Invariants Ready (solving envelopes...)</div>
          {Object.entries(partialInvariants).map(([qid, rec]) => (
            <div key={qid} className="font-mono text-[11px]">
              {rec.label ?? qid}: [{(rec.values ?? []).join(', ')}] {rec.units ?? ''}
            </div>
          ))}
        </div>
      )}

      {/* Tab selector */}
      <div className="flex gap-2 border-b border-gray-700 pb-2 mb-3 text-xs">
        {(['invariants', 'envelopes', 'hand_split', 'diagnostics'] as const).map((tab) => (
          <button
            key={tab}
            type="button"
            data-testid={`nsx-tab-${tab}`}
            onClick={() => setActiveTab(tab)}
            className={`px-2.5 py-1 rounded ${
              activeTab === tab
                ? 'bg-blue-600 text-white'
                : 'bg-gray-700/60 text-gray-300 hover:bg-gray-700'
            }`}
          >
            {tab === 'invariants'
              ? 'Invariants'
              : tab === 'envelopes'
                ? 'Envelopes'
                : tab === 'hand_split'
                  ? 'Hand Split'
                  : 'Diagnostics'}
          </button>
        ))}
      </div>

      {plotModel && (
        <div className="space-y-4" data-testid="nsx-plot-model-view">
          {/* 1. Invariants Panel */}
          <div data-testid="nsx-panel-invariants">
            <h4 className="text-xs font-semibold text-gray-300 mb-2">
              {plotModel.panels.invariants.title}
            </h4>
            {plotModel.panels.invariants.line_series.map((series) => (
              <div
                key={series.quantity_id}
                data-testid={`nsx-invariant-${series.quantity_id}`}
                className="bg-gray-900/70 rounded border border-gray-700 p-2.5 mb-2"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="text-xs text-gray-200 font-medium">
                    {series.label} ({series.units})
                  </span>
                  <span
                    className="text-[10px] font-mono bg-emerald-900/50 text-emerald-300 px-2 py-0.5 rounded"
                    data-testid={`nsx-badge-${series.quantity_id}`}
                  >
                    {series.badge_text}
                  </span>
                </div>
                <div className="text-[11px] text-gray-400 font-mono mt-1">
                  Values: [{series.values.map((v) => (v === null ? 'null' : v)).join(', ')}]
                </div>
              </div>
            ))}
          </div>

          {/* 2. Envelopes Panel */}
          <div data-testid="nsx-panel-envelopes">
            <h4 className="text-xs font-semibold text-gray-300 mb-2">
              {plotModel.panels.envelopes.title}
            </h4>
            {plotModel.panels.envelopes.envelope_series.map((band) => (
              <EnvelopeSeriesCard key={band.quantity_id} band={band} />
            ))}
          </div>

          {/* 3. Hand Split Panel */}
          <div data-testid="nsx-panel-hand_split">
            <h4 className="text-xs font-semibold text-gray-300 mb-2">
              {plotModel.panels.hand_split.title}
            </h4>
            {plotModel.panels.hand_split.envelope_series.map((band) => (
              <EnvelopeSeriesCard key={band.quantity_id} band={band} />
            ))}
          </div>

          {/* 4. Diagnostics Panel */}
          <div data-testid="nsx-panel-diagnostics">
            <h4 className="text-xs font-semibold text-gray-300 mb-2">
              {plotModel.panels.diagnostics.title}
            </h4>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs">
              {Object.entries(plotModel.panels.diagnostics.metrics).map(([key, vals]) => (
                <div
                  key={key}
                  data-testid={`nsx-diag-${key}`}
                  className="bg-gray-900/70 rounded border border-gray-700 p-2"
                >
                  <div className="text-gray-400 font-mono text-[11px]">{key}</div>
                  <div className="text-gray-200 font-mono text-[11px]">
                    [{(vals ?? []).join(', ')}]
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

export default LoopNullSpacePanel;
