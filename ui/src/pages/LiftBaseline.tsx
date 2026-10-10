/**
 * Lift Baseline page (LIFT-8 slice 3, #11748).
 *
 * Web counterpart of the desktop `LiftBaselinePanel`: shows the LIFT-1
 * cross-engine baseline for the five canonical lifts — per-engine structure
 * and start-pose measurements, cross-engine position comparisons flagged
 * against the position tolerance, per-engine phases, and the known-gap
 * ledger for the selected lift.
 */

import { useEffect, useState, type ReactNode } from 'react';
import { WorkspaceShell } from '@/components/layout/WorkspaceShell';
import {
  fetchLiftBaseline,
  fetchLiftBaselineLift,
  formatMeasurement,
  PAIR_METRICS,
  type EngineView,
  type GapEntry,
  type LiftBaselineMetadata,
  type LiftView,
  type NumericField,
  type PairMetric,
} from '@/api/lifting';

type LoadState = 'loading' | 'ready' | 'error';

function statusBadgeClass(status: PairMetric['status']): string {
  if (status === 'pass') return 'bg-emerald-700 text-emerald-50';
  if (status === 'fail') return 'bg-red-800 text-red-50';
  return 'bg-gray-700 text-gray-100';
}

function yesNo(flag: boolean | null): string {
  if (flag == null) return 'unavailable';
  return flag ? 'yes' : 'no';
}

function measurementCell(field: NumericField, digits: number, unit: string): ReactNode {
  return (
    <span title={field.reason ?? undefined}>{formatMeasurement(field, digits, unit)}</span>
  );
}

function EnginesTable({ engines }: { engines: EngineView[] }) {
  return (
    <table className="w-full text-xs border border-gray-700">
      <thead>
        <tr className="bg-gray-800 text-gray-300">
          <th className="px-2 py-1 text-left">Engine</th>
          <th className="px-2 py-1 text-left">Pack</th>
          <th className="px-2 py-1 text-left">Bodies/nq/nv</th>
          <th className="px-2 py-1 text-left">Total mass</th>
          <th className="px-2 py-1 text-left">Bar above sole</th>
          <th className="px-2 py-1 text-left">Hand-mid above sole</th>
          <th className="px-2 py-1 text-left">Smoke</th>
          <th className="px-2 py-1 text-left">Start contact</th>
        </tr>
      </thead>
      <tbody>
        {engines.map((engine) => {
          const commitShort = engine.pack.commit ? engine.pack.commit.slice(0, 10) : '—';
          return (
            <tr key={engine.engine} className="border-t border-gray-700 text-gray-200">
              <td className="px-2 py-1 capitalize">{engine.engine}</td>
              <td className="px-2 py-1" title={engine.pack.commit ?? undefined}>
                {engine.pack.repo ?? '—'}@{commitShort}
              </td>
              <td className="px-2 py-1">
                {engine.structure.n_bodies ?? '—'}/{engine.structure.nq ?? '—'}/
                {engine.structure.nv ?? '—'}
              </td>
              <td className="px-2 py-1">{measurementCell(engine.total_mass_kg, 2, 'kg')}</td>
              <td className="px-2 py-1">{measurementCell(engine.bar_above_sole_m, 3, 'm')}</td>
              <td className="px-2 py-1">
                {measurementCell(engine.hand_mid_above_sole_m, 3, 'm')}
              </td>
              <td className="px-2 py-1">
                loaded={yesNo(engine.smoke.loaded)}, stepped=
                {yesNo(engine.smoke.stepped)}, max|qvel|=
                {measurementCell(engine.smoke.max_abs_qvel, 4, '')}
              </td>
              <td className="px-2 py-1" title={engine.start_contact.reason ?? undefined}>
                {measurementCell(engine.start_contact.value_n, 2, 'N')}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function PhasesSection({ engines }: { engines: EngineView[] }) {
  return (
    <div className="flex flex-col gap-3">
      {engines.map((engine) => (
        <section key={engine.engine} className="rounded border border-gray-700 bg-gray-800 p-3">
          <h3 className="text-sm font-medium text-white mb-2 capitalize">
            {engine.engine} phases
          </h3>
          {engine.phases.length > 0 ? (
            <table className="w-full text-xs border border-gray-700">
              <thead>
                <tr className="bg-gray-900 text-gray-300">
                  <th className="px-2 py-1 text-left">Phase</th>
                  <th className="px-2 py-1 text-left">Fraction</th>
                  <th className="px-2 py-1 text-left">Hand-bar axis dist (L)</th>
                  <th className="px-2 py-1 text-left">Hand-bar axis dist (R)</th>
                </tr>
              </thead>
              <tbody>
                {engine.phases.map((phase, idx) => (
                  <tr key={`${phase.name ?? 'phase'}-${idx}`} className="border-t border-gray-700 text-gray-200">
                    <td className="px-2 py-1">{phase.name ?? '—'}</td>
                    <td className="px-2 py-1">{measurementCell(phase.fraction, 3, '')}</td>
                    <td className="px-2 py-1">
                      {measurementCell(phase.hand_bar_axis_distance_m.l, 3, 'm')}
                    </td>
                    <td className="px-2 py-1">
                      {measurementCell(phase.hand_bar_axis_distance_m.r, 3, 'm')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="text-xs text-gray-400">No phases recorded.</p>
          )}
        </section>
      ))}
    </div>
  );
}

function ComparisonsSection({ comparisons }: { comparisons: LiftView['comparisons'] }) {
  const poseNames = Object.keys(comparisons.poses);
  if (comparisons.reason) {
    return <p className="text-xs text-gray-400">{comparisons.reason}</p>;
  }
  if (poseNames.length === 0) {
    return <p className="text-xs text-gray-400">No comparisons recorded for this lift.</p>;
  }
  return (
    <div className="flex flex-col gap-3">
      {poseNames.map((pose) => {
        const pairs = comparisons.poses[pose];
        return (
          <div key={pose}>
            <h4 className="text-xs uppercase tracking-wide text-gray-400 mb-1">{pose}</h4>
            <table className="w-full text-xs border border-gray-700">
              <thead>
                <tr className="bg-gray-900 text-gray-300">
                  <th className="px-2 py-1 text-left">Pair</th>
                  {PAIR_METRICS.map((metric) => (
                    <th key={metric} className="px-2 py-1 text-left">
                      {metric}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(pairs).map(([pair, metrics]) => (
                  <tr key={pair} className="border-t border-gray-700 text-gray-200">
                    <td className="px-2 py-1">{pair}</td>
                    {PAIR_METRICS.map((metric) => {
                      const m = metrics[metric];
                      return (
                        <td key={metric} className="px-2 py-1" title={m.reason ?? undefined}>
                          <span
                            className={`px-1.5 py-0.5 rounded text-[10px] ${statusBadgeClass(m.status)}`}
                          >
                            {formatMeasurement(m, 3, 'm')} · {m.status}
                          </span>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
      })}
    </div>
  );
}

function GapsList({ gaps }: { gaps: GapEntry[] }) {
  if (gaps.length === 0) {
    return <p className="text-xs text-gray-400">No known gaps for this lift.</p>;
  }
  return (
    <ul className="flex flex-col gap-3">
      {gaps.map((gap) => (
        <li key={gap.key} className="rounded border border-gray-700 bg-gray-800 p-2">
          <div className="flex items-start justify-between gap-2">
            <h4 className="text-sm font-medium text-white">{gap.title}</h4>
            {gap.new_issue && (
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-700 text-amber-50">
                new
              </span>
            )}
          </div>
          <p className="text-[11px] text-gray-400 mt-1">
            {gap.key} · engines: {gap.engines.join(', ')} · {gap.lift_story}
          </p>
          {gap.evidence.length > 0 && (
            <ul className="text-[11px] text-gray-300 mt-1 space-y-0.5">
              {gap.evidence.map((item, idx) => (
                <li key={`${gap.key}-evidence-${idx}`}>• {item}</li>
              ))}
            </ul>
          )}
          {gap.issues.length > 0 && (
            <p className="text-[11px] text-gray-400 mt-1">{gap.issues.join(', ')}</p>
          )}
        </li>
      ))}
    </ul>
  );
}

export function LiftBaselinePage() {
  const [metaState, setMetaState] = useState<LoadState>('loading');
  const [metaError, setMetaError] = useState<string | null>(null);
  const [metadata, setMetadata] = useState<LiftBaselineMetadata | null>(null);
  const [selectedLift, setSelectedLift] = useState<string | null>(null);

  const [liftState, setLiftState] = useState<LoadState>('loading');
  const [liftError, setLiftError] = useState<string | null>(null);
  const [liftView, setLiftView] = useState<LiftView | null>(null);

  useEffect(() => {
    let cancelled = false;
    void Promise.resolve().then(async () => {
      try {
        const data = await fetchLiftBaseline();
        if (cancelled) return;
        setMetadata(data);
        setMetaState('ready');
        if (data.lifts.length > 0) {
          setSelectedLift(data.lifts[0]);
        }
      } catch (err: unknown) {
        if (cancelled) return;
        setMetaError(err instanceof Error ? err.message : 'Failed to load lift baseline');
        setMetaState('error');
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedLift) return;
    let cancelled = false;
    void Promise.resolve().then(async () => {
      setLiftState('loading');
      try {
        const data = await fetchLiftBaselineLift(selectedLift);
        if (cancelled) return;
        setLiftView(data);
        setLiftState('ready');
      } catch (err: unknown) {
        if (cancelled) return;
        setLiftError(err instanceof Error ? err.message : 'Failed to load lift view');
        setLiftState('error');
      }
    });
    return () => {
      cancelled = true;
    };
  }, [selectedLift]);

  const leftPanel = (
    <div className="flex flex-col gap-3 p-4 text-sm text-gray-200">
      <div>
        <h1 className="text-lg font-semibold text-white">Lift Baseline</h1>
        <p className="text-xs text-gray-400 mt-1">
          LIFT-1 cross-engine baseline for the five canonical lifts.
        </p>
      </div>

      {metadata && (
        <dl className="text-xs text-gray-300 space-y-1">
          <div>
            <dt className="inline text-gray-400">Generated </dt>
            <dd className="inline">{metadata.generated_utc}</dd>
          </div>
          <div>
            <dt className="inline text-gray-400">Position tolerance </dt>
            <dd className="inline">{metadata.tolerances.position_m} m</dd>
          </div>
          <div>
            <dt className="inline text-gray-400">Gaps </dt>
            <dd className="inline">{metadata.gap_count}</dd>
          </div>
        </dl>
      )}

      <label className="flex flex-col gap-1">
        <span className="text-xs uppercase tracking-wide text-gray-400">Lift</span>
        <select
          aria-label="Lift"
          className="rounded bg-gray-900 border border-gray-700 px-2 py-1"
          value={selectedLift ?? ''}
          onChange={(e) => setSelectedLift(e.target.value)}
        >
          {(metadata?.lifts ?? []).map((lift) => (
            <option key={lift} value={lift}>
              {lift}
            </option>
          ))}
        </select>
      </label>
    </div>
  );

  const rightPanel = (
    <div className="flex flex-col gap-3 p-4 text-sm text-gray-200">
      <h2 className="text-base font-semibold text-white">Known Gaps</h2>
      {liftView ? <GapsList gaps={liftView.gaps} /> : (
        <p className="text-xs text-gray-400">Select a lift to see its known gaps.</p>
      )}
    </div>
  );

  let mainContent: ReactNode;
  if (metaState === 'loading') {
    mainContent = (
      <div className="flex h-full items-center justify-center text-gray-300">
        Loading lift baseline…
      </div>
    );
  } else if (metaState === 'error') {
    mainContent = (
      <div className="flex h-full items-center justify-center text-red-300 px-6 text-center">
        {metaError ?? 'Failed to load lift baseline'}
      </div>
    );
  } else if (!selectedLift) {
    mainContent = (
      <div className="flex h-full items-center justify-center text-gray-400">
        No lifts available in the baseline receipt.
      </div>
    );
  } else if (liftState === 'error') {
    mainContent = (
      <div className="flex h-full items-center justify-center text-red-300 px-6 text-center">
        {liftError ?? 'Failed to load lift'}
      </div>
    );
  } else if (liftState === 'loading' || !liftView) {
    mainContent = (
      <div className="flex h-full items-center justify-center text-gray-300">
        Loading lift…
      </div>
    );
  } else {
    mainContent = (
      <div className="flex flex-col h-full gap-4 p-4 overflow-y-auto">
        <section className="rounded border border-gray-700 bg-gray-800 p-3">
          <h3 className="text-sm font-medium text-white mb-2">Engines</h3>
          <EnginesTable engines={liftView.engines} />
        </section>

        <section className="rounded border border-gray-700 bg-gray-800 p-3">
          <h3 className="text-sm font-medium text-white mb-2">
            Cross-engine position comparisons
          </h3>
          <ComparisonsSection comparisons={liftView.comparisons} />
        </section>

        <PhasesSection engines={liftView.engines} />
      </div>
    );
  }

  return (
    <WorkspaceShell leftPanel={leftPanel} rightPanel={rightPanel}>
      <main id="main-content" className="min-h-0 min-w-0 flex-1">
        {mainContent}
      </main>
    </WorkspaceShell>
  );
}

export default LiftBaselinePage;
