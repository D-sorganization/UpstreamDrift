/**
 * Matched Swing Results page (MS-85, #10358).
 *
 * Web counterpart of the desktop Results Browser: lists ledger runs with
 * verdict badges, GIF playback, and a three.js marker preview via
 * MocapSkeleton3D.
 */

import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react';
import { Link } from 'react-router';
import { WorkspaceShell } from '@/components/layout/WorkspaceShell';
import {
  fetchCandidatePreviewFrame,
  fetchMatchedSwingLedger,
  fetchMatchedSwingReceipt,
  fetchParityReport,
  formatMetric,
  matchedSwingAnimationUrl,
  verdictBadgeClass,
  type MatchedSwingRun,
} from '@/api/matchedSwings';
import { ImpactParametersPanel } from '@/components/analysis/ImpactParametersPanel';
import { GripWrenchCharts } from '@/components/analysis/GripWrenchCharts';
import type { MocapJoint } from '@/components/visualization/MocapSkeleton3D';

const MocapSkeleton3D = lazy(
  () => import('@/components/visualization/MocapSkeleton3D'),
);

type LoadState = 'loading' | 'ready' | 'error';

function filterRuns(
  runs: MatchedSwingRun[],
  engine: string,
  capture: string,
  lane: string,
  verdict: string,
  query: string,
): MatchedSwingRun[] {
  const q = query.trim().toLowerCase();
  return runs.filter((run) => {
    if (engine !== 'all' && run.engine !== engine) return false;
    if (capture !== 'all' && (run.capture ?? '') !== capture) return false;
    if (lane !== 'all' && run.lane !== lane) return false;
    if (verdict !== 'all' && run.verdict.toUpperCase() !== verdict.toUpperCase()) {
      return false;
    }
    if (!q) return true;
    const haystack = `${run.engine} ${run.lane} ${run.capture ?? ''} ${run.candidate_sha256 ?? ''} ${run.id} ${run.reason ?? ''}`.toLowerCase();
    return haystack.includes(q);
  });
}

function formatGateValue(value: number | null): string {
  return value != null ? value.toFixed(4) : '—';
}

export function MatchedSwingsPage() {
  const [loadState, setLoadState] = useState<LoadState>('loading');
  const [error, setError] = useState<string | null>(null);
  const [runs, setRuns] = useState<MatchedSwingRun[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [engineFilter, setEngineFilter] = useState('all');
  const [captureFilter, setCaptureFilter] = useState('all');
  const [laneFilter, setLaneFilter] = useState('all');
  const [verdictFilter, setVerdictFilter] = useState('all');
  const [search, setSearch] = useState('');
  const [previewJoints, setPreviewJoints] = useState<MocapJoint[]>([]);
  const [previewFrame, setPreviewFrame] = useState(0);
  const [previewFrameCount, setPreviewFrameCount] = useState(0);
  const [receiptJson, setReceiptJson] = useState<Record<string, unknown> | null>(null);
  const [receiptError, setReceiptError] = useState<string | null>(null);
  const [parityJson, setParityJson] = useState<Record<string, unknown> | null>(null);
  const [parityError, setParityError] = useState<string | null>(null);

  const loadPreview = useCallback(async (run: MatchedSwingRun, frameIndex: number) => {
    if (!run.capabilities.has_candidate_npz) {
      setPreviewJoints([]);
      setPreviewFrameCount(0);
      return;
    }
    try {
      const preview = await fetchCandidatePreviewFrame(run.id, frameIndex);
      setPreviewJoints(preview.joints);
      setPreviewFrame(preview.frame_index);
      setPreviewFrameCount(preview.frame_count);
    } catch {
      setPreviewJoints([]);
      setPreviewFrameCount(0);
    }
  }, []);

  const handleSelectRun = useCallback(
    (run: MatchedSwingRun) => {
      setSelectedId(run.id);
      setReceiptJson(null);
      setReceiptError(null);
      setParityJson(null);
      setParityError(null);
      void Promise.resolve().then(() => loadPreview(run, 0));
    },
    [loadPreview],
  );

  const handleResetFilters = useCallback(() => {
    setEngineFilter('all');
    setCaptureFilter('all');
    setLaneFilter('all');
    setVerdictFilter('all');
    setSearch('');
  }, []);

  const handleViewReceipt = useCallback(async (runId: string) => {
    try {
      const data = await fetchMatchedSwingReceipt(runId);
      setReceiptJson(data);
      setReceiptError(null);
    } catch (err: unknown) {
      setReceiptJson(null);
      setReceiptError(err instanceof Error ? err.message : 'Failed to load receipt');
    }
  }, []);

  const handleViewParity = useCallback(async (runId: string) => {
    try {
      const data = await fetchParityReport(runId);
      setParityJson(data);
      setParityError(null);
    } catch (err: unknown) {
      setParityJson(null);
      setParityError(err instanceof Error ? err.message : 'Failed to load parity report');
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    void Promise.resolve().then(async () => {
      try {
        const data = await fetchMatchedSwingLedger();
        if (cancelled) return;
        setRuns(data.runs);
        setLoadState('ready');
        if (data.runs.length > 0) {
          const first = data.runs[0];
          setSelectedId(first.id);
          await loadPreview(first, 0);
        }
      } catch (err: unknown) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : 'Failed to load ledger');
        setLoadState('error');
      }
    });
    return () => {
      cancelled = true;
    };
  }, [loadPreview]);

  const filteredRuns = useMemo(
    () => filterRuns(runs, engineFilter, captureFilter, laneFilter, verdictFilter, search),
    [runs, engineFilter, captureFilter, laneFilter, verdictFilter, search],
  );

  const selectedRun = useMemo(
    () => filteredRuns.find((run) => run.id === selectedId) ?? filteredRuns[0] ?? null,
    [filteredRuns, selectedId],
  );

  const engineOptions = useMemo(
    () => ['all', ...Array.from(new Set(runs.map((run) => run.engine))).sort()],
    [runs],
  );

  const captureOptions = useMemo(
    () => [
      'all',
      ...Array.from(
        new Set(runs.map((run) => run.capture).filter((c): c is string => Boolean(c))),
      ).sort(),
    ],
    [runs],
  );

  const laneOptions = useMemo(
    () => ['all', ...Array.from(new Set(runs.map((run) => run.lane))).sort()],
    [runs],
  );

  const leftPanel = (
    <div className="flex flex-col gap-3 p-4 text-sm text-gray-200">
      <div>
        <h1 className="text-lg font-semibold text-white">Matched Swing Results</h1>
        <p className="text-xs text-gray-400 mt-1">
          Ledger-backed runs with receipts, parity, and replay artefacts.
        </p>
      </div>

      <label className="flex flex-col gap-1">
        <span className="text-xs uppercase tracking-wide text-gray-400">Engine</span>
        <select
          className="rounded bg-gray-900 border border-gray-700 px-2 py-1"
          value={engineFilter}
          onChange={(e) => setEngineFilter(e.target.value)}
        >
          {engineOptions.map((engine) => (
            <option key={engine} value={engine}>
              {engine}
            </option>
          ))}
        </select>
      </label>

      <label className="flex flex-col gap-1">
        <span className="text-xs uppercase tracking-wide text-gray-400">Capture</span>
        <select
          className="rounded bg-gray-900 border border-gray-700 px-2 py-1"
          value={captureFilter}
          onChange={(e) => setCaptureFilter(e.target.value)}
        >
          {captureOptions.map((capture) => (
            <option key={capture} value={capture}>
              {capture}
            </option>
          ))}
        </select>
      </label>

      <label className="flex flex-col gap-1">
        <span className="text-xs uppercase tracking-wide text-gray-400">Lane</span>
        <select
          className="rounded bg-gray-900 border border-gray-700 px-2 py-1"
          value={laneFilter}
          onChange={(e) => setLaneFilter(e.target.value)}
        >
          {laneOptions.map((lane) => (
            <option key={lane} value={lane}>
              {lane}
            </option>
          ))}
        </select>
      </label>

      <label className="flex flex-col gap-1">
        <span className="text-xs uppercase tracking-wide text-gray-400">Verdict</span>
        <select
          className="rounded bg-gray-900 border border-gray-700 px-2 py-1"
          value={verdictFilter}
          onChange={(e) => setVerdictFilter(e.target.value)}
        >
          <option value="all">all</option>
          <option value="PASSED">PASSED</option>
          <option value="REJECTED">REJECTED</option>
          <option value="UNVERIFIED">UNVERIFIED</option>
          <option value="UNCLASSIFIED">UNCLASSIFIED</option>
        </select>
      </label>

      <label className="flex flex-col gap-1">
        <span className="text-xs uppercase tracking-wide text-gray-400">Search</span>
        <input
          className="rounded bg-gray-900 border border-gray-700 px-2 py-1"
          placeholder="engine, sha, id, reason…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </label>

      <button
        type="button"
        onClick={handleResetFilters}
        className="self-start rounded border border-gray-700 bg-gray-800 px-2 py-1 text-xs hover:border-gray-500"
      >
        Reset
      </button>

      <p className="text-xs text-gray-400">{filteredRuns.length} run(s)</p>

      <div className="flex flex-col gap-1 overflow-y-auto max-h-[50vh]">
        {filteredRuns.map((run) => (
          <button
            key={run.id}
            type="button"
            onClick={() => handleSelectRun(run)}
            className={`text-left rounded border px-2 py-2 transition ${
              selectedRun?.id === run.id
                ? 'border-blue-500 bg-gray-900'
                : 'border-gray-700 bg-gray-800 hover:border-gray-500'
            }`}
          >
            <div className="flex items-center justify-between gap-2">
              <span className="font-medium capitalize">{run.engine}</span>
              <span className={`text-[10px] px-2 py-0.5 rounded ${verdictBadgeClass(run.verdict)}`}>
                {run.verdict}
              </span>
            </div>
            <div className="text-[11px] text-gray-400 mt-1 truncate">
              {run.lane} · {run.capture ?? '—'} · {run.id.slice(0, 12)}…
            </div>
          </button>
        ))}
      </div>

      <Link
        to="/tools/cross-engine"
        className="text-xs text-blue-400 hover:text-blue-300 underline mt-2"
      >
        Open Cross-Engine Dashboard →
      </Link>
    </div>
  );

  const rightPanel = selectedRun ? (
    <div className="flex flex-col gap-3 p-4 text-sm text-gray-200">
      <div className="flex items-start justify-between gap-2">
        <div>
          <h2 className="text-base font-semibold text-white capitalize">{selectedRun.engine}</h2>
          <p className="text-xs text-gray-400">
            {selectedRun.lane} · {selectedRun.capture ?? 'no capture'} · horizon{' '}
            {selectedRun.horizon_s ?? '—'} s
          </p>
        </div>
        <span className={`text-xs px-2 py-1 rounded ${verdictBadgeClass(selectedRun.verdict)}`}>
          {selectedRun.verdict}
        </span>
      </div>

      <dl className="grid grid-cols-2 gap-2 text-xs">
        <div>
          <dt className="text-gray-400">Whole RMSE</dt>
          <dd>{formatMetric(selectedRun.metrics.whole_marker_rmse_m)}</dd>
        </div>
        <div>
          <dt className="text-gray-400">Early RMSE</dt>
          <dd>{formatMetric(selectedRun.metrics.early_marker_rmse_m)}</dd>
        </div>
        <div>
          <dt className="text-gray-400">Terminal RMSE</dt>
          <dd>{formatMetric(selectedRun.metrics.terminal_marker_rmse_m)}</dd>
        </div>
        <div>
          <dt className="text-gray-400">Club RMSE</dt>
          <dd>{formatMetric(selectedRun.metrics.club_marker_rmse_m)}</dd>
        </div>
        <div>
          <dt className="text-gray-400">Pelvis yaw</dt>
          <dd>{formatMetric(selectedRun.metrics.pelvis_yaw_rmse_rad, 'deg')}</dd>
        </div>
        <div>
          <dt className="text-gray-400">Candidate SHA</dt>
          <dd className="truncate" title={selectedRun.candidate_sha256 ?? undefined}>
            {selectedRun.candidate_sha256?.slice(0, 16) ?? '—'}…
          </dd>
        </div>
      </dl>

      {selectedRun.reason && (
        <p className="text-xs text-amber-300 border border-amber-700/40 rounded p-2">
          {selectedRun.reason}
        </p>
      )}

      <div>
        <h3 className="text-xs uppercase tracking-wide text-gray-400 mb-1">Physical Gates</h3>
        {selectedRun.gates && selectedRun.gates.length > 0 ? (
          <ul className="text-xs text-gray-300 space-y-0.5">
            {selectedRun.gates.map((gate, idx) => (
              <li key={`${gate.name}-${idx}`}>
                • {gate.name} [{gate.status}]: measured={formatGateValue(gate.measured)}{' '}
                {gate.unit} (limit={formatGateValue(gate.threshold)} {gate.unit})
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-xs text-gray-400">No physical gates evaluated for this run.</p>
        )}
      </div>

      <div className="flex flex-col gap-2">
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => void handleViewReceipt(selectedRun.id)}
            className="text-xs rounded border border-gray-700 bg-gray-800 px-2 py-1 hover:border-gray-500"
          >
            View Receipt JSON
          </button>
          <button
            type="button"
            onClick={() => void handleViewParity(selectedRun.id)}
            disabled={!selectedRun.capabilities.has_parity_report}
            className="text-xs rounded border border-gray-700 bg-gray-800 px-2 py-1 hover:border-gray-500 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            View Parity Report
          </button>
        </div>
        {receiptError && <p className="text-xs text-red-300">{receiptError}</p>}
        {receiptJson && (
          <details className="text-xs">
            <summary className="cursor-pointer text-gray-300">Receipt JSON</summary>
            <pre className="mt-1 max-h-48 overflow-auto rounded bg-gray-950 p-2 text-[11px] text-gray-300">
              {JSON.stringify(receiptJson, null, 2)}
            </pre>
          </details>
        )}
        {parityError && <p className="text-xs text-red-300">{parityError}</p>}
        {parityJson && (
          <details className="text-xs">
            <summary className="cursor-pointer text-gray-300">Parity Report JSON</summary>
            <pre className="mt-1 max-h-48 overflow-auto rounded bg-gray-950 p-2 text-[11px] text-gray-300">
              {JSON.stringify(parityJson, null, 2)}
            </pre>
          </details>
        )}
      </div>

      {/* Impact parameters relative to the target line (GCV-17, #11723) */}
      <ImpactParametersPanel runId={selectedRun.id} />
      <GripWrenchCharts runId={selectedRun.id} />
    </div>
  ) : (
    <div className="p-4 text-sm text-gray-400">Select a run to inspect metrics.</div>
  );

  let mainContent: ReactNode;
  if (loadState === 'loading') {
    mainContent = (
      <div className="flex h-full items-center justify-center text-gray-300">Loading ledger…</div>
    );
  } else if (loadState === 'error') {
    mainContent = (
      <div className="flex h-full items-center justify-center text-red-300 px-6 text-center">
        {error ?? 'Failed to load matched-swing ledger'}
      </div>
    );
  } else if (!selectedRun) {
    mainContent = (
      <div className="flex h-full items-center justify-center text-gray-400">
        No runs match the current filters.
      </div>
    );
  } else {
    mainContent = (
      <div className="flex flex-col h-full gap-4 p-4 overflow-y-auto">
        <div className="grid lg:grid-cols-2 gap-4">
          <section className="rounded border border-gray-700 bg-gray-800 p-3">
            <h3 className="text-sm font-medium text-white mb-2">GIF Playback</h3>
            {selectedRun.capabilities.has_animation_gif ? (
              <img
                src={matchedSwingAnimationUrl(selectedRun.id)}
                alt={`${selectedRun.engine} matched swing animation`}
                className="mx-auto max-h-72 rounded border border-gray-700 bg-black"
              />
            ) : (
              <p className="text-xs text-gray-400">No animation artefact for this run.</p>
            )}
          </section>

          <section className="rounded border border-gray-700 bg-gray-800 p-3 min-h-[18rem]">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-medium text-white">3D Marker Preview</h3>
              {previewFrameCount > 1 && (
                <input
                  type="range"
                  min={0}
                  max={previewFrameCount - 1}
                  value={previewFrame}
                  aria-label="Preview frame"
                  onChange={(e) => {
                    const next = Number(e.target.value);
                    setPreviewFrame(next);
                    void loadPreview(selectedRun, next);
                  }}
                  className="w-32"
                />
              )}
            </div>
            {previewJoints.length > 0 ? (
              <Suspense
                fallback={
                  <div className="h-64 flex items-center justify-center text-gray-400 text-xs">
                    Loading 3D preview…
                  </div>
                }
              >
                <div className="h-64 rounded overflow-hidden border border-gray-700">
                  <MocapSkeleton3D joints={previewJoints} />
                </div>
              </Suspense>
            ) : (
              <p className="text-xs text-gray-400">No candidate marker preview available.</p>
            )}
          </section>
        </div>
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

export default MatchedSwingsPage;
