import { useCallback, useEffect, useState } from "react";
import {
  fetchSpineAssets,
  pollSpineStudyJob,
  submitSpineStudy,
  type SpineAssetItem,
  type SpineStudyJobStatus,
} from "@/api/spineViewer";

const COUPLING_LAWS = [
  "bruno2015_regional",
  "fujii2007_lumbar",
  "uniform_reference",
] as const;

export interface SpineStudyPanelProps {
  onStudyCompleted: (receiptPath: string) => Promise<void> | void;
}

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function SpineStudyPanel({ onStudyCompleted }: SpineStudyPanelProps) {
  const [assets, setAssets] = useState<SpineAssetItem[]>([]);
  const [selectedAssetId, setSelectedAssetId] = useState("");
  const [couplingLaw, setCouplingLaw] = useState("");
  const [motionPath, setMotionPath] = useState(
    "tests/data/motion_pipeline/golden/sample.sto",
  );
  const [studyId, setStudyId] = useState("spine_study_1");
  const [jobStatus, setJobStatus] = useState<SpineStudyJobStatus | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    fetchSpineAssets()
      .then((rawList) => {
        if (!active) return;
        const list = Array.isArray(rawList) ? rawList : [];
        setAssets(list);
        if (list.length > 0) {
          const fixture = list.find(
            (a) => a.is_fixture || a.asset_id === "fixture",
          );
          setSelectedAssetId(fixture ? fixture.asset_id : list[0].asset_id);
        }
      })
      .catch((e: unknown) => active && setError(messageOf(e)));
    return () => {
      active = false;
    };
  }, []);

  const safeAssets = Array.isArray(assets) ? assets : [];
  const selectedAsset = safeAssets.find((a) => a.asset_id === selectedAssetId);

  const handleRunStudy = useCallback(async () => {
    if (!studyId || !motionPath || !selectedAssetId || isRunning) return;
    setIsRunning(true);
    setError(null);
    setJobStatus(null);

    try {
      const payload = {
        study_id: studyId,
        asset_id: selectedAssetId,
        model_id: selectedAssetId,
        coupling_law_id: couplingLaw || null,
        motion_path: motionPath,
      };
      const initial = await submitSpineStudy(payload);
      setJobStatus(initial);

      const terminal = await pollSpineStudyJob(initial.job_id, {
        onStatus: (st) => setJobStatus(st),
      });

      if (terminal.status === "completed" && terminal.receipt_path) {
        await onStudyCompleted(terminal.receipt_path);
      } else if (terminal.status === "failed") {
        setError(terminal.error ?? "Study execution failed");
      }
    } catch (e: unknown) {
      setError(messageOf(e));
    } finally {
      setIsRunning(false);
    }
  }, [
    studyId,
    motionPath,
    selectedAssetId,
    couplingLaw,
    isRunning,
    onStudyCompleted,
  ]);

  // Model status display text
  let statusText = "unknown";
  let isQualified = false;
  if (selectedAsset) {
    if (selectedAsset.is_fixture || selectedAsset.asset_id === "fixture") {
      statusText = "fixture (synthetic, unqualified)";
    } else {
      statusText = selectedAsset.status;
      isQualified = selectedAsset.status === "qualified";
    }
  }

  return (
    <div className="border-b border-gray-700 bg-gray-800/80 px-6 py-4">
      <h2 className="text-sm font-semibold text-gray-200">Run Spine Study</h2>
      <div className="mt-3 flex flex-wrap items-end gap-4">
        <label className="text-xs text-gray-300">
          Study ID
          <input
            className="ml-2 rounded bg-gray-700 px-2 py-1 text-gray-100"
            value={studyId}
            onChange={(e) => setStudyId(e.target.value)}
            disabled={isRunning}
          />
        </label>
        <label className="text-xs text-gray-300">
          Model
          <select
            className="ml-2 rounded bg-gray-700 px-2 py-1 text-gray-100"
            value={selectedAssetId}
            onChange={(e) => setSelectedAssetId(e.target.value)}
            disabled={isRunning}
          >
            {safeAssets.map((a) => (
              <option key={a.asset_id} value={a.asset_id}>
                {a.asset_id} ({a.status})
              </option>
            ))}
          </select>
        </label>
        <span
          data-testid="model-status"
          className={
            "text-xs " +
            (isQualified
              ? "text-green-400"
              : selectedAsset?.is_fixture
                ? "text-amber-400"
                : "text-red-400")
          }
        >
          Model status: {statusText}
        </span>
        <label className="text-xs text-gray-300">
          Coupling law
          <select
            className="ml-2 rounded bg-gray-700 px-2 py-1 text-gray-100"
            value={couplingLaw}
            onChange={(e) => setCouplingLaw(e.target.value)}
            disabled={isRunning}
          >
            <option value="">(model default)</option>
            {COUPLING_LAWS.map((law) => (
              <option key={law} value={law}>
                {law}
              </option>
            ))}
          </select>
        </label>
        <label className="text-xs text-gray-300">
          Swing source
          <input
            className="ml-2 rounded bg-gray-700 px-2 py-1 text-gray-100"
            value={motionPath}
            onChange={(e) => setMotionPath(e.target.value)}
            disabled={isRunning}
          />
        </label>
        <button
          type="button"
          onClick={handleRunStudy}
          disabled={isRunning || !studyId || !motionPath}
          className="rounded bg-indigo-700 px-3 py-1 text-xs font-medium text-white hover:bg-indigo-600 disabled:opacity-50"
        >
          {isRunning ? "Running..." : "Run Study"}
        </button>
      </div>
      {isRunning && (
        <div className="mt-2 text-xs text-indigo-300">
          Running study background job (status:{" "}
          {jobStatus?.status ?? "starting"})...
        </div>
      )}
      {error && (
        <div
          role="alert"
          className="mt-2 rounded border border-red-800 bg-red-900/30 px-3 py-1 text-xs text-red-300"
        >
          {error}
        </div>
      )}
    </div>
  );
}
