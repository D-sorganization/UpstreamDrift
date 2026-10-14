/**
 * Launch Monitor Analytics — Sessions panel (#11987).
 *
 * Web counterpart of the desktop "Sessions" tab
 * (`src/tools/launch_monitor_analytics/gui.py` `_on_import_files`/`import_file`
 * and `ImportMappingDialog` backed by `POST /v2/import/preview` and
 * `POST /v2/import`): preview an uploaded launch-monitor export, customize
 * session metadata and profile mapping, and import shots into client state.
 */

import {
  useCallback,
  useState,
  type ChangeEvent,
} from "react";
import {
  importSessionV2,
  previewImportV2,
  type ImportedSessionResponse,
  type ImportPreviewResponse,
} from "@/api/launchMonitorAnalytics";

const DEFAULT_STATUS =
  "Select a launch-monitor export file to preview or import a session.";

const BUTTON_CLASS =
  "rounded bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-400 text-white px-3 py-2 text-sm font-medium";

type RunState = "idle" | "running" | "done" | "error";

export function LaunchMonitorSessionsPanel({
  onSessionImported,
}: {
  onSessionImported?: (session: ImportedSessionResponse) => void;
}) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<ImportPreviewResponse | null>(null);
  const [profileId, setProfileId] = useState<string>("");
  const [sessionName, setSessionName] = useState<string>("");
  const [player, setPlayer] = useState<string>("");
  const [monitorModel, setMonitorModel] = useState<string>("");
  const [sessions, setSessions] = useState<ImportedSessionResponse[]>([]);
  const [runState, setRunState] = useState<RunState>("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [lastAction, setLastAction] = useState<"preview" | "import">("preview");

  const handleFileChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0] ?? null;
      setSelectedFile(file);
      if (!file) {
        setPreview(null);
        setProfileId("");
        setRunState("idle");
        setErrorMessage(null);
        return;
      }
      setRunState("running");
      setErrorMessage(null);
      void previewImportV2(file)
        .then((data) => {
          setPreview(data);
          setProfileId(data.detected_profile_id ?? "");
          setRunState("done");
          setLastAction("preview");
        })
        .catch((err) => {
          setRunState("error");
          setErrorMessage(err instanceof Error ? err.message : String(err));
        });
    },
    [],
  );

  const handleImport = useCallback(() => {
    if (!selectedFile || runState === "running") return;
    setRunState("running");
    setErrorMessage(null);
    void importSessionV2(selectedFile, {
      profile_id: profileId || null,
      session_name: sessionName || null,
      player: player || null,
      monitor_model: monitorModel || null,
    })
      .then((imported) => {
        setSessions((prev) => [...prev, imported]);
        setRunState("done");
        setLastAction("import");
        onSessionImported?.(imported);
      })
      .catch((err) => {
        setRunState("error");
        setErrorMessage(err instanceof Error ? err.message : String(err));
      });
  }, [
    selectedFile,
    runState,
    profileId,
    sessionName,
    player,
    monitorModel,
    onSessionImported,
  ]);

  const handleRemoveSession = useCallback((sessionId: string) => {
    setSessions((prev) => prev.filter((s) => s.session_id !== sessionId));
  }, []);

  const statusMessage = (() => {
    if (runState === "running") {
      return "Working…";
    }
    if (runState === "error") {
      return errorMessage
        ? `Session import failed: ${errorMessage}`
        : "Session import failed.";
    }
    if (runState === "done") {
      return lastAction === "import" ? "Session imported." : "Preview ready.";
    }
    return DEFAULT_STATUS;
  })();

  const statusClass =
    runState === "error"
      ? "text-red-300"
      : runState === "done"
        ? "text-emerald-300"
        : "text-gray-300";

  return (
    <section className="rounded border border-gray-700 bg-gray-800 p-3 flex flex-col gap-3">
      <h2 className="text-sm font-medium text-white">Sessions</h2>

      <div className="flex flex-wrap gap-3 items-end">
        <label className="flex flex-col gap-1">
          <span className="text-xs uppercase tracking-wide text-gray-400">
            Select Export File
          </span>
          <input
            type="file"
            accept=".csv,.tsv,.txt,.xlsx,.xls,.json"
            onChange={handleFileChange}
            className="text-xs text-gray-300"
            data-testid="lms-file-input"
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs uppercase tracking-wide text-gray-400">
            Session Name
          </span>
          <input
            type="text"
            value={sessionName}
            onChange={(e) => setSessionName(e.target.value)}
            placeholder="Optional session name"
            className="rounded bg-gray-900 border border-gray-700 px-2 py-1 text-xs text-white"
            data-testid="lms-session-name"
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs uppercase tracking-wide text-gray-400">
            Player
          </span>
          <input
            type="text"
            value={player}
            onChange={(e) => setPlayer(e.target.value)}
            placeholder="Optional player"
            className="rounded bg-gray-900 border border-gray-700 px-2 py-1 text-xs text-white"
            data-testid="lms-player"
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs uppercase tracking-wide text-gray-400">
            Monitor Model
          </span>
          <input
            type="text"
            value={monitorModel}
            onChange={(e) => setMonitorModel(e.target.value)}
            placeholder="Optional model"
            className="rounded bg-gray-900 border border-gray-700 px-2 py-1 text-xs text-white"
            data-testid="lms-monitor-model"
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs uppercase tracking-wide text-gray-400">
            Profile
          </span>
          <select
            data-testid="lms-profile-select"
            value={profileId}
            onChange={(e) => setProfileId(e.target.value)}
            className="rounded bg-gray-900 border border-gray-700 px-2 py-1 text-xs text-white"
          >
            {!preview && <option value="">(auto-detect)</option>}
            {preview?.profiles?.map((p) => (
              <option key={p.profile_id} value={p.profile_id}>
                {p.vendor} ({p.profile_id})
              </option>
            ))}
          </select>
        </label>

        <button
          type="button"
          onClick={handleImport}
          disabled={!selectedFile || runState === "running"}
          className={BUTTON_CLASS}
          data-testid="lms-import-btn"
        >
          Import Session
        </button>
      </div>

      <p data-testid="lms-status" className={statusClass}>
        {statusMessage}
      </p>

      <div data-testid="lms-sessions-list" className="flex flex-col gap-2">
        {sessions.length === 0 ? (
          <p className="text-xs text-gray-400">No imported sessions.</p>
        ) : (
          sessions.map((s) => {
            const vendorOrProfile =
              (s.manifest?.vendor as string | undefined) ??
              (s.manifest?.profile_id as string | undefined) ??
              "unknown";
            return (
              <div
                key={s.session_id}
                className="flex items-center justify-between rounded bg-gray-900 border border-gray-700 px-3 py-2 text-xs"
              >
                <div className="flex flex-col gap-0.5">
                  <span className="font-medium text-white">{s.name}</span>
                  <span className="text-gray-400">
                    ID: {s.session_id} · {s.row_count} shot(s) · {vendorOrProfile}
                  </span>
                </div>
                <button
                  type="button"
                  data-testid={`lms-remove-${s.session_id}`}
                  onClick={() => handleRemoveSession(s.session_id)}
                  className="rounded bg-red-600/80 hover:bg-red-500 text-white px-2 py-1 text-xs"
                >
                  Remove
                </button>
              </div>
            );
          })
        )}
      </div>
    </section>
  );
}

export default LaunchMonitorSessionsPanel;
