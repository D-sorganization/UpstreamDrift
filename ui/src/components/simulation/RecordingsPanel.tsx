/**
 * RecordingsPanel — list, export, and delete persisted session recordings
 * (issue #7451, export/recording parity with the desktop dashboard).
 *
 * Collapsible section: data is fetched lazily on first expand so the panel
 * adds no network traffic to the Simulation page until the user opens it.
 * Download buttons are data-driven from `GET /export/formats`, which mirrors
 * the desktop recorder's format registry — only formats whose optional
 * dependency is actually installed are offered.
 */

import { useCallback, useState } from 'react';
import {
  deleteRecording,
  fetchExportFormats,
  listRecordings,
  recordingExportUrl,
  saveRecording,
  type ExportFormats,
  type RecordingMeta,
} from '@/api/recordings';

interface RecordingsPanelProps {
  /** Disable mutating actions while a simulation is running. */
  isRunning?: boolean;
}

export function RecordingsPanel({ isRunning = false }: RecordingsPanelProps) {
  const [expanded, setExpanded] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [recordings, setRecordings] = useState<RecordingMeta[]>([]);
  const [formats, setFormats] = useState<ExportFormats>({});

  const refresh = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      const [recs, fmts] = await Promise.all([listRecordings(), fetchExportFormats()]);
      setRecordings(recs);
      setFormats(fmts);
      setLoaded(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load recordings');
    } finally {
      setBusy(false);
    }
  }, []);

  const handleToggle = useCallback(() => {
    setExpanded((open) => {
      const next = !open;
      if (next && !loaded) {
        void refresh();
      }
      return next;
    });
  }, [loaded, refresh]);

  const handleSave = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      await saveRecording();
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save recording');
      setBusy(false);
    }
  }, [refresh]);

  const handleDelete = useCallback(
    async (id: string) => {
      // eslint-disable-next-line no-alert
      if (!window.confirm(`Delete recording ${id}? This cannot be undone.`)) {
        return;
      }
      setBusy(true);
      setError(null);
      try {
        await deleteRecording(id);
        await refresh();
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to delete recording');
        setBusy(false);
      }
    },
    [refresh],
  );

  const availableFormats = Object.entries(formats).filter(
    ([, info]) => info.available,
  );

  return (
    <section aria-label="Recordings" className="border-t border-gray-700 pt-4">
      <button
        type="button"
        onClick={handleToggle}
        aria-expanded={expanded}
        className="flex w-full items-center justify-between text-sm font-semibold text-gray-400 uppercase tracking-wider mb-2 hover:text-gray-200"
      >
        <span>Recordings</span>
        <span aria-hidden="true">{expanded ? '▾' : '▸'}</span>
      </button>

      {expanded && (
        <div className="space-y-3">
          <button
            type="button"
            onClick={handleSave}
            disabled={busy || isRunning}
            className="w-full px-3 py-1.5 text-xs font-medium rounded bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Save current recording
          </button>

          {error && (
            <p role="alert" className="text-xs text-red-400">
              {error}
            </p>
          )}

          {busy && !error && (
            <p className="text-xs text-gray-400 italic">Loading…</p>
          )}

          {loaded && recordings.length === 0 && !busy && (
            <p className="text-xs text-gray-400 italic">
              No recordings yet. Run a simulation, then save it.
            </p>
          )}

          <ul className="space-y-2">
            {recordings.map((rec) => (
              <li
                key={rec.id}
                className="bg-gray-700/50 rounded-md p-2"
                data-testid={`recording-${rec.id}`}
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="min-w-0">
                    <p className="text-xs text-gray-200 font-mono truncate" title={rec.id}>
                      {rec.id}
                    </p>
                    <p className="text-[10px] text-gray-400">
                      {rec.engine ?? 'unknown engine'} · {rec.frames} frames
                      {rec.duration != null ? ` · ${rec.duration.toFixed(2)}s` : ''}
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => void handleDelete(rec.id)}
                    disabled={busy}
                    aria-label={`Delete recording ${rec.id}`}
                    className="text-xs text-red-400 hover:text-red-300 disabled:opacity-50"
                  >
                    Delete
                  </button>
                </div>
                <div className="mt-1.5 flex flex-wrap gap-1">
                  {availableFormats.map(([key, info]) => (
                    <a
                      key={key}
                      href={recordingExportUrl(rec.id, key)}
                      download={`${rec.id}${info.extension}`}
                      aria-label={`Download ${rec.id} as ${info.name}`}
                      className="px-2 py-0.5 text-[10px] rounded bg-gray-600 text-gray-200 hover:bg-gray-500"
                    >
                      {info.name}
                    </a>
                  ))}
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
