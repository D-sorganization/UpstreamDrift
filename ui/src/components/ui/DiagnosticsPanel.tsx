import { useState, useEffect, useCallback } from 'react';
import {
  isTauri,
  startBackend,
  stopBackend,
  getDiagnostics,
  type DiagnosticInfo,
} from '@/api/backend';

/**
 * Diagnostics panel that shows backend server status, Python environment info,
 * and engine loading details. Provides controls to start/stop the backend.
 *
 * Only shows Tauri-specific controls when running inside the Tauri shell.
 */
export function DiagnosticsPanel() {
  const [isOpen, setIsOpen] = useState(false);
  const [diagnostics, setDiagnostics] = useState<DiagnosticInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [backendHealth, setBackendHealth] = useState<'unknown' | 'healthy' | 'unreachable'>(
    'unknown'
  );

  const refreshDiagnostics = useCallback(async () => {
    try {
      const info = await getDiagnostics();
      setDiagnostics(info);
    } catch {
      // Not in Tauri or command failed
    }
  }, []);

  const checkBackendHealth = useCallback(async () => {
    try {
      const response = await fetch('/api/engines', { signal: AbortSignal.timeout(3000) });
      setBackendHealth(response.ok ? 'healthy' : 'unreachable');
    } catch {
      setBackendHealth('unreachable');
    }
  }, []);

  useEffect(() => {
    if (isOpen) {
      refreshDiagnostics();
      checkBackendHealth();
      const interval = setInterval(() => {
        refreshDiagnostics();
        checkBackendHealth();
      }, 5000);
      return () => clearInterval(interval);
    }
  }, [isOpen, refreshDiagnostics, checkBackendHealth]);

  const handleStart = async () => {
    setLoading(true);
    setActionError(null);
    try {
      const status = await startBackend();
      if (status.error) {
        setActionError(status.error);
      }
      // Give the server a moment to start
      setTimeout(() => {
        refreshDiagnostics();
        checkBackendHealth();
      }, 2000);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  const handleStop = async () => {
    setLoading(true);
    setActionError(null);
    try {
      await stopBackend();
      await refreshDiagnostics();
      setBackendHealth('unreachable');
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  const healthColor = {
    unknown: 'text-gray-400',
    healthy: 'text-green-400',
    unreachable: 'text-red-400',
  }[backendHealth];

  const healthLabel = {
    unknown: 'Unknown',
    healthy: 'Connected',
    unreachable: 'Unreachable',
  }[backendHealth];

  return (
    <div className="fixed bottom-4 right-4 z-50">
      {/* Toggle button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`px-3 py-1.5 rounded-lg text-xs font-mono border transition-colors ${
          backendHealth === 'healthy'
            ? 'border-green-600 bg-green-900/30 text-green-400 hover:bg-green-900/50'
            : backendHealth === 'unreachable'
              ? 'border-red-600 bg-red-900/30 text-red-400 hover:bg-red-900/50'
              : 'border-gray-600 bg-gray-800 text-gray-400 hover:bg-gray-700'
        }`}
        aria-label="Toggle diagnostics panel"
      >
        {backendHealth === 'healthy' ? '● ' : backendHealth === 'unreachable' ? '○ ' : '? '}
        Backend
      </button>

      {/* Panel */}
      {isOpen && (
        <div className="absolute bottom-10 right-0 w-80 rounded-lg border border-gray-600 bg-gray-900 text-gray-200 shadow-xl text-xs">
          <div className="flex items-center justify-between p-3 border-b border-gray-700">
            <span className="font-semibold text-sm">Diagnostics</span>
            <button
              onClick={() => setIsOpen(false)}
              className="text-gray-400 hover:text-white"
              aria-label="Close diagnostics"
            >
              x
            </button>
          </div>

          <div className="p-3 space-y-3">
            {/* Backend Health */}
            <div>
              <div className="font-medium text-gray-300 mb-1">Backend Server</div>
              <div className="space-y-1">
                <div className="flex justify-between">
                  <span>API Status</span>
                  <span className={healthColor}>{healthLabel}</span>
                </div>
                {diagnostics?.backend && (
                  <>
                    <div className="flex justify-between">
                      <span>Process</span>
                      <span
                        className={
                          diagnostics.backend.running ? 'text-green-400' : 'text-gray-500'
                        }
                      >
                        {diagnostics.backend.running
                          ? `Running (PID ${diagnostics.backend.pid})`
                          : 'Stopped'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Port</span>
                      <span>{diagnostics.backend.port}</span>
                    </div>
                    {diagnostics.backend.error && (
                      <div className="text-red-400 text-[10px] mt-1">
                        {diagnostics.backend.error}
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>

            {/* Python Environment */}
            {diagnostics && (
              <div>
                <div className="font-medium text-gray-300 mb-1">Environment</div>
                <div className="space-y-1">
                  <div className="flex justify-between">
                    <span>Python</span>
                    <span className={diagnostics.python_found ? 'text-green-400' : 'text-red-400'}>
                      {diagnostics.python_version || 'Not found'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Repo Root</span>
                    <span
                      className={diagnostics.repo_root ? 'text-green-400' : 'text-red-400'}
                      title={diagnostics.repo_root || undefined}
                    >
                      {diagnostics.repo_root ? 'Found' : 'Not found'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Server Script</span>
                    <span
                      className={
                        diagnostics.local_server_found ? 'text-green-400' : 'text-red-400'
                      }
                    >
                      {diagnostics.local_server_found ? 'Found' : 'Missing'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Runtime</span>
                    <span className="text-blue-400">{isTauri() ? 'Tauri' : 'Browser'}</span>
                  </div>
                </div>
              </div>
            )}

            {/* Actions */}
            {isTauri() && (
              <div className="flex gap-2 pt-1">
                <button
                  onClick={handleStart}
                  disabled={loading || diagnostics?.backend.running === true}
                  className="flex-1 px-2 py-1.5 rounded border border-green-600 bg-green-900/30 text-green-400 hover:bg-green-900/50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                >
                  {loading ? '...' : 'Start'}
                </button>
                <button
                  onClick={handleStop}
                  disabled={loading || !diagnostics?.backend.running}
                  className="flex-1 px-2 py-1.5 rounded border border-red-600 bg-red-900/30 text-red-400 hover:bg-red-900/50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                >
                  Stop
                </button>
                <button
                  onClick={() => {
                    refreshDiagnostics();
                    checkBackendHealth();
                  }}
                  className="px-2 py-1.5 rounded border border-gray-600 bg-gray-800 text-gray-400 hover:bg-gray-700 transition-colors"
                >
                  Refresh
                </button>
              </div>
            )}

            {actionError && (
              <div className="p-2 rounded bg-red-900/30 border border-red-700 text-red-400 text-[10px]">
                {actionError}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
