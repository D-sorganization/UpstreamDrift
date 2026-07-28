/**
 * Canonical-core estimation and comparison workspaces.
 *
 * #8081: both routes used to render only static text — a title, a shell
 * statement, a service-boundary label, and the route string. There was no
 * input, no execution action, no result, no empty state, and no service-status
 * error, so a user could not tell whether the workspace was broken, still
 * loading, or simply not built yet.
 *
 * `src/tools/canonical_core` currently ships descriptors and a PyQt6 shell
 * only — there is no estimation or comparison compute service — so the honest
 * fix is an explicit, actionable unavailable state sourced from the backend
 * rather than hardcoded in the UI. The page now calls
 * `GET /api/tools/canonical-core/{mode}/status` and renders four real states:
 * loading, service error (with retry), unavailable (with reason + next step),
 * and available (the workspace, once a service exists).
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { apiFetch } from '@/api/fetch';

type CanonicalCoreMode = 'estimation' | 'comparison';

type LoadStatus = 'loading' | 'ready' | 'error';

/** Mirrors `CanonicalCoreStatus` in `src/api/routes/canonical_core.py`. */
export interface CanonicalCoreStatus {
  tool_id: string;
  mode: string;
  name: string;
  description: string;
  web_route: string;
  capabilities: string[];
  available: boolean;
  reason: string;
  next_step: string;
}

interface CanonicalCoreShellPageProps {
  mode: CanonicalCoreMode;
}

const FALLBACK_TITLE: Record<CanonicalCoreMode, string> = {
  estimation: 'Canonical-Core Estimation',
  comparison: 'Canonical-Core Comparison',
};

/**
 * Validate the status payload at runtime.
 *
 * The backend ships separately from the UI, so a shape mismatch must surface
 * as the page's error state rather than as a render-time TypeError.
 *
 * @param raw - Parsed JSON body.
 * @returns The validated status.
 * @throws Error when a required field is missing or mistyped.
 */
// eslint-disable-next-line react-refresh/only-export-components
export function parseCanonicalCoreStatus(raw: unknown): CanonicalCoreStatus {
  if (typeof raw !== 'object' || raw === null) {
    throw new Error('Canonical-core status response was not an object');
  }
  const body = raw as Record<string, unknown>;
  if (typeof body.mode !== 'string' || body.mode.length === 0) {
    throw new Error('Canonical-core status response is missing "mode"');
  }
  if (typeof body.available !== 'boolean') {
    throw new Error('Canonical-core status response is missing "available"');
  }
  return {
    tool_id: typeof body.tool_id === 'string' ? body.tool_id : '',
    mode: body.mode,
    name: typeof body.name === 'string' ? body.name : '',
    description: typeof body.description === 'string' ? body.description : '',
    web_route: typeof body.web_route === 'string' ? body.web_route : '',
    capabilities: Array.isArray(body.capabilities)
      ? body.capabilities.filter((c): c is string => typeof c === 'string')
      : [],
    available: body.available,
    reason: typeof body.reason === 'string' ? body.reason : '',
    next_step: typeof body.next_step === 'string' ? body.next_step : '',
  };
}

export function CanonicalCoreShellPage({ mode }: CanonicalCoreShellPageProps) {
  const [status, setStatus] = useState<CanonicalCoreStatus | null>(null);
  const [loadStatus, setLoadStatus] = useState<LoadStatus>('loading');
  const [loadError, setLoadError] = useState<string | null>(null);

  const route = useMemo(() => `/tools/canonical-core/${mode}`, [mode]);

  const loadStatusReport = useCallback(async () => {
    setLoadStatus('loading');
    setLoadError(null);
    try {
      const raw = await apiFetch<unknown>(
        `/api/tools/canonical-core/${mode}/status`,
      );
      setStatus(parseCanonicalCoreStatus(raw));
      setLoadStatus('ready');
    } catch (err) {
      setStatus(null);
      setLoadStatus('error');
      setLoadError(
        err instanceof Error
          ? err.message
          : 'Failed to reach the canonical-core service',
      );
    }
  }, [mode]);

  useEffect(() => {
    queueMicrotask(() => {
      void loadStatusReport();
    });
  }, [loadStatusReport]);

  const title = status?.name || FALLBACK_TITLE[mode];

  return (
    <main className="min-h-screen bg-gray-950 text-gray-100">
      <section className="mx-auto flex min-h-screen max-w-6xl flex-col gap-6 px-4 sm:px-6 py-4 sm:py-8">
        <header className="flex flex-col gap-2 border-b border-gray-800 pb-5">
          <p className="text-sm font-medium uppercase tracking-wide text-cyan-300">
            Canonical Core
          </p>
          <h1 className="text-3xl font-semibold">{title}</h1>
          {status?.description && (
            <p className="max-w-3xl text-sm leading-6 text-gray-300">
              {status.description}
            </p>
          )}
        </header>

        {loadStatus === 'loading' && (
          <div
            className="rounded-md border border-gray-800 bg-gray-900 p-5 text-sm text-gray-300"
            data-testid="canonical-core-loading"
          >
            Checking canonical-core service status…
          </div>
        )}

        {loadStatus === 'error' && (
          <div
            className="space-y-3 rounded-md border border-red-500/40 bg-red-950/30 p-5"
            data-testid="canonical-core-error"
            role="alert"
          >
            <h2 className="text-sm font-semibold text-red-200">
              Canonical-core service unreachable
            </h2>
            <p className="break-words text-sm leading-6 text-red-200/80">
              {loadError ?? 'The service did not respond.'}
            </p>
            <p className="text-sm leading-6 text-gray-400">
              Check that the API server is running, then retry.
            </p>
            <button
              type="button"
              data-testid="canonical-core-retry"
              onClick={() => void loadStatusReport()}
              className="rounded bg-red-600 px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-red-500"
            >
              Retry
            </button>
          </div>
        )}

        {loadStatus === 'ready' && status && !status.available && (
          <div
            className="space-y-3 rounded-md border border-amber-500/40 bg-amber-950/20 p-5"
            data-testid="canonical-core-unavailable"
          >
            <h2 className="text-sm font-semibold text-amber-200">
              Workspace not available yet
            </h2>
            <p className="max-w-3xl text-sm leading-6 text-amber-100/80">
              {status.reason}
            </p>
            <div>
              <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-400">
                Next step
              </h3>
              <p className="mt-1 max-w-3xl text-sm leading-6 text-gray-300">
                {status.next_step}
              </p>
            </div>
          </div>
        )}

        {loadStatus === 'ready' && status?.available && (
          <div
            className="rounded-md border border-emerald-500/40 bg-emerald-950/20 p-5"
            data-testid="canonical-core-available"
          >
            <h2 className="text-sm font-semibold text-emerald-200">
              Service available
            </h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-gray-300">
              The canonical-core {status.mode} service is online.
            </p>
          </div>
        )}

        <div className="grid gap-4 md:grid-cols-3">
          <section className="rounded-md border border-gray-800 bg-gray-900 p-5">
            <h2 className="text-sm font-semibold text-gray-100">Shells</h2>
            <p className="mt-3 text-sm leading-6 text-gray-300">
              PyQt6 desktop and React/Tauri web surfaces are both represented by
              the shared launcher manifest.
            </p>
          </section>
          <section className="rounded-md border border-gray-800 bg-gray-900 p-5">
            <h2 className="text-sm font-semibold text-gray-100">Capabilities</h2>
            {status && status.capabilities.length > 0 ? (
              <ul
                className="mt-3 space-y-1 text-sm leading-6 text-gray-300"
                data-testid="canonical-core-capabilities"
              >
                {status.capabilities.map((capability) => (
                  <li key={capability} className="font-mono text-xs text-cyan-200">
                    {capability}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="mt-3 text-sm leading-6 text-gray-400">
                Capabilities are reported by the canonical-core service.
              </p>
            )}
          </section>
          <section className="rounded-md border border-gray-800 bg-gray-900 p-5">
            <h2 className="text-sm font-semibold text-gray-100">Route</h2>
            <p className="mt-3 break-words font-mono text-sm text-cyan-200">{route}</p>
          </section>
        </div>
      </section>
    </main>
  );
}

export default CanonicalCoreShellPage;
