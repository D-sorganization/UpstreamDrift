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

import { useEffect, useMemo, useState } from 'react';
import { apiFetch } from '@/api/fetch';
import {
  type CanonicalCoreStatus,
  parseCanonicalCoreStatus,
} from './canonicalCoreStatus';

type CanonicalCoreMode = 'estimation' | 'comparison';

type LoadStatus = 'loading' | 'ready' | 'error';

interface RequestState {
  mode: CanonicalCoreMode;
  loadStatus: LoadStatus;
  status: CanonicalCoreStatus | null;
  loadError: string | null;
}

interface CanonicalCoreShellPageProps {
  mode: CanonicalCoreMode;
}

const FALLBACK_TITLE: Record<CanonicalCoreMode, string> = {
  estimation: 'Canonical-Core Estimation',
  comparison: 'Canonical-Core Comparison',
};

export function CanonicalCoreShellPage({ mode }: CanonicalCoreShellPageProps) {
  const [requestState, setRequestState] = useState<RequestState>({
    mode,
    loadStatus: 'loading',
    status: null,
    loadError: null,
  });
  const [requestGeneration, setRequestGeneration] = useState(0);

  const route = useMemo(() => `/tools/canonical-core/${mode}`, [mode]);

  useEffect(() => {
    let active = true;

    void apiFetch<unknown>(`/api/tools/canonical-core/${mode}/status`)
      .then((raw) => {
        const status = parseCanonicalCoreStatus(raw);
        if (active) {
          setRequestState({
            mode,
            loadStatus: 'ready',
            status,
            loadError: null,
          });
        }
      })
      .catch((error: unknown) => {
        if (active) {
          setRequestState({
            mode,
            loadStatus: 'error',
            status: null,
            loadError:
              error instanceof Error
                ? error.message
                : 'Failed to reach the canonical-core service',
          });
        }
      });

    return () => {
      active = false;
    };
  }, [mode, requestGeneration]);

  const isCurrentMode = requestState.mode === mode;
  const loadStatus = isCurrentMode ? requestState.loadStatus : 'loading';
  const status = isCurrentMode ? requestState.status : null;
  const loadError = isCurrentMode ? requestState.loadError : null;

  const retry = () => {
    setRequestState({
      mode,
      loadStatus: 'loading',
      status: null,
      loadError: null,
    });
    setRequestGeneration((generation) => generation + 1);
  };

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
              onClick={retry}
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
