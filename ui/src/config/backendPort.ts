/**
 * Single source of truth for the dev-proxy backend port (issue #7163).
 *
 * Defaults to the local-topology port (8000, matching BACKEND_PORT in
 * src/api/backend.ts). The containerized topology sets VITE_BACKEND_PORT=8001.
 * Kept free of Vite/plugin imports so it is unit-testable in isolation.
 */
export const DEFAULT_BACKEND_PORT = '8000';

export function resolveBackendPort(
  env: NodeJS.ProcessEnv = process.env,
): string {
  return env.VITE_BACKEND_PORT ?? DEFAULT_BACKEND_PORT;
}
