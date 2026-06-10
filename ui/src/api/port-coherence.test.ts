/**
 * Dev-proxy / backend-port coherence (issue #7163).
 *
 * The Vite dev proxy must target the same port as BACKEND_PORT (local
 * topology) unless VITE_BACKEND_PORT overrides it (containerized topology).
 * Previously the proxy was hardcoded to 8001 while BACKEND_PORT was 8000, so
 * `npm run dev` against a default local backend got connection-refused.
 */
import { describe, it, expect } from 'vitest';

import { BACKEND_PORT } from './backend';
import {
  DEFAULT_BACKEND_PORT,
  resolveBackendPort,
} from '../config/backendPort';

describe('vite dev proxy port coherence', () => {
  it('default proxy port equals BACKEND_PORT', () => {
    expect(DEFAULT_BACKEND_PORT).toBe(String(BACKEND_PORT));
  });

  it('resolves to the default when VITE_BACKEND_PORT is unset', () => {
    expect(resolveBackendPort({} as NodeJS.ProcessEnv)).toBe(
      String(BACKEND_PORT),
    );
  });

  it('honors VITE_BACKEND_PORT for the containerized topology', () => {
    expect(
      resolveBackendPort({ VITE_BACKEND_PORT: '8001' } as NodeJS.ProcessEnv),
    ).toBe('8001');
  });
});
