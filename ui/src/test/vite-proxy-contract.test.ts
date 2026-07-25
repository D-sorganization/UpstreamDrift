/**
 * Regression tests for the Vite dev-server proxy contract (issue #8076).
 *
 * Nothing previously asserted anything about the dev proxy, so it drifted to
 * port 8001 while every other artifact (`BACKEND_PORT`,
 * `launch_upstream_drift.py --port`, `ui/README.md`, `docs/README.md`) said
 * 8000. The documented startup path therefore failed at
 * `/api/launcher/manifest` with HTTP 500.
 *
 * The `/api/ws`-before-`/api` ordering assertions are hygiene: Vite 7 upgrades
 * correctly either way, but pinning the order keeps the config from depending
 * on undocumented first-match behaviour.
 */

import { describe, it, expect } from 'vitest';
import { BACKEND_PORT } from '@/api/backend';
import {
  DEFAULT_API_PORT,
  DEV_SERVER_PORT,
  buildDevProxy,
  resolveApiPort,
} from '@/config/devProxy';

describe('dev proxy port (#8076)', () => {
  it('defaults to the frontend single-source-of-truth backend port', () => {
    expect(DEFAULT_API_PORT).toBe(BACKEND_PORT);
    expect(DEFAULT_API_PORT).toBe(8000);
  });

  it('never targets the old undocumented 8001', () => {
    for (const entry of Object.values(buildDevProxy())) {
      expect(entry.target).not.toContain('8001');
      expect(entry.target).toContain(`:${BACKEND_PORT}`);
    }
  });

  it('falls back to the default when VITE_API_PORT is unset or blank', () => {
    expect(resolveApiPort(undefined)).toBe(DEFAULT_API_PORT);
    expect(resolveApiPort('')).toBe(DEFAULT_API_PORT);
    expect(resolveApiPort('   ')).toBe(DEFAULT_API_PORT);
  });

  it('honours a valid VITE_API_PORT override', () => {
    expect(resolveApiPort('8001')).toBe(8001);
    expect(resolveApiPort(' 9123 ')).toBe(9123);
  });

  it('rejects a malformed VITE_API_PORT instead of silently defaulting', () => {
    for (const bad of ['abc', '0', '-1', '70000', '80.5', '8000extra']) {
      expect(() => resolveApiPort(bad), bad).toThrow(/VITE_API_PORT/);
    }
  });

  it('rejects an out-of-range port passed to buildDevProxy', () => {
    expect(() => buildDevProxy(0)).toThrow(/apiPort/);
    expect(() => buildDevProxy(70000)).toThrow(/apiPort/);
  });
});

describe('dev proxy websocket entries', () => {
  it('declares /api/ws before /api so ordering is explicit', () => {
    const keys = Object.keys(buildDevProxy());
    expect(keys).toContain('/api/ws');
    expect(keys).toContain('/api');
    expect(keys.indexOf('/api/ws')).toBeLessThan(keys.indexOf('/api'));
  });

  it('enables ws on every /api proxy entry', () => {
    for (const [key, entry] of Object.entries(buildDevProxy())) {
      expect(entry.ws, `${key} must set ws: true`).toBe(true);
    }
  });

  it('uses a ws:// target for the websocket entry and http:// for the rest', () => {
    const proxy = buildDevProxy();
    expect(proxy['/api/ws'].target.startsWith('ws://')).toBe(true);
    expect(proxy['/api'].target.startsWith('http://')).toBe(true);
  });

  it('points the websocket entry at the same port as the http entry', () => {
    const proxy = buildDevProxy(8123);
    expect(new URL(proxy['/api/ws'].target).port).toBe(
      new URL(proxy['/api'].target).port,
    );
  });

  it('matches the URL the simulation client actually opens', () => {
    // client.ts builds `${origin}/api/ws/simulate/${engineType}`.
    const path = '/api/ws/simulate/mujoco';
    const keys = Object.keys(buildDevProxy());
    const firstMatch = keys.find((key) => path.startsWith(key));
    expect(firstMatch).toBe('/api/ws');
  });
});

describe('dev server port', () => {
  it('stays on the documented 5180', () => {
    expect(DEV_SERVER_PORT).toBe(5180);
  });
});
