import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { KNOWN_APP_ROUTES } from './routes';

/**
 * Route parity guard (#14254). `KNOWN_APP_ROUTES` claims to be the single source
 * of truth for in-app routes, so every static `<Route path>` in `App.tsx` must be
 * listed there and vice versa. Dynamic (`:param`) and catch-all (`*`) routes are
 * not exact paths that `isKnownAppRoute()` can match, so they are excluded.
 */
const APP_SOURCE = readFileSync(resolve(__dirname, 'App.tsx'), 'utf-8');
const ROUTE_PATH = /<Route\b[^>]*?\bpath="([^"]+)"/gs;

function staticRouterPaths(source: string): string[] {
  const paths = [...source.matchAll(ROUTE_PATH)].map((match) => match[1]);
  return paths.filter((path) => path !== '*' && !path.includes(':'));
}

describe('KNOWN_APP_ROUTES parity with App.tsx', () => {
  const routerPaths = staticRouterPaths(APP_SOURCE);

  it('finds the router table in App.tsx', () => {
    expect(routerPaths).toContain('/');
  });

  it('lists every static App.tsx route', () => {
    const known = new Set<string>(KNOWN_APP_ROUTES);
    expect(routerPaths.filter((path) => !known.has(path))).toEqual([]);
  });

  it('has no entry without a matching App.tsx route', () => {
    const routed = new Set(routerPaths);
    expect(KNOWN_APP_ROUTES.filter((path) => !routed.has(path))).toEqual([]);
  });
});
