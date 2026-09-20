/**
 * Centralized route → page-title map (UI/UX #7432). Keeping titles in one place
 * (instead of a usePageTitle call duplicated in 13 page components) keeps the
 * tab/window title consistent and DRY. Matched longest-prefix-first so nested
 * tool routes resolve before the root route.
 */
export const ROUTE_TITLES: ReadonlyArray<readonly [string, string]> = [
  ['/simulation', 'Simulation'],
  ['/tools/model-explorer', 'Model Explorer'],
  ['/tools/putting-green', 'Putting Green'],
  ['/tools/video-analyzer', 'Video Analyzer'],
  ['/tools/data-explorer', 'Data Explorer'],
  ['/tools/motion-capture', 'Motion Capture'],
  ['/tools/terrain', 'Terrain'],
  ['/tools/dataset', 'Dataset Generator'],
  ['/tools/analysis', 'Analysis Tools'],
  ['/tools/character-builder', 'Character Builder'],
  ['/tools/canonical-core/estimation', 'Canonical Core — Estimation'],
  ['/tools/canonical-core/comparison', 'Canonical Core — Comparison'],
  ['/chat', 'Chat'],
  ['/', 'Dashboard'],
] as const;

/**
 * Resolves a pathname to its page title, or `null` when no known route matches
 * (the catch-all 404 page sets its own title via usePageTitle).
 */
export function titleForPath(pathname: string): string | null {
  for (const [prefix, title] of ROUTE_TITLES) {
    if (prefix === '/' ? pathname === '/' : pathname.startsWith(prefix)) {
      return title;
    }
  }
  return null;
}
