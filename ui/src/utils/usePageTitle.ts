import { useEffect } from 'react';

const APP_NAME = 'Golf Modeling Suite';

/**
 * Sets `document.title` to `"<title> — Golf Modeling Suite"` for the lifetime
 * of the calling component, restoring nothing on unmount (the next page sets
 * its own title). Satisfies WCAG 2.4.2 (Page Titled) by giving every route a
 * distinct, descriptive tab/window title.
 *
 * @param title Human-readable page name (e.g. "Simulation"). Must be non-empty.
 */
export function usePageTitle(title: string): void {
  useEffect(() => {
    const trimmed = title.trim();
    document.title = trimmed ? `${trimmed} — ${APP_NAME}` : APP_NAME;
  }, [title]);
}
