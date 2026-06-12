import { useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { titleForPath } from './routeTitles';

const APP_NAME = 'Golf Modeling Suite';

/**
 * Sets `document.title` from the centralized route map on every navigation
 * (UI/UX #7432), giving each route a distinct tab/window title (WCAG 2.4.2).
 * Unknown paths are left untouched so the 404 page's own usePageTitle wins.
 * Mount once inside the router, after <ScrollToTop />.
 */
export function RouteTitle(): null {
  const { pathname } = useLocation();
  useEffect(() => {
    const title = titleForPath(pathname);
    if (title) {
      document.title = `${title} — ${APP_NAME}`;
    }
  }, [pathname]);
  return null;
}
