import { useEffect } from 'react';
import { useLocation } from 'react-router-dom';

/**
 * Resets the window scroll position to the top whenever the route pathname
 * changes. Mount once directly inside the router. Most pages are
 * `h-screen overflow-hidden` shells, but scrollable pages (Dashboard/launcher
 * and future long pages) otherwise retain the previous page's scroll offset.
 */
export function ScrollToTop(): null {
  const { pathname } = useLocation();
  useEffect(() => {
    if (typeof window.scrollTo === 'function') {
      window.scrollTo(0, 0);
    }
  }, [pathname]);
  return null;
}
