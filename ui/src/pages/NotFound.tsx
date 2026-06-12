import { Link, useLocation } from 'react-router-dom';
import { Button } from '@/components/ui';
import { usePageTitle } from '@/utils/usePageTitle';

/**
 * Branded 404 page (UI/UX #7430). Rendered by the catch-all route so mistyped
 * or stale deep links show a recoverable message instead of a blank page.
 */
export function NotFoundPage() {
  usePageTitle('Page not found');
  const { pathname } = useLocation();

  return (
    <div className="flex h-screen items-center justify-center bg-gray-900 p-4">
      <div className="w-full max-w-md rounded-md bg-gray-800 p-8 text-center shadow-lg">
        <p className="text-5xl font-bold text-blue-500">404</p>
        <h1 className="mt-4 text-xl font-semibold text-white">Page not found</h1>
        <p className="mt-2 break-all text-sm text-gray-400">
          No route matches <span className="font-mono text-gray-300">{pathname}</span>
        </p>
        <div className="mt-6 flex items-center justify-center gap-3">
          <Link to="/">
            <Button variant="primary">Back to dashboard</Button>
          </Link>
          <Link to="/simulation">
            <Button variant="secondary">Go to simulation</Button>
          </Link>
        </div>
      </div>
    </div>
  );
}
