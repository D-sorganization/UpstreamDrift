/**
 * ErrorBoundary — React error boundary with graceful fallback UI.
 *
 * Catches JavaScript errors anywhere in the child component tree,
 * logs them, and displays a user-friendly fallback instead of a
 * blank screen.
 *
 * Addresses issue #3505 (Phase 6: Error Boundaries & Resilience).
 */

import { Component, type ReactNode } from 'react';
import { logger } from '../../utils/logger';

interface Props {
  /** Content to render when no error has occurred. */
  children: ReactNode;
  /** Optional fallback UI override. */
  fallback?: ReactNode;
  /** Optional callback when an error is caught. */
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void;
  /**
   * When any value in this array changes between renders, a latched error is
   * cleared automatically. Pass the current route (e.g. `[location.pathname]`)
   * so navigating away from a crashed page recovers it (#7434).
   */
  resetKeys?: ReadonlyArray<unknown>;
  /** Human-readable label (e.g. the route path) shown in the fallback. */
  label?: string;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

function keysChanged(
  a: ReadonlyArray<unknown> = [],
  b: ReadonlyArray<unknown> = [],
): boolean {
  return a.length !== b.length || a.some((v, i) => !Object.is(v, b[i]));
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidUpdate(prevProps: Props) {
    // Clear a latched error once a reset key (typically the route) changes, so
    // navigation recovers the boundary without a full reload.
    if (
      this.state.hasError &&
      keysChanged(prevProps.resetKeys, this.props.resetKeys)
    ) {
      this.setState({ hasError: false, error: null });
    }
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    logger.error('ErrorBoundary caught:', error, errorInfo);
    this.props.onError?.(error, errorInfo);
  }

  private handleReload = () => {
    window.location.reload();
  };

  private handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div
          className="flex items-center justify-center min-h-screen bg-gray-900 p-6"
          role="alert"
          aria-live="assertive"
          data-testid="error-boundary-fallback"
        >
          <div className="max-w-md w-full bg-gray-800 border border-red-600/40 rounded-xl p-6 shadow-xl">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-full bg-red-600/20 flex items-center justify-center flex-shrink-0">
                <svg
                  className="w-5 h-5 text-red-400"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  aria-hidden="true"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                  />
                </svg>
              </div>
              <div>
                <h2 className="text-lg font-semibold text-gray-100">Something went wrong</h2>
                <p className="text-sm text-gray-400">
                  {this.props.label
                    ? `The page "${this.props.label}" encountered an unexpected error.`
                    : 'The application encountered an unexpected error.'}
                </p>
              </div>
            </div>

            {this.state.error && (
              <div className="mb-4 p-3 bg-gray-900/80 rounded-lg border border-gray-700 overflow-auto">
                <code className="text-xs text-red-300 font-mono whitespace-pre-wrap">
                  {this.state.error.message}
                </code>
              </div>
            )}

            <div className="flex gap-3">
              <button
                onClick={this.handleReload}
                className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-blue-400"
                aria-label="Reload page"
              >
                Reload Page
              </button>
              <button
                onClick={this.handleReset}
                className="flex-1 px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-200 rounded-lg text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-gray-400"
                aria-label="Try again"
              >
                Try Again
              </button>
            </div>

            {/* Plain anchor: works even when the router context is itself dead. */}
            <a
              href="/"
              className="mt-3 block text-center text-sm text-blue-400 hover:text-blue-300 focus:outline-none focus:ring-2 focus:ring-blue-400 rounded"
            >
              Back to Dashboard
            </a>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;