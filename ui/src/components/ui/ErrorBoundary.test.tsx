/**
 * ErrorBoundary tests (issue #7434) — route-level reset + recovery.
 */

import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { ErrorBoundary } from './ErrorBoundary';

function Boom({ explode }: { explode: boolean }): React.ReactElement {
  if (explode) {
    throw new Error('kaboom');
  }
  return <div>healthy page</div>;
}

describe('ErrorBoundary', () => {
  beforeEach(() => {
    // React logs caught errors to console.error; silence it for clean output.
    vi.spyOn(console, 'error').mockImplementation(() => {});
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders the fallback (with a home link) when a child throws', () => {
    render(
      <ErrorBoundary label="/simulation">
        <Boom explode />
      </ErrorBoundary>,
    );
    expect(screen.getByTestId('error-boundary-fallback')).toBeInTheDocument();
    expect(screen.getByText(/simulation/)).toBeInTheDocument();
    const home = screen.getByText('Back to Dashboard');
    expect(home).toHaveAttribute('href', '/');
  });

  it('clears the error when a resetKey changes (navigation recovery)', () => {
    const { rerender } = render(
      <ErrorBoundary resetKeys={['/simulation']}>
        <Boom explode />
      </ErrorBoundary>,
    );
    expect(screen.getByTestId('error-boundary-fallback')).toBeInTheDocument();

    // Navigate to another route with a healthy page.
    rerender(
      <ErrorBoundary resetKeys={['/dashboard']}>
        <Boom explode={false} />
      </ErrorBoundary>,
    );
    expect(
      screen.queryByTestId('error-boundary-fallback'),
    ).not.toBeInTheDocument();
    expect(screen.getByText('healthy page')).toBeInTheDocument();
  });

  it('keeps showing the fallback while the same resetKey persists', () => {
    const { rerender } = render(
      <ErrorBoundary resetKeys={['/simulation']}>
        <Boom explode />
      </ErrorBoundary>,
    );
    rerender(
      <ErrorBoundary resetKeys={['/simulation']}>
        <Boom explode={false} />
      </ErrorBoundary>,
    );
    expect(screen.getByTestId('error-boundary-fallback')).toBeInTheDocument();
  });
});
