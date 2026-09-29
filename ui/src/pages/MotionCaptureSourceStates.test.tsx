/**
 * Regression tests for capture-source load states (issue #8080).
 *
 * Reported symptom: `/tools/motion-capture` stays on "Loading sources..."
 * indefinitely, renders no capture-source control, and offers no error or
 * retry affordance.
 *
 * Cause: the fetch swallowed every failure (`catch { /* API may not be
 * available *\/ }`) into an empty `sources` array, and the sidebar rendered the
 * loading text whenever that array was empty. Failure, "not yet loaded", and
 * "genuinely empty" were all the same observable state. The request also had no
 * timeout, so a hung API left the promise pending forever.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

vi.mock('@/api/fetch', () => ({
  apiFetch: vi.fn(),
  apiFetchForm: vi.fn(),
}));

// #8406: keep the live pose channel inert so no real WebSocket is opened.
vi.mock('@/hooks/useRealtimeChannel', () => ({
  useRealtimeChannel: vi.fn(() => ({ message: null, status: 'connecting' })),
}));

import { apiFetch } from '@/api/fetch';
import { MotionCapturePage } from './MotionCapture';
import type { CaptureSource } from './MotionCapture';

const mockedApiFetch = vi.mocked(apiFetch);

const SOURCES: CaptureSource[] = [
  {
    id: 'c3d',
    name: 'C3D File Import',
    type: 'c3d',
    available: true,
    reason: null,
    description: 'Import motion capture data from C3D files',
  },
];

/** Route non-source calls to benign empty responses. */
function routeOthers(path: string): unknown {
  if (path.includes('/skeleton/')) return [];
  if (path.endsWith('/recordings')) return [];
  throw new Error(`unexpected path ${path}`);
}

beforeEach(() => {
  mockedApiFetch.mockReset();
});

describe('capture-source error state (#8080)', () => {
  it('replaces the loading indicator with an actionable error', async () => {
    mockedApiFetch.mockImplementation(async (path: string) => {
      if (path.endsWith('/sources')) {
        throw new Error('API route not found');
      }
      return routeOthers(path);
    });

    render(<MotionCapturePage />);

    const alert = await screen.findByTestId('sources-error');
    expect(alert).toHaveTextContent('Capture sources unavailable');
    // The underlying reason is surfaced, not swallowed.
    expect(alert).toHaveTextContent('API route not found');
    // And a next step is given.
    expect(alert).toHaveTextContent(/API server is running/i);
    expect(screen.getByTestId('sources-retry')).toBeInTheDocument();

    // The core defect: the spinner must be gone.
    expect(screen.queryByTestId('sources-loading')).not.toBeInTheDocument();
    expect(screen.queryByText('Loading sources...')).not.toBeInTheDocument();
  });

  it('surfaces a timeout as an error rather than hanging forever', async () => {
    mockedApiFetch.mockImplementation(async (path: string) => {
      if (path.endsWith('/sources')) {
        throw new Error('Request timed out after 15000ms — /api/...');
      }
      return routeOthers(path);
    });

    render(<MotionCapturePage />);

    const alert = await screen.findByTestId('sources-error');
    expect(alert).toHaveTextContent(/timed out/i);
  });

  it('treats a malformed (non-array) body as an error, not an empty list', async () => {
    mockedApiFetch.mockImplementation(async (path: string) => {
      if (path.endsWith('/sources')) return { sources: [] } as unknown;
      return routeOthers(path);
    });

    render(<MotionCapturePage />);

    const alert = await screen.findByTestId('sources-error');
    expect(alert).toHaveTextContent(/not a list/i);
  });

  it('retry re-issues the request and recovers on success', async () => {
    const user = userEvent.setup();
    let attempt = 0;
    mockedApiFetch.mockImplementation(async (path: string) => {
      if (path.endsWith('/sources')) {
        attempt += 1;
        if (attempt === 1) throw new Error('Data service unreachable');
        return SOURCES as unknown;
      }
      return routeOthers(path);
    });

    render(<MotionCapturePage />);
    await screen.findByTestId('sources-error');

    await user.click(screen.getByTestId('sources-retry'));

    await waitFor(() => {
      expect(screen.getByTestId('source-c3d')).toBeInTheDocument();
    });
    expect(screen.queryByTestId('sources-error')).not.toBeInTheDocument();
    expect(attempt).toBe(2);
  });
});

describe('capture-source empty state (#8080)', () => {
  it('distinguishes a genuinely empty catalogue from still-loading', async () => {
    mockedApiFetch.mockImplementation(async (path: string) => {
      if (path.endsWith('/sources')) return [] as unknown;
      return routeOthers(path);
    });

    render(<MotionCapturePage />);

    const empty = await screen.findByTestId('sources-empty');
    expect(empty).toHaveTextContent('No capture sources configured');
    expect(empty).toHaveTextContent(/C3D/);
    expect(screen.queryByTestId('sources-loading')).not.toBeInTheDocument();
    expect(screen.queryByTestId('sources-error')).not.toBeInTheDocument();
  });
});

describe('capture-source ready state (#8080)', () => {
  it('renders the source control and drops the loading indicator', async () => {
    mockedApiFetch.mockImplementation(async (path: string) => {
      if (path.endsWith('/sources')) return SOURCES as unknown;
      return routeOthers(path);
    });

    render(<MotionCapturePage />);

    await screen.findByTestId('source-c3d');
    expect(screen.queryByTestId('sources-loading')).not.toBeInTheDocument();
    expect(screen.queryByTestId('sources-empty')).not.toBeInTheDocument();
    expect(screen.queryByTestId('sources-error')).not.toBeInTheDocument();
  });

  it('shows the loading indicator only while the request is pending', async () => {
    let resolveSources: (value: unknown) => void = () => {};
    mockedApiFetch.mockImplementation(async (path: string) => {
      if (path.endsWith('/sources')) {
        return new Promise((resolve) => {
          resolveSources = resolve;
        });
      }
      return routeOthers(path);
    });

    render(<MotionCapturePage />);

    expect(await screen.findByTestId('sources-loading')).toBeInTheDocument();

    resolveSources(SOURCES);

    await waitFor(() => {
      expect(screen.queryByTestId('sources-loading')).not.toBeInTheDocument();
    });
    expect(screen.getByTestId('source-c3d')).toBeInTheDocument();
  });
});
