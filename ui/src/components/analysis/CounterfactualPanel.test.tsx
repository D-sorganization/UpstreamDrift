/**
 * CounterfactualPanel tests (issue #7450).
 *
 * Mocks the API layer to cover capability gating, the async-task run
 * flow (start -> poll -> completed/failed), and result rendering.
 */

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { CounterfactualPanel } from './CounterfactualPanel';
import { apiFetch } from '@/api/fetch';

vi.mock('@/api/fetch', () => ({
  apiFetch: vi.fn(),
}));

// Recharts' ResponsiveContainer needs a real layout box; stub it in JSDOM.
vi.mock('recharts', async (importOriginal) => {
  const actual = await importOriginal<typeof import('recharts')>();
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
      <div style={{ width: 400, height: 300 }}>{children}</div>
    ),
  };
});

const mockedFetch = vi.mocked(apiFetch);

const SUPPORT_FULL = {
  kinds: ['control', 'drift', 'gravity', 'total', 'ztcf', 'zvcf'],
  engine: 'pendulum',
  session_available: true,
};

const RESULT = {
  kind: 'ztcf',
  times: [0, 0.01, 0.02],
  values: [
    [0.1, -0.2],
    [0.3, -0.4],
    [0.5, -0.6],
  ],
  units: 'rad/s^2',
  metadata: { n_frames: 3 },
};

beforeEach(() => {
  mockedFetch.mockReset();
});

describe('CounterfactualPanel — capability gating', () => {
  it('shows a note and disables Run when there is no session', async () => {
    mockedFetch.mockResolvedValueOnce({
      kinds: [],
      engine: null,
      session_available: false,
    });
    render(<CounterfactualPanel />);
    await waitFor(() =>
      expect(screen.getByTestId('no-session-note')).toBeInTheDocument(),
    );
    expect(screen.getByTestId('counterfactual-run')).toBeDisabled();
  });

  it('disables Run and lists supported kinds for an unsupported kind', async () => {
    mockedFetch.mockResolvedValueOnce({
      kinds: ['gravity'],
      engine: 'partial',
      session_available: true,
    });
    render(<CounterfactualPanel />);
    await waitFor(() =>
      expect(screen.getByTestId('unsupported-note')).toBeInTheDocument(),
    );
    expect(screen.getByTestId('unsupported-note').textContent).toContain('gravity');
    expect(screen.getByTestId('counterfactual-run')).toBeDisabled();
  });

  it('enables Run when the selected kind is supported', async () => {
    mockedFetch.mockResolvedValueOnce(SUPPORT_FULL);
    render(<CounterfactualPanel />);
    await waitFor(() =>
      expect(screen.getByTestId('counterfactual-run')).toBeEnabled(),
    );
    expect(screen.getByText(/engine: pendulum/)).toBeInTheDocument();
  });

  it('uses the metadata-driven HelpfulField kind selector', async () => {
    mockedFetch.mockResolvedValueOnce(SUPPORT_FULL);
    render(<CounterfactualPanel />);
    const select = await screen.findByLabelText(/Counterfactual kind/);
    expect(select.tagName).toBe('SELECT');
    expect(screen.getByRole('option', { name: 'zvcf' })).toBeInTheDocument();
  });
});

describe('CounterfactualPanel — run flow and rendering', () => {
  it('runs a task, polls status, and renders chart + summary stats', async () => {
    mockedFetch
      .mockResolvedValueOnce(SUPPORT_FULL) // kinds
      .mockResolvedValueOnce({ task_id: 't1', status: 'started', kind: 'ztcf' })
      .mockResolvedValueOnce({ status: 'completed', result: RESULT });

    render(<CounterfactualPanel />);
    const button = await screen.findByTestId('counterfactual-run');
    await waitFor(() => expect(button).toBeEnabled());
    fireEvent.click(button);

    await waitFor(() =>
      expect(screen.getByTestId('counterfactual-result')).toBeInTheDocument(),
    );
    expect(screen.getByTestId('counterfactual-chart')).toBeInTheDocument();
    // Summary stats: 3 frames x 2 DoFs, peak |accel| = 0.6 at t=0.02
    expect(screen.getByText('3 × 2')).toBeInTheDocument();
    expect(screen.getByText('0.6000')).toBeInTheDocument();
    expect(screen.getByText('0.020')).toBeInTheDocument();

    expect(mockedFetch).toHaveBeenCalledWith(
      '/api/analysis/counterfactual',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ kind: 'ztcf' }),
      }),
    );
    expect(mockedFetch).toHaveBeenCalledWith('/api/simulate/status/t1');
  });

  it('surfaces a failed task error', async () => {
    mockedFetch
      .mockResolvedValueOnce(SUPPORT_FULL)
      .mockResolvedValueOnce({ task_id: 't2', status: 'started', kind: 'ztcf' })
      .mockResolvedValueOnce({ status: 'failed', error: 'engine exploded' });

    render(<CounterfactualPanel />);
    const button = await screen.findByTestId('counterfactual-run');
    await waitFor(() => expect(button).toBeEnabled());
    fireEvent.click(button);

    await waitFor(() =>
      expect(screen.getByTestId('counterfactual-error')).toBeInTheDocument(),
    );
    expect(screen.getByTestId('counterfactual-error').textContent).toContain(
      'engine exploded',
    );
    expect(screen.queryByTestId('counterfactual-result')).not.toBeInTheDocument();
  });

  it('surfaces an HTTP 409 from the start call', async () => {
    mockedFetch
      .mockResolvedValueOnce(SUPPORT_FULL)
      .mockRejectedValueOnce(new Error('No completed simulation session'));

    render(<CounterfactualPanel />);
    const button = await screen.findByTestId('counterfactual-run');
    await waitFor(() => expect(button).toBeEnabled());
    fireEvent.click(button);

    await waitFor(() =>
      expect(screen.getByTestId('counterfactual-error')).toBeInTheDocument(),
    );
    expect(screen.getByTestId('counterfactual-error').textContent).toContain(
      'No completed simulation session',
    );
  });
});
