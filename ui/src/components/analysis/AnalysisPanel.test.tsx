/**
 * Tests for AnalysisPanel component.
 *
 * See issue #1203
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';

// Mock recharts to avoid canvas rendering in tests
vi.mock('recharts', () => ({
  LineChart: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="line-chart-mock">{children}</div>
  ),
  Line: () => <div data-testid="line-mock" />,
  XAxis: () => <div data-testid="xaxis-mock" />,
  YAxis: () => <div data-testid="yaxis-mock" />,
  CartesianGrid: () => <div data-testid="grid-mock" />,
  Tooltip: () => <div data-testid="tooltip-mock" />,
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="responsive-container-mock">{children}</div>
  ),
  Legend: () => <div data-testid="legend-mock" />,
}));

import { AnalysisPanel } from './AnalysisPanel';

describe('AnalysisPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  describe('rendering', () => {
    it('renders tab headers', () => {
      render(<AnalysisPanel isRunning={false} />);

      expect(screen.getByLabelText('View metrics')).toBeInTheDocument();
      expect(screen.getByLabelText('View plots')).toBeInTheDocument();
      expect(screen.getByLabelText('Export data')).toBeInTheDocument();
    });

    it('shows waiting message when not running', () => {
      render(<AnalysisPanel isRunning={false} />);

      expect(
        screen.getByText('Start a simulation to see metrics.'),
      ).toBeInTheDocument();
    });

    it('shows collecting message when running but no data', () => {
      // Mock fetch to return error (no engine)
      global.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 400,
        json: () => Promise.resolve({ detail: 'No engine loaded' }),
      });

      render(<AnalysisPanel isRunning={true} />);

      // Initially shows collecting message before fetch completes
      expect(
        screen.getByText('Collecting metrics...'),
      ).toBeInTheDocument();
    });
  });

  describe('tab switching', () => {
    it('switches to plots tab', () => {
      render(<AnalysisPanel isRunning={false} />);

      fireEvent.click(screen.getByLabelText('View plots'));

      expect(
        screen.getByText('Start a simulation to see time-series plots.'),
      ).toBeInTheDocument();
    });

    it('switches to export tab', () => {
      render(<AnalysisPanel isRunning={false} />);

      fireEvent.click(screen.getByLabelText('Export data'));

      expect(
        screen.getByText('Export simulation analysis data for offline processing.'),
      ).toBeInTheDocument();
    });
  });

  describe('export', () => {
    it('disables export buttons when no data', () => {
      render(<AnalysisPanel isRunning={false} />);

      fireEvent.click(screen.getByLabelText('Export data'));

      const csvButton = screen.getByLabelText('Export as CSV');
      const jsonButton = screen.getByLabelText('Export as JSON');

      expect(csvButton).toBeDisabled();
      expect(jsonButton).toBeDisabled();
    });
  });

  describe('polling (#8941)', () => {
    function statsResponse(nextSince: number, series: number[]) {
      return {
        ok: true,
        status: 200,
        headers: new Headers({ 'X-Analysis-Next-Since': String(nextSince) }),
        json: () =>
          Promise.resolve({
            sim_time: 1,
            sample_count: nextSince,
            metrics: [],
            time_series: { club_head_speed: series },
          }),
      };
    }

    async function flush() {
      await act(async () => {
        for (let i = 0; i < 5; i++) await Promise.resolve();
      });
    }

    it('sends exactly one request per tick and advances the since cursor', async () => {
      const fetchMock = vi
        .fn()
        .mockResolvedValueOnce(statsResponse(3, [1, 2, 3]))
        .mockResolvedValueOnce(statsResponse(4, [4]))
        .mockResolvedValue(statsResponse(5, [5]));
      global.fetch = fetchMock;

      render(<AnalysisPanel isRunning={true} pollInterval={500} />);
      await flush();
      expect(fetchMock).toHaveBeenCalledTimes(1);
      const firstUrl = String(fetchMock.mock.calls[0][0]);
      expect(firstUrl).toContain('/api/analysis/statistics?');
      expect(firstUrl).toContain('collect=true');
      expect(firstUrl).not.toContain('since=');

      await act(async () => {
        vi.advanceTimersByTime(500);
      });
      await flush();
      expect(fetchMock).toHaveBeenCalledTimes(2);
      expect(String(fetchMock.mock.calls[1][0])).toContain('since=3');

      await act(async () => {
        vi.advanceTimersByTime(500);
      });
      await flush();
      expect(fetchMock).toHaveBeenCalledTimes(3);
      expect(String(fetchMock.mock.calls[2][0])).toContain('since=4');
      expect(
        fetchMock.mock.calls.some(([url]) => String(url).includes('/api/analysis/metrics')),
      ).toBe(false);
    });

    it('does not poll while the simulation is stopped', async () => {
      const fetchMock = vi.fn();
      global.fetch = fetchMock;
      render(<AnalysisPanel isRunning={false} />);
      await act(async () => {
        vi.advanceTimersByTime(5000);
      });
      expect(fetchMock).not.toHaveBeenCalled();
    });

    it('stops polling on unmount', async () => {
      const fetchMock = vi.fn().mockResolvedValue(statsResponse(1, [1]));
      global.fetch = fetchMock;
      const { unmount } = render(<AnalysisPanel isRunning={true} />);
      await flush();
      unmount();
      await act(async () => {
        vi.advanceTimersByTime(5000);
      });
      expect(fetchMock).toHaveBeenCalledTimes(1);
    });
  });
});
