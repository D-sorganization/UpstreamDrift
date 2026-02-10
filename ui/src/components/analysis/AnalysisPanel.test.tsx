/**
 * Tests for AnalysisPanel component.
 *
 * See issue #1203
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';

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
});
