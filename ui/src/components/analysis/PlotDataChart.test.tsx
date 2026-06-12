/**
 * Tests for PlotDataChart — generic PlotData renderer (issue #7449).
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';

// Mock recharts to avoid canvas/SVG layout in jsdom.
vi.mock('recharts', () => ({
  LineChart: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="line-chart-mock">{children}</div>
  ),
  Line: ({ name }: { name: string }) => <div data-testid="line-mock">{name}</div>,
  XAxis: () => <div data-testid="xaxis-mock" />,
  YAxis: () => <div data-testid="yaxis-mock" />,
  CartesianGrid: () => <div data-testid="grid-mock" />,
  Tooltip: () => <div data-testid="tooltip-mock" />,
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="responsive-container-mock">{children}</div>
  ),
  Legend: () => <div data-testid="legend-mock" />,
  RadarChart: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="radar-chart-mock">{children}</div>
  ),
  Radar: ({ name }: { name: string }) => <div data-testid="radar-mock">{name}</div>,
  PolarGrid: () => <div data-testid="polar-grid-mock" />,
  PolarAngleAxis: () => <div data-testid="polar-angle-mock" />,
  PolarRadiusAxis: () => <div data-testid="polar-radius-mock" />,
}));

import { PlotDataChart } from './PlotDataChart';
import type { PlotData } from '@/api/useAnalysisPlots';

const twoSeriesFixture: PlotData = {
  plot_type: 'energies',
  title: 'Energy Analysis',
  x_label: 'Time (s)',
  y_label: 'Energy (J)',
  series: [
    {
      name: 'Kinetic Energy',
      x: [0, 0.1, 0.2],
      y: [0, 2, 4],
      units: 'J',
      metadata: {},
    },
    {
      name: 'Potential Energy',
      x: [0, 0.1, 0.2],
      y: [3, 2, 1],
      units: 'J',
      metadata: {},
    },
  ],
  metadata: { n_frames: 3 },
};

describe('PlotDataChart', () => {
  it('renders a line per series with payload-driven names and units', () => {
    render(<PlotDataChart data={twoSeriesFixture} />);

    expect(screen.getByTestId('plot-data-chart')).toBeInTheDocument();
    expect(screen.getByTestId('line-chart-mock')).toBeInTheDocument();
    const lines = screen.getAllByTestId('line-mock');
    expect(lines).toHaveLength(2);
    expect(screen.getByText('Kinetic Energy (J)')).toBeInTheDocument();
    expect(screen.getByText('Potential Energy (J)')).toBeInTheDocument();
  });

  it('renders the payload title', () => {
    render(<PlotDataChart data={twoSeriesFixture} />);
    expect(screen.getByText('Energy Analysis')).toBeInTheDocument();
  });

  it('shows the backend empty-state message when no series carry data', () => {
    const empty: PlotData = {
      plot_type: 'joint_angles',
      title: 'Joint Angles vs Time',
      x_label: 'Time (s)',
      y_label: 'Joint Angle (degrees)',
      series: [],
      metadata: { message: 'No data recorded' },
    };
    render(<PlotDataChart data={empty} />);
    expect(screen.getByTestId('plot-empty-state')).toBeInTheDocument();
    expect(screen.getByText('No data recorded')).toBeInTheDocument();
    expect(screen.queryByTestId('line-chart-mock')).not.toBeInTheDocument();
  });

  it('renders a radar chart when the payload carries the radar hint', () => {
    const radar: PlotData = {
      plot_type: 'swing_profile_radar',
      title: 'Swing Profile',
      x_label: '',
      y_label: 'Score',
      series: [
        {
          name: 'Swing Profile',
          x: [0, 1, 2, 3, 4],
          y: [80, 60, 70, 90, 50],
          units: 'score',
          metadata: {
            categories: ['Speed', 'Sequence', 'Stability', 'Efficiency', 'Power'],
          },
        },
      ],
      metadata: { chart: 'radar' },
    };
    render(<PlotDataChart data={radar} />);
    expect(screen.getByTestId('radar-chart-mock')).toBeInTheDocument();
    expect(screen.queryByTestId('line-chart-mock')).not.toBeInTheDocument();
  });
});
