/**
 * Tests for PlotsSection — selector populated from /api/analysis/plot-types
 * and plot fetching/error behavior (issue #7449).
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

const apiFetchMock = vi.fn();

vi.mock('@/api/fetch', () => ({
  apiFetch: (...args: unknown[]) => apiFetchMock(...args),
}));

// Mock recharts to avoid canvas/SVG layout in jsdom.
vi.mock('recharts', () => ({
  LineChart: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="line-chart-mock">{children}</div>
  ),
  Line: ({ name }: { name: string }) => <div data-testid="line-mock">{name}</div>,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
  Legend: () => null,
  RadarChart: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="radar-chart-mock">{children}</div>
  ),
  Radar: () => null,
  PolarGrid: () => null,
  PolarAngleAxis: () => null,
  PolarRadiusAxis: () => null,
}));

import { PlotsSection } from './PlotsSection';

const PLOT_TYPES = {
  plot_types: [
    { id: 'energies', label: 'Energies' },
    { id: 'joint_angles', label: 'Joint Angles' },
  ],
};

const ENERGIES_PLOT = {
  plot_type: 'energies',
  title: 'Energy Analysis',
  x_label: 'Time (s)',
  y_label: 'Energy (J)',
  series: [
    { name: 'Kinetic Energy', x: [0, 0.1], y: [0, 2], units: 'J', metadata: {} },
    { name: 'Potential Energy', x: [0, 0.1], y: [3, 2], units: 'J', metadata: {} },
  ],
  metadata: {},
};

describe('PlotsSection', () => {
  beforeEach(() => {
    apiFetchMock.mockReset();
  });

  it('populates the selector from /api/analysis/plot-types', async () => {
    apiFetchMock.mockResolvedValueOnce(PLOT_TYPES);
    render(<PlotsSection />);

    await waitFor(() =>
      expect(apiFetchMock).toHaveBeenCalledWith('/api/analysis/plot-types'),
    );
    expect(await screen.findByRole('option', { name: 'Energies' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Joint Angles' })).toBeInTheDocument();
  });

  it('fetches and renders plot data when a type is selected', async () => {
    apiFetchMock
      .mockResolvedValueOnce(PLOT_TYPES)
      .mockResolvedValueOnce(ENERGIES_PLOT);
    render(<PlotsSection />);

    await screen.findByRole('option', { name: 'Energies' });
    fireEvent.change(screen.getByLabelText('Plot type'), {
      target: { value: 'energies' },
    });

    await waitFor(() =>
      expect(apiFetchMock).toHaveBeenCalledWith('/api/analysis/plot-data/energies'),
    );
    expect(await screen.findByTestId('plot-data-chart')).toBeInTheDocument();
    expect(screen.getByText('Energy Analysis')).toBeInTheDocument();
    expect(screen.getAllByTestId('line-mock')).toHaveLength(2);
  });

  it('shows the backend error detail when plot data is unavailable (409)', async () => {
    apiFetchMock
      .mockResolvedValueOnce(PLOT_TYPES)
      .mockRejectedValueOnce(new Error('No simulation data available.'));
    render(<PlotsSection />);

    await screen.findByRole('option', { name: 'Energies' });
    fireEvent.change(screen.getByLabelText('Plot type'), {
      target: { value: 'energies' },
    });

    expect(await screen.findByTestId('plots-error')).toHaveTextContent(
      'No simulation data available.',
    );
    expect(screen.queryByTestId('plot-data-chart')).not.toBeInTheDocument();
  });

  it('shows an idle hint before any plot is selected', async () => {
    apiFetchMock.mockResolvedValueOnce(PLOT_TYPES);
    render(<PlotsSection />);

    expect(
      await screen.findByText(/Run a simulation, then pick a plot type/),
    ).toBeInTheDocument();
  });
});
