/**
 * Tests for AnalysisTools page.
 *
 * Renders the page against backend-shaped responses (issue #7448): the
 * mocked payloads mirror the real contracts in
 * `src/api/routes/analysis_tools.py`, and the export UI must only offer
 * formats the backend actually implements (csv/json — no xlsx/pdf).
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';

import { AnalysisToolsPage } from './AnalysisTools';
import type { MetricsSnapshot, StatisticsSummary } from './AnalysisTools';

// Backend-shaped payloads (src/api/routes/analysis_tools.py)
const metricsPayload: MetricsSnapshot = {
  status: 'ok',
  metrics: {
    sim_time: 1.25,
    max_velocity: 3.4567,
    rms_velocity: 1.2345,
    kinetic_energy: 12.5,
    joint_positions: [0.1, 0.2, 0.3],
  },
};

const statisticsPayload: StatisticsSummary = {
  sim_time: 1.25,
  sample_count: 42,
  metrics: [
    {
      metric_name: 'club_head_speed',
      current: 40.1,
      minimum: 0.0,
      maximum: 45.2,
      mean: 22.3,
      std_dev: 11.7,
    },
    {
      metric_name: 'kinetic_energy',
      current: 12.5,
      minimum: 0.0,
      maximum: 14.1,
      mean: 6.2,
      std_dev: 4.4,
    },
  ],
  time_series: { club_head_speed: [0.0, 20.0, 40.1] },
};

describe('AnalysisToolsPage', () => {
  const mockFetch = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    mockFetch.mockReset();
    vi.stubGlobal('fetch', mockFetch);
    vi.stubGlobal('URL', {
      ...URL,
      createObjectURL: vi.fn(() => 'blob:mock-url'),
      revokeObjectURL: vi.fn(),
    });
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('renders empty state before any fetch', () => {
    render(<AnalysisToolsPage />);
    expect(screen.getByText(/No metrics loaded/)).toBeTruthy();
    expect(screen.getByText(/Click "Load Statistics"/)).toBeTruthy();
  });

  it('fetches and renders the live metrics snapshot', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => metricsPayload,
    });

    render(<AnalysisToolsPage />);
    fireEvent.click(screen.getByText('Refresh Metrics'));

    await waitFor(() => {
      expect(screen.getByText('max velocity')).toBeTruthy();
    });
    expect(mockFetch).toHaveBeenCalledWith(
      '/api/analysis/metrics',
      expect.anything(),
    );
    // Scalar metric rendered with its real value
    expect(screen.getByText('3.4567')).toBeTruthy();
    // Vector metric summarized, not fabricated as a scalar
    expect(screen.getByText('[3 values]')).toBeTruthy();
  });

  it('fetches and renders backend statistics summaries', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => statisticsPayload,
    });

    render(<AnalysisToolsPage />);
    fireEvent.click(screen.getByText('Load Statistics'));

    await waitFor(() => {
      expect(screen.getByTestId('statistics-panel')).toBeTruthy();
    });
    expect(mockFetch).toHaveBeenCalledWith(
      '/api/analysis/statistics',
      expect.anything(),
    );
    expect(screen.getByText(/2 metrics summarized over 42 samples/)).toBeTruthy();
    expect(screen.getByText('club_head_speed')).toBeTruthy();
    expect(screen.getByText('45.200')).toBeTruthy(); // maximum
    expect(screen.getByText('11.700')).toBeTruthy(); // std_dev
  });

  it('shows the backend error instead of fake data when the API fails', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 400,
      statusText: 'Bad Request',
      json: async () => ({ detail: 'No physics engine loaded. Load an engine first.' }),
    });

    render(<AnalysisToolsPage />);
    fireEvent.click(screen.getByText('Refresh Metrics'));

    await waitFor(() => {
      expect(screen.getByText(/No physics engine loaded/)).toBeTruthy();
    });
    expect(screen.getByText(/No metrics loaded/)).toBeTruthy();
  });

  it('only offers export formats the backend implements (no xlsx/pdf)', () => {
    render(<AnalysisToolsPage />);
    const select = screen.getByLabelText('Format') as HTMLSelectElement;
    const values = Array.from(select.options).map((o) => o.value);
    expect(values).toEqual(['csv', 'json']);
    expect(values).not.toContain('xlsx');
    expect(values).not.toContain('pdf');
  });

  it('downloads the streamed export file and reports its real size', async () => {
    const blob = new Blob(['a,b\n1,2\n'], { type: 'text/csv' });
    mockFetch.mockResolvedValueOnce({
      ok: true,
      blob: async () => blob,
    });
    const clickSpy = vi
      .spyOn(HTMLAnchorElement.prototype, 'click')
      .mockImplementation(() => {});

    render(<AnalysisToolsPage />);
    fireEvent.click(screen.getByText('Export'));

    await waitFor(() => {
      expect(screen.getByTestId('export-result')).toBeTruthy();
    });
    expect(mockFetch).toHaveBeenCalledWith(
      '/api/analysis/export',
      expect.objectContaining({ method: 'POST' }),
    );
    const requestBody = JSON.parse(
      (mockFetch.mock.calls[0][1] as RequestInit).body as string,
    );
    expect(requestBody.format).toBe('csv');
    expect(clickSpy).toHaveBeenCalled();
    expect(screen.getByText('analysis_export.csv')).toBeTruthy();
    clickSpy.mockRestore();
  });

  it('surfaces export errors from the backend', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 400,
      statusText: 'Bad Request',
      json: async () => ({ detail: 'No data to export. Run a simulation first.' }),
    });

    render(<AnalysisToolsPage />);
    fireEvent.click(screen.getByText('Export'));

    await waitFor(() => {
      expect(screen.getByText(/No data to export/)).toBeTruthy();
    });
    expect(screen.queryByTestId('export-result')).toBeNull();
  });
});
