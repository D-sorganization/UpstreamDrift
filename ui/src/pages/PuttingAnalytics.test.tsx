import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { PuttingAnalyticsPage } from './PuttingAnalytics';

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });

const LIMITS = { objectives: ['sg', 'must_make'] };
const OPTIMIZE = {
  best: { action: { aim_offset_deg: -2, pace_ft_past: 1 }, expected_strokes: 1.5 },
  top: [{ action: { aim_offset_deg: -2, pace_ft_past: 1 }, expected_strokes: 1.5 }],
};

function stubApi(study: Response) {
  const fetchMock = vi.fn((url: string) =>
    Promise.resolve(url.endsWith('/limits') ? json(LIMITS) : study),
  );
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

function fillBaseline() {
  fireEvent.change(screen.getByLabelText('Baseline distances (ft)'), {
    target: { value: '2 5 10' },
  });
  fireEvent.change(screen.getByLabelText('Baseline putts'), {
    target: { value: '1.01 1.3 1.6' },
  });
}

describe('PuttingAnalyticsPage (#14235)', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('runs the selected study through the API and shows the desktop rows', async () => {
    const fetchMock = stubApi(json(OPTIMIZE));
    render(<PuttingAnalyticsPage />);
    expect(await screen.findByRole('option', { name: 'must_make' })).toBeInTheDocument();
    fillBaseline();
    fireEvent.click(screen.getByRole('button', { name: 'Run study' }));

    expect(await screen.findByText('aim -2.0 deg, pace 1.0 ft')).toBeInTheDocument();
    expect(screen.getByText('aim -2.0 deg, pace 1.0 ft: 1.500 strokes')).toBeInTheDocument();
    expect(screen.getByText('Done: Best aim and pace.')).toBeInTheDocument();
    const [url, init] = fetchMock.mock.calls.find(([u]) => u.endsWith('/optimize'))!;
    expect(url).toBe('/api/tools/putting-analytics/optimize');
    expect(JSON.parse((init as RequestInit).body as string).baseline).toEqual({
      distances_ft: [2, 5, 10],
      expected_putts: [1.01, 1.3, 1.6],
    });
  });

  it('posts the misread study with the reads field', async () => {
    const fetchMock = stubApi(json({ reads: 3 }));
    render(<PuttingAnalyticsPage />);
    fillBaseline();
    fireEvent.change(screen.getByLabelText('Study'), { target: { value: 'misread_cost' } });
    fireEvent.click(screen.getByRole('button', { name: 'Run study' }));
    expect(await screen.findByText('Done: Cost of misreads.')).toBeInTheDocument();
    const call = fetchMock.mock.calls.find(([u]) => u.endsWith('/misread-cost'))!;
    expect(JSON.parse((call[1] as RequestInit).body as string).reads).toHaveLength(3);
  });

  it('refuses bad input without calling the study, and reports a rejection', async () => {
    const fetchMock = stubApi(json({ detail: 'distance_ft must be <= 100' }, 400));
    render(<PuttingAnalyticsPage />);
    fireEvent.click(screen.getByRole('button', { name: 'Run study' }));
    expect(screen.getByText('Check the inputs: baseline distances is empty')).toBeInTheDocument();
    expect(fetchMock.mock.calls.some(([u]) => u.endsWith('/optimize'))).toBe(false);

    fillBaseline();
    fireEvent.click(screen.getByRole('button', { name: 'Run study' }));
    await waitFor(() =>
      expect(screen.getByRole('status')).toHaveTextContent(
        'The study failed: distance_ft must be <= 100',
      ),
    );
  });
});
