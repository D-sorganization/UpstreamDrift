/**
 * Tests for the web Spine Viewer (SPN-15c, #12815).
 *
 * The status banner must always show scientific_status and distribution_source,
 * and anything that is not exactly "qualified" must read as unqualified.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

const apiFetchMock = vi.fn();
vi.mock('@/api/fetch', () => ({
  apiFetch: (...args: unknown[]) => apiFetchMock(...args),
}));

import { SpineViewerPage } from './SpineViewer';
import { SpineStatusBanner } from '@/components/spine/SpineStatusBanner';
import { SpineHeatmap } from '@/components/spine/SpineHeatmap';
import {
  displayStatus,
  binRows,
  type SpineHeatmapPayload,
  type SpineStudyPayload,
} from '@/api/spineViewer';

function heat(quantity: string, values: number[][]): SpineHeatmapPayload {
  return {
    quantity,
    units: 'N',
    sign_convention: 'compression-positive',
    observable: 'joint_reaction',
    distribution_source: 'invented-law-a',
    scientific_status: 'unqualified',
    status_label: 'UNQUALIFIED',
    time_s: values.map((_, i) => i),
    values,
  };
}

function study(overrides: Partial<SpineStudyPayload> = {}): SpineStudyPayload {
  const heatmaps = {
    compression: heat('compression', [
      [0, 0],
      [100, 10],
      [300, 20],
    ]),
    tension: heat('tension', [
      [0, 0],
      [-100, -10],
      [-300, -20],
    ]),
    ap_shear: heat('ap_shear', [[0, 0]]),
    ml_shear: heat('ml_shear', [[0, 0]]),
  };
  return {
    study_id: 's1',
    receipt_status: 'pass',
    status_reasons: [],
    scientific_status: 'unqualified',
    distribution_source: 'invented-law-a',
    banner: 'UNQUALIFIED (run verdict: pass) | distribution_source: invented-law-a',
    levels: ['L5_S1', 'L4_L5'],
    quantities: ['compression', 'tension', 'ap_shear', 'ml_shear'],
    heatmaps,
    ...overrides,
  };
}

describe('displayStatus', () => {
  it('is qualified only for the exact string "qualified"', () => {
    expect(displayStatus('qualified')).toBe('qualified');
    for (const bad of ['unqualified', 'Qualified', 'pass', '', undefined, null]) {
      expect(displayStatus(bad as string | undefined)).toBe('unqualified');
    }
  });
});

describe('binRows', () => {
  it('keeps small inputs and bins large ones to the limit', () => {
    const rows = Array.from({ length: 1000 }, (_, i) => [i, -i]);
    const binned = binRows(rows, 100);
    expect(binned).toHaveLength(100);
    // signed value of largest magnitude per bin
    expect(binned[0][1]).toBe(-9);
    expect(binRows([[1, 2]], 100)).toEqual([[1, 2]]);
  });
});

describe('SpineStatusBanner', () => {
  it('shows unqualified status and distribution source', () => {
    render(<SpineStatusBanner study={study()} />);
    const banner = screen.getByTestId('spine-status-banner');
    expect(banner).toHaveAttribute('data-status', 'unqualified');
    expect(banner).toHaveTextContent('UNQUALIFIED');
    expect(banner).not.toHaveTextContent(/^QUALIFIED/);
    expect(screen.getByTestId('spine-distribution-source')).toHaveTextContent(
      'invented-law-a',
    );
  });

  it('never presents an unknown status as qualified', () => {
    render(
      <SpineStatusBanner
        study={study({ scientific_status: 'weird', distribution_source: '' })}
      />,
    );
    const banner = screen.getByTestId('spine-status-banner');
    expect(banner).toHaveAttribute('data-status', 'unqualified');
    expect(banner).toHaveTextContent('UNQUALIFIED');
    expect(screen.getByTestId('spine-distribution-source')).toHaveTextContent(
      'not recorded',
    );
  });

  it('shows qualified only when the payload says so, with reasons listed', () => {
    render(
      <SpineStatusBanner
        study={study({
          scientific_status: 'qualified',
          status_reasons: ['reason one'],
        })}
      />,
    );
    expect(screen.getByTestId('spine-status-banner')).toHaveAttribute(
      'data-status',
      'qualified',
    );
    expect(screen.getByText('reason one')).toBeInTheDocument();
  });
});

describe('SpineHeatmap', () => {
  it('renders a cell per time row and level with values', () => {
    render(<SpineHeatmap heatmap={heat('compression', [[0, 0], [100, 10], [300, 20]])} levels={['L5_S1', 'L4_L5']} />);
    expect(screen.getAllByTestId('spine-heatmap-cell')).toHaveLength(6);
    expect(screen.getByText('L5_S1')).toBeInTheDocument();
    expect(screen.getByTestId('spine-heatmap')).toHaveAttribute(
      'data-status',
      'unqualified',
    );
  });
});

describe('SpineViewerPage', () => {
  beforeEach(() => {
    apiFetchMock.mockReset();
  });

  it('lists receipts, opens one through the API and shows the banner', async () => {
    apiFetchMock.mockImplementation(async (url: string) => {
      if (url.includes('/receipts')) return { receipts: ['a/s1_receipt.json'] };
      return study();
    });
    render(<SpineViewerPage />);
    const option = await screen.findByRole('option', { name: 'a/s1_receipt.json' });
    expect(option).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Study receipt'), {
      target: { value: 'a/s1_receipt.json' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Open Study' }));
    await waitFor(() =>
      expect(screen.getByTestId('spine-status-banner')).toBeInTheDocument(),
    );
    const call = apiFetchMock.mock.calls.find((c) => String(c[0]).includes('/study'));
    expect(String(call?.[0])).toContain(
      'receipt=' + encodeURIComponent('a/s1_receipt.json'),
    );
    expect(screen.getByTestId('spine-status-banner')).toHaveAttribute(
      'data-status',
      'unqualified',
    );
    fireEvent.change(screen.getByLabelText('Quantity'), {
      target: { value: 'tension' },
    });
    expect(screen.getByTestId('spine-heatmap')).toHaveAttribute(
      'data-quantity',
      'tension',
    );
  });

  it('shows the server error and no heatmap when verification fails', async () => {
    apiFetchMock.mockImplementation(async (url: string) => {
      if (url.includes('/receipts')) return { receipts: ['a/s1_receipt.json'] };
      throw new Error('sha256 of s1_loads.npz does not match the receipt');
    });
    render(<SpineViewerPage />);
    await screen.findByRole('option', { name: 'a/s1_receipt.json' });
    fireEvent.change(screen.getByLabelText('Study receipt'), {
      target: { value: 'a/s1_receipt.json' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Open Study' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('sha256');
    expect(screen.queryByTestId('spine-heatmap')).not.toBeInTheDocument();
    expect(screen.queryByTestId('spine-status-banner')).not.toBeInTheDocument();
  });
});
