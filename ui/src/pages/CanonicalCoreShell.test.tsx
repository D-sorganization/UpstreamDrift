/**
 * Regression tests for the canonical-core workspaces (issue #8081).
 *
 * Reported symptom: `/tools/canonical-core/estimation` and
 * `/tools/canonical-core/comparison` rendered only static text. Neither
 * exposed an input, a dataset or engine selector, an execution action, a
 * result, an empty state, or a service-status error — so the page was
 * indistinguishable from a broken one.
 *
 * The workspaces have no compute service yet, so the contract asserted here is
 * the issue's second acceptable outcome: an explicit, actionable unavailable
 * state carrying a reason and a next step, plus a real error path when the
 * service cannot be reached.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

vi.mock('@/api/fetch', () => ({
  apiFetch: vi.fn(),
  apiFetchForm: vi.fn(),
}));

import { apiFetch } from '@/api/fetch';
import {
  CanonicalCoreShellPage,
  parseCanonicalCoreStatus,
} from './CanonicalCoreShell';

const mockedApiFetch = vi.mocked(apiFetch);

const UNAVAILABLE_ESTIMATION = {
  tool_id: 'canonical_core_estimation',
  mode: 'estimation',
  name: 'Canonical-Core Estimation',
  description: 'Workspace entry point for CC-19 estimation services.',
  web_route: '/tools/canonical-core/estimation',
  capabilities: ['canonical_core', 'estimation'],
  available: false,
  reason: 'The canonical-core estimation service is not implemented yet.',
  next_step: 'Track CC-19 for the estimation service.',
};

beforeEach(() => {
  mockedApiFetch.mockReset();
});

describe('canonical-core unavailable state (#8081)', () => {
  it.each(['estimation', 'comparison'] as const)(
    'renders reason and next step for %s',
    async (mode) => {
      mockedApiFetch.mockResolvedValue({
        ...UNAVAILABLE_ESTIMATION,
        mode,
        reason: `The canonical-core ${mode} service is not implemented yet.`,
        next_step: `Track the ${mode} service issue.`,
      });

      render(<CanonicalCoreShellPage mode={mode} />);

      const panel = await screen.findByTestId('canonical-core-unavailable');
      expect(panel).toHaveTextContent('Workspace not available yet');
      expect(panel).toHaveTextContent(`canonical-core ${mode} service`);
      expect(panel).toHaveTextContent('Next step');
      expect(panel).toHaveTextContent(`Track the ${mode} service issue.`);
    },
  );

  it('queries the status route for the mode it renders', async () => {
    mockedApiFetch.mockResolvedValue({ ...UNAVAILABLE_ESTIMATION, mode: 'comparison' });

    render(<CanonicalCoreShellPage mode="comparison" />);

    await screen.findByTestId('canonical-core-unavailable');
    expect(mockedApiFetch).toHaveBeenCalledWith(
      '/api/tools/canonical-core/comparison/status',
    );
  });

  it('renders capabilities reported by the service', async () => {
    mockedApiFetch.mockResolvedValue(UNAVAILABLE_ESTIMATION);

    render(<CanonicalCoreShellPage mode="estimation" />);

    const caps = await screen.findByTestId('canonical-core-capabilities');
    expect(caps).toHaveTextContent('canonical_core');
    expect(caps).toHaveTextContent('estimation');
  });
});

describe('canonical-core service error state (#8081)', () => {
  it('shows an actionable error with retry when the service is unreachable', async () => {
    mockedApiFetch.mockRejectedValue(new Error('API route not found'));

    render(<CanonicalCoreShellPage mode="estimation" />);

    const alert = await screen.findByTestId('canonical-core-error');
    expect(alert).toHaveTextContent('Canonical-core service unreachable');
    expect(alert).toHaveTextContent('API route not found');
    expect(alert).toHaveTextContent(/API server is running/i);
    expect(screen.getByTestId('canonical-core-retry')).toBeInTheDocument();
    expect(screen.queryByTestId('canonical-core-loading')).not.toBeInTheDocument();
  });

  it('retry re-issues the request and recovers', async () => {
    const user = userEvent.setup();
    mockedApiFetch
      .mockRejectedValueOnce(new Error('Data service unreachable'))
      .mockResolvedValueOnce(UNAVAILABLE_ESTIMATION);

    render(<CanonicalCoreShellPage mode="estimation" />);
    await screen.findByTestId('canonical-core-error');

    await user.click(screen.getByTestId('canonical-core-retry'));

    await waitFor(() => {
      expect(screen.getByTestId('canonical-core-unavailable')).toBeInTheDocument();
    });
    expect(screen.queryByTestId('canonical-core-error')).not.toBeInTheDocument();
  });

  it('treats a malformed payload as an error, not a blank shell', async () => {
    mockedApiFetch.mockResolvedValue({ unexpected: true });

    render(<CanonicalCoreShellPage mode="estimation" />);

    const alert = await screen.findByTestId('canonical-core-error');
    expect(alert).toHaveTextContent(/missing "mode"/i);
  });
});

describe('canonical-core available state (#8081)', () => {
  it('renders the workspace once a service reports available', async () => {
    mockedApiFetch.mockResolvedValue({
      ...UNAVAILABLE_ESTIMATION,
      available: true,
      reason: '',
      next_step: '',
    });

    render(<CanonicalCoreShellPage mode="estimation" />);

    expect(await screen.findByTestId('canonical-core-available')).toHaveTextContent(
      'Service available',
    );
    expect(screen.queryByTestId('canonical-core-unavailable')).not.toBeInTheDocument();
  });
});

describe('parseCanonicalCoreStatus', () => {
  it('accepts a well-formed payload', () => {
    expect(parseCanonicalCoreStatus(UNAVAILABLE_ESTIMATION).mode).toBe('estimation');
  });

  it.each([
    [null, /not an object/i],
    ['a string', /not an object/i],
    [{ available: false }, /missing "mode"/i],
    [{ mode: '' , available: false }, /missing "mode"/i],
    [{ mode: 'estimation' }, /missing "available"/i],
  ])('rejects %j', (raw, pattern) => {
    expect(() => parseCanonicalCoreStatus(raw)).toThrow(pattern as RegExp);
  });

  it('tolerates a missing capabilities list', () => {
    const parsed = parseCanonicalCoreStatus({ mode: 'estimation', available: false });
    expect(parsed.capabilities).toEqual([]);
  });
});
