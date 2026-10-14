/**
 * Tests for the BunkerShot3D workbench web page (#9545, slice 3).
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { BunkerWorkbenchPage } from './BunkerWorkbench';
import type {
  BunkerWorkbenchComparisonV1,
  BunkerWorkbenchEvaluationV1,
} from '@/api/bunkerWorkbench';

function evaluation(name: string, carry: number | null): BunkerWorkbenchEvaluationV1 {
  return {
    schema_version: 'bunker-workbench-evaluation/1',
    conventions: {},
    inputs: { design: { name } },
    tier: 'F0',
    tier_model: 'dynamic RFT',
    verdict: {
      status: 'extrapolated',
      headline: `${name}: outside the calibrated envelope`,
      caveats: [],
      reasons: ['firmness outside the calibrated range'],
      reason_count: 1,
    },
    stamp: 'F0 dynamic RFT | EXTRAPOLATED',
    refused: false,
    shot: {
      peak_force_n: 120,
      impulse_n_s: 1.5,
      entry_speed_mps: 20,
      exit_speed_mps: 12,
      max_depth_m: 0.02,
      contact_duration_s: 0.01,
      peak_inertial_fraction: 0.3,
    },
    delivered: {
      effective_loft_deg: 60,
      effective_marketed_bounce_deg: 12,
      presentation_bounce_deg: 14,
      aim_offset_deg: 1,
    },
    carry:
      carry === null
        ? null
        : { carry_m: carry, verdict_status: 'extrapolated', band_m: null, band_reasons: [] },
    unavailable: [],
    sources: {
      model: 'src.tools.bunker_shot_gui.model.WorkbenchModel',
      sand: { firmness: { basis: 'preset', source: 'medium sand preset' } },
      carry_band: null,
    },
    objective: null,
    predictive: false,
    digest: 'a'.repeat(64),
  };
}

function comparison(
  overrides: Partial<BunkerWorkbenchComparisonV1> = {},
): BunkerWorkbenchComparisonV1 {
  return {
    schema_version: 'bunker-workbench-comparison/1',
    conventions: {},
    left: evaluation('Design A', 9.5),
    right: evaluation('Design B', 8.1),
    objective: { use: 'exploratory', target_carry_m: 9, tolerance_fraction: 0.15 },
    objective_standing: {
      disposition: 'unavailable-uncalibrated',
      reasons: ['no sand-to-ball transfer qualification ships'],
      predictive: false,
    },
    ranking: {
      available: true,
      unavailable_reason: null,
      shared_points: 25,
      banded_verdict: 'separated',
      separation_m: 1.2,
      separated: true,
    },
    predictive: false,
    winner: null,
    exploratory_leader: 'Design A',
    verdict_statement: 'Design A leads on an exploratory objective',
    digest: 'b'.repeat(64),
    ...overrides,
  };
}

const { evaluateMock, compareMock } = vi.hoisted(() => ({
  evaluateMock: vi.fn(),
  compareMock: vi.fn(),
}));

vi.mock('@/api/bunkerWorkbench', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api/bunkerWorkbench')>();
  return {
    ...actual,
    evaluateBunkerDesign: evaluateMock,
    compareBunkerDesigns: compareMock,
  };
});

describe('BunkerWorkbenchPage', () => {
  beforeEach(() => {
    evaluateMock.mockReset();
    compareMock.mockReset();
  });

  it('solves design A and shows its tier, verdict, stamp and sources', async () => {
    evaluateMock.mockResolvedValue(evaluation('Design A', 9.5));
    const user = userEvent.setup();
    render(<BunkerWorkbenchPage />);

    const loft = screen.getByLabelText('Design A loft (deg)');
    await user.clear(loft);
    await user.type(loft, '58');
    await user.click(screen.getByRole('button', { name: 'Solve design A' }));

    expect(await screen.findByText('F0 dynamic RFT | EXTRAPOLATED')).toBeInTheDocument();
    expect(screen.getByText('Design A: outside the calibrated envelope')).toBeInTheDocument();
    expect(screen.getByText(/firmness: preset/)).toBeInTheDocument();
    expect(screen.getByText('9.50 m')).toBeInTheDocument();
    const request = evaluateMock.mock.calls[0][0];
    expect(request.handedness).toBe('right');
    expect(request.design).toMatchObject({ name: 'Design A', loft_deg: 58 });
    expect(request.objective).toMatchObject({ use: 'exploratory' });
  });

  it('shows an unqualified A/B lead as exploratory, never as a winner', async () => {
    compareMock.mockResolvedValue(comparison());
    const user = userEvent.setup();
    render(<BunkerWorkbenchPage />);

    await user.click(screen.getByRole('button', { name: 'Compare A/B' }));

    expect(await screen.findByText('unavailable-uncalibrated')).toBeInTheDocument();
    expect(screen.getByText('no sand-to-ball transfer qualification ships')).toBeInTheDocument();
    expect(screen.getByTestId('bunker-winner')).toHaveTextContent('Not ranked');
    expect(screen.getByTestId('bunker-exploratory-leader')).toHaveTextContent('Design A');
    expect(screen.getAllByText('F0 dynamic RFT | EXTRAPOLATED')).toHaveLength(2);
  });

  it('demotes a winner sent on a non-predictive standing', async () => {
    compareMock.mockResolvedValue(comparison({ winner: 'Design B', exploratory_leader: null }));
    const user = userEvent.setup();
    render(<BunkerWorkbenchPage />);

    await user.click(screen.getByRole('button', { name: 'Compare A/B' }));

    expect(await screen.findByTestId('bunker-winner')).toHaveTextContent('Not ranked');
    expect(screen.getByTestId('bunker-exploratory-leader')).toHaveTextContent('Design B');
  });

  it('names the winner only on a supported, predictive standing', async () => {
    compareMock.mockResolvedValue(
      comparison({
        predictive: true,
        winner: 'Design B',
        exploratory_leader: null,
        objective_standing: { disposition: 'supported', reasons: [], predictive: true },
      }),
    );
    const user = userEvent.setup();
    render(<BunkerWorkbenchPage />);

    await user.click(screen.getByRole('button', { name: 'Compare A/B' }));

    expect(await screen.findByTestId('bunker-winner')).toHaveTextContent('Design B');
    expect(screen.queryByTestId('bunker-exploratory-leader')).not.toBeInTheDocument();
  });

  it('refuses a non-numeric field instead of sending the preset value', async () => {
    const user = userEvent.setup();
    render(<BunkerWorkbenchPage />);

    const width = screen.getByLabelText('Design B sole width (mm)');
    await user.clear(width);
    await user.type(width, 'wide');
    await user.click(screen.getByRole('button', { name: 'Compare A/B' }));

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Design B sole width (mm) must be a number',
    );
    expect(compareMock).not.toHaveBeenCalled();
  });

  it('shows the route refusal instead of a result', async () => {
    compareMock.mockRejectedValue(new Error('the two designs must have different names'));
    const user = userEvent.setup();
    render(<BunkerWorkbenchPage />);

    await user.click(screen.getByRole('button', { name: 'Compare A/B' }));

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'the two designs must have different names',
    );
    expect(screen.queryByTestId('bunker-winner')).not.toBeInTheDocument();
  });
});
