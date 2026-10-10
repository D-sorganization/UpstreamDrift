/**
 * Tests for the Lift Baseline page (LIFT-8 slice 3, #11748).
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { LiftBaselinePage } from './LiftBaseline';
import type { LiftBaselineMetadata, LiftView } from '@/api/lifting';

const SAMPLE_METADATA: LiftBaselineMetadata = {
  schema: 'lift-pack-parity-baseline/v1',
  generated_utc: '2026-01-01T00:00:00+00:00',
  anthropometry: { body_mass_kg: 80 },
  tolerances: { position_m: 0.02, mass_rel: 1e-6 },
  packs: {
    mujoco: { repo: 'MuJoCo_Models', commit: 'abc123def456789', licence: 'MIT License' },
    opensim: { repo: 'OpenSim_Models', commit: 'def456abc789012', licence: 'MIT License' },
  },
  lifts: ['squat', 'deadlift'],
  gap_count: 2,
};

const SQUAT_VIEW: LiftView = {
  lift: 'squat',
  engines: [
    {
      engine: 'mujoco',
      pack: { repo: 'MuJoCo_Models', commit: 'abc123def456789', licence: 'MIT License' },
      structure: { n_bodies: 10, nq: 20, nv: 18 },
      total_mass_kg: { value: 100, reason: null },
      bar_above_sole_m: { value: 1.0, reason: null },
      hand_mid_above_sole_m: { value: 0.9, reason: null },
      smoke: { loaded: true, stepped: true, max_abs_qvel: { value: 0.1, reason: null } },
      start_contact: {
        value_n: { value: 0, reason: null },
        non_ground_normal_force_n: { value: 5, reason: null },
        n_ground_contacts: 1,
        n_non_ground_contacts: 0,
        reason: null,
      },
      phases: [
        {
          name: 'start',
          fraction: { value: 0, reason: null },
          n_targets: 2,
          hand_bar_axis_distance_m: {
            l: { value: 0.3, reason: null },
            r: { value: 0.3, reason: null },
          },
        },
      ],
    },
    {
      engine: 'opensim',
      pack: { repo: 'OpenSim_Models', commit: 'def456abc789012', licence: 'MIT License' },
      structure: { n_bodies: 9, nq: 19, nv: 17 },
      total_mass_kg: { value: null, reason: 'not recorded in the receipt' },
      bar_above_sole_m: { value: 0.98, reason: null },
      hand_mid_above_sole_m: { value: 0.91, reason: null },
      smoke: {
        loaded: true,
        stepped: false,
        max_abs_qvel: { value: null, reason: 'not recorded in the receipt' },
      },
      start_contact: {
        value_n: { value: 0, reason: null },
        non_ground_normal_force_n: { value: 4.5, reason: null },
        n_ground_contacts: 1,
        n_non_ground_contacts: 0,
        reason: null,
      },
      phases: [],
    },
  ],
  comparisons: {
    poses: {
      zero: {
        'mujoco|opensim': {
          segments_max_m: { value: 0.01, reason: null, status: 'pass', tolerance_m: 0.02 },
          hands_max_m: { value: 0.03, reason: null, status: 'fail', tolerance_m: 0.02 },
          feet_max_m: {
            value: null,
            reason: 'not recorded in the receipt',
            status: 'unavailable',
            tolerance_m: 0.02,
          },
          bar_centre_max_m: {
            value: null,
            reason: 'non-finite value in the receipt',
            status: 'unavailable',
            tolerance_m: 0.02,
          },
          com_max_m: { value: 0, reason: null, status: 'pass', tolerance_m: 0.02 },
          lifter_com_max_m: { value: 0.0199, reason: null, status: 'pass', tolerance_m: 0.02 },
        },
      },
    },
    reason: null,
  },
  gaps: [
    {
      key: 'grip_attachment',
      title: 'Right hand is not attached to the bar',
      engines: ['mujoco'],
      evidence: ['mujoco: right-hand attachment is not a bar weld'],
      issues: ['MuJoCo_Models#1'],
      lift_story: 'LIFT-3',
      new_issue: false,
    },
  ],
};

const DEADLIFT_VIEW: LiftView = {
  lift: 'deadlift',
  engines: [
    {
      engine: 'mujoco',
      pack: { repo: 'MuJoCo_Models', commit: 'abc123def456789', licence: 'MIT License' },
      structure: { n_bodies: 10, nq: 20, nv: 18 },
      total_mass_kg: { value: 100, reason: null },
      bar_above_sole_m: { value: 1.0, reason: null },
      hand_mid_above_sole_m: { value: 0.9, reason: null },
      smoke: { loaded: true, stepped: true, max_abs_qvel: { value: 0.1, reason: null } },
      start_contact: {
        value_n: { value: 0, reason: null },
        non_ground_normal_force_n: { value: 5, reason: null },
        n_ground_contacts: 1,
        n_non_ground_contacts: 0,
        reason: null,
      },
      phases: [],
    },
  ],
  comparisons: { poses: {}, reason: 'fewer than two engines available for this lift' },
  gaps: [],
};

const { fetchLiftBaselineMock, fetchLiftBaselineLiftMock } = vi.hoisted(() => ({
  fetchLiftBaselineMock: vi.fn(),
  fetchLiftBaselineLiftMock: vi.fn(),
}));

vi.mock('@/api/lifting', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api/lifting')>();
  return {
    ...actual,
    fetchLiftBaseline: fetchLiftBaselineMock,
    fetchLiftBaselineLift: fetchLiftBaselineLiftMock,
  };
});

describe('LiftBaselinePage', () => {
  beforeEach(() => {
    fetchLiftBaselineMock.mockReset();
    fetchLiftBaselineLiftMock.mockReset();
    fetchLiftBaselineMock.mockResolvedValue(SAMPLE_METADATA);
    fetchLiftBaselineLiftMock.mockImplementation(async (lift: string) =>
      lift === 'deadlift' ? DEADLIFT_VIEW : SQUAT_VIEW,
    );
  });

  it('renders metadata and auto-selects the first lift', async () => {
    render(<LiftBaselinePage />);

    expect(await screen.findByText('Lift Baseline')).toBeInTheDocument();
    expect(await screen.findByText('2026-01-01T00:00:00+00:00')).toBeInTheDocument();

    const toleranceDt = screen.getByText('Position tolerance');
    expect(toleranceDt.nextElementSibling).toHaveTextContent('0.02 m');

    const gapsDt = screen.getByText('Gaps');
    expect(gapsDt.nextElementSibling).toHaveTextContent('2');

    const select = await screen.findByRole('combobox', { name: 'Lift' });
    expect(select).toHaveValue('squat');
    await waitFor(() => {
      expect(fetchLiftBaselineLiftMock).toHaveBeenCalledWith('squat');
    });
  });

  it('renders a row for each engine of the selected lift', async () => {
    render(<LiftBaselinePage />);

    expect(await screen.findByText('mujoco')).toBeInTheDocument();
    expect(screen.getByText('opensim')).toBeInTheDocument();
  });

  it('renders "unavailable" for a null measurement and never confuses a real zero with it', async () => {
    render(<LiftBaselinePage />);

    await screen.findByText('opensim');
    const unavailableCells = await screen.findAllByText('unavailable');
    expect(unavailableCells.length).toBeGreaterThan(0);
    // mujoco's start-contact value_n is a real 0, which must render as a
    // formatted number, never as "unavailable".
    expect(screen.getAllByText('0.00 N').length).toBeGreaterThan(0);
  });

  it('shows an error state when the lift fetch rejects', async () => {
    fetchLiftBaselineLiftMock.mockReset();
    fetchLiftBaselineLiftMock.mockRejectedValue(new Error('lift view unavailable'));

    render(<LiftBaselinePage />);

    expect(await screen.findByText('lift view unavailable')).toBeInTheDocument();
  });

  it('shows a fail badge for a failing cross-engine pair metric', async () => {
    render(<LiftBaselinePage />);

    expect(await screen.findByText(/fail/)).toBeInTheDocument();
    expect(screen.getAllByText(/pass/).length).toBeGreaterThan(0);
  });

  it('fetches the new lift when the lift select changes', async () => {
    const user = userEvent.setup();
    render(<LiftBaselinePage />);

    await screen.findByText('mujoco');
    const select = screen.getByRole('combobox', { name: 'Lift' });
    await user.selectOptions(select, 'deadlift');

    await waitFor(() => {
      expect(fetchLiftBaselineLiftMock).toHaveBeenCalledWith('deadlift');
    });
    await waitFor(() => {
      expect(screen.queryByText('opensim')).not.toBeInTheDocument();
    });
  });

  it('shows an error state when the metadata fetch rejects', async () => {
    fetchLiftBaselineMock.mockReset();
    fetchLiftBaselineMock.mockRejectedValue(new Error('backend unavailable'));

    render(<LiftBaselinePage />);

    expect(await screen.findByText('backend unavailable')).toBeInTheDocument();
  });

  it('shows the comparisons reason when fewer than two engines are available', async () => {
    const user = userEvent.setup();
    render(<LiftBaselinePage />);

    await screen.findByText('mujoco');
    const select = screen.getByRole('combobox', { name: 'Lift' });
    await user.selectOptions(select, 'deadlift');

    expect(
      await screen.findByText('fewer than two engines available for this lift'),
    ).toBeInTheDocument();
  });
});
