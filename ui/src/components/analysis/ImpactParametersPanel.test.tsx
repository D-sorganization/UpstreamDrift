import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { ImpactParametersPanel } from './ImpactParametersPanel';
import * as api from '@/api/impactParameters';
import type { ImpactParametersResponse } from '@/api/impactParameters';

vi.mock('@/api/fetch', () => ({ apiFetch: vi.fn() }));

const card: ImpactParametersResponse = {
  run_id: 'r1',
  engine: 'mujoco',
  available: true,
  reason: null,
  units: 'mph',
  impact_time_s: 0.06,
  impact_time_source: 'explicit',
  frame: {},
  rows: [
    { key: 'clubhead_speed', label: 'Clubhead Speed', unit: 'mph', value: 89.5, reason: null, note: null },
    { key: 'attack_angle_deg', label: 'Attack Angle', unit: 'deg', value: -4.3, reason: null, note: null },
    { key: 'face_angle_deg', label: 'Face Angle', unit: 'deg', value: null, reason: 'face unobservable: roll', note: null },
    { key: 'smash_factor', label: 'Smash Factor', unit: '', value: null, reason: 'needs ball speed', note: null },
  ],
  d_plane: { club_path_deg: 2, face_angle_deg: null, attack_angle_deg: -4.3, dynamic_loft_deg: 12 },
};

function renderPanel() {
  return render(
    <MemoryRouter>
      <ImpactParametersPanel runId="r1" />
    </MemoryRouter>,
  );
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe('ImpactParametersPanel', () => {
  it('renders values and shows unavailable rows with reasons, never zero', async () => {
    vi.spyOn(api, 'fetchImpactParameters').mockResolvedValue(card);
    renderPanel();
    await waitFor(() => expect(screen.getByTestId('impact-clubhead_speed')).toHaveTextContent('89.5 mph'));
    expect(screen.getByTestId('impact-face_angle_deg')).toHaveTextContent('unavailable');
    expect(screen.getByTestId('impact-face_angle_deg')).toHaveAttribute('title', 'face unobservable: roll');
    expect(screen.getByTestId('impact-smash_factor')).not.toHaveTextContent('0');
    expect(screen.getByTestId('impact-diagram')).toBeInTheDocument();
    expect(screen.getByText('Open in Impact Explorer')).toHaveAttribute(
      'href',
      expect.stringContaining('clubhead_speed=89.500'),
    );
  });

  it('shows the unavailable reason for a run without a clubhead series', async () => {
    vi.spyOn(api, 'fetchImpactParameters').mockResolvedValue({
      ...card, available: false, reason: 'run carries no clubhead series', rows: [], d_plane: {},
    });
    renderPanel();
    expect(await screen.findByTestId('impact-unavailable')).toHaveTextContent('run carries no clubhead series');
  });

  it('refetches with the units toggle and target heading', async () => {
    const spy = vi.spyOn(api, 'fetchImpactParameters').mockResolvedValue(card);
    renderPanel();
    await waitFor(() => expect(spy).toHaveBeenCalledTimes(1));
    fireEvent.change(screen.getByLabelText('Units'), { target: { value: 'm/s' } });
    await waitFor(() =>
      expect(spy).toHaveBeenLastCalledWith(expect.objectContaining({ units: 'm/s' }), expect.anything()),
    );
    fireEvent.change(screen.getByLabelText('Target heading'), { target: { value: '90' } });
    await waitFor(() =>
      expect(spy).toHaveBeenLastCalledWith(
        expect.objectContaining({ targetDir: '1.000000,-0.000000' }),
        expect.anything(),
      ),
    );
  });

  it('surfaces request errors', async () => {
    vi.spyOn(api, 'fetchImpactParameters').mockRejectedValue(new Error('404 No such simulation run'));
    renderPanel();
    expect(await screen.findByRole('alert')).toHaveTextContent('404 No such simulation run');
  });
});

describe('impact parameters client helpers', () => {
  it('builds the query path and heading direction', () => {
    expect(api.buildImpactParametersPath({ runId: 'a b', targetDir: '1,0', units: 'm/s' })).toBe(
      '/api/analysis/impact-parameters?run_id=a+b&target_dir=1%2C0&handedness=right&units=m%2Fs',
    );
    expect(api.targetDirFromHeading(0)).toBe('0.000000,-1.000000');
  });
});
