/**
 * Tests for the BallFlight (Shot Tracer) page.
 *
 * Covers form validation (unit bounds from the field-metadata registry),
 * flight-model multi-select fed by the models endpoint, and results
 * rendering (metrics table + profile charts) with a mocked API.
 *
 * See issue #7456.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

// Mock the API layer.
const apiFetchMock = vi.fn();
vi.mock('@/api/fetch', () => ({
  apiFetch: (...args: unknown[]) => apiFetchMock(...args),
}));

// Mock the 3D scene (no WebGL in jsdom).
vi.mock('@/components/visualization/BallFlightScene3D', () => ({
  BallFlightScene3D: ({
    trajectories,
  }: {
    trajectories: { modelKey: string }[];
  }) => (
    <div
      data-testid="ball-flight-scene3d"
      data-trajectory-count={trajectories.length}
    />
  ),
}));

// Mock recharts to avoid layout/canvas work in tests.
vi.mock('recharts', () => ({
  LineChart: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="line-chart-mock">{children}</div>
  ),
  Line: ({ name }: { name?: string }) => (
    <div data-testid="chart-line-mock" data-name={name} />
  ),
  XAxis: () => <div data-testid="xaxis-mock" />,
  YAxis: () => <div data-testid="yaxis-mock" />,
  CartesianGrid: () => <div data-testid="grid-mock" />,
  Tooltip: () => <div data-testid="tooltip-mock" />,
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="responsive-container-mock">{children}</div>
  ),
}));

import {
  BallFlightPage,
  invalidLaunchFields,
  modelColor,
  LAUNCH_FIELD_IDS,
  MODEL_COLORS,
} from './BallFlight';
import type { BallFlightSimulationResponse, FlightModelInfo } from './BallFlight';

const MODELS: FlightModelInfo[] = [
  {
    key: 'waterloo_penner',
    name: 'Waterloo/Penner',
    description: 'Quadratic Cd/Cl model',
    reference: 'Penner (2003)',
  },
  {
    key: 'nathan',
    name: 'Nathan',
    description: 'Constant Cd/Cl model with spin decay',
    reference: 'Nathan et al. (2018)',
  },
];

function makeResult(key: string, name: string) {
  return {
    model_name: name,
    model_key: key,
    trajectory: [
      { time_s: 0, position_m: [0, 0, 0], velocity_mps: [50, 0, 20] },
      { time_s: 1, position_m: [45, 1.5, 18], velocity_mps: [40, 0.5, 5] },
      { time_s: 2, position_m: [80, 3, 0], velocity_mps: [35, 0.5, -15] },
    ],
    summary: {
      carry_m: 80.4,
      apex_m: 18.2,
      flight_time_s: 2.0,
      landing_angle_deg: 32.1,
      lateral_deviation_m: 3.0,
    },
  };
}

const SIMULATE_RESPONSE: BallFlightSimulationResponse = {
  ...makeResult('waterloo_penner', 'Waterloo/Penner'),
  results: [
    makeResult('waterloo_penner', 'Waterloo/Penner'),
    makeResult('nathan', 'Nathan'),
  ],
};

function mockApi() {
  apiFetchMock.mockImplementation((path: string) => {
    if (path === '/api/tools/ball-flight/models') {
      return Promise.resolve({ models: MODELS });
    }
    if (path === '/api/tools/ball-flight/simulate') {
      return Promise.resolve(SIMULATE_RESPONSE);
    }
    return Promise.reject(new Error(`Unexpected path: ${path}`));
  });
}

describe('invalidLaunchFields (unit-bound validation)', () => {
  const validValues = Object.fromEntries(
    LAUNCH_FIELD_IDS.map((id) => [id, '0']),
  ) as Record<(typeof LAUNCH_FIELD_IDS)[number], string>;

  beforeEach(() => {
    validValues['ball_flight.ball_speed'] = '70';
    validValues['ball_flight.launch_angle'] = '12';
    validValues['ball_flight.azimuth_angle'] = '0';
    validValues['ball_flight.spin_rate'] = '2600';
    validValues['ball_flight.spin_axis_tilt'] = '0';
    validValues['ball_flight.wind_speed'] = '0';
    validValues['ball_flight.wind_direction'] = '0';
  });

  it('accepts in-range defaults', () => {
    expect(invalidLaunchFields(validValues)).toEqual([]);
  });

  it.each([
    ['ball_flight.ball_speed', '150'], // > 100 m/s
    ['ball_flight.ball_speed', '0'], // below minimum
    ['ball_flight.launch_angle', '95'], // > 80 deg
    ['ball_flight.spin_rate', '-100'], // negative RPM
    ['ball_flight.spin_rate', '20000'], // > 15000 RPM
    ['ball_flight.wind_speed', '45'], // > 40 m/s
    ['ball_flight.wind_direction', '270'], // > 180 deg
  ] as const)('flags %s = %s as out of range', (fieldId, value) => {
    const values = { ...validValues, [fieldId]: value };
    expect(invalidLaunchFields(values)).toContain(fieldId);
  });

  it('flags non-numeric and empty values', () => {
    expect(
      invalidLaunchFields({ ...validValues, 'ball_flight.ball_speed': 'abc' }),
    ).toContain('ball_flight.ball_speed');
    expect(
      invalidLaunchFields({ ...validValues, 'ball_flight.spin_rate': '' }),
    ).toContain('ball_flight.spin_rate');
  });
});

describe('modelColor', () => {
  it('cycles through the palette', () => {
    expect(modelColor(0)).toBe(MODEL_COLORS[0]);
    expect(modelColor(MODEL_COLORS.length)).toBe(MODEL_COLORS[0]);
  });
});

describe('BallFlightPage', () => {
  beforeEach(() => {
    apiFetchMock.mockReset();
    mockApi();
  });

  it('renders explicit units on every launch-condition field', async () => {
    render(<BallFlightPage />);

    // Units come from the shared field-metadata registry (issue #7246
    // made explicit unit labels mandatory).
    expect(screen.getByText('Ball speed (m/s)')).toBeInTheDocument();
    expect(screen.getByText('Launch angle (deg)')).toBeInTheDocument();
    expect(screen.getByText('Azimuth angle (deg)')).toBeInTheDocument();
    expect(screen.getByText('Spin rate (rpm)')).toBeInTheDocument();
    expect(screen.getByText('Spin-axis tilt (deg)')).toBeInTheDocument();
    expect(screen.getByText('Wind speed (m/s)')).toBeInTheDocument();
    expect(screen.getByText('Wind direction (deg)')).toBeInTheDocument();
  });

  it('loads flight models from the shared registry endpoint', async () => {
    render(<BallFlightPage />);

    await waitFor(() => {
      expect(screen.getByText('Waterloo/Penner')).toBeInTheDocument();
      expect(screen.getByText('Nathan')).toBeInTheDocument();
    });
    expect(apiFetchMock).toHaveBeenCalledWith('/api/tools/ball-flight/models');
  });

  it('simulates selected models and renders the per-model metrics table', async () => {
    render(<BallFlightPage />);

    await waitFor(() => {
      expect(screen.getByText('Nathan')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByTestId('simulate-btn'));

    await waitFor(() => {
      expect(screen.getByTestId('metrics-table')).toBeInTheDocument();
    });

    // POST body carries the selected models list.
    const simulateCall = apiFetchMock.mock.calls.find(
      ([path]) => path === '/api/tools/ball-flight/simulate',
    );
    expect(simulateCall).toBeDefined();
    const body = JSON.parse((simulateCall![1] as RequestInit).body as string);
    expect(body.models).toEqual(['waterloo_penner', 'nathan']);
    expect(body.ball_speed_mps).toBe(70);
    expect(body.spin_rate_rpm).toBe(2600);

    // One metrics row per model with carry/apex/time/offline.
    expect(screen.getByTestId('metrics-row-waterloo_penner')).toBeInTheDocument();
    expect(screen.getByTestId('metrics-row-nathan')).toBeInTheDocument();
    expect(screen.getAllByText('80.4')).toHaveLength(2); // carry per model
    expect(screen.getAllByText('18.2')).toHaveLength(2); // apex per model

    // Charts and 3D overlay get one series per model.
    expect(screen.getByTestId('side-profile-chart')).toBeInTheDocument();
    expect(screen.getByTestId('top-profile-chart')).toBeInTheDocument();
    expect(
      screen.getByTestId('ball-flight-scene3d').dataset.trajectoryCount,
    ).toBe('2');
  });

  it('blocks simulate and warns when a value is out of range', async () => {
    render(<BallFlightPage />);

    await waitFor(() => {
      expect(screen.getByText('Nathan')).toBeInTheDocument();
    });

    const speedInput = screen.getByLabelText(/Ball speed/);
    fireEvent.change(speedInput, { target: { value: '150' } });

    expect(screen.getByTestId('validation-warning')).toHaveTextContent(
      'Ball speed',
    );
    expect(screen.getByTestId('simulate-btn')).toBeDisabled();
  });

  it('disables simulate when no model is selected', async () => {
    render(<BallFlightPage />);

    await waitFor(() => {
      expect(screen.getByText('Nathan')).toBeInTheDocument();
    });

    // Both models are pre-selected; uncheck them.
    fireEvent.click(screen.getByLabelText('Toggle Waterloo/Penner model'));
    fireEvent.click(screen.getByLabelText('Toggle Nathan model'));

    expect(screen.getByTestId('simulate-btn')).toBeDisabled();
  });

  it('surfaces API errors from simulate', async () => {
    render(<BallFlightPage />);

    await waitFor(() => {
      expect(screen.getByText('Nathan')).toBeInTheDocument();
    });

    apiFetchMock.mockImplementation((path: string) => {
      if (path === '/api/tools/ball-flight/simulate') {
        return Promise.reject(new Error('boom'));
      }
      return Promise.resolve({ models: MODELS });
    });

    fireEvent.click(screen.getByTestId('simulate-btn'));

    await waitFor(() => {
      expect(screen.getByTestId('error-message')).toHaveTextContent('boom');
    });
  });
});
