/**
 * Tests for the SwingObjectiveLab page (issue #9128).
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { SwingObjectiveLabPage } from './SwingObjectiveLab';

const apiFetchMock = vi.fn();
vi.mock('@/api/fetch', () => ({
  apiFetch: (...args: unknown[]) => apiFetchMock(...args),
}));

// Mock WorkspaceShell
vi.mock('@/components/layout/WorkspaceShell', () => ({
  WorkspaceShell: ({
    title,
    description,
    actions,
    children,
  }: {
    title: string;
    description: string;
    actions: React.ReactNode;
    children: React.ReactNode;
  }) => (
    <div data-testid="workspace-shell">
      <h1>{title}</h1>
      <p>{description}</p>
      <div data-testid="workspace-actions">{actions}</div>
      <div data-testid="workspace-children">{children}</div>
    </div>
  ),
}));

const MOCK_PRESETS = [
  {
    name: 'Tour driver (comparison default)',
    arm_mass_kg: 5.0,
    shaft_mass_kg: 0.1428,
    clubhead_mass_kg: 0.0952,
    arm_length_m: 0.65,
    club_length_m: 1.10,
    top_arm_angle_rad: 2.618,
    top_wrist_cock_rad: 1.745,
    duration_s: 0.28,
    hub_torque_nm: 250.0,
    wrist_torque_nm: 20.0,
    node_count: 21,
  },
];

const MOCK_COMPARISON_RESPONSE = {
  schema_version: '1.0.0',
  objective_keys: ['clubhead_speed', 'centrifugal'],
  units: {
    clubhead_speed: 'm/s',
    centrifugal: 'N*m*s',
  },
  raw_values: {
    clubhead_speed: { clubhead_speed: 49.7, centrifugal: 4.8 },
    centrifugal: { clubhead_speed: 48.7, centrifugal: 4.9 },
  },
  matrix: [
    [100.0, 98.5],
    [98.0, 100.0],
  ],
  torque_saturation: {
    clubhead_speed: [0.19, 1.0],
    centrifugal: [0.19, 0.81],
  },
  swing_distance: [
    [0.0, 0.14],
    [0.14, 0.0],
  ],
  is_degenerate: false,
  diagnostics: {
    clubhead_speed: { success: true, iterations: 12 },
    centrifugal: { success: true, iterations: 14 },
  },
};

describe('SwingObjectiveLabPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    apiFetchMock.mockImplementation((url: string) => {
      if (url.includes('/presets')) {
        return Promise.resolve({ presets: MOCK_PRESETS });
      }
      if (url.includes('/compare')) {
        return Promise.resolve(MOCK_COMPARISON_RESPONSE);
      }
      return Promise.reject(new Error(`Unhandled url: ${url}`));
    });
  });

  it('renders title, controls, and run button', async () => {
    render(<SwingObjectiveLabPage />);

    expect(screen.getByText('Swing Objective Lab')).toBeInTheDocument();
    expect(screen.getByTestId('preset-select')).toBeInTheDocument();
    expect(screen.getByTestId('duration-input')).toBeInTheDocument();
    expect(screen.getByTestId('hub-torque-input')).toBeInTheDocument();
    expect(screen.getByTestId('wrist-torque-input')).toBeInTheDocument();
    expect(screen.getByTestId('node-count-input')).toBeInTheDocument();
    expect(screen.getByTestId('run-comparison-btn')).toBeInTheDocument();

    await waitFor(() => {
      expect(apiFetchMock).toHaveBeenCalledWith('/tools/swing-objectives/presets');
    });
  });

  it('executes comparison and renders labelled matrix and metrics table', async () => {
    render(<SwingObjectiveLabPage />);

    const runBtn = screen.getByTestId('run-comparison-btn');
    fireEvent.click(runBtn);

    await waitFor(() => {
      expect(screen.getByTestId('cross-eval-matrix')).toBeInTheDocument();
      expect(screen.getByTestId('metrics-table')).toBeInTheDocument();
    });

    // Check that cells are explicitly labelled
    const cell00 = screen.getByTestId('matrix-cell-clubhead_speed-clubhead_speed');
    expect(cell00).toHaveTextContent('100.0%');
    expect(cell00).toHaveAttribute(
      'aria-label',
      expect.stringContaining('Optimized for Clubhead Speed, evaluated on Clubhead Speed: 100.0%')
    );

    const cell01 = screen.getByTestId('matrix-cell-clubhead_speed-centrifugal');
    expect(cell01).toHaveTextContent('98.5%');
    expect(cell01).toHaveAttribute(
      'aria-label',
      expect.stringContaining('Optimized for Clubhead Speed, evaluated on Centrifugal Release: 98.5%')
    );

    // Verify export button appears
    expect(screen.getByTestId('export-json-btn')).toBeInTheDocument();
  });

  it('surfaces plain-language degeneracy warning when is_degenerate is true', async () => {
    apiFetchMock.mockImplementation((url: string) => {
      if (url.includes('/presets')) {
        return Promise.resolve({ presets: MOCK_PRESETS });
      }
      if (url.includes('/compare')) {
        return Promise.resolve({
          ...MOCK_COMPARISON_RESPONSE,
          is_degenerate: true,
        });
      }
      return Promise.reject(new Error(`Unhandled url: ${url}`));
    });

    render(<SwingObjectiveLabPage />);
    const runBtn = screen.getByTestId('run-comparison-btn');
    fireEvent.click(runBtn);

    await waitFor(() => {
      expect(screen.getByTestId('degeneracy-warning')).toBeInTheDocument();
    });

    expect(
      screen.getByText(/Near the golfer's minimum downswing duration the constraints pin the trajectory/i)
    ).toBeInTheDocument();
  });

  it('renders error alert when API call fails', async () => {
    apiFetchMock.mockImplementation((url: string) => {
      if (url.includes('/presets')) {
        return Promise.resolve({ presets: MOCK_PRESETS });
      }
      if (url.includes('/compare')) {
        return Promise.reject(new Error('Collocation solver diverged.'));
      }
      return Promise.reject(new Error(`Unhandled url: ${url}`));
    });

    render(<SwingObjectiveLabPage />);
    const runBtn = screen.getByTestId('run-comparison-btn');
    fireEvent.click(runBtn);

    await waitFor(() => {
      expect(screen.getByTestId('error-alert')).toBeInTheDocument();
      expect(screen.getByText(/Collocation solver diverged/i)).toBeInTheDocument();
    });
  });
});
