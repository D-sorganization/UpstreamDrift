/**
 * Tests for ActuatorPanel component.
 *
 * See issue #1198
 */

import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ActuatorPanel } from './ActuatorPanel';
import type { ActuatorInfo, ActuatorPanelState } from './ActuatorPanel';

describe('ActuatorPanel types', () => {
  it('should define ActuatorInfo interface correctly', () => {
    const actuator: ActuatorInfo = {
      index: 0,
      name: 'hip_rotation',
      control_type: 'constant',
      value: 0.0,
      min_value: -3.14,
      max_value: 3.14,
      units: 'N*m',
      joint_type: 'revolute',
    };

    expect(actuator.index).toBe(0);
    expect(actuator.name).toBe('hip_rotation');
    expect(actuator.control_type).toBe('constant');
    expect(actuator.min_value).toBe(-3.14);
    expect(actuator.max_value).toBe(3.14);
    expect(actuator.units).toBe('N*m');
  });

  it('should define ActuatorPanelState interface', () => {
    const state: ActuatorPanelState = {
      n_actuators: 3,
      actuators: [
        {
          index: 0,
          name: 'joint_0',
          control_type: 'constant',
          value: 0.0,
          min_value: -100,
          max_value: 100,
          units: 'N*m',
          joint_type: 'revolute',
        },
        {
          index: 1,
          name: 'joint_1',
          control_type: 'pd_gains',
          value: 5.0,
          min_value: -50,
          max_value: 50,
          units: 'N*m',
          joint_type: 'revolute',
        },
        {
          index: 2,
          name: 'joint_2',
          control_type: 'constant',
          value: -10.0,
          min_value: -200,
          max_value: 200,
          units: 'N*m',
          joint_type: 'prismatic',
        },
      ],
      available_control_types: ['constant', 'polynomial', 'pd_gains', 'trajectory'],
      engine_name: 'mujoco',
    };

    expect(state.n_actuators).toBe(3);
    expect(state.actuators).toHaveLength(3);
    expect(state.engine_name).toBe('mujoco');
    expect(state.available_control_types).toContain('pd_gains');
  });

  it('should validate actuator value ranges', () => {
    const actuator: ActuatorInfo = {
      index: 0,
      name: 'shoulder',
      control_type: 'constant',
      value: 50.0,
      min_value: -100,
      max_value: 100,
      units: 'N*m',
      joint_type: 'revolute',
    };

    expect(actuator.value).toBeGreaterThanOrEqual(actuator.min_value);
    expect(actuator.value).toBeLessThanOrEqual(actuator.max_value);
  });

  it('should support all control types', () => {
    const types = ['constant', 'polynomial', 'pd_gains', 'trajectory'];

    for (const type of types) {
      const actuator: ActuatorInfo = {
        index: 0,
        name: 'test',
        control_type: type,
        value: 0,
        min_value: -100,
        max_value: 100,
        units: 'N*m',
        joint_type: 'revolute',
      };
      expect(actuator.control_type).toBe(type);
    }
  });

  it('should handle empty actuator panel', () => {
    const state: ActuatorPanelState = {
      n_actuators: 0,
      actuators: [],
      available_control_types: ['constant'],
      engine_name: 'none',
    };

    expect(state.n_actuators).toBe(0);
    expect(state.actuators).toHaveLength(0);
  });
});

describe('ActuatorPanel React Component', () => {
  const mockStateSmall: ActuatorPanelState = {
    n_actuators: 3,
    actuators: [
      {
        index: 0,
        name: 'shoulder_flexion',
        control_type: 'constant',
        value: 0.0,
        min_value: -3.14,
        max_value: 3.14,
        units: 'rad',
        joint_type: 'revolute',
      },
      {
        index: 1,
        name: 'hip_rotation',
        control_type: 'constant',
        value: 0.5,
        min_value: -3.14,
        max_value: 3.14,
        units: 'rad',
        joint_type: 'revolute',
      },
      {
        index: 2,
        name: 'spine_extension',
        control_type: 'constant',
        value: -0.2,
        min_value: -1.0,
        max_value: 1.0,
        units: 'rad',
        joint_type: 'revolute',
      },
    ],
    available_control_types: ['constant', 'polynomial', 'pd_gains'],
    engine_name: 'mujoco',
  };

  const mockStateLarge: ActuatorPanelState = {
    n_actuators: 22,
    actuators: Array.from({ length: 22 }, (_, i) => {
      let name = `joint_${i}`;
      if (i < 8) name = `shoulder_flex_${i}`;
      else if (i < 16) name = `hip_rot_${i}`;
      else name = `spine_ext_${i}`;
      return {
        index: i,
        name,
        control_type: 'constant',
        value: 0.0,
        min_value: -3.14,
        max_value: 3.14,
        units: 'rad',
        joint_type: 'revolute',
      };
    }),
    available_control_types: ['constant', 'polynomial'],
    engine_name: 'mujoco',
  };

  beforeEach(() => {
    vi.restoreAllMocks();
    global.fetch = vi.fn();
  });

  it('renders loading state first, then loaded actuators', async () => {
    vi.mocked(global.fetch).mockResolvedValueOnce({
      ok: true,
      json: async () => mockStateSmall,
    } as Response);

    render(<ActuatorPanel isRunning={false} />);
    expect(screen.getByText(/Loading actuators.../i)).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('shoulder_flexion')).toBeInTheDocument();
    });
    expect(screen.getByText('hip_rotation')).toBeInTheDocument();
    expect(screen.getByText('spine_extension')).toBeInTheDocument();
  });

  it('toggles collapsible Polynomial Generator Panel and configures coefficients', async () => {
    vi.mocked(global.fetch).mockResolvedValueOnce({
      ok: true,
      json: async () => mockStateSmall,
    } as Response);

    render(<ActuatorPanel isRunning={false} />);

    await waitFor(() => {
      expect(screen.getByText('shoulder_flexion')).toBeInTheDocument();
    });

    // Find the toggle button/header for Polynomial Generator
    const polyHeader = screen.getByRole('button', { name: /Polynomial Generator/i });
    expect(polyHeader).toBeInTheDocument();

    // Initially collapsed, coefficient inputs shouldn't be visible
    expect(screen.queryByLabelText(/Coefficient 0/i)).not.toBeInTheDocument();

    // Click to expand
    fireEvent.click(polyHeader);
    expect(screen.getByLabelText(/Coefficient 0/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Coefficient 1/i)).toBeInTheDocument();

    // Change coefficient value
    const input0 = screen.getByLabelText(/Coefficient 0/i);
    fireEvent.change(input0, { target: { value: '2.5' } });
    expect(input0).toHaveValue(2.5);

    // Add a coefficient
    const addBtn = screen.getByRole('button', { name: /Add Coefficient/i });
    fireEvent.click(addBtn);
    expect(screen.getByLabelText(/Coefficient 2/i)).toBeInTheDocument();

    // Remove a coefficient
    const removeBtns = screen.getAllByRole('button', { name: /Remove/i });
    // Click the last remove button to remove Coefficient 2
    fireEvent.click(removeBtns[removeBtns.length - 1]);
    expect(screen.queryByLabelText(/Coefficient 2/i)).not.toBeInTheDocument();

    // Mock the POST request for applying polynomial
    vi.mocked(global.fetch).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ status: 'ok' }),
    } as Response);

    const applyBtn = screen.getByRole('button', { name: /Apply Polynomial/i });
    fireEvent.click(applyBtn);

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith('/api/simulation/actuators', expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          actuator_index: 0, // Default to first actuator
          value: 0.0,
          control_type: 'polynomial',
          parameters: { coefficients: [2.5, 0] },
        }),
      }));
    });
  });

  it('groups and collapses segments when there are more than 20 actuators', async () => {
    vi.mocked(global.fetch).mockResolvedValueOnce({
      ok: true,
      json: async () => mockStateLarge,
    } as Response);

    render(<ActuatorPanel isRunning={false} />);

    await waitFor(() => {
      // It should display the group headers instead of showing 22 individual sliders directly
      expect(screen.getByRole('button', { name: /Upper Body \(8\)/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Lower Body \(8\)/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Core & Head \(6\)/i })).toBeInTheDocument();
    });

    // Check that we have group filters (dropdown or buttons)
    const filterSelect = screen.getByLabelText(/Filter by region/i);
    expect(filterSelect).toBeInTheDocument();

    // Choose 'Upper Body' to filter
    fireEvent.change(filterSelect, { target: { value: 'Upper Body' } });
    expect(screen.queryByText(/hip_rot_/i)).not.toBeInTheDocument();
    expect(screen.getByText('shoulder_flex_0')).toBeInTheDocument();
  });
});
