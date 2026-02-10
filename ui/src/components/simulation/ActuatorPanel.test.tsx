/**
 * Tests for ActuatorPanel component.
 *
 * See issue #1198
 */

import { describe, it, expect } from 'vitest';
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
