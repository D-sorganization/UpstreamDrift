import { describe, it, expect, beforeEach } from 'vitest';
import { act } from '@testing-library/react';
import {
  useSimulationStore,
  DEFAULT_PARAMETERS,
} from './useSimulationStore';

describe('useSimulationStore', () => {
  beforeEach(() => {
    act(() => {
      useSimulationStore.getState().resetParameters();
    });
  });

  describe('initial state', () => {
    it('starts with default parameters', () => {
      const { parameters } = useSimulationStore.getState();
      expect(parameters).toEqual(DEFAULT_PARAMETERS);
    });

    it('starts with hasRun = false', () => {
      expect(useSimulationStore.getState().hasRun).toBe(false);
    });

    it('has correct default duration', () => {
      expect(useSimulationStore.getState().parameters.duration).toBe(3.0);
    });

    it('has correct default timestep', () => {
      expect(useSimulationStore.getState().parameters.timestep).toBe(0.002);
    });
  });

  describe('setParameters', () => {
    it('merges partial updates', () => {
      act(() => {
        useSimulationStore.getState().setParameters({ duration: 5.0 });
      });

      const { parameters } = useSimulationStore.getState();
      expect(parameters.duration).toBe(5.0);
      // Other fields unchanged
      expect(parameters.timestep).toBe(0.002);
      expect(parameters.liveAnalysis).toBe(true);
      expect(parameters.gpuAcceleration).toBe(false);
    });

    it('can update multiple fields at once', () => {
      act(() => {
        useSimulationStore.getState().setParameters({
          duration: 10.0,
          gpuAcceleration: true,
        });
      });

      const { parameters } = useSimulationStore.getState();
      expect(parameters.duration).toBe(10.0);
      expect(parameters.gpuAcceleration).toBe(true);
    });
  });

  describe('replaceParameters', () => {
    it('replaces all parameters', () => {
      const newParams = {
        duration: 7.0,
        timestep: 0.001,
        liveAnalysis: false,
        gpuAcceleration: true,
      };

      act(() => {
        useSimulationStore.getState().replaceParameters(newParams);
      });

      expect(useSimulationStore.getState().parameters).toEqual(newParams);
    });
  });

  describe('markRun', () => {
    it('sets hasRun to true', () => {
      act(() => {
        useSimulationStore.getState().markRun();
      });

      expect(useSimulationStore.getState().hasRun).toBe(true);
    });
  });

  describe('resetParameters', () => {
    it('resets to defaults', () => {
      act(() => {
        useSimulationStore.getState().setParameters({ duration: 99.0 });
        useSimulationStore.getState().markRun();
      });

      act(() => {
        useSimulationStore.getState().resetParameters();
      });

      const state = useSimulationStore.getState();
      expect(state.parameters).toEqual(DEFAULT_PARAMETERS);
      expect(state.hasRun).toBe(false);
    });
  });
});
