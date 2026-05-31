import { describe, it, expect, beforeEach, vi } from 'vitest';
import { act } from '@testing-library/react';
import {
  useEngineStore,
  selectLoadedEngines,
  selectEffectiveEngine,
  selectCapabilityUnavailableReason,
  selectEngineSupportsCapability,
} from './useEngineStore';

describe('useEngineStore', () => {
  beforeEach(() => {
    // Reset the store to initial state before each test
    act(() => {
      useEngineStore.getState().resetEngines();
    });
  });

  describe('initial state', () => {
    it('starts with 7 engines in idle state', () => {
      const { engines } = useEngineStore.getState();
      expect(engines).toHaveLength(7);
      engines.forEach((e) => {
        expect(e.loadState).toBe('idle');
        expect(e.available).toBe(true);
      });
    });

    it('starts with no engine selected', () => {
      expect(useEngineStore.getState().selectedEngine).toBeNull();
    });

    it('includes MuJoCo in registry', () => {
      const mujoco = useEngineStore
        .getState()
        .engines.find((e) => e.name === 'mujoco');
      expect(mujoco).toBeDefined();
      expect(mujoco?.displayName).toBe('MuJoCo');
      expect(mujoco?.capabilities).toContain('rigid_body');
    });

    it('includes JaxSim in registry with differentiable analysis tags', () => {
      const jaxsim = useEngineStore
        .getState()
        .engines.find((e) => e.name === 'jaxsim');
      expect(jaxsim).toBeDefined();
      expect(jaxsim?.displayName).toBe('JaxSim');
      expect(jaxsim?.capabilities).toEqual([
        'rigid_body',
        'differentiable',
        'gradients',
        'parameter_sensitivity',
      ]);
    });
  });

  describe('selectEngine', () => {
    it('selects an engine by name', () => {
      act(() => {
        useEngineStore.getState().selectEngine('drake');
      });
      expect(useEngineStore.getState().selectedEngine).toBe('drake');
    });

    it('clears selection with null', () => {
      act(() => {
        useEngineStore.getState().selectEngine('drake');
      });
      act(() => {
        useEngineStore.getState().selectEngine(null);
      });
      expect(useEngineStore.getState().selectedEngine).toBeNull();
    });
  });

  describe('unloadEngine', () => {
    it('sets engine to idle', async () => {
      // Manually set an engine to loaded
      useEngineStore.setState((state) => ({
        engines: state.engines.map((e) =>
          e.name === 'mujoco' ? { ...e, loadState: 'loaded' as const } : e
        ),
      }));

      await act(async () => {
        await useEngineStore.getState().unloadEngine('mujoco');
      });

      const mujoco = useEngineStore
        .getState()
        .engines.find((e) => e.name === 'mujoco');
      expect(mujoco?.loadState).toBe('idle');
    });

    it('clears selection if unloading the selected engine', async () => {
      useEngineStore.setState({
        selectedEngine: 'mujoco',
        engines: useEngineStore.getState().engines.map((e) =>
          e.name === 'mujoco' ? { ...e, loadState: 'loaded' as const } : e
        ),
      });

      await act(async () => {
        await useEngineStore.getState().unloadEngine('mujoco');
      });

      expect(useEngineStore.getState().selectedEngine).toBeNull();
    });

    it('does not clear selection if unloading a different engine', async () => {
      useEngineStore.setState({ selectedEngine: 'mujoco' });

      await act(async () => {
        await useEngineStore.getState().unloadEngine('drake');
      });

      expect(useEngineStore.getState().selectedEngine).toBe('mujoco');
    });
  });

  describe('requestLoad', () => {
    it('sets engine to loading state immediately', async () => {
      // Mock fetch to delay
      const mockFetch = vi.fn(
        () =>
          new Promise<Response>(() => {
            /* never resolves */
          })
      );
      global.fetch = mockFetch;

      // Start loading (don't await — it won't resolve)
      const loadPromise = useEngineStore.getState().requestLoad('mujoco');

      // Should be in loading state immediately
      const mujoco = useEngineStore
        .getState()
        .engines.find((e) => e.name === 'mujoco');
      expect(mujoco?.loadState).toBe('loading');

      // Clean up
      global.fetch = vi.fn();
      // loadPromise will hang, but that's fine since the test is already passing
      void loadPromise;
    });

    it('sets engine to error on fetch failure', async () => {
      global.fetch = vi.fn(() =>
        Promise.reject(new Error('Network error'))
      );

      await useEngineStore.getState().requestLoad('mujoco');

      const mujoco = useEngineStore
        .getState()
        .engines.find((e) => e.name === 'mujoco');
      expect(mujoco?.loadState).toBe('error');
      // probeEngine wraps low-level fetch/apiFetch failures in a descriptive
      // message so the user sees which engine failed to probe.
      expect(mujoco?.error).toBe('Failed to probe engine: mujoco');
    });

    it('sets engine to loaded on successful probe + load', async () => {
      global.fetch = vi.fn((url: string | URL | Request) => {
        const urlStr = typeof url === 'string' ? url : url.toString();
        if (urlStr.includes('/probe')) {
          return Promise.resolve(
            new Response(JSON.stringify({ available: true, version: '3.1.0' }))
          );
        }
        return Promise.resolve(
          new Response(
            JSON.stringify({
              available: true,
              version: '3.1.0',
              capabilities: ['rigid_body', 'contact'],
            })
          )
        );
      }) as typeof fetch;

      await useEngineStore.getState().requestLoad('mujoco');

      const mujoco = useEngineStore
        .getState()
        .engines.find((e) => e.name === 'mujoco');
      expect(mujoco?.loadState).toBe('loaded');
      expect(mujoco?.version).toBe('3.1.0');
    });

    it('auto-selects the engine if nothing is selected', async () => {
      global.fetch = vi.fn(() =>
        Promise.resolve(
          new Response(JSON.stringify({ available: true, version: '1.0' }))
        )
      ) as typeof fetch;

      expect(useEngineStore.getState().selectedEngine).toBeNull();

      await useEngineStore.getState().requestLoad('drake');

      expect(useEngineStore.getState().selectedEngine).toBe('drake');
    });

    it('marks engine unavailable if probe returns not available', async () => {
      global.fetch = vi.fn(() =>
        Promise.resolve(
          new Response(JSON.stringify({ available: false }))
        )
      ) as typeof fetch;

      await useEngineStore.getState().requestLoad('pinocchio');

      const pinocchio = useEngineStore
        .getState()
        .engines.find((e) => e.name === 'pinocchio');
      expect(pinocchio?.loadState).toBe('error');
      expect(pinocchio?.available).toBe(false);
      expect(pinocchio?.error).toBe('Engine not installed on this system');
    });
  });

  describe('selectors', () => {
    it('reports capability support for analysis gating', () => {
      const state = useEngineStore.getState();

      expect(
        selectEngineSupportsCapability('jaxsim', 'parameter_sensitivity')(state)
      ).toBe(true);
      expect(
        selectEngineSupportsCapability('pinocchio', 'parameter_sensitivity')(
          state
        )
      ).toBe(false);
    });

    it('returns a tooltip reason when a capability is unavailable', () => {
      const state = useEngineStore.getState();

      expect(
        selectCapabilityUnavailableReason('jaxsim', 'parameter_sensitivity')(
          state
        )
      ).toBeNull();
      expect(
        selectCapabilityUnavailableReason('pinocchio', 'parameter_sensitivity')(
          state
        )
      ).toBe('Pinocchio does not support parameter_sensitivity.');
    });

    it('selectLoadedEngines returns only loaded engines', () => {
      useEngineStore.setState((state) => ({
        engines: state.engines.map((e) =>
          e.name === 'mujoco' ? { ...e, loadState: 'loaded' as const } : e
        ),
      }));

      const loaded = selectLoadedEngines(useEngineStore.getState());
      expect(loaded).toHaveLength(1);
      expect(loaded[0].name).toBe('mujoco');
    });

    it('selectEffectiveEngine returns selected engine if loaded', () => {
      useEngineStore.setState((state) => ({
        selectedEngine: 'mujoco',
        engines: state.engines.map((e) =>
          e.name === 'mujoco' ? { ...e, loadState: 'loaded' as const } : e
        ),
      }));

      expect(selectEffectiveEngine(useEngineStore.getState())).toBe('mujoco');
    });

    it('selectEffectiveEngine falls back to first loaded', () => {
      useEngineStore.setState((state) => ({
        selectedEngine: null,
        engines: state.engines.map((e) =>
          e.name === 'drake' ? { ...e, loadState: 'loaded' as const } : e
        ),
      }));

      expect(selectEffectiveEngine(useEngineStore.getState())).toBe('drake');
    });

    it('selectEffectiveEngine returns null when nothing loaded', () => {
      expect(selectEffectiveEngine(useEngineStore.getState())).toBeNull();
    });

    it('selectEffectiveEngine ignores selected engine if not loaded', () => {
      useEngineStore.setState({ selectedEngine: 'mujoco' });
      // mujoco is idle, not loaded
      expect(selectEffectiveEngine(useEngineStore.getState())).toBeNull();
    });
  });

  describe('probeAvailability (#6900)', () => {
    it('marks an engine unavailable when its probe reports not installed', async () => {
      // Backend says pinocchio is missing, everything else is installed.
      global.fetch = vi.fn((url: string | URL | Request) => {
        const urlStr = typeof url === 'string' ? url : url.toString();
        const available = !urlStr.includes('pinocchio');
        return Promise.resolve(new Response(JSON.stringify({ available })));
      }) as typeof fetch;

      // Optimistic seed: everything starts available before probing.
      expect(
        useEngineStore.getState().engines.every((e) => e.available)
      ).toBe(true);

      await act(async () => {
        await useEngineStore.getState().probeAvailability();
      });

      const pinocchio = useEngineStore
        .getState()
        .engines.find((e) => e.name === 'pinocchio');
      const mujoco = useEngineStore
        .getState()
        .engines.find((e) => e.name === 'mujoco');

      expect(pinocchio?.available).toBe(false);
      // An unavailable engine must remain unloaded (renders disabled, not loadable).
      expect(pinocchio?.loadState).toBe('idle');
      expect(mujoco?.available).toBe(true);
    });

    it('leaves the optimistic seed intact when every probe errors', async () => {
      // A backend that is still starting up (all probes reject) must not wrongly
      // disable engines that may in fact be installed — only a definitive
      // `available: false` flips the flag.
      global.fetch = vi.fn(() =>
        Promise.reject(new Error('Network error'))
      ) as typeof fetch;

      await act(async () => {
        await useEngineStore.getState().probeAvailability();
      });

      expect(
        useEngineStore.getState().engines.every((e) => e.available === true)
      ).toBe(true);
    });
  });

  describe('resetEngines', () => {
    it('resets all engines to idle and clears selection', () => {
      useEngineStore.setState((state) => ({
        selectedEngine: 'mujoco',
        engines: state.engines.map((e) =>
          e.name === 'mujoco' ? { ...e, loadState: 'loaded' as const } : e
        ),
      }));

      act(() => {
        useEngineStore.getState().resetEngines();
      });

      const state = useEngineStore.getState();
      expect(state.selectedEngine).toBeNull();
      state.engines.forEach((e) => {
        expect(e.loadState).toBe('idle');
      });
    });
  });
});
