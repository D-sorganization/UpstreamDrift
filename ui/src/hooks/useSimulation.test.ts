/**
 * Unit tests for useSimulation hook and lifecycle reducer (#14950).
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import {
  useSimulation,
  simulationLifecycleReducer,
  reduceSimLifecycle,
  initialSimulationLifecycle,
  deriveIsRunning,
  deriveIsPaused,
} from './useSimulation';

class MockWebSocket {
  static instances: MockWebSocket[] = [];
  static CONNECTING = 0; static OPEN = 1; static CLOSING = 2; static CLOSED = 3;
  url: string; readyState = MockWebSocket.CONNECTING;
  onopen: ((e: Event) => void) | null = null;
  onclose: ((e: CloseEvent) => void) | null = null;
  onmessage: ((e: MessageEvent) => void) | null = null;
  onerror: ((e: Event) => void) | null = null;
  sentMessages: string[] = [];
  constructor(url: string) { this.url = url; MockWebSocket.instances.push(this); }
  send(data: string) { this.sentMessages.push(data); }
  close(code = 1000, reason?: string) {
    this.readyState = MockWebSocket.CLOSED;
    this.onclose?.(new CloseEvent('close', { code, reason, wasClean: code === 1000 }));
  }
  simulateOpen() { this.readyState = MockWebSocket.OPEN; this.onopen?.(new Event('open')); }
  simulateMessage(data: unknown) {
    this.onmessage?.(new MessageEvent('message', { data: typeof data === 'string' ? data : JSON.stringify(data) }));
  }
  simulateError() { this.onerror?.(new Event('error')); }
  simulateUncleanClose(code = 1006) {
    this.readyState = MockWebSocket.CLOSED;
    this.onclose?.(new CloseEvent('close', { code, wasClean: false }));
  }
  static reset() { MockWebSocket.instances = []; }
  static getLastInstance() { return MockWebSocket.instances[MockWebSocket.instances.length - 1]; }
}

function simulateRunStarted(ws: MockWebSocket) {
  ws.simulateOpen();
  ws.simulateMessage({ status: 'running' });
}

describe('useSimulation lifecycle and hook contracts (#14950)', () => {
  beforeEach(() => {
    MockWebSocket.reset();
    vi.stubGlobal('WebSocket', MockWebSocket);
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ speed_factor: 1.0, status: 'ok' }),
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  describe('pure reducer lifecycle transitions', () => {
    it('supports alias reduceSimLifecycle matching simulationLifecycleReducer', () => {
      expect(reduceSimLifecycle).toBe(simulationLifecycleReducer);
    });

    it('exercises complete happy path: idle -> connecting -> starting -> running -> paused -> running -> completed', () => {
      let state = initialSimulationLifecycle;
      expect(state.phase).toBe('idle');
      expect(deriveIsRunning(state.phase)).toBe(false);
      expect(deriveIsPaused(state.phase, state.paused)).toBe(false);

      state = reduceSimLifecycle(state, { type: 'begin_connect', runId: 1 });
      expect(state.phase).toBe('connecting');
      expect(deriveIsRunning(state.phase)).toBe(false);

      state = reduceSimLifecycle(state, { type: 'socket_open', runId: 1 });
      expect(state.phase).toBe('starting');

      state = reduceSimLifecycle(state, { type: 'server_running', runId: 1 });
      expect(state.phase).toBe('running');
      expect(deriveIsRunning(state.phase)).toBe(true);
      expect(deriveIsPaused(state.phase, state.paused)).toBe(false);

      state = reduceSimLifecycle(state, { type: 'server_paused', runId: 1 });
      expect(state.phase).toBe('paused');
      expect(deriveIsRunning(state.phase)).toBe(true);
      expect(deriveIsPaused(state.phase, state.paused)).toBe(true);

      state = reduceSimLifecycle(state, { type: 'server_resumed', runId: 1 });
      expect(state.phase).toBe('running');
      expect(deriveIsPaused(state.phase, state.paused)).toBe(false);

      state = reduceSimLifecycle(state, { type: 'server_complete', runId: 1 });
      expect(state.phase).toBe('completed');
      expect(deriveIsRunning(state.phase)).toBe(false);
    });

    it('rejects illegal transitions without mutating state', () => {
      const state = initialSimulationLifecycle;
      expect(reduceSimLifecycle(state, { type: 'server_paused', runId: 1 })).toBe(state);
      expect(reduceSimLifecycle(state, { type: 'server_resumed', runId: 1 })).toBe(state);
    });

    it('handles faulted and disconnected phase transitions', () => {
      let state = reduceSimLifecycle(initialSimulationLifecycle, { type: 'begin_connect', runId: 1 });
      state = reduceSimLifecycle(state, {
        type: 'server_fault',
        runId: 1,
        message: 'Engine constraint convergence fault',
      });
      expect(state.phase).toBe('faulted');
      expect(state.failureMessage).toBe('Engine constraint convergence fault');

      state = reduceSimLifecycle(state, { type: 'socket_disconnected', runId: 1 });
      expect(state.phase).toBe('disconnected');
    });
  });

  describe('hook guards and ordering', () => {
    it('double-start while connecting is a guarded no-op', () => {
      const { result } = renderHook(() => useSimulation('mujoco'));
      act(() => { result.current.start(); });
      expect(MockWebSocket.instances).toHaveLength(1);
      expect(result.current.phase).toBe('connecting');

      act(() => { result.current.start(); });
      expect(MockWebSocket.instances).toHaveLength(1);
      expect(result.current.phase).toBe('connecting');
    });

    it('stale-socket identity guard rejects events from replaced sockets', () => {
      const { result } = renderHook(() => useSimulation('mujoco'));
      act(() => { result.current.start(); });
      const firstWs = MockWebSocket.getLastInstance()!;
      act(() => { simulateRunStarted(firstWs); });
      expect(result.current.runId).toBe(1);

      act(() => { result.current.stop(); });
      act(() => { result.current.start(); });
      const secondWs = MockWebSocket.getLastInstance()!;
      act(() => { simulateRunStarted(secondWs); });
      expect(result.current.runId).toBe(2);

      act(() => {
        firstWs.simulateError();
        firstWs.simulateUncleanClose(1006);
      });

      expect(result.current.runId).toBe(2);
      expect(result.current.isRunning).toBe(true);
      expect(result.current.phase).toBe('running');
      expect(result.current.connectionStatus).toBe('connected');
    });

    it('orders setSpeed over WebSocket stream when socket is open', async () => {
      const { result } = renderHook(() => useSimulation('mujoco'));
      act(() => { result.current.start(); });
      const ws = MockWebSocket.getLastInstance()!;
      act(() => { simulateRunStarted(ws); });

      await act(async () => { await result.current.setSpeed(2.0); });
      act(() => { result.current.pause(); });

      const actions = ws.sentMessages.map((m) => JSON.parse(m).action);
      expect(actions).toEqual(['start', 'set_speed', 'pause']);
    });

    it('exposes derived status object with phase, isConnected, isRunning, isPaused', () => {
      const { result } = renderHook(() => useSimulation('mujoco'));
      expect(result.current.status).toEqual({
        phase: 'idle',
        isConnected: false,
        isRunning: false,
        isPaused: false,
        error: null,
      });

      act(() => { result.current.start(); });
      expect(result.current.status.phase).toBe('connecting');

      const ws = MockWebSocket.getLastInstance()!;
      act(() => { simulateRunStarted(ws); });
      expect(result.current.status.phase).toBe('running');
      expect(result.current.status.isConnected).toBe(true);
      expect(result.current.status.isRunning).toBe(true);
      expect(result.current.status.isPaused).toBe(false);
    });
  });
});
