import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useSimulation, fetchEngines, type SimulationFrame } from './client';

// Create a controllable mock WebSocket
class MockWebSocket {
  static instances: MockWebSocket[] = [];
  static lastUrl: string = '';

  // WebSocket ready state constants
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  url: string;
  readyState: number = 0; // CONNECTING
  onopen: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;

  sentMessages: string[] = [];

  constructor(url: string) {
    this.url = url;
    MockWebSocket.lastUrl = url;
    MockWebSocket.instances.push(this);
  }

  send(data: string) {
    this.sentMessages.push(data);
  }

  close(code?: number, reason?: string) {
    this.readyState = 3; // CLOSED
    if (this.onclose) {
      const event = new CloseEvent('close', {
        code: code || 1000,
        reason: reason || '',
        wasClean: code === 1000,
      });
      this.onclose(event);
    }
  }

  // Test helpers
  simulateOpen() {
    this.readyState = 1; // OPEN
    if (this.onopen) {
      this.onopen(new Event('open'));
    }
  }

  simulateMessage(data: unknown) {
    if (this.onmessage) {
      this.onmessage(new MessageEvent('message', { data: JSON.stringify(data) }));
    }
  }

  simulateError() {
    if (this.onerror) {
      this.onerror(new Event('error'));
    }
  }

  simulateUncleanClose(code: number = 1006) {
    this.readyState = 3;
    if (this.onclose) {
      this.onclose(new CloseEvent('close', { code, wasClean: false }));
    }
  }

  static reset() {
    MockWebSocket.instances = [];
    MockWebSocket.lastUrl = '';
  }

  static getLastInstance(): MockWebSocket | undefined {
    return MockWebSocket.instances[MockWebSocket.instances.length - 1];
  }
}

// Install mock before tests
beforeEach(() => {
  MockWebSocket.reset();
  vi.stubGlobal('WebSocket', MockWebSocket);
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('useSimulation hook', () => {
  describe('initial state', () => {
    it('returns initial idle state', () => {
      const { result } = renderHook(() => useSimulation('mujoco'));

      expect(result.current.isRunning).toBe(false);
      expect(result.current.isPaused).toBe(false);
      expect(result.current.currentFrame).toBeNull();
      expect(result.current.frames).toEqual([]);
      expect(result.current.connectionStatus).toBe('disconnected');
    });

    it('provides control functions', () => {
      const { result } = renderHook(() => useSimulation('mujoco'));

      expect(typeof result.current.start).toBe('function');
      expect(typeof result.current.stop).toBe('function');
      expect(typeof result.current.pause).toBe('function');
      expect(typeof result.current.resume).toBe('function');
      expect(typeof result.current.setSpeed).toBe('function');
    });
  });

  describe('start simulation', () => {
    it('creates WebSocket connection with correct URL', () => {
      const { result } = renderHook(() => useSimulation('mujoco'));

      act(() => {
        result.current.start();
      });

      expect(MockWebSocket.lastUrl).toContain('/api/ws/simulate/mujoco');
    });

    it('uses different URL for different engines', () => {
      const { result } = renderHook(() => useSimulation('drake'));

      act(() => {
        result.current.start();
      });

      expect(MockWebSocket.lastUrl).toContain('/api/ws/simulate/drake');
    });

    it('sets connection status to connecting', () => {
      const { result } = renderHook(() => useSimulation('mujoco'));

      act(() => {
        result.current.start();
      });

      expect(result.current.connectionStatus).toBe('connecting');
    });

    it('sets isRunning to true when connection opens', async () => {
      const { result } = renderHook(() => useSimulation('mujoco'));

      act(() => {
        result.current.start();
      });

      const ws = MockWebSocket.getLastInstance()!;
      act(() => {
        ws.simulateOpen();
      });

      expect(result.current.isRunning).toBe(true);
      expect(result.current.connectionStatus).toBe('connected');
    });

    it('sends start action with config on open', () => {
      const { result } = renderHook(() => useSimulation('mujoco'));

      act(() => {
        result.current.start({ duration: 5.0, timestep: 0.001 });
      });

      const ws = MockWebSocket.getLastInstance()!;
      act(() => {
        ws.simulateOpen();
      });

      const sentMessage = JSON.parse(ws.sentMessages[0]);
      expect(sentMessage.action).toBe('start');
      expect(sentMessage.config.duration).toBe(5.0);
      expect(sentMessage.config.timestep).toBe(0.001);
      expect(sentMessage.config.live_analysis).toBe(true);
    });

    it('clears existing frames on new start', () => {
      const { result } = renderHook(() => useSimulation('mujoco'));

      // Simulate some existing frames
      act(() => {
        result.current.start();
      });

      const ws = MockWebSocket.getLastInstance()!;
      act(() => {
        ws.simulateOpen();
        ws.simulateMessage({ frame: 1, time: 0.1, state: {} });
      });

      expect(result.current.frames.length).toBe(1);

      // Start again
      act(() => {
        result.current.start();
      });

      const ws2 = MockWebSocket.getLastInstance()!;
      act(() => {
        ws2.simulateOpen();
      });

      expect(result.current.frames).toEqual([]);
    });
  });

  describe('receiving frames', () => {
    it('updates currentFrame on frame message', () => {
      const { result } = renderHook(() => useSimulation('mujoco'));

      act(() => {
        result.current.start();
      });

      const ws = MockWebSocket.getLastInstance()!;
      act(() => {
        ws.simulateOpen();
      });

      const frameData: SimulationFrame = {
        frame: 10,
        time: 0.2,
        state: { qpos: [0.1, 0.2, 0.3] },
        analysis: { joint_angles: [0.5, 0.3] },
      };

      act(() => {
        ws.simulateMessage(frameData);
      });

      expect(result.current.currentFrame).toEqual(frameData);
    });

    it('accumulates frames in history', () => {
      const { result } = renderHook(() => useSimulation('mujoco'));

      act(() => {
        result.current.start();
      });

      const ws = MockWebSocket.getLastInstance()!;
      act(() => {
        ws.simulateOpen();
      });

      // Send multiple frames
      for (let i = 0; i < 5; i++) {
        act(() => {
          ws.simulateMessage({ frame: i, time: i * 0.1, state: {} });
        });
      }

      expect(result.current.frames.length).toBe(5);
      expect(result.current.frames[4].frame).toBe(4);
    });

    it('limits frames history to MAX_FRAMES_HISTORY (1000)', () => {
      const { result } = renderHook(() => useSimulation('mujoco'));

      act(() => {
        result.current.start();
      });

      const ws = MockWebSocket.getLastInstance()!;
      act(() => {
        ws.simulateOpen();
      });

      // Send more than 1000 frames
      for (let i = 0; i < 1100; i++) {
        act(() => {
          ws.simulateMessage({ frame: i, time: i * 0.001, state: {} });
        });
      }

      expect(result.current.frames.length).toBe(1000);
      // Should keep the most recent frames
      expect(result.current.frames[0].frame).toBe(100);
      expect(result.current.frames[999].frame).toBe(1099);
    });

    it('handles complete status', () => {
      const { result } = renderHook(() => useSimulation('mujoco'));

      act(() => {
        result.current.start();
      });

      const ws = MockWebSocket.getLastInstance()!;
      act(() => {
        ws.simulateOpen();
      });

      act(() => {
        ws.simulateMessage({ status: 'complete' });
      });

      expect(result.current.isRunning).toBe(false);
    });

    it('handles stopped status', () => {
      const { result } = renderHook(() => useSimulation('mujoco'));

      act(() => {
        result.current.start();
      });

      const ws = MockWebSocket.getLastInstance()!;
      act(() => {
        ws.simulateOpen();
      });

      act(() => {
        ws.simulateMessage({ status: 'stopped' });
      });

      expect(result.current.isRunning).toBe(false);
    });

    it('handles paused status', () => {
      const { result } = renderHook(() => useSimulation('mujoco'));

      act(() => {
        result.current.start();
      });

      const ws = MockWebSocket.getLastInstance()!;
      act(() => {
        ws.simulateOpen();
      });

      act(() => {
        ws.simulateMessage({ status: 'paused' });
      });

      expect(result.current.isPaused).toBe(true);
    });

    it('handles malformed JSON gracefully', () => {
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => { });
      const { result } = renderHook(() => useSimulation('mujoco'));

      act(() => {
        result.current.start();
      });

      const ws = MockWebSocket.getLastInstance()!;
      act(() => {
        ws.simulateOpen();
      });

      // Send invalid JSON
      act(() => {
        if (ws.onmessage) {
          ws.onmessage(new MessageEvent('message', { data: 'not json{' }));
        }
      });

      expect(consoleSpy).toHaveBeenCalled();
      expect(result.current.isRunning).toBe(true); // Should not crash
      consoleSpy.mockRestore();
    });
  });

  describe('pause and resume', () => {
    it('sends pause action', () => {
      const { result } = renderHook(() => useSimulation('mujoco'));

      act(() => {
        result.current.start();
      });

      const ws = MockWebSocket.getLastInstance()!;
      act(() => {
        ws.simulateOpen();
      });

      // Clear messages after start
      const startMessageCount = ws.sentMessages.length;

      act(() => {
        result.current.pause();
      });

      // Find the pause message
      const pauseMessage = ws.sentMessages.slice(startMessageCount).find(
        (m) => JSON.parse(m).action === 'pause'
      );
      expect(pauseMessage).toBeDefined();
      expect(JSON.parse(pauseMessage!).action).toBe('pause');
      expect(result.current.isPaused).toBe(true);
    });

    it('sends resume action', () => {
      const { result } = renderHook(() => useSimulation('mujoco'));

      act(() => {
        result.current.start();
      });

      const ws = MockWebSocket.getLastInstance()!;
      act(() => {
        ws.simulateOpen();
      });

      act(() => {
        result.current.pause();
      });

      act(() => {
        result.current.resume();
      });

      // Find the resume message
      const resumeMessage = ws.sentMessages.find(
        (m) => JSON.parse(m).action === 'resume'
      );
      expect(resumeMessage).toBeDefined();
      expect(JSON.parse(resumeMessage!).action).toBe('resume');
      expect(result.current.isPaused).toBe(false);
    });

    it('does nothing if WebSocket not open', () => {
      const { result } = renderHook(() => useSimulation('mujoco'));

      // Don't start - no WebSocket
      act(() => {
        result.current.pause();
      });

      // Should not throw
      expect(result.current.isPaused).toBe(false);
    });
  });

  describe('speed control', () => {
    it('posts speed changes to the supported simulation speed endpoint', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            speed_factor: 2.5,
            status: 'Speed set to 2.5x',
          }),
      });

      const { result } = renderHook(() => useSimulation('mujoco'));

      act(() => {
        result.current.start();
      });

      const ws = MockWebSocket.getLastInstance()!;
      act(() => {
        ws.simulateOpen();
      });

      await act(async () => {
        result.current.setSpeed(2.5);
        await Promise.resolve();
      });

      expect(global.fetch).toHaveBeenCalledWith('/api/simulation/speed', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ speed_factor: 2.5 }),
      });
      expect(ws.sentMessages).toHaveLength(1);
      expect(JSON.parse(ws.sentMessages[0]).action).toBe('start');
    });
  });

  describe('stop simulation', () => {
    it('sends stop action and closes WebSocket', () => {
      const { result } = renderHook(() => useSimulation('mujoco'));

      act(() => {
        result.current.start();
      });

      const ws = MockWebSocket.getLastInstance()!;
      act(() => {
        ws.simulateOpen();
      });

      act(() => {
        result.current.stop();
      });

      // Find the stop message
      const stopMessage = ws.sentMessages.find(
        (m) => JSON.parse(m).action === 'stop'
      );
      expect(stopMessage).toBeDefined();
      expect(JSON.parse(stopMessage!).action).toBe('stop');
      expect(ws.readyState).toBe(3); // CLOSED
    });

    it('sets connection status to disconnected', () => {
      const { result } = renderHook(() => useSimulation('mujoco'));

      act(() => {
        result.current.start();
      });

      const ws = MockWebSocket.getLastInstance()!;
      act(() => {
        ws.simulateOpen();
      });

      act(() => {
        result.current.stop();
      });

      expect(result.current.connectionStatus).toBe('disconnected');
    });
  });

  describe('reconnection (#6896 — safe, no silent restart)', () => {
    it('does NOT silently reconnect-and-restart on an unclean close', () => {
      // Regression guard for #6896: a transient blip previously reopened the
      // socket and sent {action:'start'}, restarting the run from t=0.
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => { });
      const { result } = renderHook(() => useSimulation('mujoco'));

      act(() => {
        result.current.start();
      });

      const ws = MockWebSocket.getLastInstance()!;
      act(() => {
        ws.simulateOpen();
      });

      const instanceCountBefore = MockWebSocket.instances.length;

      act(() => {
        ws.simulateUncleanClose(1006); // Abnormal closure
      });

      // Status reflects an unrecoverable drop, not an implied continuity.
      expect(result.current.connectionStatus).toBe('lost');

      // No timer-driven reconnect: advancing time creates no new socket.
      act(() => {
        vi.advanceTimersByTime(60000);
      });
      expect(MockWebSocket.instances.length).toBe(instanceCountBefore);
      warnSpy.mockRestore();
    });

    it('does not reset frames or current frame on an unclean close', () => {
      // The core assertion #6896 asks for: reconnect/disconnect must not wipe
      // the captured timeline without explicit user action.
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => { });
      const { result } = renderHook(() => useSimulation('mujoco'));

      act(() => {
        result.current.start();
      });

      const ws = MockWebSocket.getLastInstance()!;
      act(() => {
        ws.simulateOpen();
      });

      // Capture a few frames.
      act(() => {
        ws.simulateMessage({ frame: 0, time: 0.0, state: {} });
        ws.simulateMessage({ frame: 1, time: 0.5, state: {} });
        ws.simulateMessage({ frame: 2, time: 1.0, state: {} });
      });

      expect(result.current.frames).toHaveLength(3);
      const framesBefore = result.current.frames;
      const currentBefore = result.current.currentFrame;

      act(() => {
        ws.simulateUncleanClose(1006);
      });

      // Frames and the last frame/time are preserved untouched.
      expect(result.current.frames).toBe(framesBefore);
      expect(result.current.frames).toHaveLength(3);
      expect(result.current.currentFrame).toBe(currentBefore);
      expect(result.current.currentFrame?.time).toBe(1.0);
      expect(result.current.connectionStatus).toBe('lost');
      warnSpy.mockRestore();
    });

    it('surfaces a restart-required error after an unclean close', () => {
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => { });
      const { result } = renderHook(() => useSimulation('mujoco'));

      act(() => {
        result.current.start();
      });
      const ws = MockWebSocket.getLastInstance()!;
      act(() => {
        ws.simulateOpen();
        ws.simulateUncleanClose(1006);
      });

      expect(result.current.wsError).toMatch(/restart/i);
      expect(result.current.isRunning).toBe(false);
      warnSpy.mockRestore();
    });

    it('clears the timeline only on an explicit user restart', () => {
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => { });
      const { result } = renderHook(() => useSimulation('mujoco'));

      act(() => {
        result.current.start();
      });
      let ws = MockWebSocket.getLastInstance()!;
      act(() => {
        ws.simulateOpen();
        ws.simulateMessage({ frame: 0, time: 0.0, state: {} });
        ws.simulateMessage({ frame: 1, time: 0.5, state: {} });
        ws.simulateUncleanClose(1006);
      });

      expect(result.current.frames).toHaveLength(2);
      expect(result.current.connectionStatus).toBe('lost');

      // The user explicitly restarts — only now is the timeline reset.
      act(() => {
        result.current.start();
      });
      ws = MockWebSocket.getLastInstance()!;
      act(() => {
        ws.simulateOpen();
      });

      expect(result.current.connectionStatus).toBe('connected');
      expect(result.current.frames).toEqual([]);
      expect(result.current.wsError).toBeNull();
      warnSpy.mockRestore();
    });

    it('does not reconnect on clean close', () => {
      const { result } = renderHook(() => useSimulation('mujoco'));

      act(() => {
        result.current.start();
      });

      const ws = MockWebSocket.getLastInstance()!;
      act(() => {
        ws.simulateOpen();
      });

      const instanceCountBefore = MockWebSocket.instances.length;

      act(() => {
        ws.close(1000, 'User stopped');
      });

      expect(result.current.connectionStatus).toBe('disconnected');

      act(() => {
        vi.advanceTimersByTime(5000);
      });

      // No new WebSocket should be created
      expect(MockWebSocket.instances.length).toBe(instanceCountBefore);
    });

    it('does not reconnect when stop is called', () => {
      const { result } = renderHook(() => useSimulation('mujoco'));

      act(() => {
        result.current.start();
      });

      const ws = MockWebSocket.getLastInstance()!;
      act(() => {
        ws.simulateOpen();
      });

      const instanceCountBefore = MockWebSocket.instances.length;

      act(() => {
        result.current.stop();
      });

      act(() => {
        vi.advanceTimersByTime(5000);
      });

      expect(MockWebSocket.instances.length).toBe(instanceCountBefore);
    });
  });

  describe('cleanup', () => {
    it('closes WebSocket on unmount', () => {
      const { result, unmount } = renderHook(() => useSimulation('mujoco'));

      act(() => {
        result.current.start();
      });

      const ws = MockWebSocket.getLastInstance()!;
      act(() => {
        ws.simulateOpen();
      });

      unmount();

      expect(ws.readyState).toBe(3); // CLOSED
    });

    it('creates no new socket after an unclean close, even across unmount', () => {
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => { });
      const { result, unmount } = renderHook(() => useSimulation('mujoco'));

      act(() => {
        result.current.start();
      });

      const ws = MockWebSocket.getLastInstance()!;
      act(() => {
        ws.simulateOpen();
        ws.simulateUncleanClose(1006);
      });

      // #6896: an unclean close no longer schedules a reconnect.
      expect(result.current.connectionStatus).toBe('lost');

      unmount();

      const instanceCountBefore = MockWebSocket.instances.length;

      act(() => {
        vi.advanceTimersByTime(35000);
      });

      // No new WebSocket should be created after unmount
      expect(MockWebSocket.instances.length).toBe(instanceCountBefore);
      warnSpy.mockRestore();
    });

    it('does not update state after unmount', () => {
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => { });
      const { result, unmount } = renderHook(() => useSimulation('mujoco'));

      act(() => {
        result.current.start();
      });

      const ws = MockWebSocket.getLastInstance()!;
      act(() => {
        ws.simulateOpen();
      });

      unmount();

      // This should not throw or warn about state update on unmounted component
      act(() => {
        ws.simulateMessage({ frame: 1, time: 0.1, state: {} });
      });

      // No React warning should have been logged
      consoleSpy.mockRestore();
    });
  });

  describe('engine type changes', () => {
    it('reinitializes when engine type changes', () => {
      const { result, rerender } = renderHook(
        ({ engine }) => useSimulation(engine),
        { initialProps: { engine: 'mujoco' } }
      );

      act(() => {
        result.current.start();
      });

      expect(MockWebSocket.lastUrl).toContain('mujoco');

      // Rerender with different engine
      rerender({ engine: 'drake' });

      act(() => {
        result.current.start();
      });

      expect(MockWebSocket.lastUrl).toContain('drake');
    });
  });
});

describe('fetchEngines', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  it('returns engine list on success', async () => {
    const mockEngines = [
      { name: 'mujoco', available: true, loaded: true, capabilities: [] },
    ];

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ engines: mockEngines }),
    });

    const result = await fetchEngines();

    expect(result).toEqual(mockEngines);
  });

  it('throws error on failure', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
    });

    await expect(fetchEngines()).rejects.toThrow('Failed to fetch engines');
  });
});
