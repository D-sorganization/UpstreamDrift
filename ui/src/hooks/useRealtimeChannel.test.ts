/**
 * Tests for useRealtimeChannel — reconnecting realtime WebSocket hook.
 *
 * See issue #8406
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';

vi.mock('@/api/websocketToken', () => ({
  ensureLauncherCapabilityToken: vi.fn(async () => null),
  withLauncherWebSocketToken: vi.fn((url: string) => url),
}));

import { useRealtimeChannel, buildRealtimeSubscribeUrl } from './useRealtimeChannel';

// Controllable mock WebSocket (same pattern as useSimulation.test.ts)
class MockWebSocket {
  static instances: MockWebSocket[] = [];

  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  url: string;
  readyState: number = 0;
  closeCalled = false;
  onopen: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }

  send() {}

  close(code?: number, reason?: string) {
    this.closeCalled = true;
    this.readyState = 3;
    if (this.onclose) {
      this.onclose(
        new CloseEvent('close', {
          code: code || 1000,
          reason: reason || '',
          wasClean: code === 1000,
        }),
      );
    }
  }

  simulateOpen() {
    this.readyState = 1;
    this.onopen?.(new Event('open'));
  }

  simulateMessage(data: unknown) {
    this.onmessage?.(
      new MessageEvent('message', { data: JSON.stringify(data) }),
    );
  }

  simulateRawMessage(data: string) {
    this.onmessage?.(new MessageEvent('message', { data }));
  }

  simulateUncleanClose(code: number = 1006) {
    this.readyState = 3;
    this.onclose?.(new CloseEvent('close', { code, wasClean: false }));
  }

  static reset() {
    MockWebSocket.instances = [];
  }
}

/** Flush the async connect path (token fetch microtasks) under fake timers. */
async function flushConnect() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0);
  });
}

describe('useRealtimeChannel', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    MockWebSocket.reset();
    vi.stubGlobal('WebSocket', MockWebSocket);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  describe('buildRealtimeSubscribeUrl', () => {
    it('builds a ws URL with the encoded channel query parameter', () => {
      const url = buildRealtimeSubscribeUrl('pose/canonical');
      expect(url).toContain('/api/realtime/subscribe?channel=pose%2Fcanonical');
      expect(url.startsWith('ws')).toBe(true);
    });

    it('rejects an empty channel', () => {
      expect(() => buildRealtimeSubscribeUrl('')).toThrow();
    });
  });

  it('connects to the subscribe endpoint for the channel', async () => {
    const { result } = renderHook(() => useRealtimeChannel('pose/canonical'));
    await flushConnect();

    expect(MockWebSocket.instances).toHaveLength(1);
    expect(MockWebSocket.instances[0].url).toContain(
      '/api/realtime/subscribe?channel=pose%2Fcanonical',
    );

    act(() => {
      MockWebSocket.instances[0].simulateOpen();
    });
    expect(result.current.status).toBe('connected');
  });

  it('does not connect when disabled', async () => {
    const { result } = renderHook(() =>
      useRealtimeChannel('pose/canonical', false),
    );
    await flushConnect();

    expect(MockWebSocket.instances).toHaveLength(0);
    expect(result.current.status).toBe('idle');
  });

  it('parses JSON messages and exposes the latest payload', async () => {
    const { result } = renderHook(() =>
      useRealtimeChannel<{ joints: unknown[] }>('pose/canonical'),
    );
    await flushConnect();

    const socket = MockWebSocket.instances[0];
    act(() => {
      socket.simulateOpen();
      socket.simulateMessage({ joints: [{ name: 'hips' }] });
    });
    expect(result.current.message).toEqual({ joints: [{ name: 'hips' }] });

    act(() => {
      socket.simulateMessage({ joints: [] });
    });
    expect(result.current.message).toEqual({ joints: [] });
  });

  it('keeps the last good message when a frame is unparseable', async () => {
    const { result } = renderHook(() => useRealtimeChannel('pose/canonical'));
    await flushConnect();

    const socket = MockWebSocket.instances[0];
    act(() => {
      socket.simulateOpen();
      socket.simulateMessage({ ok: true });
      socket.simulateRawMessage('{not json');
    });
    expect(result.current.message).toEqual({ ok: true });
  });

  it('reconnects with backoff after an unclean close', async () => {
    const { result } = renderHook(() => useRealtimeChannel('pose/canonical'));
    await flushConnect();
    expect(MockWebSocket.instances).toHaveLength(1);

    act(() => {
      MockWebSocket.instances[0].simulateOpen();
      MockWebSocket.instances[0].simulateUncleanClose();
    });
    expect(result.current.status).toBe('reconnecting');

    // Not yet: first backoff step is 500ms
    await act(async () => {
      await vi.advanceTimersByTimeAsync(499);
    });
    expect(MockWebSocket.instances).toHaveLength(1);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1);
    });
    expect(MockWebSocket.instances).toHaveLength(2);

    // Second consecutive failure doubles the delay to 1000ms
    act(() => {
      MockWebSocket.instances[1].simulateUncleanClose();
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(999);
    });
    expect(MockWebSocket.instances).toHaveLength(2);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1);
    });
    expect(MockWebSocket.instances).toHaveLength(3);
  });

  it('caps the reconnect backoff at 5 seconds', async () => {
    renderHook(() => useRealtimeChannel('pose/canonical'));
    await flushConnect();

    // Fail repeatedly; backoff sequence 500, 1000, 2000, 4000, 5000, 5000...
    for (let i = 0; i < 6; i++) {
      act(() => {
        MockWebSocket.instances[MockWebSocket.instances.length - 1]
          .simulateUncleanClose();
      });
      await act(async () => {
        await vi.advanceTimersByTimeAsync(5000);
      });
    }
    const count = MockWebSocket.instances.length;

    act(() => {
      MockWebSocket.instances[count - 1].simulateUncleanClose();
    });
    // 5s must now always be enough for the next attempt
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5000);
    });
    expect(MockWebSocket.instances.length).toBe(count + 1);
  });

  it('resets the backoff after a successful open', async () => {
    renderHook(() => useRealtimeChannel('pose/canonical'));
    await flushConnect();

    // Two failures push backoff to 2000ms next
    act(() => MockWebSocket.instances[0].simulateUncleanClose());
    await act(async () => {
      await vi.advanceTimersByTimeAsync(500);
    });
    act(() => MockWebSocket.instances[1].simulateUncleanClose());
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000);
    });
    expect(MockWebSocket.instances).toHaveLength(3);

    // A successful open resets the backoff to 500ms
    act(() => {
      MockWebSocket.instances[2].simulateOpen();
      MockWebSocket.instances[2].simulateUncleanClose();
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(500);
    });
    expect(MockWebSocket.instances).toHaveLength(4);
  });

  it('closes the socket and stops reconnecting on unmount', async () => {
    const { unmount } = renderHook(() => useRealtimeChannel('pose/canonical'));
    await flushConnect();

    const socket = MockWebSocket.instances[0];
    act(() => {
      socket.simulateOpen();
    });

    unmount();
    expect(socket.closeCalled).toBe(true);

    // No reconnect attempts after unmount, even well past the backoff cap
    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000);
    });
    expect(MockWebSocket.instances).toHaveLength(1);
  });

  it('cancels a pending reconnect timer on unmount', async () => {
    const { unmount } = renderHook(() => useRealtimeChannel('pose/canonical'));
    await flushConnect();

    act(() => {
      MockWebSocket.instances[0].simulateUncleanClose();
    });
    unmount();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000);
    });
    expect(MockWebSocket.instances).toHaveLength(1);
  });
});
