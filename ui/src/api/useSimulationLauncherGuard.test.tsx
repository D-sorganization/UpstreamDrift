/**
 * Regression tests for the WebSocket launcher-guard recovery path (#8077).
 *
 * Reported symptom: open `/simulation` in the local web UI, load MuJoCo, press
 * Start — the run reports "Connection lost — restart required" instantly and
 * never receives a frame.
 *
 * Root cause (confirmed against a live API on :8010):
 *   `enforce_local_websocket_guard` in `src/api/auth/ws_auth.py` rejects a
 *   local-mode upgrade that carries no launcher capability token. Because the
 *   token was cached only as a side effect of the Dashboard's
 *   `useLauncherManifest` hook, a direct visit to `/simulation` never had one.
 *   The browser surfaces that rejection as a *failed handshake* (close code
 *   1006, `wasClean: false`) rather than a 1008 close frame — hence the
 *   code-agnostic "handshake never completed" condition under test.
 *
 * The recovery must not weaken #6896: it may only fire when `onopen` never ran,
 * so no captured frames can be discarded.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { useSimulation } from './client';
import { setLauncherCapabilityToken } from './websocketToken';

class MockWebSocket {
  static instances: MockWebSocket[] = [];
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  url: string;
  readyState = 0;
  onopen: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  sentMessages: string[] = [];

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }

  send(data: string) {
    this.sentMessages.push(data);
  }

  close(code?: number) {
    this.readyState = 3;
    this.onclose?.(
      new CloseEvent('close', { code: code ?? 1000, wasClean: code === 1000 }),
    );
  }

  /** Server accepted the upgrade. */
  simulateOpen() {
    this.readyState = 1;
    this.onopen?.(new Event('open'));
  }

  /** Guard rejection: the handshake fails, so onopen never runs. */
  simulateHandshakeRejection() {
    this.readyState = 3;
    this.onclose?.(new CloseEvent('close', { code: 1006, wasClean: false }));
  }

  static reset() {
    MockWebSocket.instances = [];
  }

  static last(): MockWebSocket {
    const ws = MockWebSocket.instances[MockWebSocket.instances.length - 1];
    expect(ws, 'expected a WebSocket to have been constructed').toBeDefined();
    return ws;
  }
}

function stubManifest(token: string | null) {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: token !== null,
    json: () => Promise.resolve(token === null ? {} : { launcher_csrf_token: token }),
  } as unknown as Response);
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

beforeEach(() => {
  MockWebSocket.reset();
  setLauncherCapabilityToken(null);
  vi.stubGlobal('WebSocket', MockWebSocket);
});

afterEach(() => {
  setLauncherCapabilityToken(null);
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('useSimulation launcher-guard recovery (#8077)', () => {
  it('prefetches the launcher token on mount', async () => {
    const fetchMock = stubManifest('mount-token');

    renderHook(() => useSimulation('mujoco'));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalled();
    });
    expect(String(fetchMock.mock.calls[0][0])).toContain('/api/launcher/manifest');
  });

  it('sends the prefetched token on the first connect', async () => {
    stubManifest('mount-token');
    const { result } = renderHook(() => useSimulation('mujoco'));

    await waitFor(() => {
      expect(vi.mocked(globalThis.fetch)).toHaveBeenCalled();
    });

    act(() => {
      result.current.start();
    });

    expect(
      new URL(MockWebSocket.last().url).searchParams.get('launcher_token'),
    ).toBe('mount-token');
  });

  it('re-handshakes with a token when the upgrade was rejected without one', async () => {
    // Manifest is slow/unavailable at mount, so the first connect has no token.
    let resolveManifest: (value: Response) => void = () => {};
    const pending = new Promise<Response>((resolve) => {
      resolveManifest = resolve;
    });
    vi.stubGlobal('fetch', vi.fn().mockReturnValue(pending));

    const { result } = renderHook(() => useSimulation('mujoco'));

    act(() => {
      result.current.start();
    });
    expect(MockWebSocket.instances).toHaveLength(1);
    expect(MockWebSocket.last().url).not.toContain('launcher_token');

    // The manifest finally answers, then the guard rejects the tokenless socket.
    resolveManifest({
      ok: true,
      json: () => Promise.resolve({ launcher_csrf_token: 'late-token' }),
    } as unknown as Response);

    await act(async () => {
      MockWebSocket.instances[0].simulateHandshakeRejection();
    });

    await waitFor(() => {
      expect(MockWebSocket.instances).toHaveLength(2);
    });
    expect(
      new URL(MockWebSocket.instances[1].url).searchParams.get('launcher_token'),
    ).toBe('late-token');
  });

  it('retries at most once per user-initiated run', async () => {
    stubManifest(null); // manifest never yields a token
    const { result } = renderHook(() => useSimulation('mujoco'));

    act(() => {
      result.current.start();
    });

    await act(async () => {
      MockWebSocket.last().simulateHandshakeRejection();
    });

    // No token was obtainable, so no reconnect may happen — and the UI must
    // land on a terminal state rather than spinning on 'connecting' forever.
    expect(MockWebSocket.instances).toHaveLength(1);
    await waitFor(() => {
      expect(result.current.connectionStatus).toBe('lost');
    });
    expect(result.current.wsError).toContain('Connection lost');
  });

  it('does not retry when the handshake had already succeeded (#6896)', async () => {
    stubManifest('tok');
    const { result } = renderHook(() => useSimulation('mujoco'));
    await waitFor(() => {
      expect(vi.mocked(globalThis.fetch)).toHaveBeenCalled();
    });

    act(() => {
      result.current.start();
    });
    act(() => {
      MockWebSocket.last().simulateOpen();
    });

    await act(async () => {
      MockWebSocket.last().simulateHandshakeRejection();
    });

    // A drop after a successful open must stay 'lost' and require an explicit
    // restart — silently reconnecting would restart the run from t=0.
    expect(MockWebSocket.instances).toHaveLength(1);
    await waitFor(() => {
      expect(result.current.connectionStatus).toBe('lost');
    });
    expect(result.current.wsError).toContain('Connection lost');
  });

  it('does not retry when a token was already present at connect time', async () => {
    stubManifest('tok');
    setLauncherCapabilityToken('tok');
    const { result } = renderHook(() => useSimulation('mujoco'));

    act(() => {
      result.current.start();
    });

    await act(async () => {
      MockWebSocket.last().simulateHandshakeRejection();
    });

    // The token was not the problem, so this is a real transport failure.
    expect(MockWebSocket.instances).toHaveLength(1);
    await waitFor(() => {
      expect(result.current.connectionStatus).toBe('lost');
    });
  });
});
