/**
 * Tests for ChatPanel.
 *
 * Mocks WebSocket so we can drive server messages and assert UI state.
 *
 * See issue #3505.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, fireEvent, waitFor, act } from '@testing-library/react';
import { screen } from '@testing-library/dom';
import { ChatPanel, resolveChatUrl } from './ChatPanel';

// ---------------------------------------------------------------------------
// Mock WebSocket — local override of the global mock from setup.ts so we
// can capture sent payloads and emit messages from the server side.
// ---------------------------------------------------------------------------

interface SentFrame {
  data: string;
}

class TestWebSocket {
  static OPEN = 1;
  static instances: TestWebSocket[] = [];

  url: string;
  readyState: number = 0;
  sent: SentFrame[] = [];
  onopen: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;

  constructor(url: string) {
    this.url = url;
    TestWebSocket.instances.push(this);
    setTimeout(() => {
      this.readyState = TestWebSocket.OPEN;
      this.onopen?.(new Event('open'));
    }, 0);
  }

  send(data: string) {
    this.sent.push({ data });
  }

  close() {
    this.readyState = 3;
    this.onclose?.(new CloseEvent('close'));
  }

  // Test helper to push a server -> client frame.
  emit(payload: unknown) {
    this.onmessage?.(
      new MessageEvent('message', { data: JSON.stringify(payload) })
    );
  }
}

beforeEach(() => {
  TestWebSocket.instances = [];
  // Provide static OPEN constant on the class so component checks pass.
  Object.assign(TestWebSocket, { OPEN: 1 });
  vi.stubGlobal('WebSocket', TestWebSocket);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function waitForSocket(): Promise<TestWebSocket> {
  await waitFor(() => {
    expect(TestWebSocket.instances.length).toBeGreaterThan(0);
  });
  const socket = TestWebSocket.instances[0];
  await waitFor(() => {
    expect(socket.readyState).toBe(TestWebSocket.OPEN);
  });
  return socket;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('resolveChatUrl', () => {
  it('falls back to ws://localhost:8000 when env is unset', () => {
    expect(resolveChatUrl('new')).toMatch(/^ws:\/\/localhost:8000\/ws\/chat\/new$/);
  });

  it('rejects empty session ids', () => {
    expect(() => resolveChatUrl('')).toThrow();
  });
});

describe('ChatPanel', () => {
  it('renders header with chat icon and connection status', async () => {
    render(<ChatPanel />);
    expect(screen.getByText('Chat')).toBeInTheDocument();
    // Status starts as connecting then becomes connected.
    await waitFor(() => {
      expect(screen.getByTestId('chat-status').textContent).toMatch(/Connected/);
    });
  });

  it('disables send button until input has text', async () => {
    render(<ChatPanel />);
    await waitForSocket();
    const sendBtn = screen.getByTestId('chat-send') as HTMLButtonElement;
    expect(sendBtn.disabled).toBe(true);

    const input = screen.getByTestId('chat-input') as HTMLTextAreaElement;
    fireEvent.change(input, { target: { value: 'hello' } });
    expect(sendBtn.disabled).toBe(false);
  });

  it('sends a message and renders it in the message list', async () => {
    render(<ChatPanel />);
    const socket = await waitForSocket();

    const input = screen.getByTestId('chat-input') as HTMLTextAreaElement;
    fireEvent.change(input, { target: { value: 'hello world' } });

    const sendBtn = screen.getByTestId('chat-send');
    fireEvent.click(sendBtn);

    // The user message should appear in the rendered list.
    await waitFor(() => {
      expect(screen.getByText('hello world')).toBeInTheDocument();
    });

    // And the socket should have received the JSON payload.
    expect(socket.sent.length).toBe(1);
    const payload = JSON.parse(socket.sent[0].data) as Record<string, unknown>;
    expect(payload).toEqual({ action: 'send', message: 'hello world' });

    // Input is cleared after send.
    expect(input.value).toBe('');
  });

  it('renders streaming assistant chunks and finalizes on complete', async () => {
    render(<ChatPanel />);
    const socket = await waitForSocket();

    act(() => {
      socket.emit({ type: 'session_info', session_id: 'abc-123' });
    });

    act(() => {
      socket.emit({ type: 'chunk', content: 'Hello ' });
      socket.emit({ type: 'chunk', content: 'there!' });
    });

    await waitFor(() => {
      expect(screen.getByText('Hello there!')).toBeInTheDocument();
    });

    act(() => {
      socket.emit({ type: 'complete', session_id: 'abc-123' });
    });

    await waitFor(() => {
      expect(screen.queryByTestId('chat-streaming')).not.toBeInTheDocument();
    });
  });

  it('surfaces server error messages as system entries', async () => {
    render(<ChatPanel />);
    const socket = await waitForSocket();

    act(() => {
      socket.emit({ type: 'error', detail: 'boom' });
    });

    await waitFor(() => {
      expect(screen.getByText('boom')).toBeInTheDocument();
    });
  });

  it('forwards engine_context when provided', async () => {
    render(<ChatPanel engineContext="mujoco" />);
    const socket = await waitForSocket();

    fireEvent.change(screen.getByTestId('chat-input'), {
      target: { value: 'q' },
    });
    fireEvent.click(screen.getByTestId('chat-send'));

    await waitFor(() => {
      expect(socket.sent.length).toBe(1);
    });
    const payload = JSON.parse(socket.sent[0].data) as Record<string, unknown>;
    expect(payload.engine_context).toBe('mujoco');
  });

  it('renders message bubbles with data-role but no inline style overrides (CSS should own colors)', async () => {
    render(<ChatPanel />);
    const socket = await waitForSocket();

    // User message
    fireEvent.change(screen.getByTestId('chat-input'), { target: { value: 'hi' } });
    fireEvent.click(screen.getByTestId('chat-send'));

    await waitFor(() => {
      expect(screen.getByText('hi')).toBeInTheDocument();
    });

    // System error message
    act(() => {
      socket.emit({ type: 'error', detail: 'err' });
    });
    await waitFor(() => {
      expect(screen.getByText('err')).toBeInTheDocument();
    });

    // Assistant streaming chunk
    act(() => {
      socket.emit({ type: 'chunk', content: 'reply' });
    });
    await waitFor(() => {
      expect(screen.getByText('reply')).toBeInTheDocument();
    });

    // Every bubble must have data-role and must NOT have inline backgroundColor/borderColor/color
    const allBubbles = document.querySelectorAll('.sidekick-chat-bubble');
    expect(allBubbles.length).toBeGreaterThan(0);

    allBubbles.forEach((bubble) => {
      const el = bubble as HTMLElement;
      expect(el.dataset.role).toBeTruthy();
      expect(el.style.backgroundColor).toBe('');
      expect(el.style.borderColor).toBe('');
      expect(el.style.color).toBe('');
    });
  });
});