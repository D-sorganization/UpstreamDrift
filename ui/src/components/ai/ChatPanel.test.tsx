/**
 * Tests for ChatPanel component and useChatStore.
 *
 * Mocks WebSocket to test chunk coalescence, connection status,
 * keyboard shortcuts, and message rendering.
 *
 * See issue #3161
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, fireEvent, waitFor, act } from '@testing-library/react';
import { screen } from '@testing-library/dom';
import { ChatPanel } from './ChatPanel';
import { useChatStore } from '../../stores/useChatStore';

// ── Mock scrollIntoView (not available in jsdom) ──────────────────────────

beforeEach(() => {
  window.HTMLElement.prototype.scrollIntoView = vi.fn();
});

// ── Controllable MockWebSocket ─────────────────────────────────────────────

class MockWebSocket {
  static instances: MockWebSocket[] = [];
  static lastUrl = '';

  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;

  url: string;
  readyState = 0;
  onopen: ((e: Event) => void) | null = null;
  onclose: ((e: CloseEvent) => void) | null = null;
  onmessage: ((e: MessageEvent) => void) | null = null;
  onerror: ((e: Event) => void) | null = null;
  sentMessages: string[] = [];

  constructor(url: string) {
    this.url = url;
    MockWebSocket.lastUrl = url;
    MockWebSocket.instances.push(this);
  }

  send(data: string) {
    this.sentMessages.push(data);
  }

  close(code = 1000, reason = '') {
    this.readyState = 3;
    if (this.onclose) {
      this.onclose(new CloseEvent('close', { code, reason, wasClean: code === 1000 }));
    }
  }

  simulateOpen() {
    this.readyState = 1;
    if (this.onopen) this.onopen(new Event('open'));
  }

  simulateMessage(data: unknown) {
    if (this.onmessage) {
      this.onmessage(new MessageEvent('message', { data: JSON.stringify(data) }));
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

// ── Setup / teardown ──────────────────────────────────────────────────────

beforeEach(() => {
  MockWebSocket.reset();
  vi.stubGlobal('WebSocket', MockWebSocket);
  // Reset store state between tests
  useChatStore.setState({
    open: false,
    sessionId: null,
    messages: [],
    inputBuffer: '',
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

// ── Helpers ───────────────────────────────────────────────────────────────

function openPanel() {
  const fab = screen.getByLabelText(/open ai chat/i);
  fireEvent.click(fab);
}

function getWs(): MockWebSocket {
  return MockWebSocket.getLastInstance()!;
}

// ── Tests ─────────────────────────────────────────────────────────────────

describe('ChatPanel', () => {
  describe('FAB and open/close', () => {
    it('renders FAB button', () => {
      render(<ChatPanel />);
      expect(screen.getByLabelText(/open ai chat/i)).toBeInTheDocument();
    });

    it('shows panel when FAB is clicked', () => {
      render(<ChatPanel />);
      openPanel();
      expect(screen.getByRole('dialog')).toBeInTheDocument();
    });

    it('hides panel when X button is clicked', () => {
      render(<ChatPanel />);
      openPanel();
      fireEvent.click(screen.getByLabelText('Close chat panel'));
      expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    });

    it('toggles with Ctrl+/ keyboard shortcut', () => {
      render(<ChatPanel />);
      expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
      fireEvent.keyDown(window, { key: '/', ctrlKey: true });
      expect(screen.getByRole('dialog')).toBeInTheDocument();
      fireEvent.keyDown(window, { key: '/', ctrlKey: true });
      expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    });

    it('closes on Escape when panel is open', () => {
      render(<ChatPanel />);
      openPanel();
      fireEvent.keyDown(window, { key: 'Escape' });
      expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    });
  });

  describe('accessibility', () => {
    it('message list has role="log" and aria-live="polite"', () => {
      render(<ChatPanel />);
      openPanel();
      const log = screen.getByRole('log');
      expect(log).toHaveAttribute('aria-live', 'polite');
    });

    it('has accessible chat input', () => {
      render(<ChatPanel />);
      openPanel();
      expect(screen.getByLabelText('Chat input')).toBeInTheDocument();
    });

    it('has accessible send button', () => {
      render(<ChatPanel />);
      openPanel();
      expect(screen.getByLabelText('Send message')).toBeInTheDocument();
    });

    it('dialog has correct aria attributes', () => {
      render(<ChatPanel />);
      openPanel();
      const dialog = screen.getByRole('dialog');
      expect(dialog).toHaveAttribute('aria-label', 'AI assistant chat');
    });
  });

  describe('connection status badge', () => {
    it('shows "Connecting" status badge initially', () => {
      render(<ChatPanel />);
      openPanel();
      // WebSocket was created but not yet opened → status = 'connecting'
      expect(screen.getByLabelText(/connection status: connecting/i)).toBeInTheDocument();
    });

    it('shows "Connected" badge after WebSocket opens', async () => {
      render(<ChatPanel />);
      openPanel();
      const ws = getWs();
      act(() => ws.simulateOpen());
      await waitFor(() =>
        expect(screen.getByLabelText(/connection status: connected/i)).toBeInTheDocument(),
      );
    });
  });

  describe('chunk coalescence', () => {
    it('concatenates streaming chunks into a single assistant bubble', async () => {
      render(<ChatPanel />);
      openPanel();
      const ws = getWs();
      act(() => ws.simulateOpen());
      act(() => ws.simulateMessage({ type: 'session_info', session_id: 'sess-1' }));

      // Seed a user message so there is a prior message
      act(() => useChatStore.getState().addMessage('user', 'Hello'));

      act(() => ws.simulateMessage({ type: 'chunk', content: 'Hi ' }));
      act(() => ws.simulateMessage({ type: 'chunk', content: 'there, ' }));
      act(() => ws.simulateMessage({ type: 'chunk', content: 'world!' }));
      act(() => ws.simulateMessage({ type: 'complete', session_id: 'sess-1' }));

      await waitFor(() => {
        expect(screen.getByText('Hi there, world!')).toBeInTheDocument();
      });

      const msgs = useChatStore.getState().messages;
      const assistantMessages = msgs.filter((m) => m.role === 'assistant');
      expect(assistantMessages).toHaveLength(1);
      expect(assistantMessages[0].content).toBe('Hi there, world!');
      expect(assistantMessages[0].streaming).toBe(false);
    });

    it('sets streaming flag while chunks arrive', async () => {
      render(<ChatPanel />);
      openPanel();
      const ws = getWs();
      act(() => ws.simulateOpen());
      act(() => ws.simulateMessage({ type: 'chunk', content: 'Thinking...' }));

      await waitFor(() => {
        const msgs = useChatStore.getState().messages;
        expect(msgs[0]?.streaming).toBe(true);
      });
    });

    it('clears streaming flag after complete message', async () => {
      render(<ChatPanel />);
      openPanel();
      const ws = getWs();
      act(() => ws.simulateOpen());
      act(() => ws.simulateMessage({ type: 'chunk', content: 'Done' }));
      act(() => ws.simulateMessage({ type: 'complete', session_id: 'sess-1' }));

      await waitFor(() => {
        const msgs = useChatStore.getState().messages;
        expect(msgs[0]?.streaming).toBe(false);
      });
    });
  });

  describe('sending messages', () => {
    it('sends message payload when Enter is pressed', async () => {
      render(<ChatPanel />);
      openPanel();
      const ws = getWs();
      act(() => ws.simulateOpen());

      const textarea = screen.getByLabelText('Chat input');
      fireEvent.change(textarea, { target: { value: 'Test message' } });
      fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: false });

      await waitFor(() => {
        const sent = ws.sentMessages.find((m) => {
          const parsed = JSON.parse(m) as { action: string; message: string };
          return parsed.action === 'send' && parsed.message === 'Test message';
        });
        expect(sent).toBeDefined();
      });
    });

    it('does not send on Shift+Enter', async () => {
      render(<ChatPanel />);
      openPanel();
      const ws = getWs();
      act(() => ws.simulateOpen());

      const textarea = screen.getByLabelText('Chat input');
      fireEvent.change(textarea, { target: { value: 'Line one' } });
      fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: true });

      const sendMessages = ws.sentMessages.filter((m) => {
        try {
          return (JSON.parse(m) as { action: string }).action === 'send';
        } catch {
          return false;
        }
      });
      expect(sendMessages).toHaveLength(0);
    });

    it('clears input buffer after sending', async () => {
      render(<ChatPanel />);
      openPanel();
      const ws = getWs();
      act(() => ws.simulateOpen());

      const textarea = screen.getByLabelText('Chat input');
      fireEvent.change(textarea, { target: { value: 'Hello' } });
      fireEvent.click(screen.getByLabelText('Send message'));

      await waitFor(() => {
        expect(useChatStore.getState().inputBuffer).toBe('');
      });
    });

    it('adds user message to the message list', async () => {
      render(<ChatPanel />);
      openPanel();
      const ws = getWs();
      act(() => ws.simulateOpen());

      const textarea = screen.getByLabelText('Chat input');
      fireEvent.change(textarea, { target: { value: 'My question' } });
      fireEvent.click(screen.getByLabelText('Send message'));

      await waitFor(() => {
        expect(screen.getByText('My question')).toBeInTheDocument();
      });
    });
  });

  describe('no_provider state', () => {
    it('shows configure provider notice when server reports no API key', async () => {
      render(<ChatPanel />);
      openPanel();
      const ws = getWs();
      act(() => ws.simulateOpen());
      act(() =>
        ws.simulateMessage({
          type: 'error',
          detail: 'No API key configured for provider',
        }),
      );

      await waitFor(() => {
        // The notice paragraph (not the status badge) should appear
        expect(
          screen.getByText(/configure a provider in settings/i),
        ).toBeInTheDocument();
      });
    });
  });

  describe('error handling', () => {
    it('displays error detail as assistant message for generic errors', async () => {
      render(<ChatPanel />);
      openPanel();
      const ws = getWs();
      act(() => ws.simulateOpen());
      act(() => ws.simulateMessage({ type: 'error', detail: 'Internal server error' }));

      await waitFor(() => {
        expect(screen.getByText(/error: internal server error/i)).toBeInTheDocument();
      });
    });
  });
});

// ── Store unit tests ───────────────────────────────────────────────────────

describe('useChatStore chunk coalescence', () => {
  beforeEach(() => {
    useChatStore.setState({ messages: [], open: false, sessionId: null, inputBuffer: '' });
  });

  it('appendChunk creates a new streaming message when no active stream', () => {
    act(() => useChatStore.getState().appendChunk('Hello'));
    const msgs = useChatStore.getState().messages;
    expect(msgs).toHaveLength(1);
    expect(msgs[0].role).toBe('assistant');
    expect(msgs[0].content).toBe('Hello');
    expect(msgs[0].streaming).toBe(true);
  });

  it('appendChunk concatenates to the existing streaming message', () => {
    act(() => {
      useChatStore.getState().appendChunk('Hello');
      useChatStore.getState().appendChunk(' world');
    });
    const msgs = useChatStore.getState().messages;
    expect(msgs).toHaveLength(1);
    expect(msgs[0].content).toBe('Hello world');
  });

  it('appendChunk starts a new message if last is not a streaming assistant', () => {
    act(() => useChatStore.getState().addMessage('user', 'Question'));
    act(() => useChatStore.getState().appendChunk('Answer'));
    const msgs = useChatStore.getState().messages;
    expect(msgs).toHaveLength(2);
    expect(msgs[1].role).toBe('assistant');
    expect(msgs[1].streaming).toBe(true);
  });

  it('finaliseStream marks the last message as not streaming', () => {
    act(() => useChatStore.getState().appendChunk('Done'));
    act(() => useChatStore.getState().finaliseStream());
    const msgs = useChatStore.getState().messages;
    expect(msgs[0].streaming).toBe(false);
  });

  it('toggle flips open state', () => {
    expect(useChatStore.getState().open).toBe(false);
    act(() => useChatStore.getState().toggle());
    expect(useChatStore.getState().open).toBe(true);
    act(() => useChatStore.getState().toggle());
    expect(useChatStore.getState().open).toBe(false);
  });

  it('setInput updates inputBuffer', () => {
    act(() => useChatStore.getState().setInput('typing'));
    expect(useChatStore.getState().inputBuffer).toBe('typing');
  });

  it('clearMessages empties the messages array', () => {
    act(() => useChatStore.getState().addMessage('user', 'hello'));
    act(() => useChatStore.getState().clearMessages());
    expect(useChatStore.getState().messages).toHaveLength(0);
  });

  it('setSessionId updates sessionId', () => {
    act(() => useChatStore.getState().setSessionId('abc-123'));
    expect(useChatStore.getState().sessionId).toBe('abc-123');
  });
});
