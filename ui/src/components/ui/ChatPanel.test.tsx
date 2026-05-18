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
import { ChatPanel, resolveChatUrl, fileToBase64 } from './ChatPanel';
import { ChatMarkdown } from './chatMarkdown';

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

// ---------------------------------------------------------------------------
// Issue #5491 — attachments / markdown / paste-image / retry / quick-actions
// ---------------------------------------------------------------------------

describe('fileToBase64', () => {
  it('encodes blob bytes to base64', async () => {
    const blob = new Blob(['hi'], { type: 'text/plain' });
    const b64 = await fileToBase64(blob);
    expect(b64).toBe(typeof btoa === 'function' ? btoa('hi') : Buffer.from('hi').toString('base64'));
  });

  it('rejects non-Blob inputs', async () => {
    // @ts-expect-error — intentional misuse
    await expect(fileToBase64('not a blob')).rejects.toThrow(TypeError);
  });
});

describe('ChatMarkdown', () => {
  it('renders fenced code blocks with data-lang', () => {
    render(<ChatMarkdown source={'before\n```python\nx = 1\n```\nafter'} />);
    const codeBlocks = document.querySelectorAll('pre.sidekick-chat-codeblock');
    expect(codeBlocks.length).toBe(1);
    expect(codeBlocks[0].getAttribute('data-lang')).toBe('python');
    expect(codeBlocks[0].textContent).toContain('x = 1');
  });

  it('renders inline code, bold, italic, and links', () => {
    render(
      <ChatMarkdown source={'this is `code`, **bold**, *italic*, and a [link](https://example.com)'} />,
    );
    expect(document.querySelector('code.sidekick-chat-inline-code')?.textContent).toBe('code');
    expect(document.querySelector('strong')?.textContent).toBe('bold');
    expect(document.querySelector('em')?.textContent).toBe('italic');
    const link = document.querySelector('a.sidekick-chat-link') as HTMLAnchorElement;
    expect(link).toBeTruthy();
    expect(link.href).toBe('https://example.com/');
    expect(link.target).toBe('_blank');
    expect(link.rel).toContain('noopener');
  });

  it('never injects raw HTML', () => {
    render(<ChatMarkdown source={'<script>alert(1)</script>'} />);
    expect(document.querySelector('script')).toBeNull();
  });
});

describe('ChatPanel — attachments', () => {
  it('renders an attach button and a hidden file input', async () => {
    render(<ChatPanel />);
    await waitForSocket();
    expect(screen.getByTestId('chat-attach')).toBeInTheDocument();
    const input = screen.getByTestId('chat-file-input') as HTMLInputElement;
    expect(input).toBeInTheDocument();
    expect(input.type).toBe('file');
    expect(input.multiple).toBe(true);
  });

  it('adds a chip when a file is selected and removes it on X click', async () => {
    render(<ChatPanel />);
    await waitForSocket();
    const fileInput = screen.getByTestId('chat-file-input') as HTMLInputElement;
    const file = new File(['hello'], 'hi.txt', { type: 'text/plain' });
    await act(async () => {
      Object.defineProperty(fileInput, 'files', {
        value: [file],
        configurable: true,
      });
      fireEvent.change(fileInput);
    });
    await waitFor(() => {
      expect(screen.getByTestId('chat-attachments')).toBeInTheDocument();
    });
    const chips = screen.getAllByTestId('chat-attachment-chip');
    expect(chips.length).toBe(1);
    expect(chips[0].textContent).toContain('hi.txt');

    fireEvent.click(screen.getByTestId('chat-attachment-remove'));
    await waitFor(() => {
      expect(screen.queryByTestId('chat-attachments')).not.toBeInTheDocument();
    });
  });

  it('sends attachments in the outgoing payload', async () => {
    render(<ChatPanel />);
    const socket = await waitForSocket();
    const fileInput = screen.getByTestId('chat-file-input') as HTMLInputElement;
    const file = new File(['hello'], 'note.txt', { type: 'text/plain' });
    await act(async () => {
      Object.defineProperty(fileInput, 'files', { value: [file], configurable: true });
      fireEvent.change(fileInput);
    });
    await waitFor(() => expect(screen.queryByTestId('chat-attachments')).toBeInTheDocument());

    fireEvent.change(screen.getByTestId('chat-input'), { target: { value: 'see file' } });
    fireEvent.click(screen.getByTestId('chat-send'));

    await waitFor(() => expect(socket.sent.length).toBe(1));
    const payload = JSON.parse(socket.sent[0].data) as Record<string, unknown>;
    expect(payload.message).toBe('see file');
    const atts = payload.attachments as Array<{ name: string; mime: string }>;
    expect(atts).toHaveLength(1);
    expect(atts[0].name).toBe('note.txt');
    expect(atts[0].mime).toBe('text/plain');
  });
});

describe('ChatPanel — paste image', () => {
  it('captures image clipboard items into attachments', async () => {
    render(<ChatPanel />);
    await waitForSocket();
    const ta = screen.getByTestId('chat-input');

    const blob = new Blob([new Uint8Array([1, 2, 3, 4])], { type: 'image/png' });
    const file = new File([blob], 'pasted.png', { type: 'image/png' });

    await act(async () => {
      fireEvent.paste(ta, {
        clipboardData: {
          items: [
            { kind: 'file', type: 'image/png', getAsFile: () => file },
          ],
        },
      });
    });

    await waitFor(() => {
      expect(screen.getByTestId('chat-attachments')).toBeInTheDocument();
    });
    expect(screen.getByText('pasted.png')).toBeInTheDocument();
  });
});

describe('ChatPanel — markdown rendering', () => {
  it('renders assistant chunks as markdown (assistant only)', async () => {
    render(<ChatPanel />);
    const socket = await waitForSocket();

    act(() => {
      socket.emit({ type: 'chunk', content: 'Here is `code` and **bold**.' });
      socket.emit({ type: 'complete', session_id: 'x' });
    });

    await waitFor(() => {
      expect(document.querySelector('strong')?.textContent).toBe('bold');
    });
    expect(document.querySelector('code.sidekick-chat-inline-code')?.textContent).toBe('code');
  });

  it('user messages stay as plain text (no markdown)', async () => {
    render(<ChatPanel />);
    await waitForSocket();
    fireEvent.change(screen.getByTestId('chat-input'), {
      target: { value: '**not bold**' },
    });
    fireEvent.click(screen.getByTestId('chat-send'));
    await waitFor(() => {
      expect(screen.getByText('**not bold**')).toBeInTheDocument();
    });
    // No <strong> in the user bubble.
    const userBubble = screen.getByTestId('chat-bubble-user');
    expect(userBubble.querySelector('strong')).toBeNull();
  });
});

describe('ChatPanel — retry button', () => {
  it('resends the previous user message and drops the assistant reply', async () => {
    render(<ChatPanel />);
    const socket = await waitForSocket();

    fireEvent.change(screen.getByTestId('chat-input'), { target: { value: 'first' } });
    fireEvent.click(screen.getByTestId('chat-send'));
    await waitFor(() => expect(socket.sent.length).toBe(1));

    act(() => {
      socket.emit({ type: 'chunk', content: 'reply v1' });
      socket.emit({ type: 'complete', session_id: 's' });
    });
    await waitFor(() => expect(screen.getByTestId('chat-retry')).toBeInTheDocument());

    fireEvent.click(screen.getByTestId('chat-retry'));

    await waitFor(() => expect(socket.sent.length).toBe(2));
    const payload = JSON.parse(socket.sent[1].data) as Record<string, unknown>;
    expect(payload.message).toBe('first');
    // The old assistant message should be gone (we sliced it off).
    expect(screen.queryByText('reply v1')).not.toBeInTheDocument();
  });
});

describe('ChatPanel — quick actions', () => {
  it('renders default quick-action chips', async () => {
    render(<ChatPanel />);
    await waitForSocket();
    expect(screen.getByTestId('chat-quick-actions')).toBeInTheDocument();
    expect(screen.getByText('Explain this')).toBeInTheDocument();
    expect(screen.getByText('Summarize')).toBeInTheDocument();
  });

  it('send-mode chips dispatch a message immediately', async () => {
    render(<ChatPanel />);
    const socket = await waitForSocket();

    fireEvent.click(screen.getByText('Summarize'));

    await waitFor(() => expect(socket.sent.length).toBe(1));
    const payload = JSON.parse(socket.sent[0].data) as Record<string, unknown>;
    expect(String(payload.message)).toMatch(/summari[sz]e/i);
  });

  it('insert-mode chips append to the textarea instead of sending', async () => {
    render(<ChatPanel />);
    const socket = await waitForSocket();

    fireEvent.click(screen.getByText('Add tests'));

    expect(socket.sent.length).toBe(0);
    const ta = screen.getByTestId('chat-input') as HTMLTextAreaElement;
    expect(ta.value).toMatch(/Write tests/i);
  });

  it('honors a custom quickActions prop and hideQuickActions', async () => {
    const { rerender } = render(
      <ChatPanel quickActions={[{ label: 'Custom', prompt: 'do it', mode: 'send' }]} />,
    );
    await waitForSocket();
    expect(screen.getByText('Custom')).toBeInTheDocument();

    rerender(<ChatPanel hideQuickActions />);
    expect(screen.queryByTestId('chat-quick-actions')).not.toBeInTheDocument();
  });
});

describe('ChatPanel — reconnect button', () => {
  it('shows the reconnect button when the socket closes', async () => {
    render(<ChatPanel />);
    const socket = await waitForSocket();

    act(() => {
      socket.close();
    });

    await waitFor(() => {
      expect(screen.getByTestId('chat-reconnect')).toBeInTheDocument();
    });
  });

  it('opens a new socket when reconnect is clicked', async () => {
    render(<ChatPanel />);
    const first = await waitForSocket();
    act(() => first.close());
    await waitFor(() => expect(screen.getByTestId('chat-reconnect')).toBeInTheDocument());

    const before = TestWebSocket.instances.length;
    fireEvent.click(screen.getByTestId('chat-reconnect'));

    await waitFor(() => {
      expect(TestWebSocket.instances.length).toBeGreaterThan(before);
    });
  });
});