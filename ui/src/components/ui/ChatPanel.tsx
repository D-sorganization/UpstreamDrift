/**
 * ChatPanel — minimum-viable chat UI wired to the FastAPI chat WebSocket.
 *
 * Backend protocol (see src/api/routes/chat_ws.py):
 *   Client -> Server:
 *     {"action": "send", "message": "...", "engine_context": "mujoco"?}
 *     {"action": "history"}
 *     {"action": "new_session"}
 *   Server -> Client:
 *     {"type": "session_info", "session_id": "..."}
 *     {"type": "chunk", "content": "..."}
 *     {"type": "complete", "session_id": "..."}
 *     {"type": "history", "messages": [...]}
 *     {"type": "error", "detail": "..."}
 *
 * Connects to {VITE_API_URL || ws://localhost:8000}/ws/chat/new with
 * exponential backoff reconnect (capped at 30 s).
 *
 * See issue #3505.
 */

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
} from 'react';
import { Send, MessageSquare } from 'lucide-react';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type ChatRole = 'user' | 'assistant' | 'system';

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
}

export type ConnectionStatus =
  | 'connecting'
  | 'connected'
  | 'disconnected'
  | 'error';

interface ServerMessage {
  type: string;
  session_id?: string;
  content?: string;
  detail?: string;
  messages?: Array<{ role: ChatRole; content: string }>;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const RECONNECT_BASE_MS = 500;
const RECONNECT_MAX_MS = 30_000;

/**
 * Resolve the chat WebSocket URL.
 *
 * Precondition: returns a non-empty ws:// or wss:// URL pointing at the
 * /ws/chat/{session_id} endpoint.
 */
// eslint-disable-next-line react-refresh/only-export-components
export function resolveChatUrl(sessionId: string = 'new'): string {
  if (!sessionId || sessionId.trim().length === 0) {
    throw new Error('sessionId must be a non-empty string');
  }
  // Vite injects import.meta.env at build time. Default to localhost:8000.
  const base =
    (import.meta.env?.VITE_API_URL as string | undefined) ??
    'ws://localhost:8000';
  // Allow callers to set http(s) URLs for the API; convert to ws(s).
  const wsBase = base
    .replace(/^http:\/\//i, 'ws://')
    .replace(/^https:\/\//i, 'wss://')
    .replace(/\/+$/, '');
  return `${wsBase}/ws/chat/${sessionId}`;
}

function makeId(): string {
  // Lightweight unique id suitable for React keys.
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface ChatPanelProps {
  /** Optional engine context tag forwarded with each user message. */
  engineContext?: string;
  /** Override the resolved WebSocket URL (mostly for tests). */
  url?: string;
}

export function ChatPanel({ engineContext, url }: ChatPanelProps = {}) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [status, setStatus] = useState<ConnectionStatus>('connecting');
  const [streaming, setStreaming] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const closedByUserRef = useRef(false);
  const assistantIdRef = useRef<string | null>(null);
  const listEndRef = useRef<HTMLDivElement | null>(null);

  const targetUrl = useMemo(() => url ?? resolveChatUrl('new'), [url]);

  // Append text to the in-flight assistant message (or start a new one).
  const appendAssistantChunk = useCallback((chunk: string) => {
    setMessages((prev) => {
      const currentId = assistantIdRef.current;
      if (currentId) {
        return prev.map((m) =>
          m.id === currentId ? { ...m, content: m.content + chunk } : m,
        );
      }
      const id = makeId();
      assistantIdRef.current = id;
      return [...prev, { id, role: 'assistant', content: chunk }];
    });
  }, []);

  const handleServerMessage = useCallback(
    (raw: string) => {
      let payload: ServerMessage;
      try {
        payload = JSON.parse(raw) as ServerMessage;
      } catch {
        return;
      }

      switch (payload.type) {
        case 'session_info':
        case 'session_created':
          if (payload.session_id) {
            setSessionId(payload.session_id);
          }
          break;
        case 'chunk':
          if (typeof payload.content === 'string') {
            setStreaming(true);
            appendAssistantChunk(payload.content);
          }
          break;
        case 'complete':
          assistantIdRef.current = null;
          setStreaming(false);
          break;
        case 'history':
          if (Array.isArray(payload.messages)) {
            setMessages(
              payload.messages.map((m) => ({
                id: makeId(),
                role: m.role,
                content: m.content,
              })),
            );
          }
          break;
        case 'error':
          setMessages((prev) => [
            ...prev,
            {
              id: makeId(),
              role: 'system',
              content: payload.detail ?? 'Unknown error',
            },
          ]);
          setStreaming(false);
          assistantIdRef.current = null;
          break;
        default:
          break;
      }
    },
    [appendAssistantChunk],
  );

  // Connect (and re-connect with exponential backoff).
  useEffect(() => {
    closedByUserRef.current = false;

    const connect = () => {
      setStatus('connecting');
      let socket: WebSocket;
      try {
        socket = new WebSocket(targetUrl);
      } catch {
        setStatus('error');
        scheduleReconnect();
        return;
      }
      wsRef.current = socket;

      socket.onopen = () => {
        reconnectAttemptsRef.current = 0;
        setStatus('connected');
      };
      socket.onmessage = (event: MessageEvent) => {
        handleServerMessage(String(event.data));
      };
      socket.onerror = () => {
        setStatus('error');
      };
      socket.onclose = () => {
        setStatus('disconnected');
        if (!closedByUserRef.current) {
          scheduleReconnect();
        }
      };
    };

    const scheduleReconnect = () => {
      const attempt = reconnectAttemptsRef.current + 1;
      reconnectAttemptsRef.current = attempt;
      const delay = Math.min(
        RECONNECT_BASE_MS * 2 ** (attempt - 1),
        RECONNECT_MAX_MS,
      );
      reconnectTimerRef.current = setTimeout(connect, delay);
    };

    connect();

    return () => {
      closedByUserRef.current = true;
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      if (wsRef.current) {
        try {
          wsRef.current.close();
        } catch {
          // ignore
        }
        wsRef.current = null;
      }
    };
  }, [targetUrl, handleServerMessage]);

  // Auto-scroll to newest message. Guarded for jsdom where scrollIntoView
  // is not implemented.
  useEffect(() => {
    const node = listEndRef.current;
    if (node && typeof node.scrollIntoView === 'function') {
      node.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  const sendMessage = useCallback(() => {
    const text = input.trim();
    if (!text) return;
    const socket = wsRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      setMessages((prev) => [
        ...prev,
        {
          id: makeId(),
          role: 'system',
          content: 'Not connected. Message not sent.',
        },
      ]);
      return;
    }

    setMessages((prev) => [
      ...prev,
      { id: makeId(), role: 'user', content: text },
    ]);
    const payload: Record<string, string> = { action: 'send', message: text };
    if (engineContext) {
      payload.engine_context = engineContext;
    }
    socket.send(JSON.stringify(payload));
    assistantIdRef.current = null;
    setStreaming(true);
    setInput('');
  }, [input, engineContext]);

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    sendMessage();
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  };

  const statusColor: Record<ConnectionStatus, string> = {
    connecting: 'text-yellow-400',
    connected: 'text-green-400',
    disconnected: 'text-gray-400',
    error: 'text-red-400',
  };
  const statusLabel: Record<ConnectionStatus, string> = {
    connecting: 'Connecting...',
    connected: 'Connected',
    disconnected: 'Disconnected',
    error: 'Error',
  };

  return (
    <div
      className="sidekick-chat-surface flex flex-col h-full w-full max-w-3xl rounded-lg border shadow-xl"
      style={{
        backgroundColor: 'var(--sidekick-color-surface)',
        borderColor: 'var(--sidekick-color-border)',
        color: 'var(--sidekick-color-text)',
      }}
      data-testid="chat-panel"
    >
      {/* Header */}
      <div
        className="sidekick-chat-header flex items-center justify-between p-3 border-b"
        style={{ borderBottomColor: 'var(--sidekick-color-border)' }}
      >
        <div className="flex items-center gap-2">
          <MessageSquare
            className="w-4 h-4"
            aria-hidden="true"
            style={{ color: 'var(--sidekick-color-accent)' }}
          />
          <span
            className="font-semibold text-sm"
            aria-label="Sidekick assistant"
            title="Sidekick assistant"
          >
            Chat
          </span>
          {sessionId && (
            <span
              className="text-[10px] font-mono"
              style={{ color: 'var(--sidekick-color-text-subtle)' }}
              title={`Session ${sessionId}`}
            >
              {sessionId.slice(0, 8)}
            </span>
          )}
        </div>
        <span
          className={`text-xs font-mono ${statusColor[status]}`}
          aria-live="polite"
          data-testid="chat-status"
        >
          {status === 'connected' ? 'o ' : '. '}
          {statusLabel[status]}
        </span>
      </div>

      {/* Messages */}
      <div
        className="flex-1 overflow-y-auto p-3 space-y-2 text-sm"
        role="log"
        aria-live="polite"
        aria-label="Chat messages"
        data-testid="chat-messages"
      >
        {messages.length === 0 && (
          <div
            className="text-center pt-6"
            style={{ color: 'var(--sidekick-color-text-subtle)' }}
          >
            Start a conversation. Press Enter to send, Shift+Enter for newline.
          </div>
        )}
        {messages.map((m) => (
          <div
            key={m.id}
            className={`flex ${
              m.role === 'user' ? 'justify-end' : 'justify-start'
            }`}
            data-role={m.role}
          >
            <div
              className="sidekick-chat-bubble max-w-[80%] rounded-lg px-3 py-2 whitespace-pre-wrap break-words border"
              style={{
                backgroundColor:
                  m.role === 'user'
                    ? 'var(--sidekick-color-accent)'
                    : m.role === 'assistant'
                      ? 'var(--sidekick-color-surface-raised)'
                      : 'var(--sidekick-color-warning)',
                borderColor:
                  m.role === 'system'
                    ? 'var(--sidekick-color-warning)'
                    : 'var(--sidekick-color-border)',
                color: 'var(--sidekick-color-text)',
              }}
              data-role={m.role}
            >
              {m.content}
            </div>
          </div>
        ))}
        {streaming && (
          <div
            className="text-xs italic"
            style={{ color: 'var(--sidekick-color-text-subtle)' }}
            data-testid="chat-streaming"
          >
            assistant is typing...
          </div>
        )}
        <div ref={listEndRef} />
      </div>

      {/* Composer */}
      <form
        onSubmit={handleSubmit}
        className="sidekick-chat-composer flex items-end gap-2 p-3 border-t"
        style={{ borderTopColor: 'var(--sidekick-color-border)' }}
      >
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            status === 'connected'
              ? 'Ask a question...'
              : 'Waiting for connection...'
          }
          rows={2}
          aria-label="Message input"
          data-testid="chat-input"
          className="sidekick-focus-ring flex-1 resize-none rounded border px-2 py-1.5 text-sm focus:outline-none disabled:opacity-50"
          style={{
            backgroundColor: 'var(--sidekick-color-input)',
            borderColor: 'var(--sidekick-color-border)',
            color: 'var(--sidekick-color-text)',
          }}
          disabled={status !== 'connected'}
        />
        <button
          type="submit"
          aria-label="Send message"
          data-testid="chat-send"
          disabled={status !== 'connected' || input.trim().length === 0}
          className="sidekick-focus-ring flex items-center gap-1 px-3 py-1.5 rounded border disabled:opacity-40 disabled:cursor-not-allowed transition-colors text-sm"
          style={{
            backgroundColor: 'var(--sidekick-color-accent)',
            borderColor: 'var(--sidekick-color-accent)',
            color: 'var(--sidekick-color-selection-text)',
          }}
        >
          <Send className="w-4 h-4" aria-hidden="true" />
          Send
        </button>
      </form>
    </div>
  );
}

export default ChatPanel;
