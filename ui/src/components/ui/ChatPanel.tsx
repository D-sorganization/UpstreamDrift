/**
 * ChatPanel — chat UI wired to the FastAPI chat WebSocket.
 *
 * Features (issue #5491):
 *   - Attachment composer (file picker + drag-drop + paste-image)
 *   - Markdown rendering for assistant messages (see chatMarkdown.tsx)
 *   - Retry button on assistant turns (resends previous user message)
 *   - Quick-action preset prompts above the textarea
 *   - Reconnect button when status is error / disconnected
 *
 * Backend protocol (see src/api/routes/chat_ws.py):
 *   Client -> Server:
 *     {"action": "send", "message": "...", "engine_context": "mujoco"?,
 *      "attachments": [{"name": "...", "mime": "...", "data": "<base64>"}]?}
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
 * See issues #3505, #5491.
 */

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type ClipboardEvent,
  type DragEvent,
  type FormEvent,
  type KeyboardEvent,
} from 'react';
import { Send, MessageSquare, Paperclip, RefreshCw, X, RotateCcw } from 'lucide-react';
import { ChatMarkdown } from './chatMarkdown';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type ChatRole = 'user' | 'assistant' | 'system';

export interface ChatAttachment {
  /** Stable id for React keys + removal. */
  id: string;
  name: string;
  mime: string;
  /** Size in bytes; informational. */
  size: number;
  /** base64-encoded contents (no data: prefix). */
  data: string;
}

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  attachments?: ChatAttachment[];
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

export interface QuickAction {
  /** Display label on the chip. */
  label: string;
  /** Prompt text inserted/sent. */
  prompt: string;
  /**
   * 'send' submits immediately, 'insert' appends to the current draft
   * so the user can edit before submitting. Default: 'send'.
   */
  mode?: 'send' | 'insert';
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const RECONNECT_BASE_MS = 500;
const RECONNECT_MAX_MS = 30_000;
const MAX_ATTACHMENT_BYTES = 8 * 1024 * 1024; // 8 MiB safety cap

const DEFAULT_QUICK_ACTIONS: QuickAction[] = [
  { label: 'Explain this', prompt: 'Explain this in plain language.', mode: 'send' },
  { label: 'Summarize', prompt: 'Summarize the key points.', mode: 'send' },
  { label: 'Fix bugs', prompt: 'Review the code above and fix any bugs you find:\n\n', mode: 'insert' },
  { label: 'Add tests', prompt: 'Write tests for the code above:\n\n', mode: 'insert' },
];

/**
 * Resolve the chat WebSocket URL.
 *
 * Precondition: sessionId is non-empty.
 * Postcondition: returns a ws:// or wss:// URL pointing at /ws/chat/{sessionId}.
 */
// eslint-disable-next-line react-refresh/only-export-components
export function resolveChatUrl(sessionId: string = 'new'): string {
  if (!sessionId || sessionId.trim().length === 0) {
    throw new Error('sessionId must be a non-empty string');
  }
  const base =
    (import.meta.env?.VITE_API_URL as string | undefined) ??
    'ws://localhost:8000';
  const wsBase = base
    .replace(/^http:\/\//i, 'ws://')
    .replace(/^https:\/\//i, 'wss://')
    .replace(/\/+$/, '');
  return `${wsBase}/ws/chat/${sessionId}`;
}

function makeId(): string {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

/**
 * Convert a File or Blob into a base64 string (no data: prefix).
 *
 * Precondition: file is a Blob.
 * Postcondition: returns base64 of file's bytes; rejects on read error
 * or on files exceeding MAX_ATTACHMENT_BYTES.
 */
// eslint-disable-next-line react-refresh/only-export-components
export async function fileToBase64(file: Blob): Promise<string> {
  if (!(file instanceof Blob)) {
    throw new TypeError('fileToBase64 requires a Blob');
  }
  if (file.size > MAX_ATTACHMENT_BYTES) {
    throw new Error(`Attachment exceeds ${MAX_ATTACHMENT_BYTES} bytes`);
  }
  // Prefer arrayBuffer when available (jsdom + modern browsers).
  if (typeof file.arrayBuffer === 'function') {
    const buf = await file.arrayBuffer();
    let bin = '';
    const bytes = new Uint8Array(buf);
    for (let i = 0; i < bytes.byteLength; i += 1) {
      bin += String.fromCharCode(bytes[i]);
    }
    // btoa is available in browsers and jsdom.
    return typeof btoa === 'function'
      ? btoa(bin)
      : Buffer.from(bytes).toString('base64');
  }
  // Fallback: FileReader.
  return await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = String(reader.result ?? '');
      const idx = result.indexOf(',');
      resolve(idx >= 0 ? result.slice(idx + 1) : result);
    };
    reader.onerror = () => reject(reader.error ?? new Error('FileReader error'));
    reader.readAsDataURL(file);
  });
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface ChatPanelProps {
  /** Optional engine context tag forwarded with each user message. */
  engineContext?: string;
  /** Override the resolved WebSocket URL (mostly for tests). */
  url?: string;
  /** Replace the default quick-action chips. */
  quickActions?: QuickAction[];
  /** Hide the quick-action row entirely. */
  hideQuickActions?: boolean;
}

export function ChatPanel({
  engineContext,
  url,
  quickActions = DEFAULT_QUICK_ACTIONS,
  hideQuickActions = false,
}: ChatPanelProps = {}) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [status, setStatus] = useState<ConnectionStatus>('connecting');
  const [streaming, setStreaming] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [attachments, setAttachments] = useState<ChatAttachment[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const [reconnectNonce, setReconnectNonce] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const closedByUserRef = useRef(false);
  const assistantIdRef = useRef<string | null>(null);
  const listEndRef = useRef<HTMLDivElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const lastUserMessageRef = useRef<{
    content: string;
    attachments?: ChatAttachment[];
  } | null>(null);

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
  }, [targetUrl, handleServerMessage, reconnectNonce]);

  // Auto-scroll to newest message. Guarded for jsdom.
  useEffect(() => {
    const node = listEndRef.current;
    if (node && typeof node.scrollIntoView === 'function') {
      node.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  // -------------------------------------------------------------------------
  // Attachment helpers
  // -------------------------------------------------------------------------

  const addFiles = useCallback(async (files: FileList | File[]) => {
    const list = Array.from(files);
    for (const file of list) {
      try {
        const data = await fileToBase64(file);
        setAttachments((prev) => [
          ...prev,
          {
            id: makeId(),
            name: file.name || `pasted-${Date.now()}`,
            mime: file.type || 'application/octet-stream',
            size: file.size,
            data,
          },
        ]);
      } catch (err) {
        setMessages((prev) => [
          ...prev,
          {
            id: makeId(),
            role: 'system',
            content: `Attachment failed: ${
              err instanceof Error ? err.message : String(err)
            }`,
          },
        ]);
      }
    }
  }, []);

  const removeAttachment = useCallback((id: string) => {
    setAttachments((prev) => prev.filter((a) => a.id !== id));
  }, []);

  const handleFileInputChange = (event: ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      void addFiles(event.target.files);
      // Reset input so selecting the same file again still fires onChange.
      event.target.value = '';
    }
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragOver(false);
    if (event.dataTransfer?.files && event.dataTransfer.files.length > 0) {
      void addFiles(event.dataTransfer.files);
    }
  };

  const handleDragOver = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    if (!isDragOver) setIsDragOver(true);
  };

  const handleDragLeave = () => setIsDragOver(false);

  const handlePaste = (event: ClipboardEvent<HTMLTextAreaElement>) => {
    const items = event.clipboardData?.items;
    if (!items) return;
    const files: File[] = [];
    for (let i = 0; i < items.length; i += 1) {
      const item = items[i];
      if (item.kind === 'file' && item.type.startsWith('image/')) {
        const file = item.getAsFile();
        if (file) files.push(file);
      }
    }
    if (files.length > 0) {
      event.preventDefault();
      void addFiles(files);
    }
  };

  // -------------------------------------------------------------------------
  // Send
  // -------------------------------------------------------------------------

  const sendOverWire = useCallback(
    (text: string, atts: ChatAttachment[]) => {
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
        return false;
      }
      const payload: Record<string, unknown> = { action: 'send', message: text };
      if (engineContext) payload.engine_context = engineContext;
      if (atts.length > 0) {
        payload.attachments = atts.map((a) => ({
          name: a.name,
          mime: a.mime,
          size: a.size,
          data: a.data,
        }));
      }
      socket.send(JSON.stringify(payload));
      assistantIdRef.current = null;
      setStreaming(true);
      return true;
    },
    [engineContext],
  );

  const sendMessage = useCallback(() => {
    const text = input.trim();
    if (!text && attachments.length === 0) return;
    const atts = attachments;

    setMessages((prev) => [
      ...prev,
      { id: makeId(), role: 'user', content: text, attachments: atts.length ? atts : undefined },
    ]);
    const sent = sendOverWire(text, atts);
    if (sent) {
      lastUserMessageRef.current = { content: text, attachments: atts.length ? atts : undefined };
      setInput('');
      setAttachments([]);
    }
  }, [input, attachments, sendOverWire]);

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

  // -------------------------------------------------------------------------
  // Retry / quick-actions / reconnect
  // -------------------------------------------------------------------------

  const retryLastUserMessage = useCallback(
    (assistantId: string) => {
      const last = lastUserMessageRef.current;
      if (!last) return;
      // Drop the assistant message we're retrying (and anything after).
      setMessages((prev) => {
        const idx = prev.findIndex((m) => m.id === assistantId);
        return idx >= 0 ? prev.slice(0, idx) : prev;
      });
      sendOverWire(last.content, last.attachments ?? []);
    },
    [sendOverWire],
  );

  const handleQuickAction = (action: QuickAction) => {
    const mode = action.mode ?? 'send';
    if (mode === 'insert') {
      setInput((prev) => (prev.length > 0 ? `${prev}\n${action.prompt}` : action.prompt));
      return;
    }
    // Send mode: bypass input state to send immediately.
    setMessages((prev) => [
      ...prev,
      { id: makeId(), role: 'user', content: action.prompt },
    ]);
    const sent = sendOverWire(action.prompt, []);
    if (sent) lastUserMessageRef.current = { content: action.prompt };
  };

  const handleReconnect = () => {
    // Bumping the nonce re-runs the connect effect.
    reconnectAttemptsRef.current = 0;
    setReconnectNonce((n) => n + 1);
  };

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

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

  const showReconnect = status === 'error' || status === 'disconnected';
  const canSubmit =
    status === 'connected' && (input.trim().length > 0 || attachments.length > 0);

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
        <div className="flex items-center gap-2">
          {showReconnect && (
            <button
              type="button"
              onClick={handleReconnect}
              className="sidekick-focus-ring text-xs flex items-center gap-1 px-2 py-1 rounded border"
              aria-label="Reconnect to chat"
              data-testid="chat-reconnect"
              style={{
                borderColor: 'var(--sidekick-color-border)',
                color: 'var(--sidekick-color-text)',
              }}
            >
              <RefreshCw className="w-3 h-3" aria-hidden="true" />
              Reconnect
            </button>
          )}
          <span
            className={`text-xs font-mono flex items-center gap-1 ${statusColor[status]}`}
            aria-live="polite"
            data-testid="chat-status"
          >
            <span
              aria-hidden
              style={{
                display: 'inline-block',
                width: 8,
                height: 8,
                borderRadius: '50%',
                background:
                  status === 'connected'
                    ? 'var(--sidekick-color-success, #22c55e)'
                    : 'var(--sidekick-color-warning, #f59e0b)',
              }}
            />
            {statusLabel[status]}
          </span>
        </div>
      </div>

      {/* Messages */}
      <div
        className="flex-1 overflow-y-auto p-3 space-y-2 text-sm"
        role="log"
        aria-live="polite"
        aria-relevant="additions"
        aria-atomic="false"
        aria-label="Chat messages"
        data-testid="chat-messages"
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        style={isDragOver ? { outline: '2px dashed var(--sidekick-color-accent)' } : undefined}
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
            <div className="flex flex-col items-end gap-1 max-w-[80%]">
              <div
                className="sidekick-chat-bubble rounded-lg px-3 py-2 break-words border"
                data-role={m.role}
                data-testid={`chat-bubble-${m.role}`}
              >
                {m.role === 'assistant' ? (
                  <ChatMarkdown source={m.content} data-testid="chat-markdown" />
                ) : (
                  <span className="whitespace-pre-wrap">{m.content}</span>
                )}
                {m.attachments && m.attachments.length > 0 && (
                  <ul className="mt-1 text-xs opacity-75" data-testid="chat-bubble-attachments">
                    {m.attachments.map((a) => (
                      <li key={a.id}>
                        {a.name} ({a.mime})
                      </li>
                    ))}
                  </ul>
                )}
              </div>
              {m.role === 'assistant' && !streaming && (
                <button
                  type="button"
                  onClick={() => retryLastUserMessage(m.id)}
                  aria-label="Retry"
                  data-testid="chat-retry"
                  className="sidekick-focus-ring text-[11px] flex items-center gap-1 px-2 py-0.5 rounded border opacity-70 hover:opacity-100"
                  style={{
                    borderColor: 'var(--sidekick-color-border)',
                    color: 'var(--sidekick-color-text-subtle)',
                  }}
                >
                  <RotateCcw className="w-3 h-3" aria-hidden="true" />
                  Retry
                </button>
              )}
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

      {/* Quick-action chips */}
      {!hideQuickActions && quickActions.length > 0 && (
        <div
          className="sidekick-chat-quick-actions flex flex-wrap gap-1.5 px-3 pt-2"
          data-testid="chat-quick-actions"
        >
          {quickActions.map((qa) => (
            <button
              key={qa.label}
              type="button"
              onClick={() => handleQuickAction(qa)}
              disabled={status !== 'connected'}
              className="sidekick-focus-ring text-xs px-2 py-1 rounded border disabled:opacity-40 disabled:cursor-not-allowed"
              data-testid="chat-quick-action"
              data-mode={qa.mode ?? 'send'}
              style={{
                borderColor: 'var(--sidekick-color-border)',
                color: 'var(--sidekick-color-text)',
              }}
            >
              {qa.label}
            </button>
          ))}
        </div>
      )}

      {/* Attachment chip row (only visible when items present) */}
      {attachments.length > 0 && (
        <div
          className="sidekick-chat-attachments flex flex-wrap gap-1.5 px-3 pt-2"
          data-testid="chat-attachments"
        >
          {attachments.map((a) => (
            <span
              key={a.id}
              className="text-xs flex items-center gap-1 px-2 py-1 rounded border"
              data-testid="chat-attachment-chip"
              style={{
                borderColor: 'var(--sidekick-color-border)',
                background: 'var(--sidekick-color-input)',
              }}
            >
              {a.mime.startsWith('image/') && (
                // Render a thumbnail when we have base64 image data.
                <img
                  src={`data:${a.mime};base64,${a.data}`}
                  alt={a.name}
                  width={20}
                  height={20}
                  style={{ borderRadius: 2, objectFit: 'cover' }}
                />
              )}
              <span title={`${a.name} (${a.size}B)`}>{a.name}</span>
              <button
                type="button"
                onClick={() => removeAttachment(a.id)}
                aria-label={`Remove ${a.name}`}
                data-testid="chat-attachment-remove"
                className="sidekick-focus-ring"
                style={{ background: 'transparent', border: 'none', cursor: 'pointer', padding: 0 }}
              >
                <X className="w-3 h-3" aria-hidden="true" />
              </button>
            </span>
          ))}
        </div>
      )}

      {/* Composer */}
      <form
        onSubmit={handleSubmit}
        className="sidekick-chat-composer flex items-end gap-2 p-3 border-t"
        style={{ borderTopColor: 'var(--sidekick-color-border)' }}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          onChange={handleFileInputChange}
          aria-label="Attach files"
          data-testid="chat-file-input"
          style={{ display: 'none' }}
        />
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          aria-label="Attach file"
          data-testid="chat-attach"
          disabled={status !== 'connected'}
          className="sidekick-focus-ring flex items-center justify-center w-9 h-9 rounded border disabled:opacity-40 disabled:cursor-not-allowed"
          style={{
            borderColor: 'var(--sidekick-color-border)',
            color: 'var(--sidekick-color-text)',
          }}
        >
          <Paperclip className="w-4 h-4" aria-hidden="true" />
        </button>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          onPaste={handlePaste}
          placeholder={
            status === 'connected'
              ? 'Ask a question (paste images, drop files)...'
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
          disabled={!canSubmit}
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
