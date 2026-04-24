/**
 * ChatPanel — Floating AI assistant chat panel.
 *
 * Features:
 * - Floating panel toggled by Ctrl+/ keyboard shortcut or FAB button
 * - Streams assistant responses chunk by chunk
 * - Enter to send, Shift+Enter for newline
 * - Escape to close
 * - Connection status badge
 * - Accessible: role="log", aria-live="polite"
 *
 * See issue #3161
 */

import { useEffect, useRef, useCallback, KeyboardEvent } from 'react';
import { MessageSquare, X, Send, WifiOff, Wifi, Loader2, AlertCircle } from 'lucide-react';
import { useChatStore } from '../../stores/useChatStore';
import { useChat, type ChatConnectionStatus } from '../../api/useChat';

// ── Sub-components ────────────────────────────────────────────────────────

interface StatusBadgeProps {
  status: ChatConnectionStatus;
}

function StatusBadge({ status }: StatusBadgeProps) {
  const config: Record<
    ChatConnectionStatus,
    { label: string; className: string; icon: React.ReactNode }
  > = {
    connected: {
      label: 'Connected',
      className: 'bg-green-900/60 text-green-300 border-green-700',
      icon: <Wifi className="w-3 h-3" aria-hidden="true" />,
    },
    connecting: {
      label: 'Connecting…',
      className: 'bg-yellow-900/60 text-yellow-300 border-yellow-700',
      icon: <Loader2 className="w-3 h-3 animate-spin" aria-hidden="true" />,
    },
    reconnecting: {
      label: 'Reconnecting…',
      className: 'bg-yellow-900/60 text-yellow-300 border-yellow-700',
      icon: <Loader2 className="w-3 h-3 animate-spin" aria-hidden="true" />,
    },
    disconnected: {
      label: 'Disconnected',
      className: 'bg-gray-800 text-gray-400 border-gray-600',
      icon: <WifiOff className="w-3 h-3" aria-hidden="true" />,
    },
    failed: {
      label: 'Failed',
      className: 'bg-red-900/60 text-red-300 border-red-700',
      icon: <AlertCircle className="w-3 h-3" aria-hidden="true" />,
    },
    no_provider: {
      label: 'Configure a provider',
      className: 'bg-orange-900/60 text-orange-300 border-orange-700',
      icon: <AlertCircle className="w-3 h-3" aria-hidden="true" />,
    },
  };

  const { label, className, icon } = config[status];

  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full border text-xs font-medium ${className}`}
      aria-label={`Connection status: ${label}`}
    >
      {icon}
      {label}
    </span>
  );
}

// ── Main component ─────────────────────────────────────────────────────────

export function ChatPanel() {
  const open = useChatStore((s) => s.open);
  const toggle = useChatStore((s) => s.toggle);
  const messages = useChatStore((s) => s.messages);
  const inputBuffer = useChatStore((s) => s.inputBuffer);
  const setInput = useChatStore((s) => s.setInput);

  const { connectionStatus, sendMessage } = useChat();

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Focus textarea when panel opens
  useEffect(() => {
    if (open) {
      const timer = setTimeout(() => textareaRef.current?.focus(), 100);
      return () => clearTimeout(timer);
    }
  }, [open]);

  // Ctrl+/ to toggle; Escape to close
  useEffect(() => {
    const handleKeyDown = (e: globalThis.KeyboardEvent) => {
      if (e.ctrlKey && e.key === '/') {
        e.preventDefault();
        toggle();
      }
      if (e.key === 'Escape' && open) {
        toggle();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [open, toggle]);

  const handleSend = useCallback(() => {
    const trimmed = inputBuffer.trim();
    if (!trimmed) return;
    const sent = sendMessage(trimmed);
    if (sent) setInput('');
  }, [inputBuffer, sendMessage, setInput]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  return (
    <>
      {/* FAB button */}
      <button
        onClick={toggle}
        className={`
          fixed bottom-6 right-6 z-40 w-14 h-14 rounded-full shadow-lg
          flex items-center justify-center
          transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500
          ${open
            ? 'bg-blue-600 hover:bg-blue-700 text-white'
            : 'bg-gray-800 hover:bg-gray-700 text-blue-400 border border-gray-700'
          }
        `}
        aria-label={open ? 'Close AI chat' : 'Open AI chat (Ctrl+/)'}
        title="AI Chat (Ctrl+/)"
      >
        <MessageSquare className="w-6 h-6" aria-hidden="true" />
      </button>

      {/* Panel */}
      {open && (
        <div
          className="fixed bottom-24 right-6 z-50 w-96 max-w-[calc(100vw-3rem)]
                     flex flex-col bg-gray-900 border border-gray-700 rounded-xl
                     shadow-2xl overflow-hidden"
          style={{ height: '520px' }}
          role="dialog"
          aria-modal="false"
          aria-label="AI assistant chat"
        >
          {/* Header */}
          <div className="flex items-center gap-2 px-4 py-3 bg-gray-800/80 border-b border-gray-700 flex-shrink-0">
            <MessageSquare className="w-4 h-4 text-blue-400" aria-hidden="true" />
            <span className="text-sm font-semibold text-gray-100 flex-1">AI Assistant</span>
            <StatusBadge status={connectionStatus} />
            <button
              onClick={toggle}
              className="p-1 rounded text-gray-400 hover:text-white hover:bg-gray-700 transition-colors"
              aria-label="Close chat panel"
            >
              <X className="w-4 h-4" aria-hidden="true" />
            </button>
          </div>

          {/* No-provider notice */}
          {connectionStatus === 'no_provider' && (
            <div className="px-4 py-3 bg-orange-900/30 border-b border-orange-800 text-sm text-orange-300 flex-shrink-0">
              Configure a provider in settings to enable AI chat.
            </div>
          )}

          {/* Message list */}
          <div
            role="log"
            aria-live="polite"
            aria-label="Chat messages"
            className="flex-1 overflow-y-auto px-4 py-3 space-y-3"
          >
            {messages.length === 0 && (
              <p className="text-sm text-gray-500 text-center mt-8">
                Ask anything about your simulation…
              </p>
            )}

            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`
                    max-w-[85%] rounded-lg px-3 py-2 text-sm break-words
                    ${msg.role === 'user'
                      ? 'bg-blue-600 text-white rounded-br-none'
                      : 'bg-gray-800 text-gray-200 border border-gray-700 rounded-bl-none'
                    }
                  `}
                >
                  {msg.content}
                  {msg.streaming && (
                    <span
                      className="inline-block w-1.5 h-3.5 bg-current ml-0.5 animate-pulse align-text-bottom"
                      aria-hidden="true"
                    />
                  )}
                </div>
              </div>
            ))}

            <div ref={messagesEndRef} />
          </div>

          {/* Input area */}
          <div className="flex-shrink-0 px-3 py-3 border-t border-gray-700 bg-gray-800/50">
            <div className="flex items-end gap-2">
              <textarea
                ref={textareaRef}
                value={inputBuffer}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask a question… (Enter to send, Shift+Enter for newline)"
                rows={2}
                className="flex-1 resize-none rounded-lg px-3 py-2 bg-gray-800 border border-gray-600
                           text-sm text-gray-200 placeholder-gray-500
                           focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
                aria-label="Chat input"
                disabled={connectionStatus === 'no_provider' || connectionStatus === 'failed'}
              />
              <button
                onClick={handleSend}
                disabled={
                  !inputBuffer.trim() ||
                  connectionStatus === 'no_provider' ||
                  connectionStatus === 'failed' ||
                  connectionStatus === 'disconnected'
                }
                className="p-2 rounded-lg bg-blue-600 text-white hover:bg-blue-700
                           disabled:opacity-40 disabled:cursor-not-allowed
                           transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500"
                aria-label="Send message"
              >
                <Send className="w-4 h-4" aria-hidden="true" />
              </button>
            </div>
            <p className="mt-1 text-xs text-gray-600">
              Press <kbd className="font-mono">Ctrl+/</kbd> to toggle panel
            </p>
          </div>
        </div>
      )}
    </>
  );
}
