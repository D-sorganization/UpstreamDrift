/**
 * Chat Store — Global AI Chat State Management
 *
 * Manages the floating chat panel, WebSocket session, messages,
 * and input buffer for the AI assistant.
 *
 * @module stores/useChatStore
 */

import { create } from 'zustand';

// ── Types ─────────────────────────────────────────────────────────────────

export type MessageRole = 'user' | 'assistant';

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  /** True while assistant is still streaming this message */
  streaming?: boolean;
}

export interface ChatStoreState {
  /** Whether the chat panel is visible */
  open: boolean;
  /** Active WebSocket session ID (null until session_info received) */
  sessionId: string | null;
  /** Ordered list of messages in the conversation */
  messages: ChatMessage[];
  /** Current text in the input field */
  inputBuffer: string;
}

export interface ChatStoreActions {
  /** Toggle the chat panel open/closed */
  toggle: () => void;
  /** Set the input buffer value */
  setInput: (text: string) => void;
  /**
   * Append a streaming chunk to the last assistant message.
   * Creates a new streaming assistant message if the last message is not one.
   */
  appendChunk: (chunk: string) => void;
  /** Finalise the current streaming message (mark streaming: false) */
  finaliseStream: () => void;
  /** Add a complete message (user or assistant) */
  addMessage: (role: MessageRole, content: string) => void;
  /** Clear all messages */
  clearMessages: () => void;
  /** Set the session ID after receiving session_info from server */
  setSessionId: (id: string) => void;
}

export type ChatStore = ChatStoreState & ChatStoreActions;

// ── Helpers ───────────────────────────────────────────────────────────────

function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

// ── Store ─────────────────────────────────────────────────────────────────

export const useChatStore = create<ChatStore>((set) => ({
  open: false,
  sessionId: null,
  messages: [],
  inputBuffer: '',

  toggle: () => set((state) => ({ open: !state.open })),

  setInput: (text) => set({ inputBuffer: text }),

  appendChunk: (chunk) =>
    set((state) => {
      const last = state.messages[state.messages.length - 1];
      if (last?.role === 'assistant' && last.streaming) {
        return {
          messages: [
            ...state.messages.slice(0, -1),
            { ...last, content: last.content + chunk },
          ],
        };
      }
      // Start a new streaming assistant message
      return {
        messages: [
          ...state.messages,
          { id: generateId(), role: 'assistant', content: chunk, streaming: true },
        ],
      };
    }),

  finaliseStream: () =>
    set((state) => {
      const last = state.messages[state.messages.length - 1];
      if (last?.streaming) {
        return {
          messages: [
            ...state.messages.slice(0, -1),
            { ...last, streaming: false },
          ],
        };
      }
      return {};
    }),

  addMessage: (role, content) =>
    set((state) => ({
      messages: [...state.messages, { id: generateId(), role, content }],
    })),

  clearMessages: () => set({ messages: [] }),

  setSessionId: (id) => set({ sessionId: id }),
}));
