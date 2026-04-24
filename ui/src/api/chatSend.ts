/**
 * Chat send helper — enriches outgoing chat payloads with the user's
 * current expertise level from the UI store (#3165).
 *
 * The actual WebSocket client lives elsewhere (a parallel agent owns
 * `chat_service.py` and related adapters). This helper exists so any
 * future chat client in the UI has a single place to obtain the
 * canonical send payload.
 */

import { useUIStore } from '@/stores/useUIStore';

export interface ChatSendPayload {
  action: 'send';
  message: string;
  engine_context?: string;
  expertise_level: string;
}

export function buildChatSendPayload(
  message: string,
  engineContext?: string,
): ChatSendPayload {
  return {
    action: 'send',
    message,
    ...(engineContext ? { engine_context: engineContext } : {}),
    expertise_level: useUIStore.getState().expertiseLevel,
  };
}
