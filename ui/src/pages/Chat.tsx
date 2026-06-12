/**
 * Chat Page — hosts the ChatPanel component in a centered container.
 *
 * Wires the chat WebSocket UI from src/api/routes/chat_ws.py into a
 * routable page (#3505).
 */

import { ChatPanel } from '@/components/ui/ChatPanel';

export function ChatPage() {
  return (
    // #7419: single wrapper — ChatPanel owns its own `max-w-3xl`, so the
    // previous nested flex wrappers were redundant.
    <div className="sidekick-shell w-full h-screen flex justify-center p-2 sm:p-4">
      <ChatPanel />
    </div>
  );
}

export default ChatPage;
