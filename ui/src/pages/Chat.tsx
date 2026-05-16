/**
 * Chat Page — hosts the ChatPanel component in a centered container.
 *
 * Wires the chat WebSocket UI from src/api/routes/chat_ws.py into a
 * routable page (#3505).
 */

import { ChatPanel } from '@/components/ui/ChatPanel';

export function ChatPage() {
  return (
    <div className="sidekick-shell flex justify-center items-stretch w-full h-screen p-4">
      <div className="flex w-full max-w-3xl h-full">
        <ChatPanel />
      </div>
    </div>
  );
}

export default ChatPage;
