## Problem

Two state bugs in `ui/src/components/ui/ChatPanel.tsx`:

1. **Retry can drop attachments.** `lastUserMessageRef` is a single slot. Regular sends store `{ content, attachments }` (~line 502), but quick-action sends overwrite it with `{ content: action.prompt }` and no attachments field (~line 550). `retryLastUserMessage` (~lines 524–536) then resends `last.attachments ?? []`. Sequence: send message with image → trigger a quick action → retry the quick-action response → the retry context silently loses the attachment history. More generally, retry always replays whatever happens to be in the single ref, which may not correspond to the assistant message being retried.
2. **Reconnect leaves the streaming indicator stuck.** `handleReconnect` (~lines 553–557) only bumps the reconnect nonce. If the socket dropped mid-stream, `streaming` is still `true` and `assistantIdRef` still points at the dead message, so the "assistant is typing…" indicator persists until a new message happens to complete.

## Fix

1. Make retry message-accurate: store the originating user payload **per assistant message** (e.g. keep `{content, attachments}` on the assistant message object or in a `Map<assistantId, payload>` ref) and have `retryLastUserMessage(assistantId)` replay exactly that payload. This fixes both the attachment loss and retry-after-quick-action mismatch.
2. In `handleReconnect`, reset stream state before bumping the nonce:
   ```tsx
   setStreaming(false);
   assistantIdRef.current = null;
   reconnectAttemptsRef.current = 0;
   setReconnectNonce((n) => n + 1);
   ```
   Also mark the interrupted assistant message as errored ("response interrupted — retry?") rather than leaving it half-rendered.
3. Tests: retry after quick action replays the correct payload incl. attachments; simulated socket close mid-stream + reconnect clears the typing indicator and marks the partial message.

## Acceptance criteria

- Both repro sequences above behave correctly; new tests pass in `ChatPanel.test.tsx`.

Part of the UI/UX overhaul epic (see tracking issue).
