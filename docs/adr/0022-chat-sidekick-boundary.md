# ADR-0022: Chat / Sidekick Architecture Boundary

- Status: Accepted
- Date: 2026-05-25
- Decision Makers: UpstreamDrift core maintainers
- Related Issues/PRs: #6098, #6081 (parent), #5969 (Standalone Sidekick epic), #5967 (Agentic action layer epic), #5922 (file-size baseline)

## Context

UpstreamDrift has two parallel AI assist surfaces:
1. **In-launcher chat dock** (`src/shared/python/chat/_chat_dock_widget_qt.py`) - an embedded PyQt6 widget in the launcher. It is grandfathered in our size limits but its budget expires 2026-07-31.
2. **Standalone Sidekick** (`src/shared/python/sidekick/launcher_factory.py`, `tools_sidebar/`, etc.) - a self-contained application, driven by epic #5969 and ADR-0018.

Both surfaces share a lot of functionality (AI infrastructure, theme, tools sidebar) but live in separate file trees, risking divergence and bugs not propagating. We need to formalize the boundary and the long-term plan.

## Decision

### 1. What is canonical?
The **Standalone Sidekick** architecture is the canonical path forward. The in-launcher chat dock (`_chat_dock_widget_qt.py`) is a transitional surface. We will not maintain two separate heavy implementations. The standalone sidekick's components will eventually replace the embedded legacy implementation.

### 2. What is shared?
Both surfaces currently share (or should share):
- **Theme subsystem**: Fleet-wide color theme system in `src/shared/python/sidekick/ui/theme`.
- **AI service layer**: Core LLM providers, models, and API interfaces (currently in `src/shared/python/ai/`).
- **Tools Sidebar**: `src/shared/python/sidekick/ui/tools_sidebar/` (the subtab architecture and common tool interactions).
- **Message models**: Pydantic schemas and serialization definitions.

### 3. What is duplicated and shouldn't be?
- **Chat UI rendering**: The chat message bubbles, input field, and thread management logic currently live in `_chat_dock_widget_qt.py` and are being re-implemented or duplicated for the standalone sidekick. This logic must be abstracted into a reusable UI component.
- **Session state management**: Storing chat history, handling websocket connections, and managing memory/contexts are duplicated or overlapping between the chat dock and standalone sidekick models.

### 4. Migration plan for `_chat_dock_widget_qt.py`
Given the 2026-07-31 file-size exception expiry, the plan is to **decompose in-place and port to use shared standalone Sidekick widgets**.

We will decompose `_chat_dock_widget_qt.py` into smaller UI modules (e.g., `ChatMessageListWidget`, `ChatInputWidget`, `SessionManager`) and migrate those components into the `sidekick` module tree so both the standalone and launcher environments can consume them. A follow-up epic (#TBD) will track the granular decomposition of this 1984-line file.

## Consequences

- We formally recognize the standalone sidekick as the canonical architecture.
- We stop adding major new UI features to `_chat_dock_widget_qt.py` directly.
- The `_chat_dock_widget_qt.py` budget exception will be extended with the explicit decomposition plan until the migration is complete.
