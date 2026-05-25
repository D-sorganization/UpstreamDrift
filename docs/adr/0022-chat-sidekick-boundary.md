# ADR-0022: Chat Sidekick Boundary

Status: Accepted

## Context

Issue #6098 requires consolidating chat boundaries. See also #5922, #5967, and #5969.

## Decision

Sidekick is the canonical chat surface (`ChatPanel`, `UnifiedToolsSidebar`); the legacy chat dock (`_chat_dock_widget_qt.py`, `AIAssistantPanel`) remains a documented compatibility shell.
