# Chat UI Recovery Plan

**Parent Issue:** [#5315](https://github.com/D-sorganization/UpstreamDrift/issues/5315)
**Parent EPIC:** [#5309](https://github.com/D-sorganization/UpstreamDrift/issues/5309)
**Date Created:** 2026-05-12
**Status:** Planning

## Problem Statement

Modernized chat UI and session-history work is spread across broad repair PRs and local product changes. Some diffs include unrelated or generated churn, making review risky and slowing recovery of real user-facing chat functionality.

## Required Outcome

Recover the modern chat UI/session-history improvements as narrow PRs that land in the correct repo and are consumed by product apps.

## Scope

1. Review open chat repair PRs in Tools and UpstreamDrift
2. Split broad PRs into focused slices:
   - Shared UI components
   - Session/history storage
   - Settings management
   - Theme inheritance
   - App adapters
   - Launcher integration
3. Remove unrelated generated/binary/file-size churn
4. Ensure Tools owns reusable widgets/managers
5. UpstreamDrift and Gasification_Model own adapters only
6. Add tests for session create/load/delete, sidebar behavior, settings persistence, and theme switching
7. Validate product app launch, not just isolated unit tests

## Recovery Slices

### Slice 1: Shared UI Components (Tools Repo)

**Target Location:** `src/chat/` in Tools
**Components:**
- `ChatDockWidget` - Main chat container
- `ChatMessageBubble` - Message display component
- `ChatSidebar` - Session list sidebar
- `ChatInputArea` - Message input with controls

**Tests Required:**
- Widget instantiation tests
- Message rendering tests
- Sidebar selection behavior

### Slice 2: Session/History Storage (Tools Repo)

**Target Location:** `src/chat/session_manager.py` in Tools
**Components:**
- `SessionManager` - Create/load/delete sessions
- `SessionStore` - SQLite-backed persistence
- `ChatHistory` - Message history management

**Tests Required:**
- Session create/load/delete lifecycle
- Persistence verification
- Migration tests for schema changes

### Slice 3: Settings Management (Tools Repo)

**Target Location:** `src/chat/settings.py` in Tools
**Components:**
- `ChatSettings` - Settings schema
- `SettingsPersistence` - Settings storage
- `SettingsMigration` - Version migrations

**Tests Required:**
- Settings validation
- Persistence round-trip
- Migration path tests

### Slice 4: Theme Inheritance (Tools Repo)

**Target Location:** `src/chat/themes/` in Tools
**Components:**
- Theme loader
- Theme inheritance chain
- Theme application to chat widgets

**Tests Required:**
- Theme loading tests
- Inheritance verification
- Visual regression tests (if applicable)

### Slice 5: App Adapters (UpstreamDrift/Gasification_Model)

**Target Location:** `src/adapters/chat_adapter.py`
**Components:**
- `ChatAdapter` - Bridge between Tools and app
- Launcher integration hooks
- App-specific customizations

**Tests Required:**
- Adapter instantiation
- Launcher integration
- App-specific behavior

### Slice 6: Launcher Integration (UpstreamDrift)

**Target Location:** `src/launchers/chat_launcher.py`
**Components:**
- Chat menu entry
- Chat window management
- Keyboard shortcuts

**Tests Required:**
- Menu presence verification
- Window open/close behavior
- Shortcut activation

## TDD / DbC / LOD / DRY Principles

- **TDD:** Tests cover UI state and persistence before merge
- **DbC:** Settings/session schemas validate and migrate safely
- **LOD:** Product apps call shared UI APIs, not internals
- **DRY:** No duplicate sidebar/session managers across repos

## Acceptance Criteria

- [ ] Recovery lands as small focused PRs (max 300 lines each)
- [ ] Superseded broad PRs are closed or reduced after replacement PRs exist
- [ ] UpstreamDrift and Gasification_Model launch the recovered UI through product entry points
- [ ] All slices have test coverage > 80%
- [ ] Session persistence survives app restart
- [ ] Theme inheritance applies correctly
- [ ] Settings persist across sessions

## PR Tracking

| Slice | PR # | Status | Notes |
|-------|------|--------|-------|
| Slice 1: Shared UI | - | ⏳ Pending | - |
| Slice 2: Session Storage | - | ⏳ Pending | - |
| Slice 3: Settings | - | ⏳ Pending | - |
| Slice 4: Themes | - | ⏳ Pending | - |
| Slice 5: Adapters | - | ⏳ Pending | - |
| Slice 6: Launcher | - | ⏳ Pending | - |

## Related Issues

- #5307 - Ollama path deduplication (fixed via PR #5326)
- #5310 - Replace local shared-chat fork with Tools-owned component
- #5316 - Add end-to-end chat and launcher feature-discovery verification (PR #5327)
- #5328 - Review feedback on smoke tests (fixed)

---

*This document will be updated as recovery PRs are created and merged.*
