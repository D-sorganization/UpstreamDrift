# Shared Chat Migration Plan

**Parent Issue:** [#5310](https://github.com/D-sorganization/UpstreamDrift/issues/5310)
**Parent EPIC:** [#5309](https://github.com/D-sorganization/UpstreamDrift/issues/5309)
**Date Created:** 2026-05-12
**Status:** Planning

## Problem Statement

UpstreamDrift risks shadowing the Tools-owned chat implementation with local `src/shared/python/chat` code because source path ordering can prefer UpstreamDrift over vendored Tools. That makes chat appear present while bypassing the shared component other repos should inherit.

## Required Outcome

UpstreamDrift consumes the Tools-owned shared chat component through a declared dependency or pinned vendor path. UpstreamDrift-specific behavior lives in a thin app adapter.

## Current State

### Local Chat Implementation

UpstreamDrift contains local chat code at:
- `src/shared/python/chat/__init__.py`
- `src/shared/python/chat/chat_dock_widget.py`
- `src/shared/python/chat/chat_message_bubble.py`
- Related chat UI modules

### Tools-Owned Chat

The canonical chat implementation lives in:
- `Tools/src/chat/` (canonical source)
- May be vendored at `UpstreamDrift/vendor/Tools/src/chat/`

### Path Ordering Issue

Python's `sys.path` ordering can cause `src/shared/python/chat` to be imported before `vendor/Tools/src/chat`, shadowing the Tools-owned component.

## Migration Strategy

### Phase 1: Audit (Week 1)

**Goal:** Understand all chat-related imports and dependencies.

#### Tasks

1. **Audit Import Paths**:
   ```bash
   grep -r "from.*chat" src/ tests/ --include="*.py"
   grep -r "import.*chat" src/ tests/ --include="*.py"
   ```

2. **Audit Shared Components**:
   - `shared.python.chat` modules
   - Chat UI components
   - Codemap integration
   - AI backend connections
   - Theme inheritance

3. **Document Dependencies**:
   - Create dependency graph
   - Identify circular dependencies
   - List UpstreamDrift-specific customizations

### Phase 2: Adapter Creation (Week 2)

**Goal:** Create thin adapter layer for UpstreamDrift-specific behavior.

#### Implementation

1. **Create App Adapter** (`src/adapters/chat_adapter.py`):
   ```python
   """UpstreamDrift adapter for Tools-owned shared chat.

   This module provides UpstreamDrift-specific customizations
   while consuming the canonical Tools chat implementation.
   """

   from tools.chat import ChatDockWidget, ChatMessageBubble

   class UpstreamDriftChatAdapter:
       """Adapts Tools chat for UpstreamDrift integration."""

       def __init__(self):
           self.chat_widget = ChatDockWidget()
           self._apply_upstream_drift_customizations()

       def _apply_upstream_drift_customizations(self):
           """Apply UpstreamDrift-specific settings."""
           # Model refresh integration
           # Codebase indexing hooks
           # Response style selection
           pass
   ```

2. **Preserve Customizations**:
   - Launcher integration points
   - Model refresh hooks
   - Codebase indexing
   - Response style selection
   - Theme inheritance (via Tools)

### Phase 3: Migration (Week 3)

**Goal:** Switch imports from local to Tools-owned chat.

#### Implementation

1. **Update Import Statements**:
   ```python
   # Before
   from src.shared.python.chat import ChatDockWidget

   # After
   from tools.chat import ChatDockWidget
   # or
   from vendor.tools.chat import ChatDockWidget
   ```

2. **Update Module References**:
   - Update all `src/` files importing chat
   - Update test files
   - Update configuration files

3. **Remove Local Implementation**:
   - Move local customizations to adapter
   - Remove `src/shared/python/chat/` directory
   - Update `__init__.py` exports

### Phase 4: Testing (Week 4)

**Goal:** Verify chat functionality through product entry points.

#### Tests Required

1. **Module Resolution Tests**:
   ```python
   def test_chat_imports_from_tools():
       """Verify chat imports from Tools, not local."""
       import chat
       assert chat.__file__.startswith('vendor/tools') or \
              chat.__file__.startswith('tools')
   ```

2. **Launcher Integration Tests**:
   - Chat opens from launcher
   - Session history loads
   - Model refresh works
   - Codebase indexing triggers

3. **Product Behavior Tests**:
   - Chat displays conversation history
   - Model selection dropdown works
   - Response style selector works
   - Theme inheritance applies

### Phase 5: Documentation (Week 4)

**Goal:** Document the consumption contract.

#### Documentation Updates

1. **Update docs/SPEC.md**:
   - Document Tools ownership
   - Document adapter pattern
   - Document path ordering requirements

2. **Update README.md**:
   - Add vendor setup instructions
   - Add troubleshooting for path issues

3. **Update CI Configuration**:
   - Add vendor verification step
   - Add import path tests

## TDD / DbC / LOD / DRY Principles

- **TDD:** Tests fail against current shadowing behavior before the fix
- **DbC:** Missing/incompatible Tools fails with actionable message
- **LOD:** UpstreamDrift calls public Tools APIs, not internals
- **DRY:** No second shared chat implementation remains in UpstreamDrift

## Acceptance Criteria

- [ ] UpstreamDrift chat opens from launcher using Tools-backed component
- [ ] Module resolution is test-backed
- [ ] Existing features remain:
  - Session history
  - Response style
  - Model refresh
  - Codebase indexing
  - Theme inheritance

## PR Tracking

| Phase | PR # | Status | Notes |
|-------|------|--------|-------|
| Phase 1: Audit | - | ⏳ Pending | - |
| Phase 2: Adapter | - | ⏳ Pending | - |
| Phase 3: Migration | - | ⏳ Pending | - |
| Phase 4: Testing | - | ⏳ Pending | - |
| Phase 5: Documentation | - | ⏳ Pending | - |

## Related Issues

- #5307 - Ollama path deduplication (fixed via PR #5326)
- #5311 - Audit May 10-12 work
- #5315 - Chat UI recovery plan (PR #5333)
- #5316 - Smoke tests (PR #5327)
- #5328 - Smoke test API fix (fixed)

## Gasification_Model Notes

Gasification_Model has a similar issue tracked at:
- [Gasification_Model #3514](https://github.com/D-sorganization/Gasification_Model/issues/3514)

The migration pattern should be consistent across both repos.

---

*This document will be updated as migration PRs are created and merged.*
