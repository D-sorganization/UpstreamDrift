# Fleet Recovery Tracking Document

**Parent EPIC:** [#5309](https://github.com/D-sorganization/UpstreamDrift/issues/5309)
**Date Created:** 2026-05-12
**Status:** In Progress

## Summary

This document tracks the recovery of May 10-12, 2026 chat, codemap, theme, launcher, and biomechanics work into a controlled implementation plan across the D-sorganization fleet.

## Target Architecture

- **Tools** owns canonical shared chat, AI backend, codemap, session history, and theme packages
- **UpstreamDrift** and **Gasification_Model** consume Tools through declared dependencies or pinned vendor paths
- **Shared-folder drift** is blocked by CI, ownership manifests, and review gates
- **UpstreamDrift** exposes every supported feature in the correct launcher/menu category
- **Biomechanics model packs** (MuJoCo, Drake, Pinocchio, OpenSim) are incorporated under Biomechanics
- **Closed issues and open repair PRs** from May 10-12 are audited against working product behavior

## Child Issues Status

| Issue # | Repo | Title | Status | PR |
|---------|------|-------|--------|-----|
| #5307 | UpstreamDrift | Derive Ollama endpoint paths from host prefix | ✅ Fixed | [#5326](https://github.com/D-sorganization/UpstreamDrift/pull/5326) |
| #5309 | UpstreamDrift | [EPIC] Recover fleet chat architecture | 🔄 In Progress | - |
| #5310 | UpstreamDrift | Replace local shared-chat fork with Tools-owned component | ⏳ Pending | - |
| #5311 | UpstreamDrift | Verify May 10-12 chat/codemap/theme work landed | ⏳ Pending | - |
| #5312 | UpstreamDrift | Incorporate MuJoCo/Drake/Pinocchio/OpenSim model packs | ⏳ Pending | - |
| #5313 | UpstreamDrift | Reconcile model_pack/v1 provider schema | ⏳ Pending | - |
| #5314 | UpstreamDrift | Expose and categorize all runnable UpstreamDrift tools | ⏳ Pending | - |
| #5315 | UpstreamDrift | Recover modernized chat UI/session history | ⏳ Pending | - |
| #5316 | UpstreamDrift | Add end-to-end chat and launcher feature-discovery verification | ✅ Fixed | [#5327](https://github.com/D-sorganization/UpstreamDrift/pull/5327) |

## Completed Work

### PR #5326 - Fix Ollama Path Deduplication

**Issue:** #5307
**Branch:** `fix/ollama-v1-path-dedup`
**Changes:**
- Modified `join_url()` in `rust_core/ai_backend/src/config.rs` to detect and skip duplicate path segments
- Added `test_chat_url_handles_duplicate_path_segments()` test
- All 33 unit tests pass

**Problem Fixed:** Users configuring `ollama_host` with a `/v1` suffix (e.g., `http://localhost:11434/v1`) would get malformed URLs like `http://localhost:11434/v1/v1/chat/completions` when the default `chat_path=/v1/chat/completions` is applied.

### PR #5327 - Add Smoke Tests for Chat/Launcher Discovery

**Issue:** #5316
**Branch:** `feat/chat-launcher-smoke-tests`
**Changes:**
- Added `tests/smoke/test_chat_launcher_discovery.py` with tests for:
  - Shared chat import and public API contract
  - Launcher chat entry points
  - Model refresh configuration
  - Theme inheritance
  - Biomechanics model pack visibility
  - Codebase indexing via AI backend
  - Response style selection
- Includes manual verification checklist for UI behavior

## Remaining Work

### Issue #5310 - Replace Local Shared-Chat Fork

**Action Required:** Audit `src/shared/python/chat` and vendor/Tools paths to ensure Tools-owned component is consumed.

### Issue #5311 - Audit Matrix

**Action Required:** Create cross-repo audit matrix mapping May 10-12 work to product behavior.

### Issue #5312 - Biomechanics Model Packs

**Action Required:** Register MuJoCo_Models, Drake_Models, Pinocchio_Models, OpenSim_Models under Biomechanics category.

### Issue #5313 - Model Pack Schema Reconciliation

**Action Required:** Implement adapter for `model_pack/v1` provider manifests to normalize into UpstreamDrift contract.

### Issue #5314 - Launcher Tool Exposure

**Action Required:** Inventory and categorize all runnable tools in launcher menu.

### Issue #5315 - Chat UI Recovery

**Action Required:** Recover modernized chat UI and session history from broad repair PRs.

## Non-Negotiable Controls

1. **TDD:** Every child issue adds focused regression tests
2. **DbC:** Package boundaries have explicit contracts and actionable validation errors
3. **LOD:** Product apps consume public Tools APIs, not internals
4. **DRY:** No duplicated shared chat/codemap/theme implementations across repos
5. **Drift Control:** Shared folders have owners, manifests, CI checks, and documented exceptions

## Definition of Done

- [ ] `shared.python.chat` imports in Tools, UpstreamDrift, and Gasification_Model from one canonical implementation
- [ ] UpstreamDrift and Gasification_Model can open chat, restore sessions, apply theme inheritance
- [ ] All four biomechanics packs are discoverable under Biomechanics with exercise metadata
- [ ] Feature discovery is test-backed and hidden entries are intentional
- [ ] May 10-12 audit matrix maps relevant issue/PR work to product behavior

---

*This document is auto-updated as PRs are merged and issues are resolved.*
