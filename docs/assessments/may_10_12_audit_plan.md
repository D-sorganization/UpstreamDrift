# May 10-12 Audit Plan

**Parent Issue:** [#5311](https://github.com/D-sorganization/UpstreamDrift/issues/5311)
**Parent EPIC:** [#5309](https://github.com/D-sorganization/UpstreamDrift/issues/5309)
**Date Created:** 2026-05-12
**Status:** Matrix populated

## Problem Statement

Multiple chat, codemap, theme, and launcher-related issues/PRs were handled between May 10 and May 12, 2026, but product behavior still appears incomplete. We need evidence that tracker closure equals working functionality.

## Required Outcome

Create a cross-repo audit matrix mapping each relevant issue/PR to actual incorporation location, menu path, test coverage, and remaining gaps.

## Scope

Audit Tools, UpstreamDrift, and Gasification_Model work related to:

- Chat UI/session history
- AI/Rust backend
- Codemap/indexing
- Theme inheritance
- Launcher/menu exposure
- Shared-folder/vendor path changes

## Audit Matrix

### Required Columns

| Column                         | Description                                                |
| ------------------------------ | ---------------------------------------------------------- |
| Repo                           | Tools, UpstreamDrift, or Gasification_Model                |
| Issue/PR #                     | Link to issue or PR                                        |
| Claimed Outcome                | What the issue/PR claims to deliver                        |
| Files Changed                  | Key files modified                                         |
| Current Incorporation Location | Where the code lives now                                   |
| Product Entry Point            | Menu path or launcher entry                                |
| Test Coverage                  | Tests proving behavior                                     |
| Status                         | working, partial, missing, duplicated, superseded, blocked |
| Follow-up Link                 | Link to follow-up issue/PR                                 |

### Cross-Repo Audit Matrix

| Repo                  | Issue/PR number and title                                                                                                                    | Claimed outcome                                                                                                  | Files changed                                                                                                                                                           | Current incorporation location                                                       | Product entry point or menu path                                 | Test proving behavior                                                                                                                                                              | Status                                                                                 | Follow-up issue/PR link                                                                                                                                    |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | ---------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tools                 | [PR #2589: feat(ai-backend): migrate rust ai backend and agent tools from UpstreamDrift](https://github.com/D-sorganization/Tools/pull/2589) | Move the Rust AI backend and tool schemas into Tools as the shared owner.                                        | `rust_core/ai_backend`, `src/shared/python/ai`, `src/shared/python/chat`                                                                                                | Tools-owned shared AI/chat package.                                                  | Shared package only; no direct product menu.                     | Existing PR tests cover Rust HTTP integration and chat/service behavior.                                                                                                           | working                                                                                | [Repository_Management #1160](https://github.com/D-sorganization/Repository_Management/pull/1160)                                                          |
| Tools                 | [PR #2563: feat(codemap): repo-aware code map package + MCP server](https://github.com/D-sorganization/Tools/pull/2563)                      | Add canonical repo-aware codemap package, CLI, watcher, and MCP server.                                          | `src/shared/python/codemap`, `docs/codemap.md`, `tests/unit/codemap`                                                                                                    | Tools `codemap`, `codemap-watch`, and `codemap-mcp` console scripts.                 | CLI/MCP, not GUI.                                                | `python -m pytest tests/unit/codemap/`; manual `codemap rebuild`, `codemap search`, `codemap who-calls`.                                                                           | working                                                                                | [UpstreamDrift #5310](https://github.com/D-sorganization/UpstreamDrift/issues/5310)                                                                        |
| Tools                 | [PRs #2555, #2556, #2579, #2590: restore chat UI/theme/font work](https://github.com/D-sorganization/Tools/pull/2590)                        | Restore chat theme inheritance, FontManager, modern chat UI, and session history in the shared package.          | `src/shared/python/ai/gui`, `src/shared/python/theme`, `tests/shared/python/chat`                                                                                       | Shared `AIAssistantPanel`, `ChatDockWidget`, `ThemeManager`, and `FontManager`.      | Consumer app embeds the dock/panel; no Tools product menu found. | Existing tests cover chat dock, response-style, and theme manager behavior.                                                                                                        | partial: shared package exists, product exposure remains consumer-owned                | [Repository_Management #1159](https://github.com/D-sorganization/Repository_Management/pull/1159)                                                          |
| Tools                 | [PR #2596: fix(chat): stabilize shared package contract](https://github.com/D-sorganization/Tools/pull/2596)                                 | Define and test the public shared-chat facade that consumers may import.                                         | `src/shared/python/chat/__init__.py`, `src/shared/python/chat/models.py`, `tests/shared/python/chat/test_public_contract.py`                                            | Tools `src.shared.python.chat` public facade.                                        | Shared package only; consumers adapt it into app UI.             | `python -m pytest tests/shared/python/chat/test_models.py tests/shared/python/chat/test_public_contract.py src/shared/python/chat/tests/test_chat.py -q -o addopts=''`             | working                                                                                | [Tools #2592](https://github.com/D-sorganization/Tools/issues/2592)                                                                                        |
| Repository_Management | [PR #1160: Add Tools chat package contract documentation](https://github.com/D-sorganization/Repository_Management/pull/1160)                | Document the canonical Tools chat package contract for consumers.                                                | `docs/operations/TOOLS_CHAT_CONTRACT.md`, `src/repository_manager.py`, `tests/test_chat_contract.py`                                                                    | Contract documentation in Repository_Management, implementation owner remains Tools. | N/A; governance documentation.                                   | `tests/test_chat_contract.py` validates required contract sections and import expectations.                                                                                        | partial: merged, but follow-up needed for CI coverage command after post-merge failure | [Repository_Management follow-up PR](https://github.com/D-sorganization/Repository_Management/pulls)                                                       |
| Repository_Management | [PR #1159: shared-folder governance controls](https://github.com/D-sorganization/Repository_Management/pull/1159)                            | Add ownership and drift controls for shared-folder/component movement.                                           | `.github/CODEOWNERS`, `.github/workflows/shared-folder-drift-check.yml`, `docs/operations/SHARED_FOLDER_GOVERNANCE.md`, `scripts/drift_report.py`                       | Governance docs and drift check workflow in Repository_Management.                   | N/A; governance workflow.                                        | `python scripts/check_local_only_workflows.py`; `python scripts/drift_report.py --format json --verbose`; `python -m ruff check scripts/drift_report.py`                           | blocked: open PR awaiting CI                                                           | [Repository_Management #1159](https://github.com/D-sorganization/Repository_Management/pull/1159)                                                          |
| UpstreamDrift         | [PR #5207: consolidate codemap onto Tools implementation](https://github.com/D-sorganization/UpstreamDrift/pull/5207)                        | Replace local codemap implementation with Tools-aligned copy and wire chat codemap tools.                        | `src/shared/python/codemap`, `src/shared/python/ai/tools/codemap_tools.py`, `tests/unit/codemap_integration`                                                            | Local byte-identical copy plus adapter that prefers the Tools API.                   | Chat backend tool registry and `codemap-mcp`.                    | `tests/unit/codemap_integration/test_chat_codemap_tools.py`; manual `codemap rebuild`.                                                                                             | duplicated/superseded: canonical owner is Tools                                        | [UpstreamDrift #5310](https://github.com/D-sorganization/UpstreamDrift/issues/5310)                                                                        |
| UpstreamDrift         | [PR #5305: fix: restore modernized chat UI and theme inheritance](https://github.com/D-sorganization/UpstreamDrift/pull/5305)                | Restore modernized chat UI, refresh-models, auto-index, response-style, and theme inheritance.                   | `src/api/routes/chat_ws.py`, `src/api/services/chat_service.py`, `src/shared/python/ai/gui` plus broad unrelated churn                                                  | WebSocket actions `refresh_models` and `index_codebase`; local AI GUI copy.          | Chat UI/settings dialog and websocket chat route.                | Code evidence exists in `chat_ws.py` and `chat_service.py`; no current end-to-end product run evidence found.                                                                      | partial: backend hooks present, product proof still needed                             | [UpstreamDrift #5315](https://github.com/D-sorganization/UpstreamDrift/issues/5315), [#5316](https://github.com/D-sorganization/UpstreamDrift/issues/5316) |
| UpstreamDrift         | [PR #5327: feat: Add smoke tests for chat and launcher feature discovery](https://github.com/D-sorganization/UpstreamDrift/pull/5327)        | Add CI-safe smoke coverage for chat imports, launcher discovery, theme inheritance, model refresh, and indexing. | `tests/smoke/test_chat_launcher_discovery.py`                                                                                                                           | Smoke tests in the UpstreamDrift test suite.                                         | Manual checklist covers launcher and chat UI behavior.           | `tests/smoke/test_chat_launcher_discovery.py` skips gracefully when optional pieces are absent.                                                                                    | partial: test scaffold exists, but skip paths mean it is not full product proof        | [UpstreamDrift #5316](https://github.com/D-sorganization/UpstreamDrift/issues/5316)                                                                        |
| UpstreamDrift         | [PR #5301: feat: Add Golf Simulation Suite to launcher](https://github.com/D-sorganization/UpstreamDrift/pull/5301)                          | Expose Golf Simulation Suite and related biomechanics entry in the launcher.                                     | `src/config/models.yaml`, `src/launchers/launcher_model_handlers.py`, `src/launchers/launcher_ui_setup.py`                                                              | Launcher model registry and handler.                                                 | Launcher sidebar: `Biomechanics` -> `Golf Simulation Suite`.     | `tests/launchers/test_launcher_ui_setup.py`; `tests/launchers/test_launcher_layout_manager.py`.                                                                                    | working                                                                                | [UpstreamDrift #5314](https://github.com/D-sorganization/UpstreamDrift/issues/5314)                                                                        |
| UpstreamDrift         | [PR #5329: chore: Migrate video_analyzer and data_explorer to Tools](https://github.com/D-sorganization/UpstreamDrift/pull/5329)             | Move product tools to the Tools vendor/shared-folder path.                                                       | `vendor/ud-tools`, `src/config/models.yaml`, `src/launchers/launcher_model_handlers.py`                                                                                 | Vendor/shared-folder lookup and launcher model paths.                                | Launcher tiles `video_analyzer` and `data_explorer`.             | `scripts/check_vendor_updates.py`; launcher diagnostics reference the migrated tool ids.                                                                                           | partial: path wiring visible, runtime launch proof still needed                        | [UpstreamDrift #5314](https://github.com/D-sorganization/UpstreamDrift/issues/5314)                                                                        |
| UpstreamDrift         | [PR #5338: docs: Add Shared Chat migration plan](https://github.com/D-sorganization/UpstreamDrift/pull/5338)                                 | Document migration from local shared-chat fork to Tools-owned implementation.                                    | `docs/shared_chat_migration_plan.md`                                                                                                                                    | Planning document only.                                                              | N/A.                                                             | Document review only; no product test.                                                                                                                                             | superseded as implementation evidence; useful as plan                                  | [UpstreamDrift #5310](https://github.com/D-sorganization/UpstreamDrift/issues/5310)                                                                        |
| Gasification_Model    | [PRs #3448, #3449, #3451, #3453: codemap consolidation and chatbot adapter](https://github.com/D-sorganization/Gasification_Model/pull/3453) | Consolidate codemap onto Tools and expose it in chatbot/tool execution.                                          | `src/shared/python/codemap`, `src/integrated_process_simulator/ai/chatbot_codemap_adapter.py`, `docs/codemap-integration.md`                                            | `CodemapChatAdapter`, `GasificationToolExecutor`, local `.codemap/index.db`.         | Legacy/modern chatbot tool calls.                                | `tests/ai/test_codemap_chat_adapter.py`; manual `python scripts/codemap_rebuild.py`.                                                                                               | working adapter, duplicated shared package                                             | [Repository_Management #1159](https://github.com/D-sorganization/Repository_Management/pull/1159)                                                          |
| Gasification_Model    | [PR #3517: Refactor: Adopt shared ChatDockWidget](https://github.com/D-sorganization/Gasification_Model/pull/3517)                           | Replace legacy chat dialogs with Tools `ChatDockWidget` and keep gasification actions in an adapter.             | `src/integrated_process_simulator/ui/managers/chat_adapter.py`, `src/integrated_process_simulator/ui/managers/toolbar_dialogs.py`, `tests/unit/ui/test_chat_adapter.py` | Toolbar dialog manager resolves and mounts the shared chat adapter.                  | Toolbar -> `AI Chat Assistant`.                                  | `tests/unit/ui/test_chat_adapter.py`; `tests/integration/test_sequential_reactors_and_chatbot.py`.                                                                                 | working with adapter tests                                                             | [Gasification_Model #3514](https://github.com/D-sorganization/Gasification_Model/issues/3514)                                                              |
| Gasification_Model    | [PR #3523: fix(ui): keep chat adapter imports isolated](https://github.com/D-sorganization/Gasification_Model/pull/3523)                     | Keep `integrated_process_simulator.ui` importable when optional Tools-backed tab dependencies are absent.        | `SPEC.md`, `src/integrated_process_simulator/ui/__init__.py`                                                                                                            | Optional-dependency-safe UI package import boundary.                                 | Toolbar/chat adapter import path.                                | `python -m pytest tests/unit/ui/test_chat_adapter.py -q -o addopts=''`; `python -m ruff check src/integrated_process_simulator/ui/__init__.py tests/unit/ui/test_chat_adapter.py`. | blocked: open PR has no current-head checks attached yet                               | [Gasification_Model #3523](https://github.com/D-sorganization/Gasification_Model/pull/3523)                                                                |
| Gasification_Model    | [PRs #3447 and #3508: theme/font modernization](https://github.com/D-sorganization/Gasification_Model/pull/3508)                             | Adopt fleet FontManager/theme modernization and broader UI parity work.                                          | `src/shared/python/theme`, `src/integrated_process_simulator/ui/theme_registry.py`, frontend theme files                                                                | Shared theme package plus Gasification UI/theme registry.                            | Desktop UI theme surfaces and frontend theme assets.             | `tests/ui/test_font_manager.py`; `tests/test_theme_registry.py`.                                                                                                                   | partial: component tests exist, full visual product proof absent                       | [Gasification_Model #3500](https://github.com/D-sorganization/Gasification_Model/issues/3500)                                                              |

### Work Items to Audit

#### Chat-Related Work

| Item              | Type      | Repo                | Status              |
| ----------------- | --------- | ------------------- | ------------------- |
| ChatDockWidget    | Component | Tools               | 🔍 To Audit         |
| ChatMessageBubble | Component | Tools               | 🔍 To Audit         |
| Session History   | Feature   | Tools/UpstreamDrift | 🔍 To Audit         |
| Shared Chat Fork  | Migration | UpstreamDrift       | 📋 Plan in PR #5338 |
| Chat UI Recovery  | Recovery  | Tools/UpstreamDrift | 📋 Plan in PR #5333 |

#### AI Backend Work

| Item               | Type      | Repo          | Status              |
| ------------------ | --------- | ------------- | ------------------- |
| Rust AI Backend    | Component | UpstreamDrift | ✅ Working          |
| Ollama Adapter     | Component | UpstreamDrift | ✅ Fixed (PR #5326) |
| Path Deduplication | Fix       | UpstreamDrift | ✅ Fixed (PR #5326) |

#### Codemap/Indexing Work

| Item              | Type      | Repo          | Status      |
| ----------------- | --------- | ------------- | ----------- |
| Codebase Indexing | Feature   | UpstreamDrift | 🔍 To Audit |
| RAG Pipeline      | Component | UpstreamDrift | 🔍 To Audit |

#### Theme Work

| Item              | Type    | Repo                | Status      |
| ----------------- | ------- | ------------------- | ----------- |
| Theme Inheritance | Feature | Tools/UpstreamDrift | 🔍 To Audit |
| Theme Switching   | Feature | Tools               | 🔍 To Audit |

#### Launcher Work

| Item                   | Type    | Repo          | Status              |
| ---------------------- | ------- | ------------- | ------------------- |
| Tool Exposure          | Feature | UpstreamDrift | 📋 Plan in PR #5336 |
| Biomechanics Category  | Feature | UpstreamDrift | 📋 Plan in PR #5336 |
| Model Pack Integration | Feature | UpstreamDrift | 📋 Plan in PR #5336 |

#### Smoke Tests

| Item                    | Type  | Repo          | Status      |
| ----------------------- | ----- | ------------- | ----------- |
| Chat/Launcher Discovery | Tests | UpstreamDrift | ✅ PR #5327 |
| Smoke Test API Fix      | Fix   | UpstreamDrift | ✅ Fixed    |

## TDD / DbC / LOD / DRY Principles

- **TDD:** Every working row needs a test or manual verification command
- **DbC:** Failures classify as dependency resolution, contract validation, UI launch, or backend availability
- **LOD:** Separate shared package defects from product adapter defects
- **DRY:** Identify duplicates and assign a canonical owner

## Audit Process

### Step 1: Inventory Collection

1. **Collect Issues/PRs from May 10-12**:

   ```bash
   # UpstreamDrift
   gh issue list --created "2026-05-10..2026-05-12" --json number,title,body
   gh pr list --created "2026-05-10..2026-05-12" --json number,title,body

   # Tools
   gh issue list --repo D-sorganization/Tools --created "2026-05-10..2026-05-12"

   # Gasification_Model
   gh issue list --repo D-sorganization/Gasification_Model --created "2026-05-10..2026-05-12"
   ```

2. **Collect File Changes**:

   ```bash
   gh pr view <PR#> --json files
   ```

3. **Collect Test Files**:
   ```bash
   find tests/ -name "*.py" -newermt "2026-05-10" ! -newermt "2026-05-13"
   ```

### Step 2: Product Verification

1. **Chat Verification**:

   - [ ] Chat opens from launcher
   - [ ] Session history loads
   - [ ] Model selection works
   - [ ] Response style selection works
   - [ ] Codebase indexing triggers

2. **Launcher Verification**:

   - [ ] All tools listed
   - [ ] Categories correct
   - [ ] Biomechanics model packs visible

3. **Theme Verification**:
   - [ ] Theme inheritance applies
   - [ ] Theme switching works

### Step 3: Gap Analysis

For each work item:

1. Check if claimed outcome matches actual behavior
2. Identify missing tests
3. Identify missing documentation
4. Identify duplicates across repos
5. Classify status: working, partial, missing, duplicated, superseded, blocked

### Step 4: Follow-up Assignment

For each gap:

1. Create follow-up issue if needed
2. Assign owner
3. Set priority
4. Link to parent EPIC #5309

## Deliverables

1. **Audit Matrix** (Markdown table in this document)
2. **Gap Analysis Report** (List of missing tests, docs, features)
3. **Follow-up Issue Links** (Links to created issues)
4. **Status Summary** (Count by status category)

## Status Summary

| Status                | Count | Rows                                                                                                                                                                            |
| --------------------- | ----: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| working               |     5 | Tools AI backend, Tools codemap, Tools chat contract, UpstreamDrift launcher exposure, Gasification chat adapter                                                                |
| partial               |     7 | Tools chat UI/theme, Repository_Management contract CI follow-up, UpstreamDrift chat UI, smoke tests, vendor path migration, Gasification theme work, PR #5327 skip-based proof |
| duplicated/superseded |     2 | UpstreamDrift codemap copy, Shared Chat migration plan                                                                                                                          |
| blocked               |     2 | Repository_Management #1159, Gasification_Model #3523                                                                                                                           |

## Gap Analysis

- Canonical ownership is now clear: Tools owns shared chat, AI backend, codemap, theme/font primitives; product repos own adapters and launcher/menu exposure.
- False-green risk remains where documentation or optional smoke tests exist without a product launch proof. This affects UpstreamDrift #5315/#5316 and Gasification_Model visual theme parity.
- The UpstreamDrift local codemap and chat copies should be treated as compatibility/adapters only; new shared defects belong in Tools or Repository_Management contract/governance work.
- Repository_Management #1159 is the governance blocker for preventing future shared-folder drift. Repository_Management #1160 merged, but its post-merge coverage failure needs a narrow CI follow-up.

## PR Tracking

| Deliverable         | PR #                       | Status                  |
| ------------------- | -------------------------- | ----------------------- |
| Audit Plan Document | This PR                    | 📋 In Review            |
| Audit Matrix        | This PR                    | Complete                |
| Gap Analysis        | This PR                    | Complete                |
| Follow-up Issues    | Existing linked issues/PRs | Complete for known gaps |

## Related Issues

- #5307 - Ollama path deduplication (✅ fixed via PR #5326)
- #5310 - Shared chat migration (📋 plan in PR #5338)
- #5315 - Chat UI recovery (📋 plan in PR #5333)
- #5316 - Smoke tests (✅ PR #5327)

## Timeline

| Phase                | Duration | Target Date |
| -------------------- | -------- | ----------- |
| Inventory Collection | 1 day    | 2026-05-13  |
| Product Verification | 2 days   | 2026-05-15  |
| Gap Analysis         | 1 day    | 2026-05-16  |
| Follow-up Assignment | 1 day    | 2026-05-17  |

---

_This document will be populated with audit results and updated as follow-up issues are created._
