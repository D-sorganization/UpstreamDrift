# May 10-12 Audit Plan

**Parent Issue:** [#5311](https://github.com/D-sorganization/UpstreamDrift/issues/5311)
**Parent EPIC:** [#5309](https://github.com/D-sorganization/UpstreamDrift/issues/5309)
**Date Created:** 2026-05-12
**Status:** Planning

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

| Column | Description |
|--------|-------------|
| Repo | Tools, UpstreamDrift, or Gasification_Model |
| Issue/PR # | Link to issue or PR |
| Claimed Outcome | What the issue/PR claims to deliver |
| Files Changed | Key files modified |
| Current Incorporation Location | Where the code lives now |
| Product Entry Point | Menu path or launcher entry |
| Test Coverage | Tests proving behavior |
| Status | working, partial, missing, duplicated, superseded, blocked |
| Follow-up Link | Link to follow-up issue/PR |

### Work Items to Audit

#### Chat-Related Work

| Item | Type | Repo | Status |
|------|------|------|--------|
| ChatDockWidget | Component | Tools | 🔍 To Audit |
| ChatMessageBubble | Component | Tools | 🔍 To Audit |
| Session History | Feature | Tools/UpstreamDrift | 🔍 To Audit |
| Shared Chat Fork | Migration | UpstreamDrift | 📋 Plan in PR #5338 |
| Chat UI Recovery | Recovery | Tools/UpstreamDrift | 📋 Plan in PR #5333 |

#### AI Backend Work

| Item | Type | Repo | Status |
|------|------|------|--------|
| Rust AI Backend | Component | UpstreamDrift | ✅ Working |
| Ollama Adapter | Component | UpstreamDrift | ✅ Fixed (PR #5326) |
| Path Deduplication | Fix | UpstreamDrift | ✅ Fixed (PR #5326) |

#### Codemap/Indexing Work

| Item | Type | Repo | Status |
|------|------|------|--------|
| Codebase Indexing | Feature | UpstreamDrift | 🔍 To Audit |
| RAG Pipeline | Component | UpstreamDrift | 🔍 To Audit |

#### Theme Work

| Item | Type | Repo | Status |
|------|------|------|--------|
| Theme Inheritance | Feature | Tools/UpstreamDrift | 🔍 To Audit |
| Theme Switching | Feature | Tools | 🔍 To Audit |

#### Launcher Work

| Item | Type | Repo | Status |
|------|------|------|--------|
| Tool Exposure | Feature | UpstreamDrift | 📋 Plan in PR #5336 |
| Biomechanics Category | Feature | UpstreamDrift | 📋 Plan in PR #5336 |
| Model Pack Integration | Feature | UpstreamDrift | 📋 Plan in PR #5336 |

#### Smoke Tests

| Item | Type | Repo | Status |
|------|------|------|--------|
| Chat/Launcher Discovery | Tests | UpstreamDrift | ✅ PR #5327 |
| Smoke Test API Fix | Fix | UpstreamDrift | ✅ Fixed |

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

## PR Tracking

| Deliverable | PR # | Status |
|-------------|------|--------|
| Audit Plan Document | This PR | 📋 In Review |
| Audit Matrix | - | ⏳ Pending |
| Gap Analysis | - | ⏳ Pending |
| Follow-up Issues | - | ⏳ Pending |

## Related Issues

- #5307 - Ollama path deduplication (✅ fixed via PR #5326)
- #5310 - Shared chat migration (📋 plan in PR #5338)
- #5315 - Chat UI recovery (📋 plan in PR #5333)
- #5316 - Smoke tests (✅ PR #5327)

## Timeline

| Phase | Duration | Target Date |
|-------|----------|-------------|
| Inventory Collection | 1 day | 2026-05-13 |
| Product Verification | 2 days | 2026-05-15 |
| Gap Analysis | 1 day | 2026-05-16 |
| Follow-up Assignment | 1 day | 2026-05-17 |

---

*This document will be populated with audit results and updated as follow-up issues are created.*
