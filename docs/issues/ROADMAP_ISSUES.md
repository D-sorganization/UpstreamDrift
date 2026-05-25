# Roadmap Issues - Golf Modeling Suite

Last Updated: 2026-05-25

These issues are already tracked in GitHub and represent planned improvements.

## Phase 1: Foundation Fixes

### Phase 1.1: Fix Pytest Coverage Configuration

- **GitHub Issue**: #119
- **Status**: Open
- **Description**: Configure pytest coverage correctly for all modules

### Phase 1.2: Consolidate Dependency Management

- **GitHub Issue**: #120
- **Status**: Open
- **Description**: Unify dependency specifications across pyproject.toml files

### Phase 1.3: Implement Duplicate File Prevention

- **GitHub Issue**: #121
- **Status**: Open
- **Description**: Add checks to prevent duplicate files across engine directories

### Phase 1.4: Fix Python Version Metadata

- **GitHub Issue**: #122
- **Status**: Open
- **Description**: Ensure consistent Python version requirements

## Phase 2: Code Cleanup

### Phase 2.1: GUI Refactoring (SRP)

- **GitHub Issue**: #123
- **Status**: Open
- **Description**: Refactor GUI components to follow Single Responsibility Principle

### Phase 2.2: Archive & Legacy Cleanup

- **GitHub Issue**: #124
- **Status**: Open
- **Description**: Clean up archive and legacy code directories

### Phase 2.3: Constants Normalization

- **GitHub Issue**: #125
- **Status**: Open
- **Description**: Normalize physics constants across all engines

### Phase 2.4: Chat Entry-point Consolidation

- **GitHub Issue**: #6119
- **Status**: Open
- **Description**: Consolidate chat entry points and decompose \_chat_dock_widget_qt.py (Follow-up from ADR-0022 and #6120)

## Phase 3: Integration

### Phase 3.1: Cross-Engine Integration Tests

- **GitHub Issue**: #126
- **Status**: Open
- **Description**: Add tests that validate consistency between physics engines

### Phase 3.2: Architecture Documentation

- **GitHub Issue**: #127
- **Status**: Open
- **Description**: Document system architecture and engine interfaces

### Phase 3.3: Launcher Configuration Abstraction

- **GitHub Issue**: #128
- **Status**: Open
- **Description**: Abstract launcher configuration for easier customization

## Phase 4: Performance

### Phase 4.1: Async Engine Loading

- **GitHub Issue**: #129
- **Status**: Open
- **Description**: Implement asynchronous loading for physics engines

### Phase 4.2: Lazy Import Implementation

- **GitHub Issue**: #130
- **Status**: Open
- **Description**: Implement lazy imports to improve startup time

## Phase 5: Configuration Consolidation

### Phase 5.1: Adopt pydantic-settings and unify config

- **GitHub Issue**: #5920
- **Status**: Open
- **Description**: Consolidate the 6 config systems and 118 env vars into a unified `pydantic-settings` based Settings class, add runtime validation, generating `.env.example`, and implement secrets posture enforcement.

## Phase 6: CI/CD Infrastructure Review

### Phase 6.1: Review CI/CD Modularization and Jules-* Workflows

- **GitHub Issue**: #6099
- **Status**: Acknowledged (Recommendation-only)
- **Description**: Review modularization of `ci-standard.yml` (1,617 lines) and the proliferation of `Jules-*` workflows.
- **Path Forward**: Acknowledged the observation. Will review the size of `ci-standard.yml` and assess active/inactive `Jules-*` workflows. This is scheduled for "later" (target review date: Q4 2026) to avoid disrupting the currently stable CI system and fleet-level agent coordination protocol. Concrete follow-up issues will be filed if any structural changes or workflow retirements are decided.

## Phase 7: Documentation & Audits

### Phase 7.1: Audit Epics #4796 and #4755

- **GitHub Issue**: #6087
- **Status**: Closed
- **Description**: Epics #4796 (plot_style implementations) and #4755 (body_part_viz implementations) have been audited. The `plot_style` module has shipped the canonical `MatplotlibMarkerRenderer` and `PyQtGLMarkerRenderer`, along with its colors, colormaps, markers, resolvers, and widgets sub-packages. The `body_part_viz` module has shipped the canonical matplotlib and PyQtGL renderers, capsule/ellipsoid/cylinder/mesh shapes, and Kabsch/Procrustes/cluster fitters. Both epics are fully implemented and can be closed as completed.

## Phase 8: Whole-Repo Feature Review (2026-05-24)

### Phase 8.1: Repository hygiene & version sync

- **GitHub Issue**: #6081 (Epic A)
- **Status**: Open
- **Description**: Address version drift between `VERSION` and `SPEC.md`, remove stale `launch.bat`, and update stale project name (Golf Modeling Suite to UpstreamDrift) and paths in `docs/README.md`.

### Phase 8.2: Tool surface audit & EmbeddableTool coverage

- **GitHub Issue**: #6081 (Epic B)
- **Status**: Open
- **Description**: Audit the GUI modules in `src/tools/` for unknown functional states and expand `EmbeddableTool` contract coverage.

### Phase 8.3: Engine integration completeness

- **GitHub Issue**: #6081 (Epic C)
- **Status**: Open
- **Description**: Add MyoSuite coverage to `pose_interchange/services/` and `adapters/` for parity with `src/engines/physics_engines/myosuite/`.

### Phase 8.4: Resolve ADR-0020 Status

- **GitHub Issue**: #6081 (Follow-up)
- **Status**: Open
- **Description**: Resolve or downgrade ADR-0020 (Canonical URDF subsystem) which is stuck in Proposed state after the URDF Hardening Campaign closure.

### Phase 8.5: Test Suite Triage

- **GitHub Issue**: #6081 (Follow-up)
- **Status**: Open
- **Description**: Triage the 160 `pytest.mark.skip` and 86 `xfail` markers to identify and remove stale entries.

### Phase 8.6: Entry-Point Documentation

- **GitHub Issue**: #6081 (Follow-up)
- **Status**: Open
- **Description**: Update root `README.md` to clarify the entry-point story: web UI (`launch_golf_suite.py` / `upstream-drift`) as primary, classic PyQt launcher as fallback, and remove references to dead launchers.

### Phase 8.7: Dockerfile Consolidation

- **GitHub Issue**: #6081 (Follow-up)
- **Status**: Open
- **Description**: Decide canonical status of `Dockerfile.modular` relative to `Dockerfile` and `Dockerfile.heavy_test` to prevent architectural drift.

### Phase 8.8: Chat / Sidekick Architecture Review

- **GitHub Issue**: #6081 (Follow-up)
- **Status**: Open
- **Description**: Review Chat/Sidekick architecture boundaries (cross-ref #5969, #5967, #5922) and address the tech debt in `src/shared/python/chat/_chat_dock_widget_qt.py`.

### Phase 8.9: CI Workflow Modularization

- **GitHub Issue**: #6081 (Follow-up)
- **Status**: Open
- **Description**: Review the 81 GitHub Actions workflows (including `ci-standard.yml` and `Jules-*` proliferation) for modularization and pruning opportunities.

## Phase 9: EmbeddableTool Adoption

### Phase 9.1: Execute EmbeddableTool adoption for the 8 audited tools

- **GitHub Issue**: #6090
- **Status**: Blocked
- **Description**: Wait for the audit verdict matrix from #6089, then split execution by tool into adopt EmbeddableTool, document standalone opt-out, or remove scaffolds.
