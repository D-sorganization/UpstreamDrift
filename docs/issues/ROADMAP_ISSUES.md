# Roadmap Issues - Golf Modeling Suite

Last Updated: 2026-01-17

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
- **Description**: Review modularization of `ci-standard.yml` and the proliferation of `Jules-*` workflows.
- **Path Forward**: Acknowledged the observation. Will review the size of `ci-standard.yml` and assess active/inactive `Jules-*` workflows. This is scheduled for "later" (target review date: Q4 2026) to avoid disrupting the currently stable CI system and fleet-level agent coordination protocol. Concrete follow-up issues will be filed if any structural changes or workflow retirements are decided.
