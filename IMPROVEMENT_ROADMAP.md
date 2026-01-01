# Golf Modeling Suite - Improvement Roadmap

This document outlines the phased approach to implementing the recommendations from the Dec 27, 2025 Code Review.

## Phase 1: Stabilization (✅ Completed)
**Goal**: Fix critical issues, enable test discovery, and ensure configuration consistency.

- [x] Fix dependency version mismatch (`pyproject.toml` ruff updated).
- [x] Replace print() with logging in `model_registry.py`.
- [x] Fix test discovery (added `__init__.py`, refactored imports).
- [x] Add subprocess timeouts in `golf_launcher.py`.

## Phase 2: Quality & Hygiene (✅ Completed)
**Goal**: Clean up technical debt, remove placeholders, and consolidate duplicate code.

- [x] **Address TODO/FIXME Markers**:
  - Scan for all TODOs.
  - Resolve trivial ones inline.
  - Convert complex ones to GitHub issues.
  - Update CI to enforce no new TODOs in Python files.
- [x] **Consolidate Pendulum Models**:
  - Merge duplicate directories in `engines/pendulum_models`.
  - Create single source of truth (`engines/pendulum_models/python/`).
  - Archive legacy code (`engines/pendulum_models/archive/`).
- [x] **Improve Exception Handling**:
  - Scan codebase for bare `except Exception`.
  - Replace with specific exceptions where possible.
  - *Note: Critical paths in launcher, registry, and engine_manager addressed.*

## Phase 3: Test Coverage Expansion (✅ Completed)
**Goal**: Increase coverage from 10% to 40%+.

- [x] **Model Registry Tests**: Create unit tests for registry loading and error states.
- [x] **Launcher Integration Tests**: specific tests for `golf_launcher.py`.
- [x] **Physics Validation**: Add missing energy conservation checks for Drake/Pinocchio.

## Phase 4: Documentation & Performance (🔄 In Progress)
**Goal**: Professional polish and regression testing.

- [ ] **API Documentation**: Add docstrings to all public methods in shared/ and launchers/.
- [ ] **Performance Benchmarks**: Implement `pytest-benchmark` for ABA/forward dynamics.
- [ ] **Security Scanning**: Add pip-audit/safety to CI.

## Phase 5: Advanced Features & Optimization (⏳ Planned)
**Goal**: Enhanced user experience and performance optimization.

- [ ] **Enhanced C3D Integration**: Improve C3D viewer with advanced analysis features.
- [ ] **Docker Optimization**: Multi-stage builds for reduced image size.
- [ ] **GUI Enhancements**: Improved launcher UI with better tile management.
- [ ] **Performance Profiling**: Detailed performance analysis and optimization.
