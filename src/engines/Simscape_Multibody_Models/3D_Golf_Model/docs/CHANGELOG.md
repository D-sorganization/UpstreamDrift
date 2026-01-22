# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Comprehensive project README with setup instructions and repository structure
- Documentation index (docs/README.md) for easy navigation
- AI-Assisted Development Guide consolidating best practices
- Organized archive for historical documentation

### Changed

- Improved root README from generic template to project-specific overview
- Reorganized docs directory with clear categorization
- Consolidated duplicate development guides (cursor/copilot rules)

### Removed

- Moved session-specific summaries to archive directory
- Archived completed implementation documentation

## [2025-08-09] - Project Initialization

### Initial Setup

- Initial repository setup with MATLAB and Python structure
- Pre-commit hooks for code quality (ruff, mypy)
- Git LFS configuration for large files
- GitHub Actions CI/CD workflows
- Basic test infrastructure
- MATLAB environment setup scripts
- Safety guardrails and development guidelines

### Infrastructure Files

- `.pre-commit-config.yaml` - Automated code quality checks
- `.github/workflows/ci.yml` - Continuous integration
- `matlab/setup_matlab_environment.m` - MATLAB path and cache setup
- `matlab/run_all.m` - Master test runner
