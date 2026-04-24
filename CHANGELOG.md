# Changelog

All notable changes to UpstreamDrift will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### April 2026

#### Added

- `BaseEngineAdapter` abstract base class with `NotImplementedError` stubs for engine interchange layer (#3051, PR #3206)
- Choose-your-engine tutorial and demo script in `docs/examples/` (#3050, PR #3205)
- Launcher README cataloguing all five launchers and the canonical `upstream-drift` entry point (#3058, PR #3200)
- `docs/README.md` navigable index for the `docs/` tree (#3073)

#### Fixed

- Shrunk `pyproject.toml` mypy exclusion list; promoted suppressed modules to per-file `disable_error_code` overrides (#3207)
- Deprecation notices added to duplicate humanoid builders; `contracts.py` audited (#3057, PR #3199)
- Orphan root-level scripts (`setup_golf_suite.py`, `start_api_server.py`) relocated to `scripts/chore/` (#3070)

#### Changed

- SPEC.md module map, lock-file reference, and Python version brought in line with the actual codebase (#3071)

---

### Security (January 13, 2026)

> Note: the security hardening below was an internal engineering effort. No independent audit certificate is available at this time; the grade figures quoted previously have been removed pending a linked audit artifact.

#### Added

- `SECURITY.md`: Comprehensive security policy with reporting procedures, best practices, and compliance standards
- `docs/SECURITY_UPGRADE_GUIDE.md`: Step-by-step migration guide for API key regeneration
- `scripts/migrate_api_keys.py`: Automated migration script from SHA256 to bcrypt hashing
- `tests/unit/test_api_security.py`: Comprehensive security test suite (200+ lines)
- `engines/pendulum_models/archive/README_SECURITY_WARNING.md`: Security warnings for legacy code
- `.gitattributes`: Exclude archive code from language statistics

#### Fixed (CRITICAL)

- **API Key Security**: Upgraded from SHA256 (fast hash, brute-force vulnerable) to bcrypt (slow hash, industry standard)
  - **BREAKING CHANGE**: All API keys must be regenerated - old keys will NOT work
  - Constant-time comparison prevents timing attacks
  - Files: `api/auth/dependencies.py`
- **JWT Token Generation**: Replaced deprecated `datetime.utcnow()` with `datetime.now(timezone.utc)`
  - Python 3.12+ compatible
  - Explicit timezone handling for distributed systems
  - Files: `api/auth/security.py`, `api/auth/dependencies.py`
- **Password Logging**: Removed plaintext password from logs
  - Admin password no longer logged on startup
  - Recovery instructions provided instead
  - Files: `api/database.py`
- **Archive Code Isolation**: Added security warnings for unsafe eval() usage
  - Legacy code with code injection vulnerabilities clearly marked
  - Excluded from GitHub language statistics
  - Files: `.gitattributes`, archive README
- **Security Audit**: Made `pip-audit` blocking in CI
  - Vulnerabilities now fail CI instead of warning
  - Automated dependency security enforcement
  - Files: `.github/workflows/ci-standard.yml`

#### Documentation

- `CRITICAL_PROJECT_REVIEW.md`: 557-line adversarial security review
- `SECURITY_FIXES_SUMMARY.md`: Technical details of all security fixes
- `PR_SUMMARY.md`: Comprehensive PR summary with migration checklist

#### Compliance Achieved

- ✅ OWASP Top 10 (Authentication, Sensitive Data Exposure)
- ✅ CWE-327 (Broken Cryptography)
- ✅ CWE-532 (Sensitive Info in Logs)
- ✅ CWE-94 (Code Injection - archived/warned)
- ✅ Python Security Best Practices
- ✅ FastAPI Security Guidelines

---

### Added

- Comprehensive assessment framework (A-O) with 15 quality categories
- MyoSuite integration for musculoskeletal modeling
- OpenSim tutorials and example scripts
- AGENTS.md restored to root with Golf-specific guidelines
- Critical files protection CI workflow (prevents accidental deletion)
- Expanded pre-commit hooks (trailing whitespace, YAML validation, large file detection)

### Changed

- Updated README status from BETA to STABLE
- Removed broken GolfingRobot.png reference
- Cleaned 30+ debris files from root directory
- Updated .gitignore to prevent future accumulation
- Moved utility scripts from root to scripts/ directory
- Archived pre-Jan11-2026 assessment documents

### Fixed

- Mypy errors in plotting module
- Type annotations across physics engines

## [1.0.0] - 2026-01-10

### Added

- 5 Physics Engines: MuJoCo, Drake, Pinocchio, OpenSim, MyoSuite
- 1,563+ unit tests for comprehensive validation
- Professional PyQt6 GUI launcher
- Multi-engine comparison capabilities
- URDF generator with bundled assets

### Features

- Manipulability ellipsoid visualization
- Flexible shaft dynamics modeling
- Grip contact force analysis
- Ground reaction force processing

### Infrastructure

- Cross-engine validation framework
- Scientific plotting architecture
- Energy monitoring system
