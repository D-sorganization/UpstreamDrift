# Changelog

All notable changes to UpstreamDrift will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### April 2026

#### Added

- GitHub Actions pinning to commit SHAs for supply chain security (#3210)
- Comprehensive pytest markers added to 937+ test files for better test organization
- Aerodynamics engine integration with BallFlightSimulator for consistent wind/ball settings (#3204)

#### Fixed

- CI/CD gaps: added proper gating on build success before release artifacts (#3210)
- MyPy exclusion list reduced from 65+ to targeted disable_error_code directives (#3075, #3207)
- Pip-audit CVE ignore list cleaned up with proper dependency upgrades (#3208)
- Removed sys.path hacks from examples for clean package install (#3049, #3209)
- Pinocchio zero-force fallback for unsupported contact-force queries (#3201, #3211)
- verify_installation.py expanded with comprehensive environment checks (#3172, #3203)
- basic_flight_simulation.py example now produces output instead of silent execution (#3171)
- HelpPanel populated with substantive content and contextual bindings (#3170)
- Tutorial imports fixed: broken URLs and incorrect module paths corrected (#3169)
- UI README replaced Vite template with UpstreamDrift developer guide (#3176)
- Silent failure paths replaced with actionable error messages (#3175)
- Aerodynamics and impact models wired into physics engine step() (#3167)
- Four React Tool pages (DataExplorer, MotionCapture, VideoAnalyzer, PuttingGreen) backend implementation (#3166)
- RAG store wired on startup and glossary endpoint added (#3164, #3165)
- Tool calling enabled in chat service streaming (#3162, #3163)
- ChatPanel component shipped in React/Tauri web UI (#3161)
- AI assistant core extracted for dashboard-help integrations (#3145)
- Model editor with duplicate, parameter presets, and post-sim summary (#3174, #3190)
- Orphan root-level scripts cleaned up and reorganized (#3070)
- Test pollution eliminated: sys.modules mocking replaced with proper patching (#3212)

#### Documentation

- Updated SPEC.md for accuracy with current implementation (#3071)
- Removed orphan root scripts and tracked junk files (#3070)
- Added docs hub for navigable documentation tree (#3213)

### February-March 2026

- Initial assessment framework (Adversarial Review A-O)
- Cross-engine parity testing infrastructure
- Physics correctness validation framework
- Refactored launcher consolidation
- Humanoid builder consolidation

### Security (CRITICAL - January 13, 2026)

Critical security fixes for authentication, data exposure, and dependency vulnerabilities.

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

**Production Ready**: Previously unsuitable for production → Now production-ready ✅

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
