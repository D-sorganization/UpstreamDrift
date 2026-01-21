# Sentinel Security Journal

## 2026-01-21 - Security Audit

**Scan Results:**
- Dependencies: 0 vulnerabilities (0 H / 0 M / 0 L)
- Code Analysis (Bandit): 4 issues (0 H / 4 M / Many L)
- Pattern Scan (Semgrep): Skipped (Environment incompatibility)

**Issues Created:**
- `ISSUE_003_XML_INJECTION.md`: XML Injection in URDF tools
- `ISSUE_004_UNSAFE_URL_OPEN.md`: Unsafe URL open in Model Library

**Deferred:**
- Semgrep analysis deferred due to missing `semgrep-core` binary in the environment.
- LOW severity Bandit findings (e.g., `B101` asserts, `B603` subprocess) are logged but not escalated to issues at this time.

**Low Severity Details:**
- **B101 (Asserts):** Heavily used in test files (`tests/unit/`). Safe to ignore in tests.
- **B404/B603 (Subprocess):** Used in `tools/matlab_utilities` and `tests/verify_changes.py`. Requires careful review but not blocking.
- **B405 (xml.etree):** Imports flagged in test files alongside the Medium severity usage.
