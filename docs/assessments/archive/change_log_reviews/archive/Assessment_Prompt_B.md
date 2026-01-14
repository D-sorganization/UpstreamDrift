# Assessment B: Tools Repository Hygiene, Security & Quality Review

## Assessment Overview

You are a **principal/staff-level Python engineer** conducting an **adversarial, evidence-based** code hygiene and security review of the Tools repository. Your job is to **evaluate linting compliance, repository organization, security posture, and adherence to coding standards** as defined in AGENTS.md.

**Reference Documents**:

- `AGENTS.md` - Coding standards and agent guidelines (MANDATORY)
- `ruff.toml` - Ruff linting configuration
- `mypy.ini` - Type checking configuration
- `.pre-commit-config.yaml` - Pre-commit hooks

---

## Context: Tools Repository Quality Standards

The Tools repository must maintain high hygiene standards as a **replicant template repository** that may be cloned and used across multiple projects.

### Quality Gates (Must Pass)

| Tool       | Configuration             | Enforcement Level      |
| ---------- | ------------------------- | ---------------------- |
| Ruff       | `ruff.toml`               | Strict                 |
| Black      | Default 88 line-length    | Strict                 |
| Mypy       | `mypy.ini`                | Strict (where enabled) |
| Pre-commit | `.pre-commit-config.yaml` | Per-commit             |

### AGENTS.md Standards (Mandatory Compliance)

1. **No `print()` statements** - Use `logging` module
2. **No wildcard imports** - Explicit imports only
3. **No bare `except:` clauses** - Specific exception types
4. **Type hints required** - All public functions
5. **No secrets in code** - Use `.env` and `python-dotenv`

---

## Your Output Requirements

Do **not** be polite. Do **not** generalize. Do **not** say "looks good overall."
Every claim must cite **exact files/paths, modules, functions**, or **config keys**.

### Deliverables

#### 1. Executive Summary (1 page max)

- Overall hygiene assessment in 5 bullets
- Top 10 hygiene/security risks (ranked)
- "If CI/CD ran strict enforcement today, what fails first?"

#### 2. Scorecard (0-10)

Score each category. For every score ≤8, list evidence and remediation path.

| Category                | Description                     | Weight |
| ----------------------- | ------------------------------- | ------ |
| Ruff Compliance         | Zero violations across codebase | 2x     |
| Mypy Compliance         | Strict type safety              | 2x     |
| Black Formatting        | Consistent formatting           | 1x     |
| AGENTS.md Compliance    | All standards met               | 2x     |
| Security Posture        | No secrets, safe patterns       | 2x     |
| Repository Organization | Clean, intuitive structure      | 1x     |
| Dependency Hygiene      | Minimal, pinned, secure         | 1x     |

#### 3. Findings Table

| ID    | Severity | Category | Location | Symptom | Root Cause | Fix | Effort |
| ----- | -------- | -------- | -------- | ------- | ---------- | --- | ------ |
| B-001 | ...      | ...      | ...      | ...     | ...        | ... | S/M/L  |

**Severity Definitions:**

- **Blocker**: Security vulnerability or CI/CD-breaking violation
- **Critical**: Pervasive hygiene issue affecting multiple files
- **Major**: Significant deviation from AGENTS.md standards
- **Minor**: Isolated hygiene issue
- **Nit**: Style/consistency improvement

#### 4. Linting Violation Inventory

Run and document:

```bash
ruff check . --output-format=json
mypy . --strict
black --check .
```

| File            | Ruff Violations    | Mypy Errors | Black Issues  |
| --------------- | ------------------ | ----------- | ------------- |
| path/to/file.py | E501 (2), F401 (1) | 3 errors    | Not formatted |
| ...             | ...                | ...         | ...           |

#### 5. Security Audit

Per AGENTS.md Section Safety & Security:

| Check                        | Status | Evidence                        |
| ---------------------------- | ------ | ------------------------------- |
| No hardcoded secrets         | ✅/❌  | grep -r "password\|secret\|key" |
| .env.example exists          | ✅/❌  | File presence                   |
| No eval()/exec() usage       | ✅/❌  | grep results                    |
| No pickle without validation | ✅/❌  | grep results                    |
| Safe file I/O                | ✅/❌  | Path traversal check            |
| No SQL injection risk        | ✅/❌  | parameterized queries           |

#### 6. Refactoring Plan

Prioritized by hygiene impact:

**48 Hours** - CI/CD blockers:

- (List critical violations that would block merges)

**2 Weeks** - AGENTS.md compliance:

- (List systematic remediation tasks)

**6 Weeks** - Full hygiene graduation:

- (List long-term improvements)

#### 7. Diff-Style Suggestions

Provide ≥5 concrete hygiene fixes with before/after code examples.

---

## Mandatory Checks (Hygiene Specific)

### A. AGENTS.md Violations Hunt

For each standard in AGENTS.md, systematically check:

1. **Print Statements**:

   ```bash
   grep -rn "print(" --include="*.py" | grep -v "test_" | grep -v "#"
   ```

   - Document each occurrence
   - Propose logging replacement

2. **Wildcard Imports**:

   ```bash
   grep -rn "from .* import \*" --include="*.py"
   ```

   - Document each occurrence
   - List specific imports needed

3. **Bare Except Clauses**:

   ```bash
   grep -rn "except:" --include="*.py"
   ```

   - Document each occurrence
   - Propose specific exception types

4. **Missing Type Hints**:
   - Check public functions in each module
   - Document untyped interfaces

### B. Repository Organization Audit

Evaluate directory structure per AGENTS.md:

1. Does each tool have proper structure?

   ```
   tool_name/
   ├── __init__.py
   ├── main.py (or entry point)
   ├── README.md
   └── tests/
   ```

2. Are there orphaned files at root level?
3. Is there consistent naming (snake_case)?
4. Are there files that should be gitignored?

### C. Dependency Security Scan

1. Check for known vulnerabilities:

   ```bash
   pip-audit (if available)
   ```

2. Review dependency age and maintenance:
   - Last update dates
   - Known security advisories

3. Verify pinning strategy:
   - Are versions pinned appropriately?
   - Is there a requirements.txt or pyproject.toml?

### D. Git Hygiene

1. Are there large binary files committed?
2. Is `.gitignore` comprehensive?
3. Are there any accidentally committed secrets in history?
4. Is the commit history clean?

### E. Configuration File Audit

For each configuration file:

| File                      | Valid | Complete | Documented |
| ------------------------- | ----- | -------- | ---------- |
| `ruff.toml`               | ✅/❌ | ✅/❌    | ✅/❌      |
| `mypy.ini`                | ✅/❌ | ✅/❌    | ✅/❌      |
| `.pre-commit-config.yaml` | ✅/❌ | ✅/❌    | ✅/❌      |
| `pyproject.toml`          | ✅/❌ | ✅/❌    | ✅/❌      |

---

## Pragmatic Programmer Principles - Hygiene Focus

Apply these principles during assessment:

1. **Broken Windows Theory**: Identify first signs of code decay
2. **Design by Contract**: Verify pre/postconditions are documented
3. **Assertive Programming**: Check for appropriate assertions
4. **Topic Coupling**: Identify modules with too many connections
5. **Refactor Early, Refactor Often**: Recommend incremental improvements

---

## Output Format

Structure your review as follows:

```markdown
# Assessment B Results: Hygiene, Security & Quality

## Executive Summary

[5 bullets]

## Top 10 Hygiene Risks

[Numbered list with severity]

## Scorecard

[Table with scores and evidence]

## Linting Violation Inventory

[Comprehensive violation listing]

## Security Audit

[Security check results]

## AGENTS.md Compliance Report

[Standard-by-standard evaluation]

## Findings Table

[Detailed findings]

## Refactoring Plan

[Phased recommendations]

## Diff Suggestions

[Before/after code examples]

## Appendix: Files Requiring Attention

[Prioritized file list]
```

---

## Evaluation Criteria for Assessor

When conducting this assessment, prioritize:

1. **Security Issues** (30%): Any secrets, vulnerabilities, or unsafe patterns
2. **Linting Compliance** (25%): Ruff, Mypy, Black pass rates
3. **AGENTS.md Compliance** (25%): Adherence to established standards
4. **Organization Quality** (20%): Clean, maintainable structure

The goal is to achieve zero-violation status across all hygiene metrics.

---

_Assessment B focuses on hygiene and security. See Assessment A for architecture/implementation and Assessment C for documentation/integration._
