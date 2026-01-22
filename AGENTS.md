# AGENTS.md

## 🤖 Agent Personas & Directives

**Audience:** This document is the authoritative guide for AI agents working in this repository.

**Core Mission:**

- Write high-quality, maintainable, and secure code.
- Adhere strictly to the project's architectural and stylistic standards.
- Act as a responsible pair programmer, always verifying assumptions and testing changes.

---

## 🛡️ Safety & Security (CRITICAL)

1. **Secrets Management**:
   - **NEVER** commit API keys, passwords, tokens, or database connection strings.
   - Use `.env` files and `python-dotenv` for secrets.
   - Create `.env.example` templates for required environment variables.
2. **Code Review**:
   - Review all generated code for security vulnerabilities (SQL injection, unsafe file I/O, etc.).
   - Do not accept code you do not understand.
3. **Data Protection**:
   - Do not commit large binary files (>50MB) or personal data.

---

## 🐍 Python Coding Standards

### 1. Code Quality & Style

- **Logging vs. Print**:
  - ❌ **DO NOT** use `print()` statements for application output.
  - ✅ **USE** the `logging` module.
  - _Example_: `logger.info("Processing complete")` instead of `print("Processing complete")`.
- **Imports**:
  - ❌ **NO** wildcard imports (`from module import *`).
  - ✅ **Explicitly** import required classes/functions.
- **Exception Handling**:
  - ❌ **NO** bare `except:` clauses.
  - ✅ **Catch specific exceptions** (e.g., `except ValueError:`) or at least `except Exception:`.
- **Type Hinting**:
  - Use Python type hints for function arguments and return values.

### 2. Project Structure

```
project_name/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── src/
│   └── project_name/
│       ├── __init__.py
│       └── main.py
└── tests/
```

### 3. Testing

- Use `unittest` or `pytest`.
- Write unit tests for individual functions and integration tests for workflows.

---

## 🔢 MATLAB Coding Standards

### 1. Structure

```
matlab_project/
├── main.m
├── src/
│   ├── functions/
│   └── classes/
└── tests/
```

### 2. Best Practices

- Use clear comment blocks for function documentation.
- Avoid `.asv` and `.m~` files in commits (add to `.gitignore`).
- Use `functiontests` for testing.

---

## 🔄 Git Workflow & Version Control

### 1. Commit Messages

Use **Conventional Commits** format:

- `feat(scope): description` (New feature)
- `fix(scope): description` (Bug fix)
- `docs(scope): description` (Documentation)
- `style(scope): description` (Formatting)
- `refactor(scope): description` (Code restructuring)
- `test(scope): description` (Adding tests)
- `chore(scope): description` (Maintenance)

### 2. Branching Strategy

- `main`: Production-ready code.
- `develop`: Integration branch.
- `feature/name`: New features.
- `hotfix/name`: Critical bug fixes.

---

## 📝 Documentation

- **README.md**: Every project must have a README with Description, Installation, and Usage sections.
- **Docstrings**: Use Google or NumPy style docstrings for Python.
- **Comments**: Explain _why_, not just _what_.

---

## 🌐 Web Development Standards (HTML/CSS/JS)

### 1. HTML

- **Semantic HTML**: Use `<header>`, `<nav>`, `<main>`, `<footer>`, `<article>`, `<section>` appropriately.
- **Accessibility**: Ensure all `<img>` tags have `alt` attributes. Use ARIA labels where necessary.
- **Structure**: Maintain a clean and indented structure.

### 2. CSS

- **Naming Convention**: Use **BEM** (Block Element Modifier) for class names where possible (e.g., `.card__title--large`).
- **Responsiveness**: Design **Mobile-First**. Use media queries to adapt to larger screens.
- **Linting**: Use `stylelint` with standard config.
  - Avoid ID selectors for styling.
  - Avoid `!important`.

### 3. JavaScript

- **Modern Syntax**: Use ES6+ features (arrow functions, template literals, destructuring).
- **Variables**: Use `const` by default, `let` if reassignment is needed. ❌ **NEVER** use `var`.
- **Async/Await**: Prefer `async/await` over raw Promises/callbacks.
- **Linting**: Use `eslint`.
- **Equality**: Always use strict equality `===` and `!==`.

---

## ⚙️ C++ Coding Standards

### 1. Style Guide

- Follow the **Google C++ Style Guide**.
- **Formatting**: Use `clang-format`.
  - Indent width: 4 spaces (as seen in `.clang-format`).
  - Column limit: 0 (no hard limit, but keep it readable).
  - Brace wrapping: Allman style (braces on new line) is configured in some repos, but consistency within the specific repo is key.

### 2. Modern C++

- Use **C++11/14/17** features.
- **Memory Management**:
  - ❌ **Avoid** raw pointers (`new`/`delete`).
  - ✅ **Use** smart pointers: `std::unique_ptr` for exclusive ownership, `std::shared_ptr` for shared ownership.
- **RAII**: Use Resource Acquisition Is Initialization for resource management.

### 3. Safety

- Avoid C-style casts; use `static_cast`, `dynamic_cast`, etc.
- Initialize all variables upon declaration.

---

## 🚨 Emergency Procedures

If sensitive data is accidentally committed:

1.  **Stop** immediately.
2.  Use `git filter-branch` or BFG Repo-Cleaner to remove the file from history.
3.  Force push only if necessary and coordinated with the team.

---

## 🏗️ System Architecture & Agent Roles

**Reference:** [JULES_ARCHITECTURE.md](JULES_ARCHITECTURE.md)

This section defines the active agents within the Jules "Control Tower" Architecture. All agents must operate within their defined scope.

### 1. The Control Tower (Orchestrator)
**Role:** Air Traffic Controller
**Workflow:** `.github/workflows/Jules-Control-Tower.yml`
**Responsibilities:**
-  **Orchestrator:** Coordinates specialized agent workflows. Note that CI and Guard workflows run independently.
-  **Decision Maker:** Analyzes the event context (Triage) and dispatches the appropriate specialized worker.
-  **Loop Prevention:** Enforces `if: github.actor != 'jules-bot'` to prevent infinite recursion.

### 2. Auto-Repair (Medic)
**Role:** Fixer of Broken Builds
**Workflow:** `.github/workflows/Jules-Auto-Repair.yml`
**Triggered By:** CI Failure (Standard CI)
**Capabilities:**
-  **Read:** CI Failure Logs
-  **Write:** Fixes to syntax, imports, and simple logic errors.
-  **Constraint:** limited retries (max 3) to prevent "flailing".

### 3. Test-Generator (Architect)
**Role:** Quality Assurance Engineer
**Workflow:** `.github/workflows/Jules-Test-Generator.yml`
**Triggered By:** New PR with `.py` changes
**Capabilities:**
-  **Write:** New test files in `tests/`.
-  **Constraint:** Must not modify existing application code, only add tests.

### 4. Doc-Scribe (Librarian)
**Role:** Documentation Maintainer
**Workflow:** `.github/workflows/Jules-Documentation-Scribe.yml`
**Triggered By:** Push to `main`
**Capabilities:**
-  **Write:** Updates to `docs/` and markdown files.
-  **Mode:** "CodeWiki" - treats the codebase as a living encyclopedia.

### 5. Scientific-Auditor (The Professor)
**Role:** Peer Reviewer
**Workflow:** `.github/workflows/Jules-Scientific-Auditor.yml`
**Triggered By:** Nightly Schedule
**Capabilities:**
- **Read/Write:** Analyzes mathematical correctness; can commit reports to `docs/assessments/` or open GitHub Issues.
- **Justification:** This agent requires limited write access only to publish audit artifacts (reports and issues) so that all mathematical reviews are transparent, reproducible, and traceable over time. It remains strictly read-only with respect to source, configuration, and test code.
- **Constraints:** MUST NOT modify application source code, configuration files, or tests; MAY ONLY create or update files under `docs/assessments/` and open or comment on GitHub Issues/PRs to recommend changes.

### 6. Conflict-Fix (Diplomat)
**Role:** Merge Conflict Resolver
**Workflow:** `.github/workflows/Jules-Conflict-Fix.yml`
**Triggered By:** Manual dispatch or specific conflict events (if configured)
**Capabilities:**
-  **Write:** Merge resolution commits.
-  **Constraint:** Prioritizes "Incoming" changes unless specified otherwise.

---

## 🛠️ GitHub CLI & Workflow Reference

Always use Github CLI for making pull requests. 
Whenever you finish a task for the user, push it to remote. 
NEVER try to use GitKraken or anything other than Github CLI for Pull request creation. 
All pull requests should be verified to pass the ruff, black, and mypy requirements in the ci / cd pipeline before they are created.

### For PR Creation:
- Always check if PR already exists first using `gh pr list --state open`
- Use simple, concise titles and descriptions for initial creation
- Wrap GitHub CLI commands in powershell `-Command "..."`
- Use single quotes inside double quotes for string parameters

### For PR Management:
- Use `gh pr view [number]` to get PR details and status
- Use `gh pr checks [number]` to see CI/CD status
- Use `gh run list --branch [branch-name]` to see workflow runs
- Check for failing checks and address them systematically

### For CI/CD Issue Resolution:
- Identify failing checks using `gh pr checks`
- Examine workflow run logs using `gh run view [run-id]`
- Make fixes on the same branch and push to update the PR
- Verify fixes by checking updated CI status

### Command Templates for Future Use:

```bash
# Create PR:
powershell -Command "gh pr create --title 'Your Title' --body 'Your description'"

# Check PR status:
powershell -Command "gh pr view [PR_NUMBER]"

# Check CI/CD status:
powershell -Command "gh pr checks [PR_NUMBER]"

# List recent runs:
powershell -Command "gh run list --branch [BRANCH_NAME] --limit 5"

# View specific run:
powershell -Command "gh run view [RUN_ID]"
```

---

## 🔍 Pre-Commit Quality Checks (MANDATORY)

### Before Creating ANY PR:

**CRITICAL**: All code MUST pass linting checks locally before pushing. Failing to do so wastes CI resources and blocks PRs.

```bash
# Python files - run ALL of these before committing:
ruff check .                    # Linting errors
ruff check --fix .              # Auto-fix what can be fixed
ruff format .                   # Format code
black .                         # Additional formatting
mypy .                          # Type checking (if configured)

# Verify no issues remain:
ruff check . && echo "✓ All checks passed"
```

### Common Python Linting Issues to Avoid:

1. **Trailing whitespace on blank lines** (W293) - Use editor setting to strip trailing whitespace
2. **Unsorted imports** (I001) - Run `ruff check --fix` to auto-sort
3. **Line too long** (E501) - Break long lines, especially in data structures
4. **Missing type hints** - Add type annotations to function signatures

### Workflow/YAML Validation:

Before modifying GitHub Actions workflows, validate syntax:

```bash
# Check YAML syntax (requires yq or python-yaml)
python -c "import yaml; yaml.safe_load(open('.github/workflows/your-workflow.yml'))"

# Or use actionlint if available
actionlint .github/workflows/
```

---

## ⚠️ Shell Scripting in Workflows (CRITICAL)

### Common Pitfalls to Avoid:

1. **Unquoted variables with spaces**:
   ```bash
   # ❌ WRONG - breaks if TARGET contains spaces
   basename $TARGET

   # ✅ CORRECT - always quote variables
   basename "$TARGET"
   ```

2. **jq null coalescing operator**:
   ```bash
   # ❌ WRONG - // gets misinterpreted by shell
   jq 'first // "default"'

   # ✅ CORRECT - use if-then-else instead
   jq 'first | if . == null then "default" else . end'
   ```

3. **Heredocs in YAML**:
   ```yaml
   # ✅ CORRECT - use literal block scalar for multi-line
   run: |
     cat << 'EOF'
     Content here
     EOF
   ```

### Testing Workflow Changes:

Before pushing workflow changes:

1. **Validate YAML syntax** locally
2. **Test shell commands** in isolation
3. **Check for unquoted variables** that might contain spaces
4. **Review jq expressions** for shell quoting issues

### Reference Documentation:

See `Repository_Management/workflow-fixes/` for documented fixes and patterns to avoid.

---


### 🔄 Workflow & Automation Governance

Agents must refer to the [Workflow Tracking Document](docs/workflows/WORKFLOW_TRACKING.md) to understand available tools.
All workflows follow the Governing Workflow Guidance documented in the `Repository_Management` repository (see `docs/architecture/WORKFLOW_GOVERNANCE.md` in that repository).
The **GitHub Issue Tracker** is the primary authority for tasking and gap remediation. Check existing issues before starting work.

---


### 📂 Repository Decluttering & Organization
To maintain a clean repository root, all development-related documentation (summaries, plans, analysis reports, technical debt assessments, etc.) MUST be stored in the `docs/development/` directory. 
- **DO NOT** create new `.md` files in the root unless they are critical project-wide files (e.g., README, AGENTS, CHANGELOG).
- Prefer creating issues for task tracking rather than temporary markdown files.
