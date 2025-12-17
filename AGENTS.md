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

This section defines the active agents within the Jules "Control Tower" Architecture. All agents must operate within their defined scope.

### Core Orchestration Agents

#### 1. Jules-Control-Tower (Orchestrator)
**Role:** Air Traffic Controller
**Workflow:** `.github/workflows/Jules-Control-Tower.yml`
**Responsibilities:**
-  **Sole Trigger:** The only agent that listens to GitHub events (Push, PR, Schedule).
-  **Decision Maker:** Analyzes the event context (Triage) and dispatches the appropriate specialized worker.
-  **Loop Prevention:** Enforces `if: github.actor != 'jules-bot'` to prevent infinite recursion.

#### 2. Jules-Auto-Repair (Medic)
**Role:** Fixer of Broken Builds
**Workflow:** `.github/workflows/Jules-Auto-Repair.yml`
**Triggered By:** CI Failure (Standard CI)
**Capabilities:**
-  **Read:** CI Failure Logs
-  **Write:** Fixes to syntax, imports, and simple logic errors.
-  **Constraint:** Limited retries (max 2) to prevent "flailing".

### Quality Assurance Agents

#### 3. Jules-Test-Generator (Architect)
**Role:** Quality Assurance Engineer
**Workflow:** `.github/workflows/Jules-Test-Generator.yml`
**Triggered By:** New PR with `.py` changes
**Capabilities:**
-  **Write:** New test files in `tests/`.
-  **Constraint:** Must not modify existing application code, only add tests.

#### 4. Jules-Scientific-Auditor (The Professor)
**Role:** Peer Reviewer
**Workflow:** `.github/workflows/Jules-Scientific-Auditor.yml`
**Triggered By:** Nightly Schedule
**Capabilities:**
-  **Read-Only:** CANNOT modify code.
-  **Output:** Comments on PRs or Issues regarding mathematical correctness and physics fidelity.

#### 5. Jules-Review-Fix (Code Reviewer)
**Role:** Code Quality Reviewer
**Workflow:** `.github/workflows/Jules-Review-Fix.yml`
**Capabilities:**
-  **Read:** Code changes and PR content
-  **Write:** Review comments and suggestions

### Documentation & Maintenance Agents

#### 6. Jules-Documentation-Scribe (Librarian)
**Role:** Documentation Maintainer
**Workflow:** `.github/workflows/Jules-Documentation-Scribe.yml`
**Triggered By:** Push to `main`
**Capabilities:**
-  **Write:** Updates to `docs/` and markdown files.
-  **Mode:** "CodeWiki" - treats the codebase as a living encyclopedia.

#### 7. Jules-Archivist (Historian)
**Role:** Repository Organizer
**Workflow:** `.github/workflows/Jules-Archivist.yml`
**Capabilities:**
-  **Write:** Organizational improvements and file structure optimization

### Specialized Technical Agents

#### 8. Jules-Curie (Data Scientist)
**Role:** Scientific Computing Specialist
**Workflow:** `.github/workflows/Jules-Curie.yml`
**Capabilities:**
-  **Write:** Scientific computing improvements and data analysis enhancements

#### 9. Jules-Hypatia (Mathematician)
**Role:** Mathematical Modeling Expert
**Workflow:** `.github/workflows/Jules-Hypatia.yml`
**Capabilities:**
-  **Write:** Mathematical model improvements and algorithm optimizations

#### 10. Jules-Tech-Custodian (System Administrator)
**Role:** Infrastructure Maintainer
**Workflow:** `.github/workflows/Jules-Tech-Custodian.yml`
**Capabilities:**
-  **Write:** Infrastructure and configuration improvements

### Problem Resolution Agents

#### 11. Jules-Conflict-Fix (Diplomat)
**Role:** Merge Conflict Resolver
**Workflow:** `.github/workflows/Jules-Conflict-Fix.yml`
**Triggered By:** Manual dispatch or specific conflict events
**Capabilities:**
-  **Write:** Merge resolution commits.
-  **Constraint:** Prioritizes "Incoming" changes unless specified otherwise.

#### 12. Jules-Hotfix-Creator (Emergency Responder)
**Role:** Critical Issue Resolver
**Workflow:** `.github/workflows/Jules-Hotfix-Creator.yml`
**Capabilities:**
-  **Write:** Emergency fixes for critical issues

#### 13. Jules-Render-Healer (Graphics Specialist)
**Role:** Visualization and Rendering Expert
**Workflow:** `.github/workflows/Jules-Render-Healer.yml`
**Capabilities:**
-  **Write:** Graphics, visualization, and rendering improvements

### Supporting Infrastructure

#### Additional Workflows:
- `agent-metrics-dashboard.yml` - Agent performance monitoring
- `ci-failure-digest.yml` - CI failure analysis and reporting
- `ci-standard.yml` - Standard continuous integration
- `pr-auto-labeler.yml` - Automatic PR labeling
- `stale-cleanup.yml` - Stale issue and PR cleanup

---

## 🛠️ Quality Control Tools

The repository includes comprehensive quality control tools in the `tools/` directory:

### Core Quality Tools

#### 1. Code Quality Check (`tools/code_quality_check.py`)
**Purpose:** AI-generated code quality verification
**Features:**
- Validates code against project standards
- Color-coded terminal output for easy issue identification
- Automated quality checks for development workflow

#### 2. Scientific Auditor (`tools/scientific_auditor.py`)
**Purpose:** Scientific computation risk analysis
**Features:**
- Identifies potential numerical issues (division by zero, etc.)
- Analyzes mathematical operations for stability
- Prevents common scientific computing pitfalls

#### 3. MATLAB Quality Check (`tools/matlab_utilities/scripts/matlab_quality_check.py`)
**Purpose:** MATLAB-specific quality control
**Features:**
- Follows project's .cursorrules.md requirements
- Command-line and CI/CD integration support
- Comprehensive MATLAB code analysis

### Usage
```bash
# General code quality check
python tools/code_quality_check.py

# Scientific computation audit  
python tools/scientific_auditor.py

# MATLAB-specific quality check
python tools/matlab_utilities/scripts/matlab_quality_check.py
```