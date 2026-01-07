# Repository Safety Guardrails Guidelines

This document defines a **universal standard** for implementing safety guardrails across any repository.
It is intended for use by humans and IDE AI assistants to verify whether a project meets safety, linting,
and continuous integration (CI) quality goals.

---

## 1. Purpose

The goal of these guardrails is to:
- **Catch coding errors early** via automated linting, type-checking, and formatting.
- **Prevent low-quality or broken code** from being merged into protected branches.
- **Automate enforcement** using `pre-commit` hooks and CI status checks.
- **Standardize style and error handling** across all maintained code.

---

## 2. Required Tooling

All repositories implementing guardrails must use:

| Tool | Purpose |
|------|---------|
| **pre-commit** | Framework for running checks automatically before each commit |
| **ruff** | Fast Python linter and formatter |
| **mypy** | Static type checker for Python |
| **black** (optional) | Python code formatter (if not using ruff's formatter) |
| **pytest** | Test runner for automated unit/integration tests |
| **GitHub Actions CI** | Runs the guardrail checks on all pushes and PRs |

---

## 3. File Exclusions

Guardrails **must not block development** due to errors in:
- Legacy or archived code
- Experimental or prototype directories
- Third-party libraries checked into the repo
- Generated files

Use **consistent exclusions** in all configs (`.pre-commit-config.yaml`, `ruff.toml`, `mypy.ini`) to ignore these paths.

**Example Exclusion Patterns**:
```
^(Archive/|legacy/|old_code/|experimental/|.*Python Version/|.*Motion Capture Plotter/)
```

---

## 4. Configuration Files

Below are minimal working examples of each required config.

### 4.1 `.pre-commit-config.yaml`
```yaml
repos:
  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.4.8
    hooks:
      - id: ruff
        args: ["--fix"]
        exclude: '^(Archive/|legacy/|experimental/)'
      - id: ruff-format
        exclude: '^(Archive/|legacy/|experimental/)'

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
        exclude: '^(Archive/|legacy/|experimental/)'

default_language_version:
  python: python3.11
```

### 4.2 `ruff.toml`
```toml
target-version = "py311"
extend-exclude = [
    "Archive/",
    "legacy/",
    "experimental/"
]
fix = true
line-length = 100
```

### 4.3 `mypy.ini`
```ini
[mypy]
python_version = 3.11
ignore_missing_imports = true
exclude = (?x)(
    ^Archive/|
    ^legacy/|
    ^experimental/
)

[mypy-tests.*]
ignore_errors = true
```

---

## 5. CI Integration

Guardrails are **only effective** if enforced in CI.

### 5.1 Unified CI Workflow

This repository uses a **unified CI/CD workflow** that implements all best practices for quality checks, security scanning, and documentation validation. The workflow is defined in `.github/workflows/pr-quality-check.yml` and includes:

- **Quality checks**: Ruff linting, MyPy type checking, Black formatting, placeholder detection
- **Testing**: Multi-version Python matrix (3.9, 3.10, 3.11) with coverage reporting
- **Security**: Bandit security scanning with test directory exclusions
- **Documentation**: Pydocstyle docstring validation

**Key Features**:
- Pinned tool versions for reproducibility
- Comprehensive source directory detection
- Conditional coverage file uploads
- Proper error handling with `continue-on-error`

For implementation details, see `Repository_Management/UNIFIED_CI_APPROACH.md`.

**Setup Notes**:
1. This workflow must run **on every push and PR**.
2. In GitHub branch protection rules, require this workflow to pass before merging.
3. The workflow supports replicant branches (add to trigger branches as needed).

---

## 6. Implementation Steps

1. **Install pre-commit hooks locally**
   ```bash
   pip install pre-commit
   pre-commit install
   ```
2. **Run checks manually before committing**
   ```bash
   pre-commit run --all-files
   ```
3. **Fix any reported errors** or update exclusion patterns.
4. **Push branch and open a PR**.
5. **Verify CI passes** before merging.

---

## 7. Completion Checklist

✅ `.pre-commit-config.yaml` present with `ruff` + `mypy` + exclusions
✅ `ruff.toml` present with matching exclusions
✅ `mypy.ini` present with matching exclusions
✅ GitHub Actions CI workflow present and passing
✅ Branch protection rules require CI to pass
✅ All maintained source files pass `pre-commit` locally and in CI
✅ Legacy, experimental, and third-party code excluded from checks

---

## 8. Final Notes

- Avoid **bypassing** guardrails unless absolutely necessary (`--no-verify` should be rare).
- Exclusion lists should be **minimal** and periodically reviewed.
- CI should reflect **exactly** what runs locally in `pre-commit` to avoid drift.

---
