---
inclusion: always
---

# CI-Matched Local Testing Standards

## 🎯 Core Principle
**NEVER claim tests "pass locally" without running them in a CI-equivalent environment.**

## 📋 Mandatory Pre-Push Checklist

Before pushing ANY code changes, you MUST run these commands in the exact order shown:

### 1. Quality Gate Checks (from repository root)

```bash
# Navigate to repository root
cd /path/to/repository

# Run Ruff with exact CI configuration
ruff check .

# Run Black with exact CI configuration  
black --check .

# Run MyPy (if applicable to changed files)
mypy . --ignore-missing-imports
```

### 2. Test Execution (matching CI environment)

```bash
# Run tests with proper PYTHONPATH
PYTHONPATH=. python -m pytest tests/ -v
```

### 3. Verification Commands

```bash
# Verify no uncommitted formatting changes
git diff --exit-code

# Verify all files are properly formatted
ruff check . && black --check .
```

## 🔧 Tool Version Verification

Before ANY testing session, verify tool versions match CI:

```bash
# Check versions
ruff --version    # Must be: 0.14.10
black --version   # Must be: 25.12.0
python --version  # Prefer: 3.11.x (CI uses 3.11)

# If versions don't match, note the discrepancy in your response
```

## 🚫 Prohibited Statements

**NEVER say:**
- ❌ "All tests pass locally"
- ❌ "Ruff/Black checks pass"
- ❌ "Everything looks good locally"

**INSTEAD say:**
- ✅ "Ran CI-equivalent checks: [list specific commands and results]"
- ✅ "Verified with exact CI configuration: [show output]"

## 🔄 CI Configuration Reference

Current CI configuration:
- **Python Version**: 3.11
- **Ruff Version**: 0.14.10
- **Black Version**: 25.12.0
- **MyPy Version**: 1.13.0
- **Environment**: Ubuntu latest (Linux)
