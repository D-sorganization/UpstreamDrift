# Configuration Issues - Quick Reference

## Critical Issues (Fix Immediately)

### 1. MyPy Version Mismatch
| Location | Current | Should Be | Impact |
|----------|---------|-----------|--------|
| `.github/workflows/ci-standard.yml` | `mypy==1.13.0` | `mypy==1.19.1` | Type checking uses different version than local dev |
| `requirements-dev.lock` | `mypy==1.19.1` | - | Lock file is correct |

**Action:** Update ci-standard.yml line 35

### 2. NumPy Version Constraint Conflict
| File | Constraint | Status |
|------|-----------|--------|
| `pyproject.toml` | `numpy>=1.26.4,<3.0.0` | Too loose |
| `environment.yml` | `numpy>=1.26.4,<2.0.0` | Correct |
| `requirements.txt` | `numpy>=2.0.1` | CONFLICTS with environment.yml |
| `requirements.lock` | `numpy==1.26.4` | Correct |

**Action:** Update requirements.txt to match environment.yml `>=1.26.4,<2.0.0`

### 3. Unexpected Flask Dependency
| File | Content | Status |
|------|---------|--------|
| `requirements.txt` line 25 | `flask>=3.0.0` | ❌ Flask not in pyproject.toml |

**Cause:** Likely copy-paste from template
**Action:** Verify if Flask is needed; if not, remove immediately

### 4. Duplicate Configuration Files
| File | Conflicts With | Status |
|------|---|---|
| `mypy.ini` | `pyproject.toml [tool.mypy]` | ⚠️ Both exist, mypy uses .ini |
| `pytest_improvements.ini` | `pyproject.toml [tool.pytest.ini_options]` | ⚠️ Both exist, duplicate markers |

**Action:** Delete both .ini files entirely

---

## Important Issues (Fix This Sprint)

### 5. Port Configuration Mismatch
| Component | Port | Status |
|-----------|------|--------|
| `Dockerfile` | EXPOSE 8000 | Correct |
| `Dockerfile.unified` | Health check localhost:8000 | Correct |
| `docker-compose.yml` backend | 8001:8001 | ❌ Mismatch |
| `docker-compose.yml` health check | localhost:8001 | Inconsistent |

**Action:** Standardize on 8000 everywhere OR document which is correct

### 6. Python Version Build Mismatch
| File | Python Version |
|------|---|
| `Dockerfile` line 25 | `python=3.12` |
| `environment.yml` line 34 | `python=3.11` |

**Action:** Align to 3.11 (environment.yml is authoritative)

### 7. Missing pytest in CI
| Location | Issue | Status |
|----------|-------|--------|
| `ci-standard.yml` | No pytest job | Tests never run in CI! |
| `Makefile` line 57-58 | pytest exists locally | Only runs if you run `make test` |

**Action:** Add pytest job to ci-standard.yml

### 8. Ruff Version Constraint Too Loose
| File | Constraint | Should Be |
|------|-----------|-----------|
| `pyproject.toml` | `ruff>=0.1.0` | `ruff==0.14.10` |
| `.pre-commit-config.yaml` | `ruff==0.14.10` | Correct |
| `.github/workflows/ci-standard.yml` | (installs from dependencies) | Would use wrong version |

**Action:** Pin ruff in pyproject.toml to 0.14.10

---

## Configuration Files Quick Guide

### What Each Does (Authoritative Source)

| File | Purpose | Authoritative? |
|------|---------|---|
| `pyproject.toml` | Project metadata + dependencies | ✅ YES (modern standard) |
| `requirements.txt` | Pip requirements (manual) | ⚠️ LEGACY (for pip install compatibility) |
| `requirements.lock` | Pinned prod dependencies | ✅ Generated (keep in sync) |
| `requirements-dev.lock` | Pinned dev dependencies | ✅ Generated (keep in sync) |
| `environment.yml` | Conda environment | ✅ Alternative to pip |
| `setup.py` | Legacy setuptools | ❌ MINIMAL WRAPPER ONLY |
| `mypy.ini` | MyPy config | ❌ DELETE (use pyproject.toml) |
| `pytest_improvements.ini` | Pytest config | ❌ DELETE (use pyproject.toml) |
| `.pre-commit-config.yaml` | Git hooks | ✅ Active + correct |
| `docker-compose.yml` | Local dev containers | ⚠️ Inconsistent with Dockerfile |

### Configuration Consistency Verification

```bash
# Check if configurations align
grep "numpy" pyproject.toml requirements.txt environment.yml requirements.lock
grep "ruff" pyproject.toml .pre-commit-config.yaml .github/workflows/ci-standard.yml
grep "mypy" requirements-dev.lock .github/workflows/ci-standard.yml
```

---

## Redundant Files to Remove

```
❌ mypy.ini                    (duplicate of pyproject.toml [tool.mypy])
❌ pytest_improvements.ini     (duplicate of pyproject.toml [tool.pytest.ini_options])
❌ requirements.txt            (should use requirements.lock instead)
⚠️  One of: Dockerfile vs Dockerfile.unified (choose which is primary)
```

---

## Testing Configuration State

```bash
# Check mypy version inconsistency
pip show mypy | grep Version
cat requirements-dev.lock | grep mypy==
grep "mypy==" .github/workflows/ci-standard.yml

# Check if pytest runs in CI
grep -A 20 "quality-gate:" .github/workflows/ci-standard.yml | grep pytest

# Check for duplicate configs
ls -la mypy.ini pytest_improvements.ini 2>/dev/null && echo "REMOVE THESE"

# Validate CI jobs
grep "runs-on:" .github/workflows/ci-standard.yml
grep "pytest\|test" .github/workflows/ci-standard.yml
```

---

## Changes by Team

### DevOps/Platform Team
1. Fix mypy version in CI (1.13.0 → 1.19.1)
2. Add pytest job to ci-standard.yml
3. Fix numpy version constraints
4. Align Docker ports
5. Choose authoritative Dockerfile

### Build/Python Team
1. Delete mypy.ini and pytest_improvements.ini
2. Pin ruff in pyproject.toml
3. Verify Flask dependency (delete if unintended)
4. Update Makefile lint target to include mypy

### Documentation Team
1. Update README with correct setup steps
2. Clarify Dockerfile usage (which is for what)
3. Document dependency management strategy
4. Add troubleshooting section for "works locally, fails in CI"

---

## Testing These Fixes

```bash
# After fixing mypy version
python -m pytest tests/unit/ -v  # Should use 1.19.1

# After fixing numpy versions
pip install -r requirements-dev.lock  # Should resolve cleanly

# After removing duplicate configs
mypy . --config-file pyproject.toml  # Should read from pyproject.toml

# After updating ruff
ruff check .  # Should use 0.14.10

# After adding pytest to CI (test locally first)
python -m pytest tests/ -v -m "not slow"
```

---

## Related Files by Category

### Build System
- `pyproject.toml` - **Source of truth for dependencies**
- `build_hooks.py` - UI bundling during wheel build
- `setup.py` - Legacy wrapper
- `Makefile` - Development convenience targets

### Dependency Management
- `requirements.txt` - Should remove
- `requirements.lock` - Keep (prod deps)
- `requirements-dev.lock` - Keep (dev deps)
- `environment.yml` - Alternative for conda

### CI/CD
- `.github/workflows/ci-standard.yml` - Primary CI (has bugs)
- `.pre-commit-config.yaml` - Git hooks (mostly correct)
- `.github/workflows/nightly-cross-engine.yml` - Extended tests

### Docker
- `Dockerfile` - Builder + runtime + training stages
- `Dockerfile.unified` - Unified with micromamba
- `docker-compose.yml` - Local dev setup (port mismatch)

### Configuration Tools
- `pyproject.toml [tool.ruff]` - Linter rules
- `pyproject.toml [tool.mypy]` - Type checker (authoritative)
- `pyproject.toml [tool.pytest.ini_options]` - Test runner (authoritative)
- `pyproject.toml [tool.black]` - Formatter (reference only)

---

## Before/After Checklist

### After Fixes
- [ ] All version numbers consistent across files
- [ ] No duplicate config files
- [ ] pytest runs in CI
- [ ] Docker ports aligned
- [ ] mypy uses same version locally and in CI
- [ ] Lock files regenerated and committed
- [ ] Makefile updated
- [ ] No Flask in requirements.txt
- [ ] All tests pass locally and in CI
- [ ] Documentation updated

---

**Last Updated:** March 7, 2026
**Review Tool:** Comprehensive infrastructure audit
