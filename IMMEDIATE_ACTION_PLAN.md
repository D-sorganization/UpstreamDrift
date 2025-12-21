# Golf Modeling Suite - Unified Immediate Action Plan
**Created:** December 20, 2025  
**Based On:** Unified Professional Assessment v1.0  
**Priority:** CRITICAL - Complete before next release

---

## Quick Reference: Your Current State

**Dual Rating Explanation:**
- **Production Deployment:** 5.5/10 (NOT READY - 6 blockers)
- **Research Platform:** B+ (82/100) (READY for beta use)

**The Difference:**
- Production view → Security-first, polish-critical
- Research view → Architecture-first, capability-focused
- **Both are correct** for their contexts

**Your Target:** Get to production-ready v1.0 in **6 months**

---

## 🔴 PHASE 1: Critical Blockers (1-2 Days)

### Must complete before ANY public release

| Priority | Task | Time | Blocker Level |
|----------|------|------|---------------|
| #1 | Remove junk files | 5 min | CRITICAL |
| #2 | Fix TODO violation | 30 min | HIGH |
| #3 | Security audit (eval/exec) | 4 hrs | CRITICAL |
| #4 | Fix Docker root user | 1 hr | MEDIUM |
| #5 | Add dependency bounds | 1 hr | MEDIUM |

**Total Phase 1 Time: ~7 hours (~1 work day)**

---

## Task #1: Remove Junk Files (5 minutes)

### The Problem
```bash
$ ls -la =*
-rw-r--r-- 1 diete 197609 0 Dec 20 15:48 =1.0.14
-rw-r--r-- 1 diete 197609 0 Dec 20 15:48 =1.3.0
-rw-r--r-- 1 diete 197609 0 Dec 20 15:48 =10.0.0
# ... 9 total files
```

**Root Cause:** Malformed pip: `pip install package == version` (extra space)

### The Fix
```bash
cd c:/Users/diete/Repositories/Golf_Modeling_Suite

# Remove junk files
rm =1.0.14 =1.3.0 =10.0.0 =2.31.0 =2.6.0 =3.0.0 =3.9.0 =4.8.0 =6.6.0

# Update .gitignore to prevent recurrence
echo -e "\n# Prevent malformed pip artifacts\n=*" >> .gitignore

# Commit
git add -u
git add .gitignore
git commit -m "Remove pip installation artifact files and prevent recurrence"
```

### Why This Matters
**This single issue drops your rating from B+ to 5.5/10** in production context
- First impression for evaluators
- Indicates lack of quality control
- May confuse file searches
- Unprofessional appearance

---

## Task #2: Fix TODO Violation (30 minutes)

### The Problem
**File:** `launchers/unified_launcher.py:103`
```python
# TODO: Read from version file or package metadata
return "1.0.0-beta"
```

**Violations:**
1. Your coding rules ban `TODO` in production code
2. Hardcoded version creates drift from `pyproject.toml`

### The Fix

**Option 1: Quick Fix (5 min)**
```python
def get_version(self) -> str:
    """Get suite version.
    
    Note: Version defined in pyproject.toml and shared/__init__.py.
    Using shared package version to avoid pkg_resources dependency.
    """
    import sys
    from pathlib import Path
    
    # Add shared to path
    shared_path = Path(__file__).parent.parent / "shared" / "python"
    if str(shared_path) not in sys.path:
        sys.path.insert(0, str(shared_path))
    
    try:
        import shared
        return shared.__version__
    except (ImportError, AttributeError):
        return "1.0.0-beta"  # Fallback if not properly installed
```

**Option 2: Proper Fix (30 min)**
```python
def get_version(self) -> str:
    """Get suite version from package metadata.
    
    Returns:
        Version string (e.g., "1.0.0-beta")
        
    Note:
        Primary source: Package metadata (installed package)
        Fallback: shared.__version__ (development mode)
        Last resort: Hardcoded default
    """
    # Try package metadata first (installed package)
    try:
        from importlib.metadata import version
        return version("golf-modeling-suite")
    except Exception:
        pass
    
    # Try shared package (development mode)
    try:
        from shared import __version__
        return __version__
    except (ImportError, AttributeError):
        pass
    
    # Last resort fallback
    return "1.0.0-beta"
```

### Verification
```bash
# Test that version is correct
python -c "from launchers.unified_launcher import UnifiedLauncher; launcher = UnifiedLauncher(); print(launcher.get_version())"
# Should output: 1.0.0
```

---

## Task #3: Security Audit - Eval/Exec (4 hours)

### The Problem
**Your Coding Rules Violation:**
> "Security: NO eval() or exec()"

**Files Affected:** Multiple launchers contain `eval()/exec()` calls

### Discovery Phase (30 min)

```bash
# Find all eval/exec usage
cd c:/Users/diete/Repositories/Golf_Modeling_Suite

grep -rn "eval(" --include="*.py" launchers/
grep -rn "exec(" --include="*.py" launchers/

# Create audit report
echo "# Eval/Exec Security Audit" > SECURITY_AUDIT.md
echo "Date: $(date)" >> SECURITY_AUDIT.md
echo "" >> SECURITY_AUDIT.md
echo "## Found Usage:" >> SECURITY_AUDIT.md
grep -rn "eval(" --include="*.py" launchers/ >> SECURITY_AUDIT.md
grep -rn "exec(" --include="*.py" launchers/ >> SECURITY_AUDIT.md
```

### Refactoring Pattern (3 hours)

**For Each Instance:**

**BEFORE (Security Risk):**
```python
# Dynamic signal connection - INSECURE
def connect_signals(self):
    for widget_name, handler_name in self.connections.items():
        exec(f"self.{widget_name}.clicked.connect(self.{handler_name})")
```

**AFTER (Secure):**
```python
# Explicit mapping - SECURE
def connect_signals(self):
    # Map widget names to actual widget objects
    widget_map = {
        "run_button": self.run_button,
        "stop_button": self.stop_button,
        "reset_button": self.reset_button,
        "configure_button": self.configure_button,
    }
    
    # Map handler names to actual methods
    handler_map = {
        "on_run": self.on_run,
        "on_stop": self.on_stop,
        "on_reset": self.on_reset,
        "on_configure": self.on_configure,
    }
    
    # Connect explicitly
    for widget_name, handler_name in self.connections.items():
        widget = widget_map.get(widget_name)
        handler = handler_map.get(handler_name)
        
        if widget is None:
            raise ValueError(f"Unknown widget: {widget_name}")
        if handler is None:
            raise ValueError(f"Unknown handler: {handler_name}")
        
        widget.clicked.connect(handler)
```

**For Expression Parsing:**
```python
# BEFORE (eval risk)
result = eval(user_input)

# AFTER (safe)
import ast
try:
    result = ast.literal_eval(user_input)  # Only literals
except (ValueError, SyntaxError) as e:
    raise ValueError(f"Invalid expression: {user_input}") from e
```

### Add Security Scanning (30 min)

**Create `.github/workflows/security.yml`:**
```yaml
name: Security Scan

on:
  push:
    branches: [main, master]
  pull_request:
    branches: [main, master]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install security tools
        run: |
          pip install bandit safety
      
      - name: Run Bandit (code security)
        run: |
          bandit -r shared/ launchers/ -ll
      
      - name: Run Safety (dependency vulnerabilities)
        run: |
          safety check --json
```

### Verification
```bash
# Manually run security checks
pip install bandit safety

# Check for security issues
bandit -r launchers/ shared/ -ll

# Should show: No issues found (after fixes)
```

---

## Task #4: Fix Docker Root User (1 hour)

### The Problem
Containers run as root by default (security risk)

### The Fix

**File:** `engines/physics_engines/mujoco/Dockerfile` (and others)

**BEFORE:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

CMD ["python", "main.py"]
```

**AFTER:**
```dockerfile
FROM python:3.11-slim

# Security: Create non-root user
RUN groupadd -r golfuser && \
    useradd -r -g golfuser -u 1000 golfuser && \
    mkdir -p /app /home/golfuser && \
    chown -R golfuser:golfuser /app /home/golfuser

WORKDIR /app

# Install dependencies as root (needed for apt/pip)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY --chown=golfuser:golfuser . .

# Switch to non-root user
USER golfuser

CMD ["python", "main.py"]
```

### Apply to All Dockerfiles
```bash
# Find all Dockerfiles
find . -name "Dockerfile" -type f

# Apply pattern to each:
# - engines/physics_engines/mujoco/Dockerfile
# - engines/physics_engines/drake/Dockerfile  
# - engines/physics_engines/pinocchio/Dockerfile
# - Root Dockerfile (if exists)
```

---

## Task #5: Add Dependency Bounds (1 hour)

### The Problem
**File:** `pyproject.toml`

**Unbounded Dependencies:**
```toml
sympy>=1.12                # ⚠️ No upper bound
mujoco>=3.2.3              # ⚠️ No upper bound
pathlib2>=2.3.6            # ❌ Unnecessary for Python 3.11+

[engines]
pin>=2.6.0                 # ⚠️ No upper bound
```

### The Fix

**Edit `pyproject.toml`:**
```toml
dependencies = [
    "numpy>=1.26.4,<3.0.0",
    "pandas>=2.2.3,<3.0.0",
    "matplotlib>=3.8.0,<4.0.0",
    "scipy>=1.13.1,<2.0.0",
    "sympy>=1.12,<2.0.0",          # ✅ Added upper bound
    "PyQt6>=6.6.0,<7.0.0",
    "mujoco>=3.2.3,<4.0.0",        # ✅ Added upper bound
    "pyyaml>=6.0.1",
    # "pathlib2>=2.3.6",          # ❌ REMOVED (built-in in Python 3.11+)
    "defusedxml>=0.7.1",
]

[project.optional-dependencies]
engines = [
    "drake>=1.22.0,<2.0.0",
    "pin>=2.6.0,<3.0.0",           # ✅ Added upper bound
    "pin-pink>=1.0.0,<4.0.0",
    "qpsolvers>=1.10.0,<5.0.0",
    "osqp>=0.6.0,<2.0.0",
    "meshcat>=0.0.18",
]
```

### Verification
```bash
# Test dependency resolution
pip install -e .
pip install -e ".[dev,all]"

# Should install without conflicts
```

### Remove pathlib2 Imports
```bash
# Find any pathlib2 usage
grep -rn "from pathlib2" --include="*.py"
grep -rn "import pathlib2" --include="*.py"

# Replace with standard pathlib
# from pathlib2 import Path  →  from pathlib import Path
```

---

## 🟡 PHASE 2: High Priority (3 weeks)

### These enable v1.0 production release

**Week 1: Testing & Validation**
- [ ] Create physics validation suite (1 week)
  - Energy conservation tests
  - Cross-engine consistency
  - Known analytical solutions
- [ ] Increase test coverage to 50%+ (3 days)
  - Add launcher integration tests
  - Engine switching workflows
  - I/O format tests

**Week 2: Documentation**
- [ ] Generate API documentation (3 days)
  - Configure Sphinx autodoc
  - Build in CI
- [ ] Create tutorial notebooks (4 days)
  - Basic simulation
  - Parameter optimization
  - Multi-engine comparison

**Week 3: Integration & Polish**
- [ ] Integrate engine probes into GUI (2 days)
  - Display diagnostic results
  - Add "Fix It" workflows
- [ ] Improve error messages (2 days)
  - Audit all exceptions
  - Add actionable instructions
- [ ] Final quality pass (1 day)
  - Run all checks
  - Fix remaining issues

---

## 🟢 PHASE 3: Community & Growth (Ongoing)

### Post-v1.0 improvements

- [ ] Performance benchmarking
- [ ] Expand MyPy coverage (gradual)
- [ ] Community building (Discord, forum)
- [ ] Published validation study
- [ ] Conference presentations

---

## Verification Checklist

### Before Committing Phase 1 Fixes

```bash
cd c:/Users/diete/Repositories/Golf_Modeling_Suite

# 1. Clean repository
ls =* 2>/dev/null && echo "❌ Junk files still present" || echo "✅ Clean"

# 2. No TODOs
grep -r "TODO" --include="*.py" launchers/ shared/ && echo "❌ TODOs found" || echo "✅ No TODOs"

# 3. No eval/exec (except justified)
grep -r "eval(" --include="*.py" launchers/ shared/ && echo "⚠️ Verify justified" || echo "✅ No eval"
grep -r "exec(" --include="*.py" launchers/ shared/ && echo "⚠️ Verify justified" || echo "✅ No exec"

# 4. Code quality
black . && echo "✅ Black passed"
ruff check . && echo "✅ Ruff passed"
mypy . && echo "✅ MyPy passed"

# 5. Tests pass
pytest && echo "✅ Tests passed"

# 6. Security scan
bandit -r shared/ launchers/ -ll && echo "✅ Bandit passed"
```

### Success Criteria

All checks must pass:
- ✅ No junk files (`=*`)
- ✅ No TODO/FIXME/HACK in production code
- ✅ No unjustified eval/exec
- ✅ All dependencies bounded
- ✅ Docker uses non-root user
- ✅ Black formatting passes
- ✅ Ruff linting passes
- ✅ MyPy type checking passes
- ✅ Pytest suite passes
- ✅ Bandit security scan passes

---

## Timeline Summary

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| **Phase 1: Critical** | 1-2 days | Clean, secure codebase |
| **Phase 2: High Priority** | 3 weeks | v1.0 RC (production candidate) |
| **Phase 3: Community** | 2-3 months | v1.0 Stable |
| **Phase 4: Growth** | 6-12 months | v2.0 Mature |

**Total to Production-Ready v1.0: ~6 months**

---

## Quick Start: Do This NOW

```bash
cd c:/Users/diete/Repositories/Golf_Modeling_Suite

# 1. Clean up (5 min)
rm =*
echo -e "\n# Prevent malformed pip artifacts\n=*" >> .gitignore
git add -u .gitignore
git commit -m "Remove pip artifacts and prevent recurrence"

# 2. Run quality checks
black .
ruff check .
mypy .
pytest

# 3. Review security
pip install bandit
bandit -r launchers/ shared/ -ll

# 4. Plan next steps
cat UNIFIED_PROFESSIONAL_ASSESSMENT.md | grep "PHASE 1"
```

---

## Getting Help

**Assessment Documents:**
- `UNIFIED_PROFESSIONAL_ASSESSMENT.md` - Full analysis
- `IMMEDIATE_ACTION_PLAN.md` - This file
- `PROFESSIONAL_ASSESSMENT.md` - Original deployment view
- `ARCHITECTURE_IMPROVEMENTS_PLAN.md` - Detailed technical plan

**Questions?**
- Refer to UNIFIED_PROFESSIONAL_ASSESSMENT.md Section 12 (Recommendations)
- Check ARCHITECTURE_IMPROVEMENTS_PLAN.md for implementation details

---

**Status:** 🔴 CRITICAL Phase 1 Required  
**Next Milestone:** Clean codebase (1-2 days)  
**Ultimate Goal:** Production v1.0 (6 months)  
**Current Rating:** 5.5/10 (production) / B+ (research)  
**Target Rating:** 9/10 (production) / A (research)
