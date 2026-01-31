#!/bin/bash
#
# Create GitHub issues from Code Quality Assessment 2026-01-31
#
# Prerequisites:
#   - GitHub CLI (gh) installed and authenticated
#   - Run: gh auth login
#
# Usage:
#   ./scripts/create_assessment_issues.sh [--dry-run]
#

set -e

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE - No issues will be created ==="
fi

# Check gh is installed and authenticated
if ! command -v gh &> /dev/null; then
    echo "Error: GitHub CLI (gh) is not installed"
    echo "Install with: brew install gh (macOS) or apt install gh (Ubuntu)"
    exit 1
fi

if ! gh auth status &> /dev/null; then
    echo "Error: Not authenticated to GitHub"
    echo "Run: gh auth login"
    exit 1
fi

echo "Creating 15 issues from Code Quality Assessment..."
echo ""

# Issue 1: Critical - eval() security
create_issue_1() {
    local title="[Security] Replace unsafe eval() in signal_toolkit/fitting.py"
    local labels="security,critical,python"
    local body=$(cat <<'EOF'
## Summary

The signal_toolkit/fitting.py file contains an unsafe `eval()` call that poses a code injection vulnerability.

## Location

- **File:** `src/shared/python/signal_toolkit/fitting.py`
- **Pattern:** `return eval(expression, {"__builtins__": {}}, local_dict)  # noqa: S307`

## Risk

**CRITICAL** - Code injection vulnerability if `expression` comes from user input. The Bandit security check has been explicitly bypassed with `noqa: S307`.

## Recommended Fix

1. Replace with `ast.literal_eval()` for simple expressions
2. Or use `simpleeval` library (already in project dependencies)
3. Remove the `noqa` comment after fix is applied

## Migration Steps

1. Audit all callers of this function to understand input sources
2. Replace eval with safe evaluation library
3. Add input validation for expression format
4. Remove noqa comment
5. Run Bandit to verify fix

## Effort Estimate

**Small** - Focused change in single file

---
*Generated from Code Quality Assessment 2026-01-31*
EOF
)

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY] Would create: $title"
    else
        gh issue create --title "$title" --body "$body" --label "$labels"
        echo "Created: $title"
    fi
}

# Issue 2: Critical - pickle security
create_issue_2() {
    local title="[Security] Remove pickle deserialization in motion_optimization.py"
    local labels="security,critical,python"
    local body=$(cat <<'EOF'
## Summary

The motion_optimization.py file uses `allow_pickle=True` with numpy load, enabling arbitrary code execution from untrusted files.

## Location

- **File:** `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/motion_optimization.py`
- **Pattern:** `data = np.load(filename, allow_pickle=True)`

## Risk

**CRITICAL** - Arbitrary code execution from untrusted pickle files. Pickle deserialization is inherently unsafe for untrusted data.

## Recommended Fix

1. Convert data format to `.npz` without pickle
2. Or use `safetensors` library for model weights
3. Add file provenance validation if pickle must be used
4. Document trusted file sources

## Migration Steps

1. Audit all `allow_pickle=True` usages in codebase
2. Identify data formats that can be converted
3. Create migration script for existing files
4. Update loading code to use safe format
5. Add warning/validation for any remaining pickle usage

## Effort Estimate

**Medium** - Requires data format migration

---
*Generated from Code Quality Assessment 2026-01-31*
EOF
)

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY] Would create: $title"
    else
        gh issue create --title "$title" --body "$body" --label "$labels"
        echo "Created: $title"
    fi
}

# Issue 3: High - God module refactoring
create_issue_3() {
    local title="[Refactor] Break up golf_launcher.py god module (2,539 LOC)"
    local labels="refactoring,maintainability,python"
    local body=$(cat <<'EOF'
## Summary

The golf_launcher.py file at 2,539 lines of code is a "god module" that handles too many responsibilities, violating the Single Responsibility Principle.

## Location

- **File:** `src/launchers/golf_launcher.py`
- **Size:** 2,539 LOC

## Current Responsibilities (Mixed Concerns)

- UI/Window management
- Docker environment checks
- Environment dialogs
- Splash screens
- Model card widgets
- Engine initialization
- Configuration handling

## Risk

**HIGH** - High coupling leads to:
- Difficult testing (can't test components in isolation)
- Merge conflicts when multiple developers work on launcher
- Onboarding friction for new developers
- Harder to reason about code behavior

## Recommended Refactoring

Extract into focused modules:
1. `docker_manager.py` - Docker environment checks
2. `environment_dialog.py` - Environment configuration UI
3. `splash_screen.py` - Splash screen component
4. `model_card_widget.py` - Model card UI component
5. `launcher_core.py` - Core launcher logic

## Migration Strategy

Use **Strangler Fig Pattern**:
1. Extract one component at a time
2. Create new module with extracted code
3. Import from new module in golf_launcher.py
4. Maintain backward compatibility during transition
5. Add tests for each extracted component

## Effort Estimate

**Large** - Multiple extraction cycles needed

---
*Generated from Code Quality Assessment 2026-01-31*
EOF
)

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY] Would create: $title"
    else
        gh issue create --title "$title" --body "$body" --label "$labels"
        echo "Created: $title"
    fi
}

# Issue 4: High - Dependency lockfile
create_issue_4() {
    local title="[DevOps] Complete Python dependency lockfile"
    local labels="dependencies,ci/cd"
    local body=$(cat <<'EOF'
## Summary

The `requirements.lock` file is incomplete (13 entries) while the project has 100+ actual dependencies. Additionally, `requirements.txt` uses loose `>=` constraints without upper bounds.

## Location

- **File:** `requirements.lock` - Only 13 entries
- **File:** `requirements.txt` - Uses `>=` without upper bounds
- **File:** `pyproject.toml` - Some deps have good bounds, others don't

## Risk

**HIGH**:
- Non-reproducible builds between environments
- Silent breakage from uncontrolled dependency updates
- Security vulnerabilities from unexpected package versions
- CI/CD inconsistency

## Recommended Fix

1. Generate complete lockfile using `pip-compile` or `pip freeze`
2. Add upper bounds to critical dependencies
3. Ensure consistency between requirements.txt, requirements.lock, and pyproject.toml

## Migration Steps

1. Run `pip freeze > requirements.lock.new` in clean environment
2. Compare with current lock file
3. Test build against new lock
4. Update Dependabot to use lock file
5. Add CI check that lock file is up-to-date

## Effort Estimate

**Medium** - Requires testing across environments

---
*Generated from Code Quality Assessment 2026-01-31*
EOF
)

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY] Would create: $title"
    else
        gh issue create --title "$title" --body "$body" --label "$labels"
        echo "Created: $title"
    fi
}

# Issue 5: High - Accessibility
create_issue_5() {
    local title="[A11y] Add accessibility attributes to React components"
    local labels="accessibility,frontend"
    local body=$(cat <<'EOF'
## Summary

The React frontend lacks proper accessibility (a11y) attributes, creating WCAG compliance gaps and excluding users who rely on assistive technologies.

## Location

- **Directory:** `ui/src/components/`
- **Affected files:** SimulationControls.tsx, EngineSelector.tsx, Scene3D.tsx

## Specific Issues

1. **No ARIA attributes** - Buttons lack `aria-label` descriptions
2. **Missing role attributes** - Custom button-like elements need roles
3. **No keyboard navigation indicators** - Focus styles not visible
4. **No live regions** - Status updates not announced to screen readers
5. **Canvas without fallback** - Three.js Canvas has no alternative content

## Risk

**HIGH**:
- WCAG 2.1 AA compliance violations
- Potential legal liability (ADA, Section 508)
- Excluded users (blind, low-vision, motor impairments)
- Failed accessibility audits

## Recommended Fix

```tsx
// SimulationControls buttons should have:
<button aria-label="Start simulation" ...>

// EngineSelector should use:
<fieldset>
  <legend>Physics Engine</legend>
  <div role="group">
```

## Migration Steps

1. Add `eslint-plugin-jsx-a11y` to ESLint config
2. Run linter to identify all violations
3. Fix existing violations (add ARIA, roles, focus styles)
4. Add accessibility tests using `@testing-library/jest-dom`
5. Add keyboard navigation support

## Effort Estimate

**Medium** - Requires updates across all components

---
*Generated from Code Quality Assessment 2026-01-31*
EOF
)

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY] Would create: $title"
    else
        gh issue create --title "$title" --body "$body" --label "$labels"
        echo "Created: $title"
    fi
}

# Issue 6: High - Frontend tests
create_issue_6() {
    local title="[Testing] Add React component tests"
    local labels="tests,frontend"
    local body=$(cat <<'EOF'
## Summary

The React frontend has minimal test coverage - only `ui/src/api/client.test.ts` exists. No component render tests or interaction tests are present.

## Location

- **Current tests:** `ui/src/api/client.test.ts` (95 lines)
- **Missing tests for:**
  - `pages/Simulation.tsx`
  - `components/simulation/EngineSelector.tsx`
  - `components/simulation/SimulationControls.tsx`
  - `components/visualization/Scene3D.tsx`

## Risk

**HIGH**:
- UI regressions undetected
- Unsafe to refactor components
- Poor confidence in deployments
- No documentation of expected behavior

## Test Priority

1. SimulationControls - button interactions
2. EngineSelector - selection and loading states
3. Simulation page - integration of components
4. Scene3D - render without errors

## Migration Steps

1. Add `@testing-library/react` and `@testing-library/jest-dom` dependencies
2. Create test files for each component
3. Test critical user flows first (engine selection, simulation start/stop)
4. Add tests to CI pipeline
5. Target 80% component coverage

## Effort Estimate

**Medium** - New test infrastructure needed

---
*Generated from Code Quality Assessment 2026-01-31*
EOF
)

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY] Would create: $title"
    else
        gh issue create --title "$title" --body "$body" --label "$labels"
        echo "Created: $title"
    fi
}

# Issue 7: High - Docker hardening
create_issue_7() {
    local title="[Security] Harden main Dockerfile (non-root user, pinned base)"
    local labels="security,docker"
    local body=$(cat <<'EOF'
## Summary

The main Dockerfile has several security issues: uses `:latest` tag, runs as root, no multi-stage build, no health check.

## Location

- **File:** `/Dockerfile` (root)
- **Compare to:** `Dockerfile.unified` (production-ready example)

## Security Issues

1. **`:latest` base image** - Non-deterministic builds, no version pinning
2. **No non-root user** - Container runs as root, container escape risk
3. **No multi-stage build** - Bloated image with build tools in production
4. **No health check** - No container health monitoring

## Recommended Fix

Follow patterns from `Dockerfile.unified`:

```dockerfile
FROM continuumio/miniconda3:24.11.2 AS base  # Pin version

RUN useradd -m -s /bin/bash golfer
USER golfer

HEALTHCHECK --interval=30s --timeout=10s \
  CMD curl -f http://localhost:8000/api/health || exit 1
```

## Migration Steps

1. Pin base image to specific version
2. Add non-root user creation
3. Switch to multi-stage build (builder + runtime)
4. Add health check
5. Test build and runtime
6. Update CI to use new Dockerfile

## Effort Estimate

**Medium** - Template exists in Dockerfile.unified

---
*Generated from Code Quality Assessment 2026-01-31*
EOF
)

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY] Would create: $title"
    else
        gh issue create --title "$title" --body "$body" --label "$labels"
        echo "Created: $title"
    fi
}

# Issue 8: Medium - MyPy coverage
create_issue_8() {
    local title="[TypeScript] Reduce mypy exclusion list (10+ directories)"
    local labels="typing,python"
    local body=$(cat <<'EOF'
## Summary

The mypy configuration excludes 10+ directories from type checking, reducing type safety and IDE support.

## Location

- **File:** `pyproject.toml` (lines 108-125)
- **File:** `mypy.ini`

## Excluded Directories

- `src/shared/python/signal_toolkit/`
- `src/shared/python/plotting/`
- `src/shared/python/ui/qt/widgets/`
- `src/shared/python/dashboard/tests/`
- `src/api/` (entire API!)
- And more...

## Risk

**MEDIUM**:
- Type errors slip through to runtime
- IDE support degraded in excluded areas
- Gradual type coverage erosion
- Harder refactoring

## Migration Strategy

1. Pick one directory per sprint
2. Start with most-used modules (src/api/ is high priority)
3. Add type hints incrementally
4. Remove from exclusion list when clean
5. Track progress in this issue

## Progress Tracking

- [ ] `src/api/` - Highest priority, 5,400+ LOC
- [ ] `src/shared/python/signal_toolkit/`
- [ ] `src/shared/python/plotting/`
- [ ] `src/shared/python/ui/qt/widgets/`
- [ ] Other directories...

## Effort Estimate

**Large (ongoing)** - Multi-sprint effort

---
*Generated from Code Quality Assessment 2026-01-31*
EOF
)

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY] Would create: $title"
    else
        gh issue create --title "$title" --body "$body" --label "$labels"
        echo "Created: $title"
    fi
}

# Issue 9: Medium - Exception handling
create_issue_9() {
    local title="[Reliability] Replace generic exception catches with specific handlers"
    local labels="error-handling,python"
    local body=$(cat <<'EOF'
## Summary

50+ instances of `except Exception:` in API routes mask specific errors and make debugging difficult.

## Location

- **File:** `src/api/routes/simulation.py:45` - `except Exception as exc:`
- **File:** `src/api/server.py:231-235` - Generic exception catch
- **File:** `src/api/server.py:246` - Broad exception in startup_event

## Risk

**MEDIUM**:
- Masks specific errors (e.g., TimeoutError vs MemoryError)
- Harder to debug production issues
- Poor error messages returned to API clients
- Silent failures may go unnoticed

## Migration Steps

1. Identify exception types in each handler
2. Add specific catches for known exceptions
3. Use structured error codes from `error_codes.py`
4. Keep generic Exception as last resort with proper logging
5. Add tests for each error path

## Effort Estimate

**Medium** - Systematic review of all handlers

---
*Generated from Code Quality Assessment 2026-01-31*
EOF
)

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY] Would create: $title"
    else
        gh issue create --title "$title" --body "$body" --label "$labels"
        echo "Created: $title"
    fi
}

# Issue 10: Medium - Unused deps
create_issue_10() {
    local title="[Cleanup] Remove unused frontend dependencies (Zustand, react-hook-form, zod)"
    local labels="dependencies,frontend"
    local body=$(cat <<'EOF'
## Summary

The frontend `package.json` includes dependencies that are not imported anywhere in the codebase.

## Location

- **File:** `ui/package.json`

## Unused Dependencies

- `zustand@^5.0.10` - State management library (not imported)
- `react-hook-form@^7.71.1` - Form library (not imported)
- `zod@^3.25.36` - Schema validation (not imported)

## Risk

**MEDIUM**:
- Larger bundle size (affects load time)
- Increased security surface area
- Confusing for developers
- Dependency update burden

## Migration Steps

1. Run `npx depcheck` to verify unused packages
2. Decide: remove or implement
3. If removing: `npm uninstall zustand react-hook-form zod`
4. Verify build still works
5. Update documentation if features were planned

## Effort Estimate

**Small** - Simple package removal

---
*Generated from Code Quality Assessment 2026-01-31*
EOF
)

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY] Would create: $title"
    else
        gh issue create --title "$title" --body "$body" --label "$labels"
        echo "Created: $title"
    fi
}

# Issue 11: Medium - CVE documentation
create_issue_11() {
    local title="[Security] Document CVE exception remediation plans"
    local labels="security,documentation"
    local body=$(cat <<'EOF'
## Summary

Security scanning ignores specific CVEs without documented remediation plans or expected fix dates.

## Location

- **File:** `.pre-commit-config.yaml` - Ignores CVE-2024-23342
- **File:** `.github/workflows/ci-standard.yml` - Ignores CVE-2026-0994

## Ignored CVEs

1. **CVE-2024-23342** (ecdsa) - Ignored, no fix available from upstream
2. **CVE-2026-0994** (protobuf) - Ignored, transitive dependency from dm_control

## Risk

**MEDIUM**:
- Security vulnerabilities remain unaddressed
- Audit failures (compliance requirements)
- No tracking of when to re-evaluate
- No documentation of mitigations

## Recommended Fix

1. Document each CVE exception in `SECURITY.md`
2. Include: Why ignored, mitigation measures, expected fix date, risk assessment
3. Create tracking issues for each CVE
4. Review monthly

## Effort Estimate

**Small** - Documentation task

---
*Generated from Code Quality Assessment 2026-01-31*
EOF
)

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY] Would create: $title"
    else
        gh issue create --title "$title" --body "$body" --label "$labels"
        echo "Created: $title"
    fi
}

# Issue 12: Medium - WebSocket reconnection
create_issue_12() {
    local title="[Reliability] Add WebSocket reconnection with exponential backoff"
    local labels="reliability,frontend"
    local body=$(cat <<'EOF'
## Summary

The WebSocket client disconnects without auto-reconnection logic, causing poor user experience on network issues.

## Location

- **File:** `ui/src/api/client.ts`
- **Function:** `useSimulation()` hook

## Current Behavior

- WebSocket closes on error/disconnect
- No automatic reconnection
- No exponential backoff
- User must manually refresh page
- In-flight simulation data may be lost

## Risk

**MEDIUM**:
- Poor UX on network instability
- Lost simulation data
- User frustration
- No connection status indicator

## Migration Steps

1. Add reconnection state management
2. Implement exponential backoff (1s, 2s, 4s, 8s, 16s, 30s max)
3. Add connection status indicator to UI
4. Handle in-flight message recovery
5. Add tests for reconnection scenarios

## Effort Estimate

**Medium** - State management changes

---
*Generated from Code Quality Assessment 2026-01-31*
EOF
)

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY] Would create: $title"
    else
        gh issue create --title "$title" --body "$body" --label "$labels"
        echo "Created: $title"
    fi
}

# Issue 13: Low - pytest consolidation
create_issue_13() {
    local title="[DRY] Consolidate duplicate pytest.ini configurations"
    local labels="maintainability,tests"
    local body=$(cat <<'EOF'
## Summary

Identical `pytest.ini` configurations are duplicated across 4+ engine subdirectories.

## Location

- `src/engines/Simscape_Multibody_Models/2D_Golf_Model/python/pytest.ini`
- `src/engines/Simscape_Multibody_Models/3D_Golf_Model/pytest.ini`
- `src/engines/physics_engines/drake/python/pytest.ini`
- `src/engines/physics_engines/mujoco/python/pytest.ini`

## Risk

**LOW**:
- DRY violation
- Configuration drift risk
- Maintenance burden when updating settings

## Migration Steps

1. Audit all pytest.ini files for differences
2. Consolidate common settings in pyproject.toml
3. Use conftest.py for engine-specific needs
4. Remove duplicate pytest.ini files
5. Test each engine's tests still run correctly

## Effort Estimate

**Small** - Straightforward consolidation

---
*Generated from Code Quality Assessment 2026-01-31*
EOF
)

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY] Would create: $title"
    else
        gh issue create --title "$title" --body "$body" --label "$labels"
        echo "Created: $title"
    fi
}

# Issue 14: Low - Tailwind cleanup
create_issue_14() {
    local title="[Cleanup] Extract repeated Tailwind button classes"
    local labels="frontend,maintainability"
    local body=$(cat <<'EOF'
## Summary

Button styling classes are repeated 4+ times in SimulationControls component.

## Location

- **File:** `ui/src/components/simulation/SimulationControls.tsx`
- **Lines:** 16-22, 27-32, 35-40, 43-48

## Risk

**LOW**:
- Inconsistent styling risk
- Harder to update button styles globally
- Verbose component code

## Recommended Fix

Extract to utility object:

```tsx
const buttonStyles = {
  base: 'flex items-center justify-center gap-2 font-semibold py-2 px-4 rounded transition-colors',
  primary: 'bg-green-600 hover:bg-green-700 text-white',
  danger: 'bg-red-600 hover:bg-red-700 text-white',
};
```

## Effort Estimate

**Small** - Single file change

---
*Generated from Code Quality Assessment 2026-01-31*
EOF
)

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY] Would create: $title"
    else
        gh issue create --title "$title" --body "$body" --label "$labels"
        echo "Created: $title"
    fi
}

# Issue 15: Low - Docker scanning
create_issue_15() {
    local title="[Security] Add Docker image scanning to CI (Trivy)"
    local labels="security,ci/cd"
    local body=$(cat <<'EOF'
## Summary

No container security scanning is configured in CI workflows, leaving Docker images unaudited for vulnerabilities.

## Location

- **Missing from:** `.github/workflows/ci-standard.yml`
- **Missing from:** All CI workflows

## Risk

**LOW** (but important for security posture):
- Container vulnerabilities undetected until production
- Base image vulnerabilities inherited silently
- No visibility into image security status

## Recommended Fix

Add Trivy scanner to CI workflow:

```yaml
- name: Run Trivy vulnerability scanner
  uses: aquasecurity/trivy-action@master
  with:
    image-ref: 'upstreamdrift:ci'
    format: 'sarif'
    severity: 'CRITICAL,HIGH'
    exit-code: '1'
```

## Migration Steps

1. Add Trivy action to CI workflow
2. Run initial scan to triage findings
3. Set severity threshold (start with CRITICAL only)
4. Fix or accept findings
5. Gradually increase threshold to HIGH

## Effort Estimate

**Small** - Standard GitHub Action addition

---
*Generated from Code Quality Assessment 2026-01-31*
EOF
)

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY] Would create: $title"
    else
        gh issue create --title "$title" --body "$body" --label "$labels"
        echo "Created: $title"
    fi
}

# Main execution
echo "=== Critical Priority (2 issues) ==="
create_issue_1
sleep 1
create_issue_2
sleep 1

echo ""
echo "=== High Priority (5 issues) ==="
create_issue_3
sleep 1
create_issue_4
sleep 1
create_issue_5
sleep 1
create_issue_6
sleep 1
create_issue_7
sleep 1

echo ""
echo "=== Medium Priority (5 issues) ==="
create_issue_8
sleep 1
create_issue_9
sleep 1
create_issue_10
sleep 1
create_issue_11
sleep 1
create_issue_12
sleep 1

echo ""
echo "=== Low Priority (3 issues) ==="
create_issue_13
sleep 1
create_issue_14
sleep 1
create_issue_15

echo ""
echo "=== Complete ==="
echo "Created 15 issues from Code Quality Assessment 2026-01-31"
