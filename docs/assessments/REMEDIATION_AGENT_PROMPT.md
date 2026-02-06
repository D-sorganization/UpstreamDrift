# Agent Prompt: Fix Critical Assessment Issues

## Context

The UpstreamDrift repository assessment (2026-02-03) identified 6 critical issues requiring remediation. This prompt provides instructions for systematically fixing all issues.

## Issue Summary

| Priority | Issue                                                   | Category      |
| -------- | ------------------------------------------------------- | ------------- |
| P0       | Test suite collection failures in headless environments | Testing       |
| P0       | 79 files flagged for potential hardcoded secrets        | Security      |
| P1       | 3/4 tutorial files are placeholders                     | Documentation |
| P1       | Bandit security findings (MD5, SQL, yaml.load)          | Security      |
| P2       | No automated release pipeline                           | CI/CD         |
| P2       | 6 files not Black-formatted                             | Code Quality  |

---

## Task Instructions

### Phase 1: Quick Wins (P2 Issues)

#### 1.1 Fix Black Formatting (P2)

Run Black on the 6 non-compliant files:

```bash
black src/api/dependencies.py src/api/server.py \
  src/engines/physics_engines/pendulum/python/pendulum_physics_engine.py \
  src/engines/physics_engines/pinocchio/python/pinocchio_physics_engine.py \
  src/engines/physics_engines/myosuite/python/myosuite_physics_engine.py \
  src/engines/physics_engines/opensim/python/opensim_physics_engine.py
```

Commit with message: `style: apply Black formatting to 6 non-compliant files`

#### 1.2 Create Automated Release Pipeline (P2)

Create `.github/workflows/release.yml` with the following requirements:

- Trigger on tag push matching `v*.*.*`
- Build Python package using `build`
- Publish to PyPI using `twine` (use `PYPI_API_TOKEN` secret)
- Create GitHub Release with auto-generated notes
- Include checksums for artifacts

Example structure:

```yaml
name: Release
on:
  push:
    tags: ["v*.*.*"]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install build twine
      - run: python -m build
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
      - uses: softprops/action-gh-release@v2
        with:
          generate_release_notes: true
          files: dist/*
```

Commit with message: `ci: add automated PyPI release workflow`

---

### Phase 2: Documentation (P1)

#### 2.1 Complete Tutorial Placeholders (P1)

The following files need complete tutorials:

- `docs/tutorials/content/02_placeholder.md` → `02_first_simulation.md`
- `docs/tutorials/content/03_placeholder.md` → `03_engine_comparison.md`
- `docs/tutorials/content/04_placeholder.md` → `04_video_analysis.md`

For each tutorial:

1. Research the codebase to understand the relevant functionality
2. Write step-by-step instructions with code examples
3. Include expected outputs and troubleshooting tips
4. Add screenshots or diagrams where helpful

**Tutorial 02: First Simulation**

- Cover: Loading a model, configuring parameters, running simulation, viewing results
- Use MuJoCo engine as primary example
- Include complete runnable code snippets

**Tutorial 03: Engine Comparison**

- Cover: Available engines, when to use each, cross-engine validation
- Show how to run same simulation across multiple engines
- Compare results and explain differences

**Tutorial 04: Video Analysis**

- Cover: Video pose pipeline, MediaPipe integration, marker mapping
- Show end-to-end workflow from video to simulation
- Include sample video references

Commit with message: `docs: complete tutorial series (02-04)`

---

### Phase 3: Security (P0/P1)

#### 3.1 Audit Potential Hardcoded Secrets (P0)

Search for the 79 flagged files:

```bash
grep -r -l "password\|secret\|api_key\|token" --include="*.py" src/
```

For each file:

1. Determine if the string is actually a secret vs. variable name/documentation
2. If real secret: Move to environment variable, update `.env.example`
3. If false positive: Add `# nosec` comment with justification or ignore in config
4. If test fixture: Ensure it's clearly fake (use `test_`, `fake_`, `dummy_` prefix)

Create a tracking checklist in `docs/assessments/issues/SECRET_AUDIT_CHECKLIST.md`:

```markdown
# Secret Audit Checklist

## Files Reviewed

- [ ] file1.py - Status: [False Positive | Remediated | N/A]
- [ ] file2.py - ...
```

Commit with message: `security: audit and remediate hardcoded secrets`

#### 3.2 Fix Bandit Security Findings (P1)

Run Bandit and address each finding:

```bash
bandit -r src -ll -ii -f json > bandit_report.json
```

Common fixes:

1. **MD5 usage**: Replace with SHA-256 for security-sensitive hashing, or add `# nosec B303` if used for non-security purposes (checksums, caching)
2. **String-formatted SQL**: Use parameterized queries with placeholders
3. **yaml.load**: Replace with `yaml.safe_load()` or use `yaml.load(data, Loader=yaml.SafeLoader)`

For each finding:

- If security risk: Fix the code
- If false positive: Add `# nosec BXXX` with comment explaining why
- Document all suppressions in `docs/assessments/BANDIT_SUPPRESSIONS.md`

Commit with message: `security: remediate Bandit findings`

---

### Phase 4: Testing (P0)

#### 4.1 Fix Test Suite Headless Failures (P0)

Investigate and fix test collection failures:

1. **Identify failing tests**:

```bash
QT_QPA_PLATFORM=offscreen pytest --collect-only 2>&1 | grep -E "(ERROR|FAILED)"
```

2. **Common headless issues to fix**:

   - GUI imports at module level: Move to function level or guard with `if TYPE_CHECKING:`
   - Missing display: Add `@pytest.mark.requires_gl` marker and skip in CI
   - Missing fixtures: Ensure conftest.py provides all required fixtures

3. **Create minimal reliable test slice**:
   Create `tests/conftest.py` additions or `pytest.ini` configuration:

```ini
[pytest]
markers =
    requires_gl: marks tests requiring OpenGL/display (deselect with '-m "not requires_gl"')
    headless_safe: marks tests verified to work in headless mode
```

4. **Add CI job for headless testing**:
   Update `.github/workflows/ci-standard.yml` to run with:

```yaml
- run: pytest -m "not requires_gl" --ignore=tests/integration/isolated
```

5. **Fix or skip problematic imports**:
   - Guard PyQt6 imports: `if os.environ.get('DISPLAY') or os.environ.get('QT_QPA_PLATFORM'):`
   - Use lazy imports for heavy GUI modules

Commit with message: `test: fix headless test collection and add requires_gl markers`

---

## Verification Checklist

After completing all fixes, verify:

- [ ] `black --check src/` passes with no reformatting needed
- [ ] `bandit -r src -ll` shows no high/medium issues (or all are documented)
- [ ] `pytest --collect-only` succeeds in headless environment
- [ ] All 4 tutorial files have substantive content (>500 words each)
- [ ] No actual secrets in codebase (grep shows only variable names/docs)
- [ ] Release workflow syntax is valid: `gh workflow view release.yml`
- [ ] CI passes on a test PR

---

## Commit Strategy

Make atomic commits for each logical change:

1. `style: apply Black formatting to 6 non-compliant files`
2. `ci: add automated PyPI release workflow`
3. `docs: add Tutorial 02 - First Simulation`
4. `docs: add Tutorial 03 - Engine Comparison`
5. `docs: add Tutorial 04 - Video Analysis`
6. `security: audit and remediate hardcoded secrets`
7. `security: remediate Bandit findings`
8. `test: fix headless test collection and add requires_gl markers`

Create a single PR with all commits titled:
**"fix: remediate all critical assessment issues (P0-P2)"**

---

## Success Criteria

The following metrics should improve after remediation:

- Black compliance: 765/765 files (was 759/765)
- Bandit issues: 0 high/medium (document all suppressions)
- Tutorial completion: 4/4 (was 1/4)
- Headless test collection: 100% success
- Release automation: Functional workflow

Update `docs/assessments/assessment_summary.json` with new scores after completion.
