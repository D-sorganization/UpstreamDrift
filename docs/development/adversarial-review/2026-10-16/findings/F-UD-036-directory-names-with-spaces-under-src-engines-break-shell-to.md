# F-UD-036

**Directory names with spaces under `src/engines/...` break shell tooling**

| Severity | Category | Confidence | Effort | Status   |
| -------- | -------- | ---------- | ------ | -------- |
| ⬜ Low   | hygiene  | high       | S      | proposed |

**Locations:** `AGENTS.md:146`

## Finding

- Severity: Low - Category: hygiene
- Confidence: high
- Location: AGENTS.md:146-151
- Evidence: `find src -name "*.py" | xargs wc -l` emits ~150 `No such file or directory` lines for `Motion Capture Plotter/` and
  `Simscape Multibody Data Plotters/Python Version/`; the same paths are not importable as packages.
- Failure scenario: every ad-hoc shell audit (and some CI `find | xargs` steps, e.g. the placeholder grep at
  ci-standard.yml:1073) silently skips these files unless `-print0` is used — the placeholder step does use `-print0`; the
  bandit step at :789 does; `scripts/*.py` using `glob` are fine. Human one-liners are not.
- Why a newer engineer misses this: MATLAB does not care about spaces.
- Proposed change: rename to `motion_capture_plotter/`, `simscape_multibody_data_plotters/python_version/` and leave shims.
- Verification: `find src -name "* *" | wc -l` → 0.
- Effort: S

## Draft Pull Request

**Title:** `chore(engines): rename directories containing spaces under src/engines`

### Why This Matters (The Pitch)

Directories like 'Motion Capture Plotter/' and 'Simscape Multibody Data Plotters/Python Version/' make find | xargs wc -l emit about 150 No such file errors and skip those files silently. They are also not importable as packages. The placeholder grep and bandit steps use -print0, but human one-liners do not. Renaming to snake_case and leaving shims removes the trap.

### Proposed PR Body

## Problem

`AGENTS.md:146-151`: `find src -name "*.py" | xargs wc -l` emits ~150 `No such file or directory` lines for `Motion Capture Plotter/` and `Simscape Multibody Data Plotters/Python Version/`; the ci-standard placeholder grep (`:1073`) and bandit step (`:789`) use `-print0`.

## Change

- [ ] Rename to `motion_capture_plotter/` and `simscape_multibody_data_plotters/python_version/` and leave shims

## Verification

`find src -name "* *" | wc -l` → 0.

## Risk and Rollback

Low to medium: MATLAB or scripts that hard-code the old paths need updating. Rollback is a revert.

## Handoff

None

Refs: adversarial review 2026-10-16, F-UD-036

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
