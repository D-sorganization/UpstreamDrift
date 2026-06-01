---
title: "Pre-push pytest hook uses wrong Python venv (OpenSim_Models instead of UpstreamDrift)"
labels: [bug, ci-infrastructure, fleet-followup]
priority: high
discovered_in: PR claude/test-coverage-improvements (branch)
discovered_at: 2026-06-01
reporter: claude
status: open
related_to: .pre-commit-config.yaml, .gaai/core/GAAI.md
---

## Summary

The pre-push hook `pytest (unit tests)` defined in `.pre-commit-config.yaml`
uses the bare command `python -m pytest ...`. On a machine with multiple
`python` shims on `PATH` (in particular, a stale
`C:\Users\diete\Repositories\OpenSim_Models\.venv\Scripts\python.EXE`
appearing first in `where.exe python`), the hook fires with the _wrong_
interpreter:

```
C:\Users\diete\Repositories\OpenSim_Models\.venv\Scripts\python.EXE: No module named pytest
```

The `pytest` package is not installed in the OpenSim_Models `.venv` because
that virtualenv was created before `pytest` was added to `dev` dependencies
(or by a different `pip install` flow). The hook therefore aborts with a
hard failure — every `git push` from this machine fails until the user
manually `pip install`s `pytest` into the wrong venv.

## Evidence

```
$ where.exe python
C:\Users\diete\Repositories\OpenSim_Models\.venv\Scripts\python.exe   <-- first on PATH
C:\Python314\python.exe
C:\Users\diete\AppData\Local\Microsoft\WindowsApps\python.exe
C:\Users\diete\AppData\Local\Python\bin\python.exe

$ C:\Python314\python.exe -c "import pytest; print(pytest.__version__)"
9.0.3
```

The C:\Python314 interpreter has pytest installed; the
OpenSim_Models `.venv` does not.

## Recommended fix

In `.pre-commit-config.yaml`, replace the bare `python` reference with a
project-relative discovery: e.g.

```yaml
entry: |
  $(git rev-parse --show-toplevel)/.venv-312/bin/python -m pytest
  tests/unit/dbc
  tests/unit/core
  tests/unit/utils
  -x -q --tb=short -m "not slow and not integration"
  -p no:xvfb
  --override-ini="addopts=..."
```

…or — better — drop the pre-push pytest step entirely and rely on
CI's `pytest` lane (`.github/workflows/ci-standard.yml`). The pre-push
hook is duplicating work the CI matrix already does, and a local
env-mismatch shouldn't block land-able PRs.

## Acceptance criteria

- `git push` on this machine does _not_ fail because pytest is missing
  from the OpenSim_Models `.venv`.
- CI's `pytest` lane continues to enforce the test floor.

## Related

- `.pre-commit-config.yaml` — line defining the `pytest (unit tests)` hook.
- `Repository_Management/docs/FLEET_HOOK_STANDARDS.md` — fleet policy on
  pre-commit / pre-push hooks.
- `CLAUDE.md` — "When a hook is legitimately broken" / "When a hook fails
  on something you didn't touch" guidance.
