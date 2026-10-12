# Analytic Muscle Replay Input Contract Turnover

**Issue:** #12392, under the ongoing #11792 muscle-recruitment workstream
**Branch:** `feat/analytic-replay-input-12392`
**State:** input-contract implementation; no execution worker or native result

## Change

Added `decode_analytic_replay_spec` at
`src/shared/python/motion_matching/analytic_muscle_replay_spec.py`. It parses
bounded UTF-8 JSON into frozen nested records, rejecting duplicate/unknown or
missing keys, invalid types and domains, non-finite values, malformed vectors,
non-monotonic/off-grid clocks, invalid exact rate limits, duplicate IDs, and
invalid derived initial musculotendon length. The public record fixes one
rotational constant-moment-arm coordinate, explicit SI unit labels, no contact
or constraints, and semi-implicit Euler as the declared future fixture update
label. It retains raw commands, previous raw commands, activation values and
the activation floor as separate inputs.

The schema is capped at 65,536 bytes, eight muscles, 128 steps, and 128 UTF-8
bytes per identifier. Optional producer kind and revision are syntax-only
metadata; no revision is resolved or accepted as an implementation pin. The
parser performs no force calculation, activation update, integration, replay,
solver call, or native SDK import. The fresh-process test checks forbidden module components before importing
the parser, decodes a valid record, and checks again afterward.

## Validation

The first vendor-bootstrap failure was setup failure, not a RED test. A later
missing-parser `ModuleNotFoundError` was the intended RED condition, but its
outer launcher did not retain a reliable inner pytest exit code. The completed
standard-conftest focused run used:

```text
python -B -m pytest -o 'addopts=' -q tests/unit/motion_matching/test_analytic_muscle_replay_spec.py
32 passed, 1 existing DeprecationWarning
```

Ruff check and Ruff format check passed for the parser and its test. Canonical
diff-scoped mypy passed for the new parser source file, with the worktree root
as `MYPYPATH`, via:

```text
python -B Repository_Management/shared_scripts/run_mypy_diff.py src/shared/python/motion_matching/analytic_muscle_replay_spec.py
Success: no issues found in 1 source file
```

A separate direct `python -m mypy <one file>` attempt followed imports across
the repository and reported 57 errors in 36 files; it was not the configured
diff-scoped gate. It is retained as a wrong-scope attempt, not reported as a
parser failure or as a passing check. No full `pre_pr` run or native execution
is claimed here.

## Limits and Next Step

This decoder is a physical-input schema, not an execution receipt. Passing it
does not prove source/runtime identity, capability, force balance, activation
updates, trajectory behavior, qualification, or parity. The next step is a
separately pinned worker and replay-result contract that consumes this immutable
record and reports its own runtime, implementation, and complete state/output
provenance. All 17 registered model/variant rows remain unqualified; this
source-specific schema changes no qualification status and makes no parity
claim for other engines. Calculation inventory and manual release remain
blocked.
