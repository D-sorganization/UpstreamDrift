## Problem

A read-only pre-PR check of phase-two commit `32877b6ecc` found 33 Mypy errors across ten validation helper files and twelve Python files needing Ruff formatting. These are owned shot-pattern validation helpers/tests; the scientific producer remains source-frozen at `4023111d2a` during supplemental v6 rendering. Existing numerical acceptance must retain the exact executed source bytes and receipt identities.

## Acceptance Criteria

- Preserve immutable executed helper bytes and original review receipts with verified hashes before correcting runnable helper sources.
- Correct type narrowing, optional GUI/import handles, typed containers and callable instrumentation without broad ignores, exclusions, weakened tolerances or hook bypasses.
- Apply scoped formatting and lint repairs; do not alter frozen scientific/reporting producer sources or rerun simulations.
- Add meaningful regression/negative checks for changed validation behavior and rerun relevant helper checks; distinguish historical execution evidence from newly qualified sources.
- Run canonical pre-PR gates against a fixed final commit and disclose unrelated baseline failures separately.
- Deliver through a ready PR with a change fragment and current handoff; leave this issue open until merged acceptance.

## Evidence

Root readiness log: `/tmp/p2-prepr-readiness-20261010-safe.txt`; format log: `/tmp/p2-readiness-format.txt`. The canonical runner requires `PYTHONSAFEPATH=1` plus Repository_Management on `PYTHONPATH` so the checkout's local `scripts` package does not shadow the runner. The broader test/gate run remains in progress; no full pass is claimed.
