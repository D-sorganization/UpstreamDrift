# Muscle Qualification Evidence Guard Turnover

## Identity and Scope

Repository: UpstreamDrift. Worktree: `C:/Users/diete/Repositories/Worktrees/feedback-muscle-evidence-11791`. Branch: `fix/feedback-muscle-evidence-11791`. Baseline: `44f091273884439bef43f61b90c1c10f68b77726`. Current commit: `SELF`; resolve with `git rev-parse HEAD`. Test-first commit: `d66e31ed2d`. PR: not created. Scoped child #11810 under F07 #11791 and #11784; this is a narrow evidence-integrity correction, not completion of either parent. The child claim was clear and a codex lease was posted before publication.

Owned paths: `tour_matching/muscle_qualification.py`, focused tests, calculation reference and this turnover. The coordinating task owns `muscle_replay.py`; do not modify it here. No capture or native full-body model is needed for this guard. No unrelated/user-owned changes observed.

## Failing Behavioral Evidence

The original orchestration invented a `QUALIFIED_SHORT_REPLAY` from fixed reserve/pelvic metrics, and set geometry, moment-arm, equilibrium and activation audit booleans without performing those audits. A standalone metrics container also defaulted to a qualified status.

Using `C:/Users/diete/Repositories/.venv-feedback-opensim/Scripts/python.exe`:

```powershell
python3 -m pytest tests/opensim/test_muscle_cmc.py --override-ini='addopts=' -q -k 'does_not_invent_native_evidence or constructing_short_replay_metrics'
```

Result: two expected assertion failures, 32 deselected. Test collection first identified an uninitialized Tools submodule; it was restored offline from the coordinating worktree at the unchanged pin `3678409fc51024150ab28970b72e3b468935f345` before recording this red result. No remote call or provider pin change was required.

## Implemented Boundary and Green Evidence

The orchestrator now preserves the actual anatomy/parameter audit outcomes, returns `short_replay_receipt=None`, and leaves unperformed audit pass fields false. Replay metrics default to `UNVERIFIED_NATIVE_REPLAY`; a container or explicitly supplied status is not verification authority. Lower-level audits remain available. The compatibility duration argument no longer invents runtime observations.

The focused command above now passes within the full public fixture-independent test selection:

```powershell
python3 -m pytest tests/opensim/test_muscle_cmc.py --override-ini='addopts=' -q -m 'not requires_mocap_fixtures'
```

Result: 31 passed, 3 private-fixture tests deselected, using the Python 3.12 provider above and separately the system Python 3.13 pre-PR provider. No private captures or native qualification experiment were needed. Ruff check/format, design-manual governance and specification-fragment validation passed. All five central pre-PR gates passed: lint/format, diff mypy, affected tests, Semgrep/import policy and policy/fragment validators. Its default combined root/src MYPYPATH initially produced a duplicate-module mypy error; the successful complete rerun used command-scoped `MYPYPATH=(Get-Location).Path`, preserving all repository type settings. Tests ran serially with `PYTEST_ADDOPTS='-n 0 --override-ini=addopts= -m "not requires_mocap_fixtures"'`.

The governed manual chapter 14 describes this boundary. Its inventory blocker remains provisional; no publication approval or native scientific qualification is claimed.

## Next Bounded Step

Publish the scoped guard through normal PR protections. Continue actual native model/state/input/contact verification in F07/F08. The requested full native qualification and independent excitation replay remain open.
