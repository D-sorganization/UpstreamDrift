# Muscle Qualification Evidence Guard Turnover

## Identity and Scope

Repository: UpstreamDrift. Worktree: `C:/Users/diete/Repositories/Worktrees/feedback-muscle-evidence-11791`. Branch: `fix/feedback-muscle-evidence-11791`. Baseline: `44f091273884439bef43f61b90c1c10f68b77726`. Current test commit: `SELF`; resolve with `git rev-parse HEAD`. PR: not created. Governing task: F07 #11791 under #11784; this is a narrow evidence-integrity correction, not completion of either issue.

Owned paths: `tour_matching/muscle_qualification.py`, focused tests, calculation reference and this turnover. The coordinating task owns `muscle_replay.py`; do not modify it here. No capture or native full-body model is needed for this guard. No unrelated/user-owned changes observed.

## Failing Behavioral Evidence

The orchestration currently invents a `QUALIFIED_SHORT_REPLAY` from fixed reserve/pelvic metrics, and sets geometry, moment-arm, equilibrium and activation audit booleans without performing those audits. A standalone metrics container also defaults to a qualified status.

Using `C:/Users/diete/Repositories/.venv-feedback-opensim/Scripts/python.exe`:

```powershell
python3 -m pytest tests/opensim/test_muscle_cmc.py --override-ini='addopts=' -q -k 'does_not_invent_native_evidence or constructing_short_replay_metrics'
```

Result: two expected assertion failures, 32 deselected. Test collection first identified an uninitialized Tools submodule; it was restored offline from the coordinating worktree at the unchanged pin `3678409fc51024150ab28970b72e3b468935f345` before recording this red result. No remote call or provider pin change was required.

## Next Bounded Step

Remove fabricated runtime results from the orchestrator, preserve actual anatomy/parameter audit outcomes, and leave all unperformed audits and native replay explicitly unqualified. Run focused regressions and the existing pure qualification tests, update the governed manual/registry and fragment, and publish through normal PR protections. The requested full native qualification remains open.
