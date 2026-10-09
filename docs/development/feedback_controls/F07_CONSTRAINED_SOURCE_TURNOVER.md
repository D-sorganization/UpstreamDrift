# F07 Constrained-Source Readiness Turnover

Child #11923, branch `feat/feedback-constrained-source-11923`, current commit
`SELF`. Parent #11791/#11792 and epic #11784 remain open. Source ownership is
`tour_matching/native_constraint_state.py` and its native tests; canonical
chapter 32 records equations, state limits and reproduction.

The observer and fresh-source wrapper now preserve actual native coordinate
locks/prescription/clamping/dependency, constraint enforcement, scalar actuator
values, named/registered state, derivatives and constraint residuals. The wrapper
owns XML bytes, restores exact achieved named values and preserves all native
initialization defaults. No arbitrary full-state restart is admitted. Actual
lock targets are unverified even when a caller declaration matches current q.

TDD: the new module was absent (RED), then eight native cases passed. Two further
RED cases required explicit assistance observations and a non-overridable
unqualified result. The unchanged source exposed an older-XML native log handle
inside temporary directories; an original version40000 fixture reproduced the
Windows cleanup failure (RED), then owned-file-only cleanup passed (GREEN).
Current scoped suite: 18 native tests and 11 governance tests passed under
Python 3.12/OpenSim 4.6, serially. Four added RED cases require a mapping for
declared targets even when an invalid container is empty.

Actual unchanged source evaluation reproduces 1348 prepared named states exactly,
549 registered discrete values, 1760 options, 54 locks, two couplers, 29 nonmuscle
actuators and G dimensions 56 by 171. The retained local receipt is
`docs/development/feedback_controls_planning/native_model_evaluation/constrained_source_audit_11923_final.json`
under the fleet workspace, not a committed source asset or private capture.
Source/loaded hashes match prior preparation. Resource/DLL closure, actual lock
targets, physiology, contact and capture qualification remain blocked.

Native source investigation also retains IL_L4 approximately 48 kN and multifidus
approximately 62 times maximum-isometric force after equilibrium; native minimum
activation does not resolve them. Review source assembly/parameter/neutral-pose
provenance before tuning, preserving original failures and licensed notices.

Run `python -m pytest tests/opensim/test_native_constraint_state.py -o addopts='' -q`
with the actual OpenSim runtime. Default Python 3.13 skips native tests and is not
native evidence. Ruff format/check, changed-source mypy, all five central
`pre_pr --base-ref origin/main` gates, change-fragment validation and manual
governance pass. The default Python 3.13 mapped run skips all 18 native cases;
the separate actual runtime run above supplies native evidence. The final API
was rerun on the unchanged candidate with matching source/loaded/prepared-state
identities. Protected CI and publication remain pending until recorded; no
bypass or scientific completion is implied. The current change-fragment workflow
satisfies SPEC/central handoff freshness without editing shared queue hotspots.

## Main Integration After Native Geometry and Torque Replay

The October 9 merge of `origin/main` at
`5b21b80a9ce7f1b088ec73c38d19314666738958` retained this observer's
source and test bytes, plus the already merged native torque replay (#11836)
and native OpenSim geometry (#11911). The only content conflicts were the
calculation registry and manual index; both now retain all three provisional
references. The pinned Tools T01 submodule is
`2e7665111b06f92ffbfe178b92d74d6a81c95388`.

On the integrated tree, 18 actual OpenSim observer cases and 11 manual
governance cases passed under the owned Python 3.12/OpenSim 4.6 runtime with
the checkout on `PYTHONPATH`. Thirteen native MuJoCo replay cases passed
separately. Twenty native OpenSim geometry cases passed when the existing
Python 3.12 host `cv2` was appended only for that test process, after the
owned runtime's NumPy had loaded. The five scoped central gates passed;
their default Python 3.13 test mapper skipped all 18 OpenSim cases and is
not counted as native evidence. No observer behavior, model threshold,
anatomical result or physical acceptance was changed by this integration.
