# F02 Distributed Feedback Turnover (#11786)

F02 adds a phase-gated controller core on `feat/f02-distributed-feedback-11786`
in the isolated `UpstreamDrift-f02-11786` worktree. F01 PR #11808 has merged;
F02 PR #11816 now targets main. This integration includes main through the
ground-reaction integration squash `b7d4189c9470a0bdabf778e6ade584f42111a49e`.
The F02 PR body is the authoritative final commit/checks/handoff record.

The controller consumes an existing MOSAIC TVLQR gain sequence and frozen
nominal actuator torques. It records feedforward, feedback, requested and
post-limit applied torque separately. Its exact simulated-state information
pattern, tangent-state difference, phase/task priority, actuator order,
unit-constant transmission, unactuated root and one-sample-per-native-step
clock are explicit contracts. The contact allocator wraps the existing
contact/grip QP as a feasibility prediction only; the native plant alone
produces its reactions. Do not route predicted QP force to an own-contact
native solver as an external load.

Test-first evidence: the new unit test initially failed to import the module;
the bounded suite now covers analytic restoring versus reversed sign,
saturation and slew rate, priority/phase conflict, all five task groups,
wrapped-angle and `nq != nv` differences, hidden root and permutation
rejection, missing/indefinite adapters, feasible and infeasible QP cases,
and a one-step native MuJoCo torque motor. Reproduce with
`python -m pytest tests/unit/motion_matching/test_distributed_feedback.py -q`.
No owner capture, private media, EMG, muscle excitation, measured ground/grip
force, native full-body contact or continuous replay was used. The MuJoCo
fixture is a single hinge; its result does not qualify the human swing.

Next integrator owners should provide native tangent difference, mass inverse,
framed task Jacobians, loaded actuator/transmission identity and native
contact dynamics. Exact post-allocation torque can then be exported through
Tools T01 and independently replayed under F01/F09 gates. State-dependent
transmission and muscle excitation require dedicated controllers/input
schemas; do not relabel this direct torque path as muscle control. The
calculation registry remains blocked for release.

## Main Integration Validation

The main integration preserves the controller and its behavior-test bytes.
Both SPEC sections and all canonical manual chapters/inventory blockers were
retained; the governance test uses main's live canonical-source count. The
vendored Tools checkout matches the merged-main gitlink `2e7665111b06f92ffbfe178b92d74d6a81c95388`.
The canonical divergence inventory is regenerated without attribution edits.
Twenty-seven focused tests passed in the installed native MuJoCo environment,
including the real hinge step and eleven manual-governance cases. This does not
extend the original synthetic/one-joint scope or qualify full-model parity.

The merge-group seam regression exposed on the Drake branch is also carried
forward here: generic fallback checks still reject owned cluster gaps, while
an explicitly enabled Tools Lab overlay must resolve the canonical mocap leaf.
A genuinely missing Lab module remains unavailable. This is the existing test
correction from `8687f9222c`, with no import-loader or controller source change.

Latest-main integration also preserves the controller/test bytes, incorporates
the already merged shared ground-reaction work, and resolves only generated
divergence inventories. The installed-native controller/governance suite passed
all 27 tests again on that combined tree.

The latest-main push hook exposed 50 array-type errors in its isolated mypy
runtime, which lacked NumPy entirely. Annotation-only probing did not resolve
the imported array types and was reverted. The maintained correction adds the
existing `requirements-dev.lock` NumPy pin (`2.2.6`) to the mypy hook environment.
The original physics annotations and runtime behavior are preserved; no typing
ignore or validation bypass is introduced. The two initially failing modules
pass the corrected isolated hook. Full push checks remain mandatory.
