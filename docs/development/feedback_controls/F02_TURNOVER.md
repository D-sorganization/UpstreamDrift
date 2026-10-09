# F02 Distributed Feedback Turnover (#11786)

F02 adds a phase-gated controller core on `feat/f02-distributed-feedback-11786`
in the isolated `UpstreamDrift-f02-11786` worktree. This branch is stacked on
F01 commit `f994983da9` and PR #11808 until F01 merges. The F02 PR body is
the authoritative record of final commit, checks and handoff state.

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
