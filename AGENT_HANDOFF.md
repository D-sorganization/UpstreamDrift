# Motion-Matching Handoff

## Tour Baselines TB-06: Constrained Upper-Body Golfer (#10591)

Branch `feat/10591-upper-body-baseline`; parent epic [#10584](https://github.com/D-sorganization/UpstreamDrift/issues/10584); program [#10363](https://github.com/D-sorganization/UpstreamDrift/issues/10363). Lease holder: `codex` (session `01a0ca1f-c09b-78a1-8878-37402a7d4eec`).

- Commit `0d620ecf9` establishes `upper_body_replay`: constrained native replay, separately recorded seven actuator torques and four constraint reactions, exact terminal source frame, native kinematic points, explicit planar marker attachments, rigid capture-frame embedding, and standard physical marker metrics only on an identical body-target clock.
- The evaluator does not synthesize 3D attachments, infer capture transforms, interpolate source clocks, or claim acceptance. A capture-specific fit must calibrate fixed geometry, feasible `q0`/`v0`, marker attachments, and bounded torque controls before it can produce Driver or Iron evidence.
- Source boundaries are `physics_golfer.py`, `golfer_constraints.py`, `simulation_golfer.py`, and `model_registry.py`. Do not edit `vendor/ud-tools`.
- Focused validation: `python3 -m pytest tests/unit/pendulum_simulator/test_upper_body_replay.py tests/unit/pendulum_simulator/test_golfer_constraints.py tests/unit/test_simulation_golfer.py -q -n 0 --no-cov --timeout=60`; scoped Ruff check and format pass.
- Next step: implement bounded torque fitting and capture-specific calibration over the explicit TB-06 replay and marker contracts.

## Tour Baselines TB-07: Reconcile Existing Reference and Full-Body Results (#10592) [MERGED]

PR [#10730](https://github.com/D-sorganization/UpstreamDrift/pull/10730) merged as `22b7fd7a534ddfbc0b1db943e6443fdc9ac7efb3`. The TB-04 Driver and Iron receipts remain `DISQUALIFIED`; coverage and the unified ledger report them as rejected rather than qualified.

## Required Before Continuing

- Read `AGENTS.md`, `CLAUDE.md`, and `docs/development/DEVELOPMENT_LOG.md`.
- Keep #10591 open until a merged PR demonstrably meets its acceptance criteria. A replay foundation or a rejected candidate is not qualification.
- Manual governance: UP-D0 (#9066) and UP-D1 (#9067) remain release blockers. Edit only the `manuals/upstreamdrift` QMD source and run `python3 -m scripts.check_design_manual_governance` for governed changes.
- Update this handoff, the development-log entry, and exactly one `SPEC.md` change-log row for every substantive PR.
