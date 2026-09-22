# Motion-Matching Handoff

Deferred external validation: six Board plans live in `docs/development/planning/`.
Software remains active; no physical evidence is supplied. See the current
`docs/development/HANDOFF.md` for scope, prior #9546 closure and publication gates.

## Tour Baselines TB-06: Constrained Upper-Body Golfer (#10591)

Branch `feat/10591-upper-body-capture`; parent epic [#10584](https://github.com/D-sorganization/UpstreamDrift/issues/10584); program [#10363](https://github.com/D-sorganization/UpstreamDrift/issues/10363). Lease holder: `codex` (session `01a0ca1f-c09b-78a1-8878-37402a7d4eec`).

- PR [#10732](https://github.com/D-sorganization/UpstreamDrift/pull/10732) merged as `234b8dc146c5b2c24d80cf9eb4d09d37e68c1b72`. It establishes `upper_body_replay`: constrained native replay, separately recorded seven actuator torques and four constraint reactions, exact terminal source frame, native kinematic points, explicit planar marker attachments, rigid capture-frame embedding, and standard physical marker metrics only on an identical body-target clock.
- The evaluator does not synthesize 3D attachments, infer capture transforms, interpolate source clocks, or claim acceptance. A capture-specific fit must calibrate fixed geometry, feasible `q0`/`v0`, marker attachments, and bounded torque controls before it can produce Driver or Iron evidence.
- PR [#10733](https://github.com/D-sorganization/UpstreamDrift/pull/10733) merged as `91fff0cc582d5a906bfb0530d7bda2d8c67aaf6c`. PR [#10735](https://github.com/D-sorganization/UpstreamDrift/pull/10735) merged as `930e1edcbd1ec60d7deb83907694662e8496feb7`, superseding stale/conflicted #10645 with native closed-loop fitting and continuous-control replay. Its controls, residual policy, and finite bounded-solver tolerances are shared through `motion_matching.bernstein_controls`.
- `upper_body_capture.py` now retains native C3D source-clock metadata and produces one fixed rigid capture plane from declared markers. On the six shoulder, elbow, and wrist markers, the real Driver and Iron traces yield 110.4 mm and 112.7 mm plane RMSE respectively. This is a diagnostic, not a flattened projection or qualification result.
- PR [#10740](https://github.com/D-sorganization/UpstreamDrift/pull/10740) merged as `56ad6a8dcd13221699ec14442b13c1df6fc5729a`. Its source-clock, planarity lower-bound, and campaign-receipt contracts retain reproducible Driver and Iron receipts under `docs/plans/tour_baselines/evidence/`: 110.4 mm and 112.7 mm irreducible normal RMSE, respectively, above the profile 55 mm 3D marker ceiling. Both outcomes are rejected and indexed in `reports/matched_swing_ledger.json`.
- Source boundaries are `physics_golfer.py`, `golfer_constraints.py`, `simulation_golfer.py`, and `model_registry.py`. Do not edit `vendor/ud-tools`.
- Focused validation: `python3 -m pytest tests/unit/motion_matching/test_bernstein_controls.py tests/unit/engines/physics_engines/pendulum/test_golfer_fit.py tests/unit/pendulum_simulator/test_upper_body_replay.py -q -n 0 --no-cov --timeout=60` (32 passed; manufactured optimizer regression 9.2 s); scoped Ruff, format, and `python3 scripts/ci/check_dry_duplication_gate.py` pass.
- Campaign conclusion: do not calibrate planar marker attachments, q0/v0, or torque controls against these captures because their measured normal residual alone exceeds the full 3D acceptance ceiling. A spatial upper-body topology would require a separately scoped follow-up; this planar topology retains both rejected outcomes.

## Tour Baselines TB-07: Reconcile Existing Reference and Full-Body Results (#10592) [MERGED]

PR [#10730](https://github.com/D-sorganization/UpstreamDrift/pull/10730) merged as `22b7fd7a534ddfbc0b1db943e6443fdc9ac7efb3`. The TB-04 Driver and Iron receipts remain `DISQUALIFIED`; coverage and the unified ledger report them as rejected rather than qualified.

## Required Before Continuing

- Read `AGENTS.md`, `CLAUDE.md`, and `docs/development/DEVELOPMENT_LOG.md`.
- Keep #10591 open until a merged PR demonstrably meets its acceptance criteria. A replay foundation or a rejected candidate is not qualification.
- Manual governance: UP-D0 (#9066) and UP-D1 (#9067) remain release blockers. Edit only the `manuals/upstreamdrift` QMD source and run `python3 -m scripts.check_design_manual_governance` for governed changes.
- Update this handoff, the development-log entry, and exactly one `SPEC.md` change-log row for every substantive PR.
