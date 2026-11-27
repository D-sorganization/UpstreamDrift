# Shaft Bend Kinematic Fit — Calculation Reference (C10 Standalone Slice)

This record describes the **kinematic** shaft-bend least-squares fit in
`src/shared/python/motion_matching/shaft_bend_fit.py`. It satisfies the
calculation-level reference asked for in the #15464 independent review, with
an explicitly **narrow** acceptance scope. End-to-end hand-model validation
(#15150 C13, impact-card and flight comparison, bent-shaft bundle) remains on
the hand-model track and is **not** claimed here.

## Assumptions

- The emitted `club_shaft_chain` lumped model and MJCF primitive order (Rx
  then Rz about club x and z) are the kinematic truth for marker propagation.
- Twist is not fitted; head roll is confounded with face-roll calibration.
- Targets are club-frame marker positions after rigid removal via shaft markers
  (`club_frame_targets`).
- Bend is a sum of static beam shape amplitudes per plane (`shape_basis`), with
  `max_tip_slope_deg` limiting the **combined** tip slope (sum of amplitudes per
  plane), not each amplitude independently when `shapes_per_plane = 2`.

## Method

Whole-trajectory `scipy.optimize.least_squares` over shape amplitudes with
marker residuals, second-difference smoothness, ridge toward straight, optional
contact breaks, and box bounds. With two shapes per plane the solver uses
variables `(t, u)` per plane with amplitudes `(u, t - u)` so the box bound on
`t` enforces the documented combined tip-slope limit.

## Acceptance Criteria (This Slice Only)

Software acceptance for the standalone solver is **only** the unit suite:

```bash
python3 -m pytest tests/unit/motion_matching/test_shaft_bend_fit.py -m unit --timeout=120
```

That suite covers synthetic recovery, MuJoCo kinematic parity, identifiability,
contact breaks, contracts, and the #15464 two-shape combined tip-slope bound
regressions. A green run is necessary and sufficient for this slice; it is **not**
physical acceptance of a captured swing fit.

## Out of Scope (Deferred)

- Real-swing bend fit with rigid-club double-counting resolved (C23 club-root
  refit). The spec half of C23 is done: `apply_shaft_chain` now attaches the
  shaft markers (the club cluster outside the head triad) to chain link 0
  through the `ClubheadShaft` frame, so in the engine a bend past joint 0 moves
  only the head markers (`tests/unit/motion_matching/test_shaft_chain_markers.py`).
  The refit of the lead wrist and closure with the club root following the
  shaft markers still needs the private capture and is not run here.
- Bent-shaft bundle, impact-card, or flight before/after comparison (#15150 C13).
- Any Explorer or live playback integration.

## Limitations

Kinematic only; no shaft elasticity dynamics, no contact forces, no
cross-engine replay receipt.