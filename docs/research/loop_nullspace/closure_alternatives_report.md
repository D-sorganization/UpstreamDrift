# Closure Alternatives With Re-Derived Kinematics

Research report for NSX-72 ([#12590](https://github.com/D-sorganization/UpstreamDrift/issues/12590),
epic #12593). Exploratory. Label `CLOSURE_ALTERNATIVE_COMPARISON`. The
calculation-level statement is Section "Closure Alternatives With Re-Derived
Kinematics" of [`loop_nullspace.tex`](loop_nullspace.tex); the contract is the
NSX-72 section of [`contracts.md`](contracts.md).

## Question

NSX-26 kept the welded kinematics of a recorded trajectory and only removed
reaction rows. A grip that really is a shaft-twist revolute or a ball joint is
a different closure: it keeps its own rows, so it admits a different set of
motions, has its own pose, velocity and acceleration residuals, and (for a
joint that adds coordinates) a different unactuated balance. This report
re-derives four grip models, each as its own closure, puts the same
trajectories to each, and states where the answer differs from NSX-26.

## Summary of Findings

1. **On a trajectory the weld admits, a rows-only grip model reproduces NSX-26
   exactly.** Counts and envelope widths of the shaft-twist revolute and the
   trail ball equal the NSX-26 `SHAFT_FREE5` and `BALL3` numbers to `1e-9`,
   on the spatial fixture and on all 199 anthro frames (124/124/118 feasible,
   75/75/81 infeasible, as in `restriction_anthro_stride1.json`). The
   restriction result is therefore the true dynamics of those grips, for that
   trajectory.
2. **What NSX-26 could not see is the motion each grip admits.** The same
   trajectories are admitted or refused per model with the model's own
   residuals (Table 1). The weld refuses a shaft twist at velocity residual
   1.3 and acceleration residual 1.7; the revolute admits it at 1e-15. The
   admitted velocity space grows from 8 (weld) to 9 (revolute) to 11 (ball)
   dimensions on the spatial fixture and from 35 to 36 to 38 on the anthro
   model.
3. **A ball joint at the lead hand is a different system, not a restriction.**
   It adds three unactuated coordinates. Its allocation nullity is 3 and
   its moment multipliers are determined (median envelope width 0), whereas
   the trail ball has nullity 3 with moment rows absent. Its torque need on the
   welded motion (7.46 N m) differs from the trail ball's (7.64 N m), and it
   cannot represent, or be judged on, the welded-coordinate twist and tilt
   motions that the other models share.
4. **The null-space dimension follows the rows exactly where a golden exists.**
   Allocation nullity 6, 5, 3 for weld, revolute, trail ball on the fixed-base
   fixture equals the exact `restriction_counts` golden. The lead ball (3) is a
   structural count, not an exact golden.
5. **Where conclusions differ from NSX-26.** (a) NSX-26 could only say a motion
   was "weld-consistent"; here a motion the weld refuses is judged on its own
   model (the twist motion is admitted by the revolute). (b) NSX-26's
   `BALL3` fixes only reactions; here it is also the closure whose position
   residual is second order in a tilt step (4.5e-9) while the revolute's is
   first order (4.7e-5). (c) The lead ball changes the envelope, not just
   narrows it (force-multiplier width 17.6 N against 13.1 N for the trail ball
   at an 8 N m box). Nothing else in the NSX-26 conclusion changes: restriction
   still narrows or leaves envelopes unchanged, and a tight capacity prior is
   still what makes frames infeasible.

None of this says which grip a golfer uses.

## Models

All closure rows are in the shaft-first grip frame (`x` along the club),
ordered `fx fy fz tx ty tz`. The weld pose residual is
`Phi_6(q) = log6(T_lead^-1 T_trail)` with the lead grip frame on the club.

| Model                  | Closure kind  | Rows kept          | Pose residual                    | Coordinates |
| ---------------------- | ------------- | ------------------ | -------------------------------- | ----------- |
| `WELD`                 | `WELD6`       | `fx fy fz tx ty tz` | all six components of `Phi_6`    | 14          |
| `SHAFT_TWIST_REVOLUTE` | `SHAFT_FREE5` | `fx fy fz ty tz`   | `Phi_6` without the twist `tx`   | 14          |
| `TRAIL_BALL`           | `BALL3`       | `fx fy fz`         | the three position components    | 14          |
| `LEAD_BALL`            | `WELD6`       | `fx fy fz tx ty tz` | `Phi_6(q, beta)`, six components | 17          |

`LEAD_BALL` hangs the club on a z-y-x ball (`beta`, three unactuated
coordinates) at the lead hand's grip point; the trail hand welds to the club.
The first three share the arm coordinates and differ only in the rows; their
Jacobians are row selections of the weld rows, which the tests confirm
independently against a central difference of each model's own pose residual
(maximum difference below `1e-6`). The shaft is the fixture's grip `z` axis
(the bare club lies on it); the module permutes it to `x` so the NSX-26 shaft
check applies.

Motion classes at the closed posture (all rates satisfy the acceleration level
for the rows they must satisfy):

- `WELDED` is in the null space of all six weld rows;
- `TWIST` is in the null space of the five revolute rows but turns the hand
  about the shaft;
- `TILT` is in the null space of the three force rows only;
- `OWN` is in the null space of the model's own rows.

## Admission per Model

Fixed-base spatial fixture, seed 7, 20 frames, gates `1e-6` at every level
(never loosened). Table 1 gives the maximum scaled residuals of the model's own
rows (position / velocity / acceleration); `-` marks a refusal.

| Trajectory | `WELD`                        | `SHAFT_TWIST_REVOLUTE`         | `TRAIL_BALL`                   | `LEAD_BALL`                   |
| ---------- | ----------------------------- | ------------------------------ | ------------------------------ | ----------------------------- |
| `WELDED`   | admit (2e-16 / 2e-15 / 2e-15) | admit (2e-16 / 2e-15 / 2e-15)  | admit (1e-16 / 2e-15 / 1e-15)  | admit (2e-16 / 2e-15 / 2e-15) |
| `TWIST`    | refuse (2e-16 / 1.3 / 1.7)    | admit (2e-16 / 1e-15 / 2e-15)  | admit (1e-16 / 1e-15 / 9e-16)  | refuse (2e-16 / 1.3 / 1.7)    |
| `TILT`     | refuse (2e-16 / 1.6 / 2.9)    | refuse (2e-16 / 1.4 / 2.3)     | admit (1e-16 / 9e-16 / 9e-16)  | refuse (2e-16 / 1.6 / 2.9)    |

Table 1. Table entries are model by trajectory, all models judged on the
welded-coordinate motion (embedded with a locked ball for `LEAD_BALL`).

A refusal is never repaired. Judged on its own motion (`OWN`) every model
admits its trajectory at 1e-15. The `LEAD_BALL` own motion rotates the ball
(rates above `1e-3`); no other model has those coordinates, so asking for it
raises "not representable" rather than mapping it.

Position level. The probe takes three steps of `1e-4` to `2e-4` along a twist
or tilt direction scaled to a unit weld rate, with zero rates, and evaluates
each model's own pose residual (maximum over the steps):

| Probe direction | `WELD`  | `SHAFT_TWIST_REVOLUTE` | `TRAIL_BALL` | `LEAD_BALL` |
| --------------- | ------- | ---------------------- | ------------ | ----------- |
| `TWIST`         | 2.0e-4 (refuse) | 9.3e-9 (admit) | 1.7e-9 (admit) | 2.0e-4 (refuse) |
| `TILT`          | 1.9e-4 (refuse) | 4.7e-5 (refuse) | 4.5e-9 (admit) | 1.9e-4 (refuse) |

Table 2. A model admits a direction when the rows it keeps close to second
order in the step; a model whose dropped rows the direction would violate at
first order refuses it. The probe is finite-step and local to one posture. It
is not a statement about a swing.

## Null-Space Dimensions

| Quantity                                   | `WELD` | `SHAFT_TWIST_REVOLUTE` | `TRAIL_BALL` | `LEAD_BALL` |
| ------------------------------------------ | ------ | ---------------------- | ------------ | ----------- |
| Admitted velocity dimension, spatial       | 8      | 9                      | 11           | 11          |
| Admitted velocity dimension, anthro        | 35     | 36                     | 38           | n/a         |
| Allocation nullity, spatial fixed base     | 6      | 5                      | 3            | 3           |
| Allocation nullity, anthro (range)         | 6-18   | 5-17                   | 3-15         | n/a         |
| Golden                                     | exact  | exact                  | exact        | structural  |

Table 3. The rows-only counts on the fixed-base and floating-root-no-feet variants
agree with the exact sympy golden `restriction_counts` (tested). `LEAD_BALL` has `n=17`,
`n_a=14`, `n_u=3`, `m=6`, `rank J_c = 6`, `rank J_cu^T = 3`, hence gauge 0 and
attainable torque variation 3, a generic-rank count. A floating root with no
feet cannot balance a model on this fixture (the closure Jacobian has zero root
columns, 3e-16, because both arms hang on the root) and is
recorded as `BALANCE_INCONSISTENT`, as in NSX-26.

## Dynamics and Envelopes

Declared prior, identical for every model: a symmetric torque box on the 14 arm
joints (`H-TORQUE-CAPACITY`, every slot ASSUMED, a synthetic fixture and not a
golfer). The smallest box holding all 20 welded frames, by a min-max LP per
frame, is 5.75 N m (weld), 6.10 (revolute), 7.64 (trail ball) and 7.46 (lead
ball). The boxes below bracket these numbers.

Welded trajectory, 20 frames. Entries are feasible frames, then the median
envelope width (torque in N m over the 14 joints; force multiplier in N).

| Box     | `WELD`                | `SHAFT_TWIST_REVOLUTE` | `TRAIL_BALL`        | `LEAD_BALL`         |
| ------- | --------------------- | ---------------------- | ------------------- | ------------------- |
| 6 N m   | 20; 6.26; 25.6 N      | 16; 3.76; 11.7 N       | 0 (all infeasible)  | 0 (all infeasible)  |
| 8 N m   | 20; 16.0; 75.9 N      | 20; 14.7; 72.8 N       | 20; 2.18; 13.1 N    | 20; 2.73; 17.6 N    |
| 12 N m  | 20; 24.0; 130.5 N     | 20; 24.0; 120.5 N      | 20; 12.3; 44.0 N    | 20; 12.5; 44.0 N    |

Table 4. Conflict sets of the infeasible frames name `H-TORQUE-CAPACITY` only.
No frame is balance-inconsistent on the fixed base. Torque-width ratio against
the weld: 0.49 / 0.96 / 1.00 (revolute), n/a / 0.18 / 0.56 (trail ball),
n/a / 0.23 / 0.57 (lead ball) at the three boxes. Moment multipliers (median width over the kept moment rows): weld 8.6, 25.8,
47.7 N m; revolute 8.5, 29.5, 50.0 N m. The revolute value can exceed the weld
value because the median runs over two moment rows (`ty tz`) against three; it
is not a pointwise comparison. The trail ball has no moment rows; the lead
ball's moment multipliers are fixed by the club's rotational balance (median
width 0).

Anthro record (199 frames, welded trajectory, NSX-26 priors): the revolute and
trail ball admit the trajectory with velocity residual below 3e-14 and
acceleration residual below 3e-8; the position level is skipped (no closure
plant), as in NSX-26. Feasible/infeasible counts: weld 124/75, revolute 124/75,
trail ball 118/81, equal to NSX-26. Median torque-envelope width ratio against
the weld: 1.000 (revolute), 0.991 (trail ball). Conflict sets name the assumed
capacity prior only.

## Where Conclusions Differ From NSX-26

| Topic                         | NSX-26 (restriction)                              | NSX-72 (re-derived)                                                            |
| ----------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------ |
| Admitted trajectories         | Only the welded one                               | Each model admits its own null space; weld refuses twist, revolute admits it   |
| Residuals reported            | Weld rows                                         | Position, velocity and acceleration of each model's own rows                   |
| Welded trajectory dynamics    | Restricted rows                                   | Identical for the rows-only models (a check, not an assumption)                |
| Admitted motion dimension     | Unchanged                                         | Grows 8, 9, 11 (spatial) and 35, 36, 38 (anthro)                               |
| Lead-hand ball                | Not expressible                                   | Own coordinates, nullity 3, determined moments, different force envelope       |
| Position level                | Skipped on anthro                                 | Evaluated on the spatial fixture; second order versus first order per model    |

The practical reading: the NSX-26 numbers can be quoted as the dynamics of a
revolute or ball grip **for the same welded trajectory**. They cannot be quoted
as evidence that a trajectory is feasible for that grip in general, and they say
nothing about a ball at the lead hand.

## Limitations

- **No grip measurements.** Whether a real grip transmits a twist moment, how
  large a palm moment is, or whether the lead hand slips needs the golfer's
  hand-club contact data. None is available, none is invented, and every model
  states its assumption in `GRIP_MODEL_SPECS[...].limitation`.
- **`LEAD_BALL` exists only on the synthetic spatial fixture.** It adds three
  coordinates and a ball-joint topology; the full-body spec builder supports one
  weld closure per spec, so the anthro model cannot host it. The anthro
  comparison covers the rows-only models only.
- **Rows-only anthro models are the same closure with fewer rows.** For a
  welded recorded trajectory they therefore agree with NSX-26 by construction;
  no anthro trajectory of the revolute or ball motion class exists.
- **Closed-posture frames.** Spatial frames sit at the fixture's closed posture
  with different rates; the position level is exercised only by the finite-step
  probe, not by a swing.
- **Synthetic boxes.** The 6, 8 and 12 N m boxes are declared to bracket the
  fixture's need; they are not golfer capacities. Rates are random members of
  the stated null spaces and are not golf swings.
- **Hand split** needs hand geometry along the shaft, which the spatial fixture
  does not define (it places the grip frame at the trail hand and no hand
  points along the shaft), so it is reported on the anthro model only, where it is unchanged
  from NSX-26 for the rows-only models.
- **No scientific qualification.** The ledger records software behaviour only.

## Reproduce

```bash
python3 -m pytest tests/unit/loop_nullspace/test_closure_alternatives.py -n 0 -q
python3 scripts/benchmarks/loop_nullspace_closure_alternatives.py \
    docs/research/loop_nullspace/records/closure_alternatives_spatial.json
LOOP_NULLSPACE_RESTRICTION_STRIDE=1 \
LOOP_NULLSPACE_ALTERNATIVES_JSON=docs/research/loop_nullspace/records/closure_alternatives_anthro_stride1.json \
python3 -m pytest tests/integration/test_loop_nullspace_closure_alternatives_anthro.py -n 0 -q
```

Records: `records/closure_alternatives_spatial.json`,
`records/closure_alternatives_anthro_stride1.json`.

## Non-Claims

Exploratory. Not evidence about the true grip mechanics or about any golfer.
Each grip model is a declared alternative and none is validated. Capacities and
priors are declared hypotheses and are identical for every model. A refused or
balance-inconsistent trajectory is a finding and is never repaired. No
tolerance was loosened.
