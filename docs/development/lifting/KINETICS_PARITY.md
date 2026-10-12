# Lift Kinetics Parity Report (LIFT-7, #11747)

This report records cross-engine parity of lift kinetics computed from the
same LIFT-5 same-input bundle. It is software parity evidence only: equal
numbers in every engine do not establish that the numbers are physically
right. Scientific qualification stays with the design-manual governance
pathway.

## Status

| Quantity                                        | Back Squat                                                       | Deadlift, Bench, Snatch, Clean and Jerk |
| ----------------------------------------------- | ---------------------------------------------------------------- | --------------------------------------- |
| GRF per foot and net, CoP, free moment (GCV-1)  | Parity in MuJoCo, Drake, Pinocchio, OpenSim                      | Pending LIFT-5 bundles                  |
| GRF moment about the CoM                        | Pending (adapters expose no CoM)                                 | Pending                                 |
| Per-hand wrench and equivalent couple (GCV-7/8) | Parity in MuJoCo and Drake; unavailable in Pinocchio and OpenSim | Pending                                 |
| Joint moments and powers                        | Pending                                                          | Pending                                 |
| Bar-system energy                               | Pending (no barbell body in the bundle)                          | Pending                                 |
| L5/S1 compression and shear                     | Pending                                                          | Pending                                 |
| Plots sheet per lift                            | Pending                                                          | Pending                                 |

The static-hold bar wrench of the four external model packs (LIFT-4, #11744)
is reported separately by `tests/unit/lifting/pack_audit/test_bar_hold_parity.py`.

## Method

`src/shared/python/lifting/kinetics_parity.py` evaluates, at chosen reference
states of the bundle, in each engine:

- the shared contact law on that engine's own sphere kinematics, grouped by
  foot through the shared ground-reaction core
  (`ground_reaction_breakdown`, GCV-1): per-foot and net force, centre of
  pressure and free moment;
- the per-hand wrench on the grip from the engine's closure KKT multiplier,
  with the holding hand from club Newton-Euler (GCV-7/8). An engine whose
  same-input adapter has no multiplier extraction reports the hand wrench as
  unavailable with a reason, never as zero.

State `k` uses `reference_q[k]`, `reference_v[k]` and the effort applied from
it. `scripts/measure_lift_kinetics_parity.py` writes the record
`evidence/back_squat/kinetics_parity.json` (every 20th state of the 200-step
window, 11 states).

## Tolerances

Fixed before measuring and not loosened: the acceleration-parity level L1 of
epic #11605 that the GCV-8 grip test already uses,
`|dX| <= 1e-6 + 1e-8 * max|X|` (N or N m, with the net force as scale), and
1e-8 m for the CoP.

## Back Squat Results

Largest difference from MuJoCo over the 11 states (source commit
`17f4735560`, bundle spec SHA-256 `36702e05...`):

| Quantity              | Drake   | Pinocchio   | OpenSim     |
| --------------------- | ------- | ----------- | ----------- |
| Left foot force (N)   | 3.2e-10 | 2.2e-10     | 2.0e-10     |
| Right foot force (N)  | 1.8e-10 | 1.8e-10     | 3.2e-10     |
| Net force (N)         | 2.3e-10 | 3.2e-10     | 4.7e-10     |
| Net CoP (m)           | 1.7e-14 | 9.8e-15     | 1.1e-14     |
| Net free moment (N m) | 6.1e-12 | 3.5e-12     | 5.9e-12     |
| Left hand force (N)   | 1.7e-11 | unavailable | unavailable |
| Right hand force (N)  | 2.8e-11 | unavailable | unavailable |
| Grip net force (N)    | 1.1e-11 | unavailable | unavailable |
| Grip couple (N m)     | 7.2e-13 | unavailable | unavailable |

Every available quantity agrees to round-off, five or more orders of
magnitude inside the tolerance.

## Unexplained Observations

The MuJoCo values for scale show a net vertical GRF of 608 N to 2030 N
(0.57 to 1.91 times the 1062 N spec weight) and a horizontal net force of up
to 524 N over the 0.2 s window around the squat bottom. The bundle carries no
barbell load, so these magnitudes are not established as physical: they come
from the penalty contact law at the tracked reference states (the IK held the
lowest foot sphere up to 0.54 mm below the ground). Parity in all engines
shows only that every engine evaluates the same law on the same kinematics.

## Reproduce

```bash
MUJOCO_GL=egl python3 scripts/measure_lift_kinetics_parity.py
python3 -m pytest -o addopts="" tests/unit/lifting/test_squat_kinetics_parity.py
```
