# Lift Kinetics Parity Report (LIFT-7, OSL-8, #11747, #13045)

This report records cross-engine parity of lift kinetics computed from the
same LIFT-5 same-input bundle. It is software parity evidence only: equal
numbers in every engine do not establish that the numbers are physically
right. Scientific qualification stays with the design-manual governance
pathway.

## Status

| Quantity                                        | Back Squat                                                       | Deadlift, Bench, Snatch, Clean and Jerk                          |
| ----------------------------------------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------- |
| GRF per foot and net, CoP, free moment (GCV-1)  | Parity in MuJoCo, Drake, Pinocchio, OpenSim                      | Parity in MuJoCo, Drake, Pinocchio, OpenSim (OSL-8, #13045)      |
| GRF moment about the CoM                        | Pending (adapters expose no CoM)                                 | Pending                                                          |
| Per-hand wrench and equivalent couple (GCV-7/8) | Parity in MuJoCo and Drake; unavailable in Pinocchio and OpenSim | Parity in MuJoCo and Drake; unavailable in Pinocchio and OpenSim |
| Joint moments and powers                        | Unavailable (same-input adapters have no ID pipeline; #13045)   | Unavailable (same-input adapters have no ID pipeline; #13045)    |
| Bar-system energy                               | Pending (no barbell body in the bundle)                          | Pending                                                          |
| L5/S1 compression and shear                     | Pending                                                          | Pending                                                          |
| Plots sheet per lift                            | Pending                                                          | Pending                                                          |

The static-hold bar wrench of the four external model packs (LIFT-4, #11744)
is reported separately by `tests/unit/lifting/pack_audit/test_bar_hold_parity.py`.

## Method

`src/shared/python/lifting/kinetics_parity.py` evaluates, at chosen reference
states of each registered lift bundle, in each engine:

- the shared contact law on that engine's own sphere kinematics, grouped by
  foot through the shared ground-reaction core
  (`ground_reaction_breakdown`, GCV-1): per-foot and net force, centre of
  pressure and free moment;
- the per-hand wrench on the grip from the engine's closure KKT multiplier,
  with the holding hand from club Newton-Euler (GCV-7/8). An engine whose
  same-input adapter has no multiplier extraction reports the hand wrench as
  unavailable with a reason, never as zero.

State `k` uses `reference_q[k]`, `reference_v[k]` and the effort applied from
it. `scripts/measure_lift_kinetics_parity.py` writes the records
`evidence/<lift>/kinetics_parity.json` (every 20th state of the 200-step
window, 11 states) across all five barbell lifts.

## Tolerances

Fixed before measuring and not loosened: the acceleration-parity level L1 of
epic #11605 that the GCV-8 grip test already uses,
`|dX| <= 1e-6 + 1e-8 * max|X|` (N or N m, with the net force as scale), and
1e-8 m for the CoP.

## Back Squat Results

Largest difference from MuJoCo over the 11 states of `back_squat_bundle.npz`
(bundle spec SHA-256 `aedbf64e...`):

| Quantity              | Drake   | Pinocchio   | OpenSim     |
| --------------------- | ------- | ----------- | ----------- |
| Left foot force (N)   | 1.3e-10 | 1.2e-10     | 9.6e-11     |
| Right foot force (N)  | 1.1e-10 | 1.3e-10     | 8.5e-11     |
| Net force (N)         | 2.3e-10 | 2.5e-10     | 1.5e-10     |
| Net CoP (m)           | 4.0e-15 | 5.2e-15     | 4.3e-15     |
| Net free moment (N m) | 6.9e-12 | 4.5e-12     | 5.6e-12     |
| Left hand force (N)   | 3.4e-11 | unavailable | unavailable |
| Right hand force (N)  | 6.7e-10 | unavailable | unavailable |
| Grip net force (N)    | 6.4e-10 | unavailable | unavailable |
| Grip couple (N m)     | 2.0e-10 | unavailable | unavailable |

Every available quantity agrees to round-off, five or more orders of
magnitude inside the tolerance.

## Per-Lift Results Across All Five Lifts

Largest difference from MuJoCo over the 11 states for all registered lifts:

| Lift           | Engine    | Net GRF (N) | Net CoP (m) | Net Free Moment (N m) | Grip Net Force (N) | Grip Couple (N m) |
| -------------- | --------- | ----------- | ----------- | --------------------- | ------------------ | ----------------- |
| Back Squat     | Drake     | 2.33e-10    | 4.01e-15    | 6.89e-12              | 6.42e-10           | 2.02e-10          |
| Back Squat     | Pinocchio | 2.53e-10    | 5.18e-15    | 4.47e-12              | unavailable        | unavailable       |
| Back Squat     | OpenSim   | 1.51e-10    | 4.28e-15    | 5.64e-12              | unavailable        | unavailable       |
| Deadlift       | Drake     | 2.96e-10    | 1.24e-14    | 3.54e-12              | 3.00e-11           | 3.57e-11          |
| Deadlift       | Pinocchio | 1.82e-10    | 1.13e-14    | 2.82e-12              | unavailable        | unavailable       |
| Deadlift       | OpenSim   | 1.86e-10    | 1.44e-14    | 3.52e-12              | unavailable        | unavailable       |
| Bench Press    | Drake     | 1.95e-10    | 6.75e-14    | 2.97e-13              | 9.45e-11           | 1.13e-10          |
| Bench Press    | Pinocchio | 9.12e-11    | 2.06e-14    | 1.91e-13              | unavailable        | unavailable       |
| Bench Press    | OpenSim   | 4.88e-11    | 3.31e-14    | 3.60e-13              | unavailable        | unavailable       |
| Snatch         | Drake     | 3.40e-10    | 1.39e-14    | 1.86e-11              | 4.23e-11           | 5.72e-11          |
| Snatch         | Pinocchio | 2.07e-10    | 1.36e-14    | 1.04e-11              | unavailable        | unavailable       |
| Snatch         | OpenSim   | 1.25e-10    | 1.47e-14    | 6.85e-12              | unavailable        | unavailable       |
| Clean and Jerk | Drake     | 3.57e-10    | 1.01e-14    | 4.10e-12              | 9.08e-11           | 1.18e-10          |
| Clean and Jerk | Pinocchio | 2.73e-10    | 1.36e-14    | 2.69e-12              | unavailable        | unavailable       |
| Clean and Jerk | OpenSim   | 2.75e-10    | 1.52e-14    | 3.46e-12              | unavailable        | unavailable       |

In all five lifts:
- Net GRF agrees within $3.6\times10^{-10}$\,N (bound $\sim 10^{-5}$\,N).
- Net CoP agrees within $6.8\times10^{-14}$\,m (bound $10^{-8}$\,m).
- Drake grip wrench agrees within $6.5\times10^{-10}$\,N force and $2.1\times10^{-10}$\,N\,m couple.
- Pinocchio and OpenSim adapters report grip wrench as unavailable with exact reason `"pinocchio/opensim same-input adapter has no closure-multiplier grip analysis"`.

## Inverse Dynamics and Joint Moments

The same-input adapters for MuJoCo, Drake, and Pinocchio (`VectorPlant` in
`src/shared/python/motion_matching/same_input/plant.py` and the engine models in
`src/engines/physics_engines/*/python/`) implement forward-dynamics evaluation
(`accelerations(q, v, tau)`), closed-loop weld residuals, contact sphere
kinematics, and weld-closure multiplier extraction. None of the same-input
adapters implement an inverse-dynamics joint moment pipeline on bundle states
with applied external contact and grip wrenches. In contrast, OpenSim evaluates
inverse dynamics via its dedicated `InverseDynamicsSolver` pipeline
(`opensim_full_body_lifts.py`), applying external ground and hand wrenches to
`calcn_l`/`calcn_r` and `hand_l`/`hand_r`.

Per OSL-8 scope rules, joint moments are recorded as unavailable across the
same-input adapters with the reason `"same-input adapters have no inverse-dynamics joint moment evaluation"`,
rather than building an unvetted ID pipeline.

## Unexplained Observations

The MuJoCo values for scale show a net vertical GRF of 608 N to 2030 N
(0.57 to 1.91 times the 1062 N spec weight) and a horizontal net force of up
to 524 N over the 0.2 s window around the squat bottom. The bundle carries no
barbell load, so these magnitudes are not established as physical: they come
from the penalty contact law at the tracked reference states (the IK held the
lowest foot sphere up to 0.54 mm below the ground). Parity in all engines
shows only that every engine evaluates the same law on the same kinematics.
For bench press, feet contact the ground at the start pose (Fz = 642 N) and
leave the floor during mid-press (Fz = 0 N, CoP = None).

## Reproduce

```bash
MUJOCO_GL=egl python3 scripts/measure_lift_kinetics_parity.py
python3 -m pytest -o addopts="" tests/unit/lifting/test_squat_kinetics_parity.py
```

