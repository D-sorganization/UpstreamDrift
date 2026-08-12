# Proximal-to-Distal Evidence Schema V2

## Purpose

The v2 evidence contract makes the model ladder falsifiable without changing
the observable at each fidelity tier. It supplements the planar
`JointTransferTrajectory`; it does not reinterpret or invalidate the existing
v1 hand-path evidence.

The canonical implementation is
`src/shared/python/biomechanics/interaction_evidence.py`. The preregistered
prediction registry is
`data/model_completion_predictions.json`.

## Prediction Contract

Every primary prediction declares, before its outcome is evaluated:

- a stable prediction and hypothesis identifier;
- the estimand and intervention;
- the expected result and a result that would falsify it;
- competing explanations and negative controls;
- required model tiers and a preoutcome numerical tolerance; and
- one status: `untested`, `supported`, `contradicted`, or `inconclusive`.

An unexecuted model tier remains `untested`. A missing observable or
non-identifiable parameter produces `inconclusive`, not support. Status changes
must cite a versioned evidence bundle and exact code commit.

## Named Spatial Interface Contract

`SpatialWrenchTrajectory` stores arrays with sample count `T` and interface
count `I`:

| Quantity                                        |       Shape | Layout                     |
| ----------------------------------------------- | ----------: | -------------------------- |
| Time                                            |      `(T,)` | seconds                    |
| Reference Position                              | `(T, I, 3)` | Cartesian metres           |
| Total, Drift, Control, and Optional ZVCF Wrench | `(T, I, 6)` | `[Fx, Fy, Fz, Mx, My, Mz]` |
| Compatible Twist                                | `(T, I, 6)` | `[vx, vy, vz, wx, wy, wz]` |

Each named interface declares proximal and distal bodies, frame, reference
point, and action direction. Wrench power is

$$
P = \mathbf F \mathbin{\cdot} \mathbf v
  + \mathbf M \mathbin{\cdot} \boldsymbol\omega.
$$

Total, drift, and control are evaluated at the same state and must satisfy

$$
\mathcal W_{\mathrm{total}}
= \mathcal W_{\mathrm{drift}}
+ \mathcal W_{\mathrm{control}}.
$$

ZVCF remains a separately defined diagnostic and is never inserted into that
identity.

## Reference and Frame Transformations

When the reference moves from $O$ to $P$, the wrench and twist move together:

$$
\mathbf M_P = \mathbf M_O
- (\mathbf r_P-\mathbf r_O) \times \mathbf F,
\qquad
\mathbf v_P = \mathbf v_O
+ \boldsymbol\omega \times (\mathbf r_P-\mathbf r_O).
$$

The transformation must preserve total power. A frame change must use a
proper rotation with determinant `+1` on positions, forces, moments, linear
velocities, and angular velocities. Reflections are rejected.

## Numerical Tolerances

Numerical tolerances are evidence, not convenient constants. Each phase must
calibrate its primary tolerance before evaluating its preferred outcome. The
provided calibration helper uses the difference between the two finest
predeclared discretizations multiplied by a safety factor and records both an
absolute and result-scaled relative bound. Analytical closure and manufactured
solutions may impose a stricter independent floor.

If a solver, contact model, filter, or event detector changes, the applicable
tolerance is recalibrated and versioned. A tolerance must not be loosened after
inspecting whether a scientific prediction passed.

## Migration and Compatibility

Planar adapters may embed their two Cartesian components in the first two
spatial axes and their scalar couple/angular rate in the third rotational
axis. Migration must retain the original v1 artifact, model tier, signs,
reference point, and source hashes. Existing consumer snapshots remain valid
until a reviewed v2 evidence export supersedes them.

## Mandatory Invariants

Every applicable adapter and evidence writer tests:

1. finite, strictly increasing time and dimensionally valid arrays;
2. unique named interfaces and explicit action ownership;
3. same-state wrench and power reconstruction;
4. reference-transport and proper-frame power invariance;
5. action–reaction and virtual-work consistency;
6. numerical convergence and explicit undefined/unsupported states; and
7. traceability from prediction to evidence bundle, inputs, solver, and commit.
