# Constraint and Internal-Force Diagnostics

## Purpose

Constraint rank, force-allocation rank, and sensor rank answer different
questions. This diagnostic keeps their domains explicit so that a nullspace in
one map cannot be used as evidence about an unobserved quantity in another. It
consolidates three existing authorities and adds adverse analytical cases. It
does not infer participant force, muscle action, scapular strategy, or coaching
guidance.

## Three Different Nullspaces

For the planar closed loop, the kinematic constraint is

$$
J_c(q)\dot q=0.
$$

The right nullspace of $J_c$ contains locally feasible generalized velocities.
It contains no hand forces and cannot determine constraint multipliers.
Dynamics, applied loads, constitutive laws, and a solved closed state remain
necessary.

For two three-axis point forces, the instantaneous measurement equation is

$$
\begin{bmatrix}F\\M_O\end{bmatrix}
=A_F(r_1,r_2;O)
\begin{bmatrix}F_1\\F_2\end{bmatrix}.
$$

The right nullspace of $A_F$ contains individual-force allocations invisible
to the net club wrench. For separated contacts, the invisible mode is the
equal-and-opposite component along the contact line. This is a measurement
ambiguity, not evidence that a golfer produces that mode. If both hands supply
complete six-axis wrenches, twelve inputs map to six net-wrench outputs; six
allocation directions remain unobservable without bilateral sensing or an
independently justified constitutive model.

## Dimensionally Declared Conditioning

Raw closure Jacobians mix angular and translational generalized coordinates.
The report therefore evaluates $J_cS$, where $S$ maps dimensionless normalized
coordinates to declared increments of 1 rad for each angular coordinate and
0.75 m for each translation. Positive scaling preserves exact rank and
nullity, but not singular values or condition number. Three alternative
translation scales are retained to expose that dependence.

Raw wrench maps likewise mix force and moment. The report divides net-moment
rows by a fixed 0.10 m reference length. For the full two-hand wrench map,
input moments also use a force-times-0.10 m scale. The resulting map is
dimensionless; its conditioning remains conditional on the declared length.

## Adverse Geometries and Tolerances

At the declared regular planar geometry, the scaled $4\times5$ closure map has
rank 4 and nullity 1. An analytical alignment with both arm angles zero and
grip angle $\pi/2$ has rank 3 and nullity 2. The latter is an algebraic
Jacobian singularity only: constraint-position closure, anatomical feasibility,
and occurrence in a human swing have not been established. Near the alignment,
rank remains 4 while the scaled condition number grows; its magnitude changes
with the coordinate scale.

Coincident point contacts remove all lever-arm moment information, leaving
rank 3 and nullity 3. A separated 0.20 m point-force map has rank 5 and nullity
1 under the declared normalization. At a 1 micrometre separation, changing the
relative SVD tolerance from $10^{-8}$ to $10^{-6}$ changes the numerical rank
decision from 5 to 3. Binary rank labels therefore require the tolerance and
adverse geometry to remain part of the evidence.

## Spatial Prescribed-State Boundary

The subject-scaled spatial source retains local constraint rank 6 at its
prescribed states, but every state fails the 5 mm anatomical contact-closure
gate. Full local row rank at an open state is not a feasible bilateral
trajectory, a solved contact force, or human evidence. Its legacy condition
values are retained in the source artifact but are not compared here because
that artifact lacks an explicit generalized-coordinate scale contract.

## Falsifiers and Remaining Work

The consolidated evidence fails if:

1. the regular planar map loses rank 4/nullity 1;
2. the analytical alignment does not gain a feasible-velocity direction;
3. coincident contacts retain moment observability above rank 3;
4. separated point forces lose their axial allocation ambiguity;
5. a scale-free condition number is reported for a dimensionally mixed map; or
6. a kinematic or measurement nullspace is promoted to participant force,
   muscle action, or strategy evidence.

Future work must locate scaled singularity margins on feasible closed
trajectories, propagate sensor and contact-position uncertainty, test compliant
and distributed contacts, and acquire governed bilateral six-axis
measurements.

## Reproduction

```powershell
python -m scripts.research.proximal_distal_energy.run_constraint_internal_force_diagnostics write
python -m scripts.research.proximal_distal_energy.run_constraint_internal_force_diagnostics validate
python -m pytest -q -n 0 tests/research/test_constraint_internal_force_diagnostics.py tests/research/test_constraint_internal_force_diagnostics_evidence.py
```

The machine-readable authority is
[`data/constraint_internal_force_diagnostics.json`](data/constraint_internal_force_diagnostics.json).
