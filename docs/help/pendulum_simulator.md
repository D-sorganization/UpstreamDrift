---
title: Pendulum Simulator
tile_id: pendulum_simulator
status: complete
---

# Pendulum Simulator

## Purpose

Build a reduced mechanical model of a golf swing - two segments, three
segments, or a full upper body - drive it with joint torques, and watch what
the resulting dynamics do. This is the teaching and intuition tile: it exists
so you can change one mass, length, or torque profile and see the consequence
for proximal-to-distal sequencing, clubhead speed, and joint loads, with the
governing matrices shown alongside the animation.

## Inputs

The window has four tabs, each a separate model with its own controls:

| Tab                | Model                                          | Degrees of freedom |
| ------------------ | ---------------------------------------------- | ------------------ |
| Double Pendulum    | arms + shaft, clubhead as a point mass at tip  | 2                  |
| Triple Pendulum    | three-segment chain (shoulder, elbow, wrist)   | 3                  |
| Golfer Upper Body  | hub, two scapula/arm chains, shared club, closed kinematic loop | 8 DOF with 4 loop constraints, so 4 independent |
| Screw Kinematics   | screw-theory visualisation of the active pose  | n/a                |

Physical parameters, from `PendulumParams` in
`src/shared/python/pendulum_simulator/physics.py` (the triple and golfer models
have analogous parameter sets):

| Parameter              | Symbol | Unit        | Typical value in code |
| ---------------------- | ------ | ----------- | --------------------- |
| Arm (segment 1) mass   | m1     | kg          | ~5.0                  |
| Shaft (segment 2) mass | m2     | kg          | ~0.30                 |
| Arm length             | L1     | m           | ~0.65                 |
| Shaft length           | L2     | m           | ~1.10                 |
| Clubhead mass          | mClub  | kg          | ~0.20                 |
| Gravity                | g      | m/s^2       | 9.80665, toggleable to 0 |
| Viscous damping        | b1, b2 | N*m*s/rad   | 0.0                   |
| Coulomb friction       | mu1, mu2 | N*m       | 0.0                   |

Initial conditions and drive:

- Joint angles [rad]: theta1 is the absolute angle of segment 1 from downward
  vertical, positive counterclockwise; phi is segment 2 relative to segment 1,
  same sign convention. Both zero means hanging straight down at equilibrium.
- Joint angular velocities [rad/s], entered as `dtheta1` and `dphi`.
- Joint torque profiles [N*m], authored per joint. A function generator dialog
  is available for shaping them.
- Joint limits: angle bounds [rad] with a contact stiffness [N*m/rad] and
  damping [N*m*s/rad] (defaults 500.0 and 20.0).
- Torque saturation: symmetric magnitude clamps [N*m] per joint, unbounded by
  default.
- Animation playback speed, 0.05 to 20.0 times real time.

## Outputs

- Animated 2D rendering of the segment chain over the simulated swing.
- Full state trajectory: time [s], joint angles [rad], joint angular velocities
  [rad/s].
- Mass (inertia) matrix M(q) terms, the Coriolis/centrifugal vector
  C(q, qdot)*qdot, and the gravity torque vector G(q), displayed with physical
  labels and live at the selected sample.
- Jacobians for the active model, with an equations popup documenting the
  dynamics and Jacobian derivations.
- Torque history plots and a torque preview for the authored profiles.
- Analysis tab: 2D line plots of any registered extracted quantity against any
  other, and 3D parameter-sweep surfaces over the active model.
- Perturbation panel and counterfactual/swing-comparison dialogs for
  side-by-side runs.
- Screw-theory visualisation of the active configuration.

## Method

The double-pendulum model is a Lagrangian formulation in relative coordinates
with segment 1 as arms (shoulder to wrist), segment 2 as shaft (wrist to
clubhead), and the clubhead as a point mass at the tip
(`src/shared/python/pendulum_simulator/physics.py`). Dynamics are assembled
from an explicit mass matrix, Coriolis/centrifugal vector, gravity vector, and
a dissipative term combining viscous and Coulomb friction; joint limits are
enforced through a Hermite smoothstep penalty on penetration depth and velocity
rather than a hard stop.

The triple model extends the same construction to three segments
(`physics_triple.py`). The golfer model
(`physics_golfer.py`) is a closed-loop upper-body chain: a fixed pivot, a
massless standoff to a hub, two upper-body (scapula) segments to the shoulders,
independent arm chains through elbow and hand, and both hands attached to a
shared club segment. Attaching two wrists to one club imposes 2 x 2 = 4 scalar
constraints on 8 DOFs, leaving 4 independent. Models are registered through
`model_registry.py` rather than hardcoded, each declaring its DOF count, state
size, parameter class, and simulation runner.

Background on the pendulum model family is in
[../engines/pendulum.md](../engines/pendulum.md). Identifiability limits of the
double-pendulum parameterisation are analysed in
[../research/proximal_distal_energy_transfer/DOUBLE_PENDULUM_IDENTIFIABILITY.md](../research/proximal_distal_energy_transfer/DOUBLE_PENDULUM_IDENTIFIABILITY.md).

## Limitations

- **Everything this tile produces is model output, not measurement.** It is a
  reduced-order mechanical analogue of a swing, not a swing.
- Planar and reduced by construction. The double and triple models are
  in-plane chains; segments are rigid, joints are ideal revolutes, and the
  clubhead is a point mass. No shaft bending, no clubface, no three-dimensional
  out-of-plane motion in those two models.
- No ball and no impact. The simulation ends with the mechanism's state; it
  does not produce launch conditions or a trajectory. Chain to the
  [Swing to Flight Pipeline](swing_flight_pipeline.md) for that.
- No muscles, no tendons, no activation dynamics, and no metabolic cost. Joint
  torques are prescribed inputs, not solved from a musculoskeletal model.
- Parameters are entered by hand and are not fitted to any subject. The
  "typical" values in the code are defaults, not anthropometric measurements.
- The identifiability review above is the standing caveat on inferring
  parameters from this model class: do not read fitted masses or lengths off it
  as physical facts.
- The registry description mentions the double, triple, and golfer models; the
  fourth tab, Screw Kinematics, is a kinematic visualisation surface and runs
  no dynamics of its own.
- Matplotlib is needed for the analysis plots; without it that panel degrades
  to a placeholder while the simulation still runs.

## See Also
- [Pendulum Models](../engines/pendulum.md)
- [Double Pendulum Identifiability](../research/proximal_distal_energy_transfer/DOUBLE_PENDULUM_IDENTIFIABILITY.md)
- [Swing Objective Lab](swing_objective_lab.md) - optimises this model class against competing objectives
- [Simulation Controls](simulation_controls.md)
- [Analysis Tools](analysis_tools.md)
- [Engine Selection Guide](engine_selection.md)
