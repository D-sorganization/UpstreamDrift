# JaxSim Floating-Base Velocity Conventions

Issue #6652 adds an explicit convention contract for floating-base dynamics
terms before JaxSim adapters are allowed to feed `h`, `g`, or drift terms into
`PhysicsEngine`.

## Canonical Suite Contract

- Spatial vectors are always ordered as `[angular; linear]`, matching
  `SPATIAL_JACOBIAN_ORDER = ("angular", "linear")`.
- The suite-normalized velocity representation is `inertial`.
- Angular velocity units are `rad/s`; linear velocity units are `m/s`.
- Gravity is stored as an inertial/world-frame vector in `m/s^2`, with the
  default `(0.0, 0.0, -9.80665)`.
- The base frame is the root body frame whose rotation maps body-frame vectors
  into inertial/world-frame vectors.

## Supported Representations

`body_fixed`
: Angular and linear components are both expressed in the base body frame.

`inertial`
: Angular and linear components are both expressed in the inertial/world frame.

`mixed`
: Angular components are expressed in the base body frame, while linear
components are expressed in the inertial/world frame. This is the JaxSim
default and must be normalized before cross-engine dynamics comparisons.

## Adapter Rule

Backend adapters must call `normalize_floating_base_velocity` or
`convert_floating_base_velocity` at the boundary where native base velocities
enter engine-core dynamics. Public dynamics functions must document their units,
base frame, gravity direction, and velocity representation.

For the analytic single-floating-body case used by the tests, the base origin is
the center of mass. Gravity therefore contributes only a linear force term, and
the gyroscopic bias torque is `omega x I omega`, expressed in the angular frame
implied by the selected representation.
