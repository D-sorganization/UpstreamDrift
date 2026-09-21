# BunkerShot3D Contact Regimes and Coupled Rotation

**Issue #9544 (epic #9541).** What the solver may say about the club, the sand and the
ball together, at each fidelity tier, and what it refuses.

The ground truth for every statement here is the code and its tests:
[`tests/bunkershot3d/ball/test_contact_regimes_9544.py`](../../tests/bunkershot3d/ball/test_contact_regimes_9544.py)
and
[`tests/bunkershot3d/solvers/test_rotation_coupling_9544.py`](../../tests/bunkershot3d/solvers/test_rotation_coupling_9544.py).
Nothing below is a measurement. Both tiers remain at NASA-STD-7009B Validation 0 of 4
([credibility statement](credibility.md)); no published bunker-shot measurement of ball
launch, head deceleration or ejecta mass exists to qualify any regime against
([validation roadmap](validation-roadmap.md)).

## Reconciliation Against #8733

Delivered and in the tree: the F1 ball as a rigid circular plane-strain section
(`solvers/mpm/ball.py`), sand-flux-onto-ball reach (`solvers/mpm/ballreach.py`), the
whole-shot F1 march (`solvers/mpm/wholeshot.py`), the Rankine plastic-limit check
(`solvers/mpm/limit_states.py`), and manufactured-solution / temporal order-of-accuracy
studies (`solvers/mpm/order_of_accuracy.py`). What remains open from #8733 is F2 and
reference qualification, which this issue does not claim — see the F2 record below.

## Supported Regimes

`bunkershot3d.ball.classify_contact_regime` decides which of four outcomes a strike fell
into, from what the F0 march and the divot measurement already report
(`StrikeOutcome.from_shot`). `compute_bunker_launch` refuses every regime but the splash.

| Regime (`ContactType`) | Decided by                                                                              | Launch                                                |
| ---------------------- | --------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| `NO_HIT`               | never engaged; or sole left the sand behind a ball sunk to the surface                  | refused — nothing reached the ball                    |
| `BURIED_NO_RELEASE`    | engaged, sole never came back above the surface                                         | refused — no exit crossing, no divot, no release      |
| `THIN` (direct strike) | sole entered inside the ball's footprint, or left the sand behind a ball standing proud | refused — an impact problem, outside the splash model |
| `SPLASH`               | entered behind the footprint, still submerged when it reached the ball                  | splash partition (`ball/splash.py`), uncalibrated     |

The discriminants are the sole path's stations against the ball's footprint. They are
declared geometric conventions: the F0 half space has no crater and no free surface, so
the sand cushion between face and ball is **not** a quantity it has. A `SPLASH` verdict
says the geometry permits a splash; it does not measure one. Direct contact is never
routed through the splash partition, and a workbench point in any refused regime reports
"carry: ..." with the reason rather than a number.

## Rotation Modes and the Coupled Boundary

`ShotSettings.rotation_mode` names who owns the head's rotation during the march.

- **`RotationMode.PRESCRIBED`** (default): the delivered angular velocity is held for the
  whole strike. The support — shaft, grip, golfer — is taken to react the entire sand
  moment. `vandv.conservation.support_angular_impulse` puts that reaction on the ledger
  (`-∫τ dt`), and `prescribed_driver_work` the work the driver did holding the rotation
  (`-∫τ·ω dt`). Translation is free, so the support's linear impulse is zero by the same
  idealisation. The gyroscopic term `ω × Iω` is not included: F0 carries no inertia
  tensor.
- **`RotationMode.COUPLED`**: a caller-supplied `RotationCoupling` receives the sand
  wrench every step and returns the angular velocity for the next. A shaft model, an
  adapter over the existing `kinematics.coupling.CoupledDoublePendulum` (which already
  maps a clubhead wrench to joint torques), or the scalar-inertia oracle in the tests all
  fit the boundary; what any of them does with the moment is its own model, and the
  rotational ledger is theirs — the prescribed-mode ledger refuses a coupled trace. The
  shipped coupling is the boundary plus test oracles, not a calibrated shaft.

**Wrench conventions at the boundary.** `forces_n[k]` is the sand force in world axes.
`torques_n_m[k]` is the sand moment about the body-frame origin `positions_m[k]`, world
axes. Shift with `Wrench.about(point)` before comparing with a moment taken about the grip
or the centre of mass; the shift is `τ(p₂) = τ(p₁) + (p₁ − p₂) × F`, and its sign is
asserted by test. The coupling's answer is applied with the same explicit ordering as the
translation: the orientation is advanced by the returned velocity, exactly as the position
is advanced by the updated one.

**Zero-sand recovery.** With no sand wrench the coupled march and the prescribed march
agree bit for bit. Under a constant torque, the recorded angular velocity is the
closed-form `ω_k = ω₀ + k·dt·τ/I` to round-off, and the exit spin read off the last two
orientations equals the last recorded angular velocity. Under a viscous coupling the swept
angle converges to its closed form at first order in the step, which is the order of the
explicit scheme.

## What F1 May and May Not Expose

F1 is plane strain. Its ball is an infinite cylinder, its flux onto the ball is per unit
out-of-plane width, and it may visualise qualitative sand transport toward the ball
(`SliceFidelity.EXTRUDED`, `BallContactSplit.is_qualitative`). It **may not** report a
ball launch (`RefusedQuantity.BALL_LAUNCH`) or any heel-toe / lateral distribution
(`RefusedQuantity.OUT_OF_PLANE`); both raise, and the regime tests re-assert that they
still do. This is [ADR-0044](../adr/0044-out-of-plane-fidelity-for-bunkershot3d.md)'s
durable position, not a stopgap.

## Claims That Need Genuine 3-D Sand/Ball Contact

The following cannot be produced by F0 or F1 and are **not claimed** at any tier:

- quantitative spherical ball launch (speed, angle, spin) from resolved sand-ball contact;
- lateral sand flow, heel-toe force split, or any out-of-plane distribution;
- the sand cushion thickness between face and ball in a direct or mixed strike;
- crater memory and free-surface evolution across a shot.

An optional higher-fidelity backend (F2, 3-D MPM; F3, DEM) would have to declare, before
any of these is quoted:

| Requirement                  | What must be recorded                                                                                                                                          |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Compute cost                 | ADR-0044 measured 30–90 min/shot on a Maxwell+ GPU for F2; F1 costs 24–38 ms/step at a few thousand particles. The CPU F0 default (< 50 ms/shot) stays usable. |
| Resolution / coarse-graining | Bulk resolution of 1–2 mm (ADR-0033) against a 0.3 mm grain; DEM at true scale needs 2.1 × 10⁸ grains, so any tractable run is a coarse-grained proxy.         |
| Constitutive calibration     | Drucker–Prager / μ(I) constants remain `BORROWED_ANALOGUE`; the drained shear box is the measurement that flips `friction_angle_deg` (validation roadmap).     |
| Contact ordering             | Sand-on-sole before sand-on-ball, with the anti-tunnelling swept test; the ordering must be stated per step, not left to the solver's loop order.              |
| Timestep limits              | CFL on the continuum sound speed (MPM) or Rayleigh/Courant (DEM), enforced not advisory (`backends/stability.py`).                                             |

A reference solver is not experimental truth: it calibrates F0's constants against a
declared simulated target, and a `CONVENTION` fit is not a `MEASURED` one. Ball flight is
integrated only after a `SPLASH` verdict and a completed exit; every other regime keeps
its refusal.
