# Pointwise Acceleration Audit Handoff

## Scope

Read-only evaluation at all **307 saved scalar states** from run59, using the
same model and run19 native polynomial profile. No integrations or production
edits were performed. This is sampled pointwise evidence, not trajectory
acceptance or a bound between samples.

## Findings

- Largest scalar-versus-actual-manifold-RHS native acceleration difference:
  **2.18034e-6 rad/s²**, LSInputZ at **0.7861111111 s**, compared with scalar
  acceleration 539073.4924 rad/s². Direct native acceleration mapping differs by
  2.18209e-6 rad/s² at the same sample.
- The direct route and actual current-state actuator route differ by at most
  **1.74623e-9** in native acceleration. Maximum tangent-effort roundtrip error
  is **1.25056e-12**. At that sample, native position/rate roundtrip errors are
  4.44e-15 and 4.43e-12 respectively.
- Repeating each exact state after evaluating a different saved state produces
  **exactly zero acceleration difference** for both the scalar model and the
  actual manifold RHS route across all 307 probes. This test found no solver
  history dependence for the tested sequence; it does not prove history
  independence under every possible calling pattern.
- Maximum componentwise error scaled by `1+abs(reference acceleration)` is
  **1.18733e-9** over the complete sampled audit.

At 0.7861111111 s, unscaled SI mass-matrix condition numbers are 6.66264e7
(scalar) and 6.72366e7 (manifold); the minimum eigenvalues are approximately
1.37970e-6. Constraint Schur condition is 1.52772e6 for both. KKT condition is
1.50250e6 for scalar and 8.81817e4 for manifold. The constraint Jacobians have
six nonzero sampled singular values, with minima 0.27544 and 0.33316 respectively.
These condition numbers mix translational and rotational SI coordinates and
must not be interpreted as dimensionless physical quality or global bounds.

The audit does not expose a gross same-state effort-routing or history defect.
It also does not prove the propagation mechanism behind nonmonotonic trajectory
errors. The next diagnostic should measure perturbation propagation through the
transition, rather than infer global stability from pointwise agreement or
sampled condition numbers. Existing acceptance status remains unchanged.
The rejected run19 fixture's extreme native-rate/acceleration spikes should
inform later joint trajectory/control optimization, not an unreviewed physics
change or waiver of parity gates.

## Reproduction

`inspect_acceleration.py` checks direct native mapping, the actual current-state
actuator conversion with the original q0 branch reference, and both repeated
calls after a different prior state. It evaluates M using actual Pinocchio CRBA,
J using the actual weld Jacobian API, and computes unscaled Schur/KKT conditioning.
No inertia padding or constraint modification is used. `diagnostic.json` records
input/script hashes, full time series, coordinate maxima and sampled matrices.
Ruff passes. `raw-source.zip` preserves the exact executed script/report bytes.

Run the following from this artifact directory, or supply a full local script
path to scp:

```powershell
scp inspect_acceleration.py controltower:/C:/Users/diete/inspect_manifold_acceleration_10043_19.py
ssh -o BatchMode=yes controltower 'wsl -d ControlTower-Runner -- env PYTHONPATH=/home/dieterolson/native-manifold-10043-18 /home/dieterolson/simscape-pinocchio-9967/.venv/bin/python /mnt/c/Users/diete/inspect_manifold_acceleration_10043_19.py --model /mnt/c/Users/diete/native_geometry_spec_9967.json --candidate /mnt/c/Users/diete/native-ms-fit-9967-19/returned-candidate.json --states /mnt/c/Users/diete/native-manifold-replay-9967-59/scalar.npz --output /mnt/c/Users/diete/native-manifold-acceleration-repeat.json'
```
