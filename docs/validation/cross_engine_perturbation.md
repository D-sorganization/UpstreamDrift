# Unified Cross-Engine Perturbation Study

Issue #13795 (SAP-6, epic #13789) re-runs the #2021 study. The receipt is [`cross_engine_perturbation_receipt.json`](cross_engine_perturbation_receipt.json), status `measured`, and the code is `src/shared/python/pendulum_simulator/unified_perturbation_study.py`.

## Setup

Every engine ran the one setup recorded per engine in the receipt:

- `TRIPLE_PENDULUM_SPEC` (SHA-256 `cde02870…e7c89f`);
- its single gravity constant, 9.80665 m/s²;
- 1 ms RK4, converged per SAP-3 (#13792);
- zero-order-hold control with zero base torque;
- additive Gaussian noise of 0.1 N·m per joint, seed 42 + trial;
- 10 trials of 1.5 s, starting at rest from (−1.5, 0.5, 0.2) rad.

The engines are two real backends. Each computes its own state, tip velocity and energy:

- **`in_house_triple`** (`TriplePendulumDynamics`): energy is ½vᵀM(q)v plus gravity on the segment centres of mass, and the tip velocity is analytic.
- **`mujoco_triple`** (MuJoCo 3.8 on the spec-generated XML): energy is `mj_energyPos` + `mj_energyVel`, and the tip velocity is `mj_objectVelocity`.

## Results

The cross-engine CV is the CV of the two engines' trial means. The within-engine trial CV is the noise-induced spread, which is identical for both engines to four digits. It is the sensitivity band the cross-engine CV is read against.

| Metric | Cross-engine CV | Within-engine trial CV |
| ------ | --------------- | ---------------------- |
| total_energy_final | 1.117e-16 | 1.548e-04 |
| end_effector_speed_final | 0.000e+00 | 1.028e-03 |
| peak_end_effector_speed | 1.067e-16 | 3.294e-04 |

The engine means are 89.970186 J final energy, 5.816501 m/s final tip speed and 11.771680 m/s peak tip speed. No mean is near zero, so no CV is undefined (#12807). No CV threshold was set or tuned.

## Closure Evidence (#2021)

#2021 recorded cross-engine CVs of 1.21 to 1.50. On the unified setup the cross-engine CV is at floating-point round-off. That is about twelve orders of magnitude below the noise spread of the study itself.

The #2021 divergence was therefore a setup artefact, not a physics disagreement. Its causes were mismatched step sizes, unstandardised parameters and stub engines being counted.

## Limitations

- **Two engines only.** Drake, Pinocchio and MyoSuite have no canonical-spec triple-pendulum adapter, so they are not claimed here.
- **One spec, one integrator.** Both models come from one spec and use classical RK4. Agreement at round-off shows that the setup is unified and the two implementations match. It is not independent model validation, and an error in the spec would be shared by both.
- **Setup is declared, not read back.** g, dt and the integrator are declared by the harness and set through each adapter's `configure`. They are not read back from the engine (`setup_attestation`).
- **The CC-11 report is unchanged.** `cross_engine_v1.md` still reports zero comparisons.

To regenerate, run `scripts/validation/generate_cross_engine_perturbation_receipt.py` with the command in its docstring. `tests/unit/engines/pendulum/test_unified_perturbation_receipt.py` validates the receipt and this table, and its slow test checks that a regeneration reproduces the CVs.
