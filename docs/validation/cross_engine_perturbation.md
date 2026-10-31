# Unified Cross-Engine Perturbation Study

- Issue: #13795 (SAP-6, epic #13789); re-runs the study behind #2021
- Receipt: [`cross_engine_perturbation_receipt.json`](cross_engine_perturbation_receipt.json), schema `upstreamdrift.cross_engine_perturbation_receipt.v1`, status `measured`
- Code: `src/shared/python/pendulum_simulator/unified_perturbation_study.py`

## Setup

Every engine ran the one setup the receipt records per engine:

| Field              | Value                                                    |
| ------------------ | -------------------------------------------------------- |
| Spec               | `TRIPLE_PENDULUM_SPEC`, SHA-256 `cde02870…e7c89f`        |
| Gravity            | 9.80665 m/s² (the spec's single constant)                |
| Step and method    | 1 ms RK4, converged per SAP-3 (#13792)                   |
| Control            | zero-order hold per step, zero base torque               |
| Noise              | additive Gaussian, 0.1 N·m per joint, seed 42 + trial    |
| Trials, duration   | 10 trials of 1.5 s from (−1.5, 0.5, 0.2) rad at rest     |

Engines, both real backends:

- `in_house_triple`: `TriplePendulumDynamics`. Energy is ½vᵀM(q)v from its own mass matrix plus gravity on each segment centre of mass. Tip velocity is the analytic derivative of the tip position.
- `mujoco_triple`: MuJoCo 3.8 on `TRIPLE_PENDULUM_XML` (generated from the same spec). Energy is `mj_energyPos` + `mj_energyVel`. Tip velocity is `mj_objectVelocity` of the club-head geom.

## Results

Cross-engine CV is the CV of the two engines' trial means. The within-engine trial CV is the noise-induced spread, which is the same for both engines to three digits. It is the sensitivity band the cross-engine CV is read against.

| Metric | Cross-engine CV | Within-engine trial CV |
| ------ | --------------- | ---------------------- |
| total_energy_final | 1.117e-16 | 1.548e-04 |
| end_effector_speed_final | 0.000e+00 | 1.028e-03 |
| peak_end_effector_speed | 1.067e-16 | 3.294e-04 |

Engine means: final energy 89.970186 J, final tip speed 5.816501 m/s, peak tip speed 11.771680 m/s. The two engines agree to the last printed digit. No metric has a near-zero mean, so no CV is undefined (#12807). No CV threshold was set or tuned.

## Closure Evidence (#2021)

#2021 recorded cross-engine CVs of 1.21 to 1.50. On the unified setup the cross-engine CV is at floating-point round-off (≤ 1.2e-16). That is about twelve orders of magnitude below the noise-induced spread of the study itself. The divergence was a setup artefact:

- mismatched step sizes;
- unstandardised parameters;
- stub engines counted as engines.

It was not a disagreement in the physics. This receipt is the closure evidence for #2021.

## Limitations

- Two engines. Drake, Pinocchio and MyoSuite have no canonical-spec triple-pendulum adapter, so they were not run. They are not claimed here.
- Both models are generated from one spec and integrated with classical RK4. Agreement at round-off therefore shows the setup is unified and the two dynamics implementations match. It is not independent model validation, and an error in the spec itself would be shared.
- g, dt and the integrator are declared by the harness and set through each adapter's `configure`. They are not read back from the engine (`setup_attestation`).
- `cross_engine_v1.md` (CC-11 differential report) is unchanged. It still reports zero comparisons, because its harness and adapters were never merged.

## Regeneration

```bash
PYTHONPATH=.:src/engines/pendulum_models/python python3 -m scripts.validation.generate_cross_engine_perturbation_receipt
```

Add `--check` to fail when the committed receipt is stale. `tests/unit/engines/pendulum/test_unified_perturbation_receipt.py` validates the receipt and this table.
