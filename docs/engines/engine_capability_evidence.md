# Engine Capability Evidence and Documentation Governance

<!-- AUTO-GENERATED — do not edit by hand. -->
<!-- Regenerate with: python3 -m scripts.companion_evidence render-docs -->

Generated from `scripts/config/companion_capability_evidence.v1.json` and
`scripts/config/companion_documentation.v1.json` (registry v1.0.0,
issue #9193). Every row states a software fact only: a capability is
`qualified` when it names an exact test or artifact executed by the
declared gate, and `unqualified` otherwise. Nothing here is scientific
validation, a tolerance, a calculation, or engineering approval; those
remain with their governed authorities (#9064, #9070). Review freshness,
content hashes, and immutable links are exported per commit by
`scripts/companion_catalog.py`, not copied into this page.

## Engines

### Drake (`drake`)

- Support tier: `extended`
- Runtime availability: `conditional` — Requires the `drake` optional extra (`pip install -e ".[dev,drake]"`); the required pull-request lane does not install it.
- Documentation: `engine-capability-evidence`, `engine-drake`, `engine-support-tiers`, `user-optional-features`

| Capability                 | Title                                         | Evidence state | Evidence | Gate | Reason                                                                                                                                                                       |
| -------------------------- | --------------------------------------------- | -------------- | -------- | ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `counterfactual_ztcf_zvcf` | Zero-torque and zero-velocity counterfactuals | unqualified    | —        | —    | No gate executes the Drake counterfactual surface against the real runtime.                                                                                                  |
| `cross_engine_parity`      | Cross-engine agreement                        | unqualified    | —        | —    | The nightly cross-engine workflow installs Drake but executes validator and conformance tests against stub adapters only; real-engine agreement is owned by epic #9964.      |
| `forward_dynamics_rollout` | Forward dynamics rollout                      | unqualified    | —        | —    | tests/unit/test_drake_physics_engine.py skips when Drake is absent and no required or nightly gate installs Drake and executes an engine-level rollout test.                 |
| `inverse_dynamics`         | Inverse dynamics                              | unqualified    | —        | —    | tests/unit/test_drake_physics_engine.py::test_drake_physics_engine_compute_inverse_dynamics skips when Drake is absent; no gate records it passing against the real runtime. |

Limitations:

- Agreement between engines is not established by this catalog.
- Counterfactual attribution is descriptive on the same trajectory; it is not a validated forward prediction.
- Extended tier: validated through scheduled cross-engine workflows, not every required pull-request check.

### MuJoCo (`mujoco`)

- Support tier: `supported`
- Runtime availability: `conditional` — Installed by the default `dev` profile (`requirements-dev.lock`); interactive rendering additionally needs a GL-capable host.
- Documentation: `engine-capability-evidence`, `engine-mujoco`, `engine-support-tiers`, `user-optional-features`

| Capability                    | Title                                         | Evidence state | Evidence                                                                                                                                                                                                                                                       | Gate                                | Reason                                                                                                                                               |
| ----------------------------- | --------------------------------------------- | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `counterfactual_ztcf_zvcf`    | Zero-torque and zero-velocity counterfactuals | qualified      | `tests/unit/engines/mujoco/test_counterfactuals.py::TestCounterfactualPhysics::test_ztcf_plus_counterfactual_equals_observed`<br>`tests/unit/engines/mujoco/test_counterfactuals.py::TestCounterfactualPhysics::test_zvcf_plus_counterfactual_equals_observed` | `.github/workflows/ci-standard.yml` | —                                                                                                                                                    |
| `cross_engine_parity`         | Cross-engine agreement                        | unqualified    | —                                                                                                                                                                                                                                                              | —                                   | The nightly cross-engine workflow executes validator and conformance tests against stub adapters only; real-engine agreement is owned by epic #9964. |
| `drift_control_decomposition` | Drift-control acceleration decomposition      | qualified      | `tests/unit/engines/mujoco/test_drift_control.py::TestDriftControlDecomposer::test_superposition_principle`                                                                                                                                                    | `.github/workflows/ci-standard.yml` | —                                                                                                                                                    |
| `forward_dynamics_rollout`    | Forward dynamics rollout                      | qualified      | `tests/unit/engines/mujoco/test_continuous_torque_simulate.py::test_simulate_with_canonical_golfer_xml_rollout`<br>`tests/unit/engines/mujoco/test_continuous_torque_simulate.py::test_determinism_back_to_back`                                               | `.github/workflows/ci-standard.yml` | —                                                                                                                                                    |
| `inverse_dynamics`            | Inverse dynamics                              | qualified      | `tests/unit/engines/mujoco/test_inverse_dynamics.py::TestInverseDynamicsSolver::test_compute_required_torques`                                                                                                                                                 | `.github/workflows/ci-standard.yml` | —                                                                                                                                                    |

Limitations:

- Agreement between engines is not established by this catalog.
- Counterfactual attribution is descriptive on the same trajectory; it is not a validated forward prediction.
- Evidence covers a single-pendulum reference model only.
- Evidence covers the double-pendulum reference model only.
- Evidence uses the canonical golfer model with the engine's default semi-implicit Euler integrator; it does not establish agreement with other engines.

### MyoSuite (`myosuite`)

- Support tier: `experimental`
- Runtime availability: `conditional` — Requires the `biomechanics` optional extra; experimental tier with best-effort local validation only.
- Documentation: `engine-capability-evidence`, `engine-myosuite`, `engine-support-tiers`, `user-optional-features`

| Capability                 | Title                    | Evidence state | Evidence | Gate | Reason                                                                                                |
| -------------------------- | ------------------------ | -------------- | -------- | ---- | ----------------------------------------------------------------------------------------------------- |
| `forward_dynamics_rollout` | Forward dynamics rollout | unqualified    | —        | —    | No gate installs MyoSuite; tests/unit/test_myosuite_adapter.py exercises a mocked muscle system only. |
| `muscle_actuation`         | Muscle-driven actuation  | unqualified    | —        | —    | The muscle surface is a stub and its adapter tests are mock-backed; no real-runtime evidence exists.  |

Limitations:

- Experimental tier: missing capabilities and stub implementations are expected.

### OpenSim (`opensim`)

- Support tier: `experimental`
- Runtime availability: `conditional` — Requires the `biomechanics` optional extra; experimental tier with best-effort local validation only.
- Documentation: `engine-capability-evidence`, `engine-opensim`, `engine-support-tiers`, `user-optional-features`

| Capability                 | Title                    | Evidence state | Evidence | Gate | Reason                                                                                                 |
| -------------------------- | ------------------------ | -------------- | -------- | ---- | ------------------------------------------------------------------------------------------------------ |
| `forward_dynamics_rollout` | Forward dynamics rollout | unqualified    | —        | —    | No gate installs OpenSim; tests/unit/test_opensim_physics_engine.py patches a mocked `opensim` module. |
| `muscle_biomechanics`      | Muscle biomechanics      | unqualified    | —        | —    | The OpenSim muscle-biomechanics surface remains a stub; no real-runtime evidence exists.               |

Limitations:

- Experimental tier: missing capabilities and stub implementations are expected.

### Pinocchio (`pinocchio`)

- Support tier: `extended`
- Runtime availability: `conditional` — Requires the `pinocchio` optional extra (`pip install -e ".[dev,pinocchio]"`); the required pull-request lane does not install it.
- Documentation: `engine-capability-evidence`, `engine-pinocchio`, `engine-support-tiers`, `user-optional-features`

| Capability                 | Title                                         | Evidence state | Evidence | Gate | Reason                                                                                                                                                                                                   |
| -------------------------- | --------------------------------------------- | -------------- | -------- | ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `counterfactual_ztcf_zvcf` | Zero-torque and zero-velocity counterfactuals | unqualified    | —        | —    | No gate executes the Pinocchio counterfactual surface against the real runtime.                                                                                                                          |
| `cross_engine_parity`      | Cross-engine agreement                        | unqualified    | —        | —    | The nightly cross-engine workflow installs Pinocchio but executes validator and conformance tests against stub adapters only; real-engine agreement is owned by epic #9964.                              |
| `forward_dynamics_rollout` | Forward dynamics rollout                      | unqualified    | —        | —    | tests/unit/test_pinocchio_physics_engine.py skips when Pinocchio is absent and no required or nightly gate installs Pinocchio and executes an engine-level rollout test.                                 |
| `inverse_dynamics`         | Inverse dynamics                              | unqualified    | —        | —    | tests/unit/test_pinocchio_physics_engine.py::test_pinocchio_physics_engine_compute_ztcf and its RNEA/CRBA siblings skip when Pinocchio is absent; no gate records them passing against the real runtime. |

Limitations:

- Agreement between engines is not established by this catalog.
- Counterfactual attribution is descriptive on the same trajectory; it is not a validated forward prediction.
- Extended tier: validated through scheduled cross-engine workflows, not every required pull-request check.

## Known Gaps

| Gap                                  | Issue | Scope   | Summary                                                                                                                                                                                            |
| ------------------------------------ | ----- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `launcher-model-registry-divergence` | #8853 | catalog | The web launcher manifest and the local model registry still disagree on program membership; `summary.single_source_program_records` counts programs that exist in only one of the two registries. |

## Documentation Records

| ID                                      | Title                                                   | Source                                                                             | Status | Owner                     | Last reviewed | Review due |
| --------------------------------------- | ------------------------------------------------------- | ---------------------------------------------------------------------------------- | ------ | ------------------------- | ------------- | ---------- |
| `adr-0011-plot-style-toolkit`           | ADR 0011: Plot-Style Toolkit                            | `docs/adr/0011-plot-style-toolkit.md`                                              | active | upstreamdrift-maintainers | —             | —          |
| `adr-0031-launch-monitor-shot-schema`   | ADR 0031: Canonical Launch Monitor Shot Schema          | `docs/adr/0031-launch-monitor-canonical-shot-schema.md`                            | active | upstreamdrift-maintainers | —             | —          |
| `adr-0043-companion-provider-authority` | ADR-0043: Companion Manifest Provider Authority         | `docs/adr/0043-companion-manifest-provider-authority.md`                           | active | upstreamdrift-maintainers | 2026-09-17    | 2027-03-17 |
| `engine-capability-evidence`            | Engine Capability Evidence and Documentation Governance | `docs/engines/engine_capability_evidence.md`                                       | active | upstreamdrift-maintainers | 2026-09-17    | 2027-03-17 |
| `engine-capability-matrix`              | Physics Engine Capability Matrix                        | `docs/engines/engine_capabilities.md`                                              | active | upstreamdrift-maintainers | —             | —          |
| `engine-drake`                          | Drake Engine                                            | `docs/engines/drake.md`                                                            | active | upstreamdrift-maintainers | —             | —          |
| `engine-mujoco`                         | MuJoCo Engine                                           | `docs/engines/mujoco.md`                                                           | active | upstreamdrift-maintainers | —             | —          |
| `engine-myosuite`                       | MyoSim Engine (MyoSuite)                                | `docs/engines/myosim.md`                                                           | active | upstreamdrift-maintainers | —             | —          |
| `engine-opensim`                        | OpenSim Engine                                          | `docs/engines/opensim.md`                                                          | active | upstreamdrift-maintainers | —             | —          |
| `engine-pendulum`                       | Pendulum Models                                         | `docs/engines/pendulum.md`                                                         | active | upstreamdrift-maintainers | —             | —          |
| `engine-pinocchio`                      | Pinocchio Engine                                        | `docs/engines/pinocchio.md`                                                        | active | upstreamdrift-maintainers | —             | —          |
| `engine-support-tiers`                  | Supported Engine Tiers                                  | `docs/engines/support_tiers.md`                                                    | active | upstreamdrift-maintainers | 2026-09-17    | 2027-03-17 |
| `engine-tier-policy`                    | Engine Tier Policy                                      | `docs/operations/tier-policy.md`                                                   | active | upstreamdrift-maintainers | —             | —          |
| `model-explorer-attachment-manifests`   | Model Explorer Attachment Manifests                     | `docs/model_explorer/attachment-manifests.md`                                      | active | upstreamdrift-maintainers | —             | —          |
| `ops-companion-publication`             | Companion Publication and Acquisition                   | `docs/operations/companion-publication.md`                                         | active | upstreamdrift-maintainers | 2026-09-17    | 2027-03-17 |
| `research-counterfactual-framework`     | The Timing Question and the Counterfactual Framework    | `docs/research/proximal_distal_energy_transfer/chapters/_ch04_counterfactuals.qmd` | active | upstreamdrift-maintainers | —             | —          |
| `troubleshooting-installation`          | Installation Troubleshooting Guide                      | `docs/troubleshooting/installation.md`                                             | active | upstreamdrift-maintainers | —             | —          |
| `user-installation`                     | Installation                                            | `docs/user_guide/installation.md`                                                  | active | upstreamdrift-maintainers | —             | —          |
| `user-launch-monitor-analytics`         | Launch Monitor Analytics                                | `docs/user_guide/launch_monitor_analytics.md`                                      | active | upstreamdrift-maintainers | —             | —          |
| `user-optional-features`                | Installing Optional Features                            | `docs/user_guide/installing_optional_features.md`                                  | active | upstreamdrift-maintainers | —             | —          |
