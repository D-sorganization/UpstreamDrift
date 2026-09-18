# Rajagopal2015 Muscle CMC — Post-MVP Path (issue #4296)

This note describes how to bring up the optional 80-muscle Rajagopal2015
CMC pipeline scaffolded by `muscle_analysis.py`. The scaffold is
**strictly additive** to the joint-torque MVP described in
`OPENSIM_PARITY_SPEC.md` §1–§7 and never participates in the default-CI
joint-torque path.

## Why this is gated

Per the parent roadmap (#4134) and #4296, body-marker mocap fixtures and
the Rajagopal2015 muscle-enabled `.osim` are **not shipped** with this
repository. Until they land, the affected tests must:

1. Fail loudly with a typed reason that names the missing file.
2. Skip cleanly under `-m "not requires_mocap_fixtures"`.
3. Never silently green an untested code path.

`muscle_analysis.MuscleFixturesUnavailableError` is the canonical typed
error; it embeds the absolute path that was probed.

## Asset checklist

You need:

1. **Rajagopal2015 model.** The 80-muscle lower-extremity OpenSim model
   from Rajagopal _et al._ 2015, distributed via SimTK / the upstream
   `opensim-models` repo. Download it yourself and respect the upstream
   license. Place the `.osim` at:

   ```
   <fixtures-root>/Rajagopal2015.osim
   ```

2. **Body-marker mocap kinematics.** A short (sub-second) `.mot` or
   `.sto` kinematics fixture compatible with the Rajagopal2015 marker
   set. The CMC smoke test loads it from:

   ```
   <fixtures-root>/kinematics/smoke.mot
   ```

3. **Marker set metadata.** The required marker names are listed in
   `muscle_analysis.RAJAGOPAL2015_REQUIRED_MARKERS` and validated by
   `muscle_analysis.validate_marker_trajectory`.

`<fixtures-root>` defaults to:

```
tests/fixtures/mocap/rajagopal2015/
```

Override with the `UPSTREAMDRIFT_MOCAP_FIXTURES_ROOT` environment
variable when you stage assets elsewhere.

## Running the gated tests

The new tests live at `tests/opensim/test_muscle_cmc.py`.

| Selector                                                   | Behaviour                                |
| ---------------------------------------------------------- | ---------------------------------------- |
| `pytest tests/opensim/test_muscle_cmc.py`                  | DbC tests run; opt-in tests collected    |
| `pytest -m "requires_mocap_fixtures"`                      | Loud-fails until fixtures present        |
| `pytest -m "not requires_mocap_fixtures"`                  | Default-CI selector — fixtures-free path |
| `pytest -m "requires_opensim and requires_mocap_fixtures"` | Full integration smoke                   |

## Provenance and licensing

- Rajagopal _et al._ 2015 model — IEEE TBME, distributed via SimTK under
  the project's stated terms. Do **not** redistribute the `.osim` from
  this repository.
- OpenSim Python bindings — Apache 2.0.
- Mocap kinematics fixtures — must be derived from data the contributor
  has the rights to share. Document provenance alongside any fixture
  added under `tests/fixtures/mocap/`.

## Non-regression contract

- The joint-torque MVP path under
  `src/engines/physics_engines/opensim/python/motion_matching/` is **not**
  touched by this scaffold.
- `build_rajagopal2015_muscle_model` and `run_cmc_smoke` are pure
  additions; importing `muscle_analysis` does not import `opensim` at
  module load.

## OpenSim Golf Model Muscle Qualification (OG-08, #10402)

Under OpenSim epic #10394 / #10363, muscle and tendon extensions are qualified
without replacing the certified joint-torque baseline:

1. **Explicit Anatomy Scope:**
   - A lower-extremity model (e.g. Rajagopal2015 80-muscle) is **not** a full golf
     muscle model. Claiming `full_body_golf` with lower limbs alone raises
     `UnsupportedAnatomyClaimError`.
   - Full golf coverage requires explicit lower extremity, torso/spine,
     shoulder/scapula, arm/forearm, and wrist/hand musculature, with declared
     omissions (e.g. rigidly affixed head/neck per #10394) verified by
     `audit_anatomy_coverage`.
2. **Strength Scaling Disclaimer:**
   - The standard OpenSim scale tool scales segment lengths and MTU geometries,
     but does **not** qualify or scale maximum isometric force ($F_{\text{max}}$).
     PCSA or body mass scaling laws must be applied explicitly with parameter provenance.
3. **Moment Arm Qualification:**
   - Muscle moment arms are audited against the virtual work finite-difference
     path-length derivative: $r_{\text{FD}} = -\frac{l_{MT}(q + \Delta q) - l_{MT}(q - \Delta q)}{2 \Delta q}$.
     Deviations exceeding tolerance raise `MomentArmDerivativeMismatchError`.
4. **Activation Dynamics & Static Equilibrium:**
   - Initial muscle-tendon states must satisfy static equilibrium ($F_{\text{fiber}}\cos\alpha = F_{\text{tendon}}$)
     before forward simulation or replay; non-equilibrated states raise `UninitializedTendonStateError`.
5. **Acceptance & Validation Governance:**
   - Receipts report reserve actuator torques and pelvic residuals ($F_x, F_y, F_z, M_x, M_y, M_z$).
   - Muscle-complete status remains `IN_PROGRESS_QUALIFICATION` and independent
     scientific validation remains `PENDING_10375` until clinical force validation completes.
