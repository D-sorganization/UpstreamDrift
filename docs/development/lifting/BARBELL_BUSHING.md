# Barbell on the Compliant Grip Bushing (LIFT-4, #11744)

This record covers a rigid barbell held at both hands of the full-body
same-input spec through the OSV-7 compliant grip bushing. It is applied to
the back-squat bundle. These are software checks, not a scientific
qualification. The design-manual inventory blocker is
`UP-D1-lift-barbell-bushing-inventory`.

## Model

- **Bar.** `Barbell` in `src/shared/python/lifting/barbell_bushing.py`:
  - The total mass is the lift's load: 100 kg, from the bundle's
    `optimizer.bar_mass_kg`.
  - The shaft is the IWF bar: 20 kg and 2.20 m.
  - The plate centroid offset of 0.70 m from the centre is an engineering
    assumption.
  - Axial inertia is the shaft only, because the sleeves spin freely.
  - Transverse inertia is 48.28 kg·m² at 100 kg.
- **Coupling.** `BarbellBushingPlant` appends six bar coordinates to any
  same-input `VectorPlant` (MuJoCo, Drake, Pinocchio, OpenSim): centre-of-mass
  position, plus body-fixed X-Y-Z angles relative to the attachment
  orientation.
  - The hand frames are the spec's `Grip` frame on the club body, with the
    OSV-7 left and right grip offsets.
  - Each hand's bushing wrench uses the shared OpenSim `BushingForce` law
    (`bushing_wrench`) and acts on the bar.
  - The reaction reaches the body through the transpose of a fourth-order
    finite-difference Jacobian of that engine's own forward kinematics
    (h = 1e-3). Its error against MuJoCo `mj_jac` is 1.4e-12.
  - The bar follows Newton–Euler. Bar efforts must be zero.
- **Stiffness and damping.** Stiffness uses the OSV-7 defaults (1e6 N/m,
  1600 N·m/rad). Damping uses the OSV-7 modal design at ζ = 0.7, evaluated
  for the bar's mass and inertia.
- **Body control.** The body follows the pipeline computed-torque tracking
  controller. Its efforts get a bar-weight feedforward `Jᵀ m g` added at the
  hands. The bar starts moving rigidly with the hands, offset by the hanging
  static deflection `K⁻¹ m g`.

## Tolerances (Fixed Before the Runs)

| Check                                                            | Threshold                                    |
| ---------------------------------------------------------------- | -------------------------------------------- | --- | --------------------- | --- | -------------- |
| Hold: mean vertical ground reaction over the last 0.5 s of 1.5 s | body + bar weight within 1 %                 |
| Hold: hand vertical forces                                       | bar weight within 1 %, left share 0.5 ± 0.01 |
| Hold: peak bushing deflection                                    | ≤ 3 mm                                       |
| Bundle replay in MuJoCo (first 5 steps)                          | ≤ 1e-8                                       |
| Cross-engine parity, every 20th state                            | `                                            | dX  | <= 1e-6 + 1e-8 \* max | X   | ` (L1, #11605) |
| Unit: static hold on the plant, Newton's third law, Jacobian     | 1 %, 1e-6 N, 1e-6                            |

## Results

### Cross-Engine Parity

Evidence: `evidence/back_squat/barbell_parity.json`. The table gives the
maximum difference from MuJoCo at 11 states of the loaded bundle. Every value
is within L1.

| Quantity             | MuJoCo Scale | Drake  | Pinocchio | OpenSim |
| -------------------- | ------------ | ------ | --------- | ------- |
| Body acceleration    | 2863.6       | 1.5e-6 | 8.4e-7    | 1.1e-6  |
| Bar acceleration     | 211.1        | 3.3e-8 | 1.9e-8    | 3.4e-8  |
| Hand force L / R (N) | 1007 / 1012  | 1.1e-7 | 1.1e-7    | 9.4e-8  |
| Hand moment (N·m)    | 13.9         | 2.3e-9 | 2.3e-9    | 2.8e-9  |

### Loaded Bundle

Evidence: `evidence/back_squat/back_squat_barbell_bundle.npz`.

- The back-squat window is re-tracked in MuJoCo with the bar on.
- The body arrays are an ordinary same-input bundle. The bar reference,
  bar/bushing record and attachment pose are under `provenance["barbell"]`.
- Over the window:
  - per-hand vertical bushing force is 450–1012 N;
  - the left share is 0.494–0.502;
  - peak deflection is 1.0 mm.
- Re-tracking with the bar changes the motion. Individual coordinates differ
  from the unloaded reference by up to 1.3 rad (shoulder and wrist inputs,
  MTP).

### Quasi-Static Hold

Evidence: `evidence/back_squat/barbell_hold.json`. **The hold fails.**

- For the first 0.35 s the vertical ground reaction is about body + bar
  weight (2043 N), and the hands carry about 980 N, split evenly.
- After that the stance tips forward:

| Hold Measure                           | Value  |
| -------------------------------------- | ------ |
| Mean vertical GRF / weight, last 0.5 s | 0.34   |
| Hand vertical sum / bar weight         | 0.39   |
| Left share                             | −0.11  |
| Peak deflection                        | 2.2 mm |

**Cause.** `hold_balance_geometry` pins it, and
`test_hold_failure_is_the_forward_tip` asserts it:

- The golf spec's hands hang the bar in front of the body.
- At the standing pose the body's own centre of mass lies between the heel
  and forefoot spheres.
- With the bar added, the combined centre of mass is ahead of both forefoot
  spheres.
- The controller's balance term steers the body centre of mass, not the
  combined one. Even a combined-COM target would need the body centre of mass
  behind the heels in this posture.

The two hold-threshold tests stay as strict expected failures with unchanged
thresholds.

## What Is and Is Not Checked

|                           | Status                                                                                                   |
| ------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Checked (software)**    | The bushing law and its Newton's-third-law transfer to the body root                                     |
|                           | Static hand load equals the weight feedforward                                                           |
|                           | A bar under its own weight hangs with the static deflection                                              |
|                           | Four engines agree on coupled accelerations and per-hand wrenches                                        |
|                           | The loaded bundle replays deterministically in MuJoCo                                                    |
| **Not checked / not met** | Vertical GRF ≈ body + bar weight in a quasi-static hold (fails, see above)                               |
|                           | Back-squat geometry: bar on the back, grip width. The spec's hands are 0.08 m apart along the club axis. |
|                           | A distributed hand contact model                                                                         |
|                           | The other four lifts                                                                                     |
|                           | Full open-loop replay in engines other than MuJoCo                                                       |
|                           | Joint moments, spine loads, measured bar forces, uncertainty                                             |

## Reproduction

The commands are headless and run from the repository root. The bundle and
parity runs take tens of minutes.

```bash
MUJOCO_GL=egl python3 scripts/generate_barbell_squat_evidence.py \
    --part hold --part bundle --part parity --source-commit <sha>
python3 -m pytest -o addopts="" -n 0 \
    tests/unit/lifting/test_barbell_bushing.py \
    tests/unit/lifting/test_barbell_evidence.py
```
