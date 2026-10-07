# Musculoskeletal Golf Swing Research Notes

Editable research source: [musculoskeletal_swing.tex](musculoskeletal_swing.tex)
(compile with `tectonic`; the PDF is not committed). Issue #11617, epic #11605.
Result status: `STATIC_OPTIMIZATION_ONLY_NOT_QUALIFIED`.

## Phase 2 Summary (Spec-Driven, Exact Dynamics)

Inputs are the dynamically consistent same-input bundles (driver, iron): 44
coordinates at 1 ms, reference `q`, `v` and efforts. Ground reactions are exact
(the deterministic contact law evaluated on the OpenSim skeleton), not estimated.

| Check | Driver | Iron |
| --- | --- | --- |
| ID vs bundle efforts, leg/pelvis/trunk, median / p95 abs (N m) | 0.01 / 1.1 | 0.01 / 0.9 |
| ID RMS before / after club-head speed peak (N m) | 0.8 / 5.5 | 0.3 / 4.1 |
| ID peak error (N m), at stiff foot-contact transients | 80.7 | 71.2 |
| Body origins vs MuJoCo parity model (mm), excl. fixed convention offsets | 0.0 | 0.0 |
| Peak total vertical force (body weights) | 3.01 | 3.00 |
| Leg reserve RMS / peak (N m) | 34.3 / 452 | 35.6 / 417 |
| Leg reserve RMS before / after speed peak (N m) | 22.9 / 53.2 | 14.3 / 63.2 |

Phase 1 (IK inputs, estimated GRF): leg reserve peak 2134 N m, pelvis residual up
to 22 316 (RMS up to 1457). Phase 2 has no pelvis residual.

Fixed convention offsets between the exports are pose independent: the club body
(OpenSim origin at the head), the grip standoff (12.7 mm) and the tibia frame
(9.3 mm).

### What Is Muscle-Driven

- Muscle-driven: 14 leg coordinates (80 Rajagopal muscles, rigid tendon), solved per
  frame by bounded least squares on midpoint states every 5 ms.
- Torque stand-ins: arms, scapulae, neck, spine and torso keep the reference
  efforts (no muscles exist for them). The arm efforts include the grip-weld
  reaction and are not pure demand.
- Reserves remain at the high-demand instants (hip flexion, adduction, rotation);
  left hip rotation has a one-sided muscle torque range in most poses, which is an
  open modelling observation.

Not qualified: reserves are non-zero, tendons are rigid, only leg muscles exist and
no independent open-loop forward replay was done.

## Upper-Limb and Trunk Muscle Models

Surveyed locally (official `opensim-org/opensim-models` submodule, commit
`d9b05d4`, SHA-256 prefix in brackets): `Rajagopal/RajagopalLaiUhlrich2023.osim`
(80 leg muscles) [8f30d0b6], `Hamner/FullBodyModel_Hamner2010_v2_0.osim` (93
lower-limb muscle paths, torque-driven upper body) [cb854785],
`Arm26/arm26.osim` (planar, 6 muscles) [e2224d00], `WristModel/wrist.osim`
(25 forearm muscles) [66488a78]. None provides a 3D shoulder or trunk muscle set.

Candidates not downloaded (a download needs the user's explicit approval; licence
and checksum must be recorded from the official page at that time): MoBL-ARMS
(Saul et al. 2015, SimTK `upexdyn`), the Christophy et al. 2012 lumbar model and the
Hamner 2010 full-body model. Graft plan: obtain from the official source; map
shoulder, elbow and wrist coordinates by name to the spec `LScap/LS/LE/LF/LW`
and `R*` coordinates; recover frames from the scapula/clavicle joint frames as for the
pelvis (`musculoskeletal_graft.py`); validate muscle lengths against the source
model (target below 5 mm RMS, `musculoskeletal_graft_validation.py`) before any
solve; separate the grip-weld reaction before solving the arm coordinates.

## Reproduction

```bash
MUJOCO_GL=egl MPLBACKEND=Agg QT_QPA_PLATFORM=offscreen PYTHONPATH=.:src \
python3 scripts/run_musculoskeletal_spec_swing.py --bundle driver.npz \
  --out-dir /tmp/msk2_driver --receipt receipt_v2_driver.json \
  --v1-receipt docs/development/full_body_models/evidence/musculoskeletal/receipt.json
```

Receipts: `docs/development/full_body_models/evidence/musculoskeletal/receipt_v2_*.json`.

## MyoFullBody Full-Body Muscle Study (Epic #11642)

Editable research source: [myofullbody_swing.tex](myofullbody_swing.tex) (compile
with `tectonic`; the PDF is not committed). It covers the model and licences, the
spec-to-MyoFullBody orientation mapping, ROM clamps, the static optimisation, the
inverse-dynamics consistency check, reserves, failed experiments and the
reproduction commands. Receipts:
`docs/development/full_body_models/evidence/myofullbody/receipt_{driver,iron}.json`.
Result status for both swings: `NOT_QUALIFIED` (the reserves exceed 10 percent of
the effort RMS in the trunk, legs and arms). Renders: `~/Videos/Parity Audit/musculoskeletal_fullbody/`.

## Build Locally

CI typesets this reference with a pinned Tectonic (`.github/workflows/latex-references.yml`)
whenever a `docs/research/**/*.tex` file changes. To reproduce it:

```bash
tectonic -X compile --outdir build docs/research/musculoskeletal_swing/musculoskeletal_swing.tex
# or, with a TeX Live install:
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build docs/research/musculoskeletal_swing/musculoskeletal_swing.tex
```
