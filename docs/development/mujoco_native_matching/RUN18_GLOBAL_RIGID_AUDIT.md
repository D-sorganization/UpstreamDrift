# Run18 Global Marker Registration Diagnostic

## Main Finding

Common translation and rotation explain little of the late tracking error.
At0.85seconds, a best global rigid registration lowers Euclidean observed-marker
RMS from103.949mm to99.505mm:91.63%of the original squared error remains.
At0.8seconds,92.59%remains. The late error therefore cannot be explained primarily
as a common global pose offset of the predicted marker constellation.

This does not prove which controls should change: root forces and torques also
alter internal motion through coupled forward dynamics. It weakens the case for
prioritizing root bounds solely on a presumed global translation/rotation error.
Inspect articulation and geometry/attachment disagreement alongside the active
run19 outcome before choosing the next controlled bound-release experiment.

## Counterfactual Decomposition

All25modeled markers are observed at each selected frame. RMS uses equal-weight
Euclidean marker distances and exactly the existing valid mask. No scaling,
reflection, interpolation, frame reset or acceptance-metric replacement occurs.
The rigid transform maps predicted points toward target points.

| Time(s) | Original RMS(mm) | After Translation(mm) | After Rigid(mm) | Translation Correction XYZ(mm) | Rotation(deg) | Squared Error Remaining |
| ------- | ---------------- | --------------------- | --------------- | ------------------------------ | ------------- | ----------------------- |
| 0.60    | 27.750           | 24.135                | 22.825          | [-0.013, -8.509, 10.729]       | 1.070         | 67.66%                  |
| 0.70    | 48.001           | 40.082                | 37.746          | [-4.810, -19.351, 17.318]      | 1.479         | 61.84%                  |
| 0.80    | 58.551           | 56.539                | 56.339          | [-9.416, -11.320, 3.847]       | 0.710         | 92.59%                  |
| 0.85    | 103.949          | 102.247               | 99.505          | [4.529, 13.651, -12.004]       | 4.865         | 91.63%                  |

Translation is target centroid minus predicted centroid. The report also records
R and t for target approximately prediction @ R.T + t; that t is measured about
the world origin and differs from the centroid shift when R is nonidentity.
The rotation is the proper Kabsch optimum, using the existing shared
body_part_viz.fitters.\_kabsch.kabsch_rotation utility. RMS reuses observed_rms.

At0.85seconds, the largest residuals after global rigid alignment are LElbowOut
177.4mm, HeadTop160.4mm, LShoulderBack159.5mm, BackLeft148.7mm and LUArmHigh143.2mm.
The remaining error is not a pure articulation measure: fixed marker attachment
geometry, averaging/noise and articulation all contribute. This registration
has no forward-reachability or joint-closure guarantee and is not a new match.

## Evidence and Reproduction

`evidence/run18-global-rigid-report.json` retains every transform, original and
registered RMS, squared-error decomposition, observed labels and per-marker
residuals. Target coordinates and masks are checked against the exact original
payload; candidate identity matches the independent replay receipt, and the
unmodified terminal RMS reproduces that receipt within1e-12m. Source and input
hashes, original capture hash and model identity are recorded.

```powershell
python3 docs/development/mujoco_native_matching/audit_rigid_marker_errors.py `
  --run <raw-root>/native-ms-fit-9967-18 `
  --target <raw-root>/prefix-1200ms-sextic-01/driver_marker_payload.json `
  --output <NEW-json-path>
```

Run from this worktree with PYTHONPATH set to its root. Raw root is
C:/Users/diete/Repositories/simscape-tour-checkpoints. Output is exclusive.
The exact report, trajectory, target, replay receipt and scripts are retained in
`evidence/raw-run18-global-rigid-audit.zip`. Three tests first failed on the
missing audit module, then passed: pure translation, proper rotation with
nonrigid scale residual, and missing-marker/insufficient-geometry handling.
Ruff passes. No dynamics or optimization ran and no trajectory was modified.
