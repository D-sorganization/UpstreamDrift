# Full-Capture Static Diagnostic Checkpoint

This is a completed diagnostic, not a full-swing forward simulation. The original
run102 candidate remains unchanged and rejected at 0.85 s.

## Native Static Curve

static_curve.tar.gz contains the exact executed audit.py, corrected_native.py,
constrained_marker_pose.py and full receipt.json (all coordinates, predictions,
targets, validity, yaw/closure/convergence, local bound activity). Its SHA256 is
in summary.json. No unpacked receipt is duplicated in this directory.

Remote directory: /home/dieterolson/native-static-curve-20260917-01 on ControlTower,
WSL ControlTower-Runner. Completed with exit 0 in 143.434 seconds: 38 sampled frames,
304 solves. Eight solves per frame: two starts at exact yaw, -5%, +5%, and free yaw.
Inputs and runtime match the preceding terminal_feasibility_20260917 audit; the
script additionally reads inputs/driver_marker_payload_9967.json from the fitter
runtime. All prefix target points/masks were checked against the saved replay.
No torque fitting, model changes or joint-limit qualification occurred.

Search boxes are translation +/-0.25 m and rotation +/-1 rad. At times through
0.85 s the box is centered on the saved replay pose; later it is centered on the
previous sampled best allowed static pose. These are local continuation searches.
The best allowed results have no active bounds through 1.20 s, but cases at 1.25,
1.30, 1.35, 1.40, 1.45 and 1.55 s do. The 1.25 s spike is **not evidence of a
physical error floor**. Valid marker counts also vary (24 at 1.25 s, 22 at
1.45–1.55 s, 19 at the final sample). One of eight solves at 1.45 s did not converge.
Preserve masks and inspect these limitations before comparing later rows.

Selected best yaw-valid RMS: 33.819 mm at 0.80 s; 39.763 mm at 0.85 s;
41.911 mm at 0.90 s; 46.294 mm at 1.05 s. All later results are local static
witnesses, not global minima or forward-dynamics candidates. Do not extrapolate a
successful full-swing match from any single row below 35 mm.

To reproduce, verify the archive hash, extract into a NEW remote directory, and
use the interpreter/PYTHONPATH command documented in the preceding audit README.
The source contains exact original paths. Do not overwrite completed directories.

## Independent Rigidity Diagnostic

rigid_curve.py reuses the tested shared rigidity provider over all 654 frames.
It reads the tracked run102 candidate and archived original C3D-derived payload.
rigid_curve.json records conditional independent-body lower bounds and observed
same-body pair-distance ranges. Reproduce from the repository root to a new file:

```text
python docs/development/simscape_tour_matching/native_evidence/static_curve_20260917/rigid_curve.py --repo . --output <NEW_OUTPUT_JSON>
```

Seven existing rigidity tests pass. Missing entire body observations are represented
as null RMS, not zero; the first diagnostic attempt rejected nonfinite JSON, and
the completed script now serializes missing values explicitly before opening output.
The rigid relaxation gives 23.677 mm at 0.85 s, below the articulated local result;
it cannot certify that a connected pose meeting 35 mm exists.

Head and back markers share Hub. BackLeft–HeadSide model separation is 0.42184 m,
while target separation ranges from 0.34968 to 0.45929 m across valid observations.
Rigid motion cannot reproduce a changing pair distance exactly. Fixed recalibration
can change the compromise but cannot reproduce every such distance. Determine
whether target processing, real relative motion or model assumptions explain this;
do not silently add a neck or remove markers. Full-capture calibration and coherent
model-variant decisions remain next-agent work.
