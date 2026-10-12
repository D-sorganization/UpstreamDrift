---
issue: 11744
summary: "LIFT-4: barbell held at both hands through the OSV-7 compliant bushing on the back-squat same-input bundle, with per-hand bushing wrench, a recorded quasi-static hold and four-engine parity"
---

`BarbellBushingPlant` wraps any same-input `VectorPlant` with a rigid 100 kg
barbell held at the spec's grip frames by the OSV-7 bushing (shared law,
stiffness defaults, zeta 0.7 modal damping for the bar). The back-squat window
re-tracked with the bar (`back_squat_barbell_bundle.npz`) matches MuJoCo in
Drake, Pinocchio and OpenSim within L1 at every 20th state: body acceleration
within 1.5e-6 (scale 2864), per-hand bushing force within 1.1e-7 (scale
1012 N). The quasi-static hold does not pass: with the golf spec's hands the
bar hangs in front, the combined centre of mass is ahead of both forefoot
spheres, and the stance tips forward after about 0.4 s (mean vertical ground
reaction 0.34 of body plus bar weight over the last 0.5 s). The hold checks
are strict expected failures with unchanged thresholds; carrying the bar on
the back needs a torso-mounted bar frame in the spec.
