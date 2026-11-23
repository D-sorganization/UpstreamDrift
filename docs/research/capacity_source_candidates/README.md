# Upper-Limb and Toe Capacity Source Candidates

This research note identifies primary measurements that may inform later
source review. **No model capacity, verification status, or feasibility gate is
changed.** Group-mean laboratory measurements are not subject-specific swing
limits. The existing `src/config/loop_nullspace_registry.json` remains
authoritative for current values and their present `UNVERIFIED` labels.
The coordinate signs below are the reference-pose conventions in
`src/shared/python/motion_matching/coordinate_sign_map.py`; joint gimbal axes
and generalized-force mappings change with pose.

## Forearm and Wrist

[Rezzoug, Hernandez and Gorce (2021)](https://doi.org/10.3390/biomechanics1010008),
*Biomechanics* 1:102–117, Tables 2–3, studied seven right-handed male sports
science students (20.0 ± 2.7 years). The protocol instrumented the right upper
limb, but does not explicitly identify the limb used for the torque test.
Biodex isometric torque used the larger of two 4-second
trials in a seated, strapped posture: elbow flexion 71.3 ± 6.7°, forearm
pronation 75.1 ± 7.9°, wrist deviation 4.5 ± 5.2°. Mean ± SD torque was
9.3 ± 2.3 N m pronation and 11.8 ± 5.3 N m supination; wrist ulnar and radial
deviation were 17.1 ± 8.0 and 14.4 ± 7.1 N m. These are candidates for
forearm `LFInput`/`RFInput` negative and positive, and wrist `LWInputX`/
`RWInputX` negative and positive, respectively. Mapping either side and
transferring a static population mean to a dynamic swing require review.

[Peleg et al. (2025)](https://doi.org/10.3390/jfmk10040377), *J. Funct.
Morphol. Kinesiol.* 10:377, Table 1, studied 20 men and 20 women, all
right-handed, testing the right radiocarpal joint at 90°/s through 5° radial
to 40° ulnar deviation. Men produced concentric peaks 26.5 ± 6.6 N m ulnar
and 14.9 ± 3.4 N m radial; eccentric peaks were 30.3 ± 7.1 and 18.9 ± 3.9 N m.
These inform dynamic directional comparisons, not isometric limits or a
validated full torque–velocity envelope. See the LaTeX reference for posture.

## Shoulder, Scapula and Toes

[Nicolozakes et al. (2021)](https://doi.org/10.1249/MSS.0000000000002708),
*Med. Sci. Sports Exerc.* 53:2354–2362, Table 1, tested the dominant right
arm of eight healthy men and nine women, with scapula clamped, shoulder 90°
abducted and 20° horizontally adducted, elbow flexed 90°. Male neutral-rotation
horizontal adduction/abduction were 68 ± 24 / 45 ± 16 N m; with 90° external
rotation they were 42 ± 10 / 38 ± 13 N m. These directions are plausibly
related to `LSInputX`/`RSInputX` (positive horizontal abduction at the model
reference pose), but the posture, restraint and moving gimbal axes preclude
direct substitution. The paper also reports both internal/external rotations.
Rezzoug reports flexion/extension 71.6 ± 17.0 / 87.5 ± 25.7 N m and
internal/external rotation 53.9 ± 12.0 / 41.7 ± 5.8 N m, not the model's
horizontal axis; its shoulder and wrist coordinate systems differ. Flexion/
extension may inform shoulder Y and internal/external rotation shoulder Z,
but their coordinate-force mappings and moving axes require review.

[Williams et al. (2009)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2953350/),
*N. Am. J. Sports Phys. Ther.* 4:200–209, measured protraction and retraction
forces in 30 healthy women aged 22–28. They are external forces in N or kg,
not torques about the model's independent scapular rotation coordinates;
neither a moment arm nor a valid generalized-force mapping is established.

[Hile et al. (2023)](https://doi.org/10.3390/s23104654), *Sensors* 23:4654,
Table 4, tested 38 adults (18 men), seated with 90° knee flexion and neutral
ankle. It measured great-toe extension and flexion **force** (N), not MTP
torque. The negative `mtp_angle_l/r` direction is toe extension, but the
model's single joint represents all toes and the required moment arm and
lesser-toe contribution are not established. The registry's existing Saeki
2021 reference addresses flexion, not extension. Do not convert these forces
to N m without defensible subject/device geometry and a reviewed joint map.

## Next Evidence Review

The cited candidates were absent from the inspected registry by author and
DOI. Preserve them as source leads, check measured side, joint axes and posture
against the pinned v2 model, and separately evaluate population and dynamic
transfer. Do not promote an `UNVERIFIED` direction or change a capacity on the
basis of this note. No physiological or performance qualification is claimed.
