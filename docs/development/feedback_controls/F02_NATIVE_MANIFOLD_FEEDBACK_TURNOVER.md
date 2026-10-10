# F02 Native Manifold Feedback Turnover (#11946)

This child connects the existing F02 distributed-feedback controller to the
F05c native MuJoCo 3.8 floating-root fixture. It does not change the shared
F02 law or create another replay protocol. `NativeMuJoCoTangentModel` uses
`mj_differentiatePos` for current-minus-reference configuration error and
expands native `qM` for the tangent inverse mass. The adapter admits only the
contact-free, gravity-free 9/8/2 Euler fixture and the ordered direct unit
hip/knee motors. The six-dimensional root receives no actuator command.

The native runner samples exact simulated $(q,v)$ once per 0.01 s step,
applies the F02 TVLQR law and bounded allocator, and holds the actual
post-limit torque. The pinned Tools T01 bundle contains a complete
`mjSTATE_INTEGRATION` initial state, model/provider/policy/input identities
and the executed ZOH motor history. A fresh MuJoCo run independently replays
those torques. It receives no controller or observation and must match every
full integration-state row within $10^{-12}$.

The test-first sequence had a missing-module failure, then six actual native
tests and two receipt-parser tests passed. A quaternion sign-flip gives zero
native rotation error despite a nonzero Euclidean four-vector difference;
malformed quaternion, channel permutation and contact geometry reject.
The 12-step initial hip error is 0.4 rad. Native feedback finishes at
0.1260837119 rad versus 0.4 rad for the frozen-zero-input run. A separate
saturation trial requests $(4,-4)$ N m and proves that only the native motor
bounds $(2,-1.5)$ N m enter the saved bundle and independent replay. The
source-hashed fixture evidence is
`F02_NATIVE_MANIFOLD_RECEIPT_MJ38.json`. It is a deterministic synthetic
software-boundary result, not a measured-golfer or runtime qualification.

The moving-reference test begins with a separate native replay of varying
admissible two-motor torque. It verifies each F05c native tangent Euler
derivative against that replay's next state, builds MOSAIC TVLQR gains
along the nonconstant physical trajectory, and gives F02 its exact
replayed nominal states and frozen feedforward. At one held-out 0.25 rad
hip perturbation, the feedforward-only trajectory ends 0.0750848 rad from
the moving target and feedback ends 0.0246020 rad away. The teacher and
frozen-run input SHA-256 identities match; the feedback-applied input SHA
differs. Both runs independently reproduce every full native integration-
state row exactly. This is a real native moving-reference handoff and input
separation, but the teacher input was not optimized or fitted to capture.
F03's existing sparse-collocation spike is one-DOF and has no native 9/8/2
optimizer export yet.

Reproduce using an environment with MuJoCo 3.8.0 and the pinned Tools T01
contract:

```bash
F02_NATIVE_RECEIPT=1 python -m pytest -q tests/unit/motion_matching/test_native_distributed_feedback.py -o addopts= -o junit_family=legacy --junitxml=/tmp/f02-native-manifold.xml
python -m pytest -q tests/unit/motion_matching/test_f02_native_receipt.py -o addopts=
python -m scripts.f02_native_manifold_receipt --junit /tmp/f02-native-manifold.xml --host-alias <local-host-label> --output docs/development/feedback_controls/F02_NATIVE_MANIFOLD_RECEIPT_MJ38.json
```

The parser refuses missing/failed native tests, incomplete or malformed
identities, worse feedback or a replay difference above tolerance. The
receipt binds source and model/policy/input hashes but does not establish
anatomical geometry, real contact/grip, muscle physiology, marker accuracy,
full-swing continuous replay, six-engine parity or a hard solver deadline.
F02 #11786, F05 #11789 and manual release remain open. This child stacks
on F05c PR #11922 and must stay unarmed until that prerequisite reaches
`main` and the PR base is retargeted.
