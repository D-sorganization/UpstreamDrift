# Frozen Training Registration Turnover

## Scope and Ownership

Child #11912 extends geometry #11903 under F07 #11791 and F08 #11792. Canonical
authority is chapter29. The existing Kabsch provider, frozen capture contract,
static marker calibration and native bounded IK are reused. No new capture
parser, anatomical alias or physics provider was created. Full implementation
and scientific acceptance remain open.

## Evidence and Interpretation

TDD first exposed mutable registration arrays, missing target-degeneracy
validation and the absent frozen registration API. Tests now prove that changing
nontraining observations cannot change the fitted transform or identity, and
reject missing anchors/provenance, invalid training frames and source-frame
mismatch. Immutable byte-backed arrays resist mutation through aliases and
write-flag changes. A native PinJoint fixture verifies independently generated
marker motion after registration; source-axis tests reject mismatched bodies.

The original and registered private candidate receipts are retained separately.
Source initial pelvis_ty is 0.9299999991 m, not zero; all other selected initial
pelvis coordinates are zero. The diagnostic explicitly separates centroid/yaw
registration from training-frame local offsets. Radii reduce to 0.110–0.188 m,
but RMS remains 37.649 mm and the final sample 65.096 mm with a source-bound hit.
This is not an anatomical fit or ground calibration. Later poses consume their
own observations in IK and are not independent predictions. Anatomical admission
is missing-correspondence; waist and ASIS/PSIS aliases are never inferred.

Actual source anatomy, locks, forces and couplers are unchanged. All detailed
capture transforms, source identifiers, offsets and poses stay in the private
clone. The receipt binds source/loaded/native runtime/provider identities,
original clock, training frame, transform, native reference pose and fit bounds.
The earlier unregistered receipt and failed unbounded LM experiment remain.

## Validation and Next Work

Pure registration, existing registration and governance tests: 31 passed in
Python 3.13, including a later RED/GREEN missing-source-identity regression.
Native fixture plus frozen registration contract: 17 passed in OpenSim 4.6 /
Python 3.12. The older real-C3D test cannot share the native OpenSim process:
the existing ezc3d/OpenSim DLL conflict reproduces there, so it is verified in
the separate capture runtime. This is a runtime isolation boundary, not a
successful combined-SDK claim. Ruff, changed-source mypy and manual governance
pass; the inventory remains empty and release blocked.

The central diff mapper selected an unrelated nonlinear-controller registration
receipt test: 76 passed, one native-runtime skip and one failure. Its committed
receipt hashes for requirements.lock, ode_backend.py and double_pendulum.py
differ from their current HEAD blobs; none is changed here. This is not a
Windows line-ending discrepancy. The stale evidence is preserved and the broad
affected-test gate is not claimed green. Lint/format, diff mypy, Semgrep/import
policy and policy/fragment gates passed. Separate changed-source mypy includes
the newly added files, which were untracked at the first central invocation.

Next requires independently justified marker/body correspondences, source-frame
and ground registration, training-frozen anthropometry/offsets, and uncertainty
validation before capture-level anatomical claims. Continue nonpelvic bindings,
muscle/contact state policy and full-state excitation replay without weakening
passive-load, rigid-tendon, wrist/grip, reserve or coupler gates. No approved
calculation inventory or manual release is asserted.

## Native Chart Follow-Up

The final rigid-cluster optimum has 6.421 mm RMS, while neither equivalent
intrinsic Z-X-Y angle branch fits the unchanged source root ranges. Native
three-pose convention checks agree within 6.67e-16. The principal branch exceeds
pelvis_rotation; the other exceeds pelvis_tilt/list. This isolates a current
gauge/source-chart restriction without declaring a physiological limit, changing
anatomy or choosing a new registration from withheld poses. Detailed receipts
remain private. Local reproducible diagnostic drivers and aggregate reasoning
are in workspace staging `feedback_controls_planning/NATIVE_520_READINESS_NEXT.md`.

The repaired geometry parent contributes tracking comments, simpler attribute
access and a narrowly owned parameter-budget exception, with no numerical
policy changes. After merging it, 28 native/contract/governance tests pass.
The actual candidate rerun has exactly identical achieved q and RMS with the
new native provider hash; the earlier receipt is preserved separately. The
new change-fragment YAML metadata is validated by the complete policy/fragment
gate. Normal publication hooks pass; the unrelated stale research receipt
remains a disclosed broad-test failure.

## Shared Native Fixture Correction

PR #11916 unit CI failed because collecting the native geometry module before
the registration module scoped its plugin-exported `native_pin` fixture to the
first module. The two-file local collection reproduced the missing-fixture
error. Move the single fixture to `tests/opensim/conftest.py` and remove the
test-module plugin declaration; geometry and registration now share ordinary
directory fixture ownership without copying the model or changing assertions.
The native OpenSim 4.6 registration/contract run passes 17 tests. Full-suite
protected CI remains pending after the correction; Qt canvas tracebacks in
the prior job were not the reported failure.
