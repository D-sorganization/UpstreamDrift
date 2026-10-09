---
issue: 11912
summary: "Freeze training-only native capture registration with explicit geometry lineage"
dl_state: "in_review"
next_step: "Qualify anatomical correspondences and ground registration before independent muscle-driven capture replay."
title: "Frozen Training Capture Registration"
owner: "codex"
branch: "feat/feedback-frozen-registration-11912"
paths: "src/engines/physics_engines/opensim/python/tour_matching/frozen_registration.py"
---

Freeze explicit training-only rigid registration with source/frame lineage and
immutable transforms. Reuse native geometry and existing calibration/IK for an
explicitly unqualified pelvis coordinate-gauge diagnostic; retain missing
anatomical correspondence, residuals, source bounds and private evidence.

Share the native PinJoint fixture through directory conftest so combined
geometry/registration collection retains fixture visibility in the unit gate.
