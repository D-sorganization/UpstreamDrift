---
issue: 11815
summary: "Add explicit model-owned native contact policies and complete wrench evidence to muscle replay"
dl_state: "in_review"
next_step: "Qualify full-body anatomy, calibrated ground/grip contact and full-horizon private mocap replay."
title: "Native Muscle Contact Replay"
owner: "codex"
branch: "feat/feedback-native-contact-11815"
paths: "src/engines/physics_engines/opensim/python/tour_matching/muscle_replay.py"
---

# Native Muscle Contact Replay

- Add explicit native contact-force path policies and complete immutable world-frame wrench evidence to independent muscle excitation replay.
- Reuse the native OpenSim force/torque provider; reject unlisted forces, unsupported contact geometry and incomplete evidence.
- Validate actual synthetic compliant muscle/contact integration and tighter-accuracy replay; full-body anatomy, grip and private mocap acceptance remain open.
