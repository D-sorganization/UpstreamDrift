---
issue: 11900
summary: "Add native Pinocchio frozen torque replay through the existing engine"
dl_state: "in_progress"
next_step: "Qualify shared-adapter regressions and required production model bindings."
title: "Native Pinocchio Replay Development Boundary"
owner: "codex"
branch: "feat/feedback-pinocchio-native-11900"
paths: "src/engines/physics_engines/pinocchio/python/native_torque_replay.py"
---

Restore complete native manifold configuration and tangent velocity, bind
source/provider/integrator identity and independently replay frozen unit motor
torques. Reuse the existing public integration kernel and canonical Tools
schema. Record actual native RED/GREEN, refinement and unqualified full-model
and capture gates in chapter25 and turnover.
