---
issue: 11838
summary: "Add restricted native Drake frozen torque replay"
dl_state: "in_review"
next_step: "Integrate native candidates and qualify refinement, contact and full-model parity."
title: "Native Drake Replay Development Boundary"
owner: "codex"
branch: "feat/feedback-drake-native-11838"
paths: "src/engines/physics_engines/drake/python/native_torque_replay.py"
---

Add restricted native Drake full-discrete-state frozen torque replay through
the canonical Tools bundle. Bind actual source/provider/parameter/solver and
input policy, audit native effort, and reject unsupported model/state modes.
Document TDD failures, native synthetic evidence and remaining parity gates
in canonical chapter22, inventory, SPEC and dedicated turnover.
