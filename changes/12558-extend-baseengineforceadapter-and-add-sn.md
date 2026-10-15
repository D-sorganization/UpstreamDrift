---
issue: 12558
summary: "Extend BaseEngineForceAdapter and add snapshot_from_adapter with Capability.CLOSED_LOOP_SNAPSHOT and constrained KKT parity"
---

- Add `Capability.CLOSED_LOOP_SNAPSHOT` to `src/shared/python/engine_core/capabilities.py`.
- Extend `BaseEngineForceAdapter` and `PinocchioForceAdapter` with closed-loop snapshot extraction (`compute_closure_drift`, `closure_wrench_frame`, `club_inertia_split`, `foot_contacts_snapshot`) and constrained KKT acceleration parity (`constraintDynamics`).
- Implement `snapshot_from_adapter` and `CapabilityUnavailableError` in `src/shared/python/loop_nullspace/snapshot.py` and canonicalize closure Jacobians in `MultiEngineTorqueAllocator`.
