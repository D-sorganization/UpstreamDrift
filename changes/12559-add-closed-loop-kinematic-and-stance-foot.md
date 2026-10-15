---
issue: 12559
summary: "Add closed-loop kinematic and stance-foot contact admission gate with AdmittedTrajectory and RepairRecord"
---

- Implement `AdmissionGates`, `FrameAdmissionDiagnostics`, `RepairRecord`, `AdmissionVerdict`, `AdmittedTrajectory`, `AdmissionReport`, and `admit` in `src/shared/python/loop_nullspace/admission.py`.
- Enforce scaled position, velocity, and acceleration closure gates, stance-foot contact velocity checks, derivative noise-floor and time-synchronization verification, and reversible `project_to_closure` repair.
