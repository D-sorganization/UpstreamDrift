---
issue: 12151
summary: "Add assistance-explicit OpenSim scalar replay with native force and work accounting"
---

# Assistance-Explicit Native Replay

Prerequisite of #12147. Introduce a separate T01 profile for fixed-path native equilibrium muscles and
CoordinateActuators. Preserve strict muscle-only admission and share the owned
native integration loop. Bind role, bounds, gains, units, source and complete
state; verify actual applied controls and physical time. Return immutable native
force/power and sampled work with exact input/state/policy lineage. Synthetic
fixtures do not qualify full-model capture matching or physiology.
