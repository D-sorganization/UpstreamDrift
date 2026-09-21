# Shadow Tracker Implementation Rules

Follow repository rules and
[Shadow Tracker Agent Handoff](../../../../docs/plans/shadow_tracker/AGENT_HANDOFF.md).
Before implementation, read the selected work package and its dependency issues.

- Write and run failing tests before every new behavior; preserve red/green evidence.
- Reuse canonical camera/state, body geometry, controls and engine interfaces.
- Validate contracts including finite values, dimensions, units, unknowns and ownership.
- Keep orchestration behind narrow protocols; no private engine access or deep chains.
- Separate observed masks, inferred kinematics, forward candidates and qualified results.
- Never fabricate markers, metre errors, physical time, confidence or engine success.
- Never accept a dynamics result without fresh continuous replay and physics audits.
- A planning directory is not an implemented capability; do not advertise it as ready.
