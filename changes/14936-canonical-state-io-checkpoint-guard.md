# Change fragment: ADR-0026 canonical state I/O (#14936)

Issue: #14936

## Summary

Wire `get_canonical_state` / `set_canonical_state` on `BasePhysicsEngine` for MuJoCo and
Pinocchio via `pose_interchange` adapters; checkpoints record the short `engine_type` and
`restore_checkpoint` rejects cross-engine restores; `CheckpointManager.restore` raises on
checksum failure.

## SPEC touchpoints

- Engine plugin checkpoint contract (state convention metadata)
- Canonical Core EPIC #6772 / ADR-0026 acceptance (CC-7 round-trip, cross-engine guard)
