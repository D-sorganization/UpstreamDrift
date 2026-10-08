
# Active: Shared Grip Wrench Core — #11713

- Branch `claude/gcv-7-grip-wrench`; epic #11706. Module `src/shared/python/biomechanics/grip_wrench.py` (hand-on-club wrench, midpoint net force and couple, contact-moment/free-torque split, per-hand MOF, club-local frame, `split_method`).
- Simscape fixtures (`tests/unit/engines/simscape/test_force_channels.py`) have only total hand force, LH MOF and midpoint couple, no per-hand forces, so the Simscape cross-check is deferred to GCV-9 (#11715).
- Next: engine adapters populate `ContactReaction.grip_wrench` through `to_contact_reaction_wrench`.
