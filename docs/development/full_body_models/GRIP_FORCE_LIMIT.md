# Grip Force Limit (HRG-HX4)

> **Warning: `grip_force_limit_n` is NOT `grip_strength_deg`.** `GripPose.grip_strength_deg` is a visual roll offset in degrees (see [`HANDS_AUDIT.md`](HANDS_AUDIT.md)). `grip_force_limit_n` is a force cap in newtons. Never convert one into the other.

Issue: #13761 (epic #13757).

## Field

- `GripPose.lead_grip_force_limit_n` and `GripPose.trail_grip_force_limit_n`,
  read through `GripPose.grip_force_limit_n(hand)`.
- The default is the explicit string `"unlimited"`, so configuration and
  capability reports show it. Changing that default needs its own migration
  decision and validated limits.
- A finite value must be a finite number greater than 0 N
  (`src/shared/python/grip_contact/grip_force_limit.py`). Anything else raises.
- Units: newtons, a cap on the magnitude of one hand's attachment force. No
  torque limit is defined. No physiological value is claimed.

## Enforcement Under the Option C Ruling

The cap is enforced only in the Python bushing law. MuJoCo pad contact, and
every other engine or mechanism that cannot enforce a finite limit, raises.

`force_limit_enforcement` in `src/config/grip_capability_inventory.json`
records, per engine, whether `pad_contact` and `weld_bushing` enforce a finite
limit. `GripCapabilityInventory.require_grip_force_limit_enforceable` raises
`ValueError` for a finite limit on any engine or mechanism whose flag is not
`true`, including an engine that is not listed (fail closed). `"unlimited"`
always passes. `force_limit_enforcement_report()` states the default and each
engine's flags.

Current state: `weld_bushing` is `true` for MuJoCo and Pinocchio (HRG-HX4 Part 3, closing #13761), where `ClubOnBushings` enforces the force cap via `bushing_wrench(..., force_limit_n=...)`. `pad_contact` and all other engines remain `false` (fail closed).
