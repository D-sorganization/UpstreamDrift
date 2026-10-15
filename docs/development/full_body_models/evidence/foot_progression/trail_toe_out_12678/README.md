# Trail-Foot Toe-Out at Address: Owner Style Target (R3, #12683)

Parent epic: #12678. The engine is MuJoCo, and the stage is the calibrated address
(inverse kinematics only). This is not a dynamics or replay result.

## Decision

The owner chose **decision B** on #12678 (2026-10-15). The trail (right) foot at
address gets a **style target of about 20 deg** of toe-out. The target is
recorded as owner-set, not measured.

- The value is `foot_progression.TRAIL_TOE_OUT_STYLE_DEG`. It is defined from the
  OSV-4 documented per-foot value `DEFAULT_TOE_OUT_DEG` (20 deg), so the literal
  appears once.
- `foot_progression.resolve_foot_target` is the only place that applies the
  target.
  - In `style` mode, the trail foot gets the style target (source `style`). The
    lead foot keeps the capture measurement (source `capture`), or the flagged
    default where the markers are unreliable (source `default`).
  - In `capture` mode, a reliable capture measurement wins for both feet. That is
    the "capture wins" rule. This mode is bit-for-bit the previous resolver;
    `test_capture_mode_matches_the_legacy_resolver_bit_for_bit` checks this.
- `--foot-progression` now defaults to `style`
  (`address_feet.CANONICAL_FOOT_PROGRESSION`). The canonical commands pass no
  flag, so they now turn the trail foot out.
  - `off`, `capture` and `default` are unchanged.
  - A programmatic `Namespace` without the attribute still resolves to `off`.
- Every receipt foot entry gains `target_source` (`capture`, `default` or
  `style`).

## Gauges (Capture Values, Unchanged)

The three capture gauges and the per-frame spread were measured by OSV-4 and
#12109 (`../osv4_final`, `../pelvis_alignment_12109`). This change does not touch
them.

- The trail target from the corrected ankle-to-toe axis is +4.03 deg on capture-A
  and -0.37 deg on capture-B.
- The forefoot normal differs from it by 5.1 deg and 4.0 deg.
- Decision B does not depend on a gauge being biased: the target is a style
  choice.
- New synthetic tests cover the three gauges at known toe-out, plus a
  left-handed mirror case. They are in `tests/unit/motion_matching/test_foot_progression.py`.

## Measured Address Toe-Out

Angles are in degrees, from the calcn-to-toes axis, with positive meaning toes out.

Measured at base commit `16e8846229` plus this change, on one Windows host. Full
numbers are in [`address_toe_out_summary.json`](address_toe_out_summary.json).

| Run                            | Mode                  | Lead  | Trail     | `hip_rotation_r` | Address RMS (mm) |
| ------------------------------ | --------------------- | ----- | --------- | ---------------- | ---------------- |
| capture-A driver, before       | `off` (old default)   | 29.08 | -2.93     | 10.21            | 10.95            |
| capture-A driver, capture mode | `capture`             | 16.52 | 3.74      | -18.20           | 9.62             |
| capture-A driver, after        | `style` (new default) | 16.64 | **19.80** | -34.38           | 10.54            |
| capture-B 7-iron, before       | `off` (old default)   | 33.46 | -9.81     | 3.17             | 8.42             |
| capture-B 7-iron, capture mode | `capture`             | 15.28 | -0.81     | -23.32           | 9.71             |
| capture-B 7-iron, after        | `style` (new default) | 15.05 | **13.53** | **-40.00**       | 12.19            |

- **Driver.** The trail foot is within 0.2 deg of the 20 deg target, and the
  address marker RMS improves by 0.4 mm. The lead foot equals the capture-mode
  lead (0.12 deg apart; same target).
- **7-iron.** The target is **not met**. `hip_rotation_r` sits on its -40 deg
  external-rotation limit (`coordinate_ranges_deg`, unchanged), so the trail foot
  stops at 13.5 deg. That is 6.5 deg short of the target, and the address RMS
  rises 3.8 mm against `off`.
  - From the capture-mode pose (-23.3 deg, trail -0.8 deg), about -47 deg of hip
    rotation would be needed at the measured sensitivity of about 0.85 deg of foot
    yaw per degree.
  - The range was not widened and no tolerance was changed.
  - Reaching 20 deg on this capture needs an owner decision: widen the trail
    hip's external-rotation range (a modeling change), accept about 13.5 deg on
    the iron, or take the trail foot from yaw elsewhere in the chain. R4's knee
    flex changes the leg coupling and should be re-measured first.

The shared FK seed (`seed_document_feet`) reaches 20 deg on both documents
(`test_style_seed_turns_the_trail_foot_out_and_keeps_the_lead_foot`). The limit
appears only when the marker fit also has to hold the capture's leg posture.

## Not Done Here

The canonical ground-support receipts (`anthro_driver_seeds`,
`anthro_iron_seeds_zmp`) were **not** regenerated. The 7-iron result above needs
the owner decision first, and the receipts are canonically made on ControlTower
(`ud-sim`), where host-to-host solver variation is 0.3 to 0.5 mm. Regenerate
both receipts once the 7-iron decision is recorded. The regeneration is
serialized before R4 (#12684).

## Design Manual

`manuals/upstreamdrift` has no address-stage or foot-progression chapter, and
`calculation-registry.json` registers no calculation for it (the governance check
reports 0 registered calculations). The calculation-level reference stays
`docs/development/full_body_models/DESIGN_DECISIONS.md` §15, which this change
updates.

## Reproduce

```bash
# PYTHONPATH must contain the repo root and src/
python - <<'PY'
import sys
from src.shared.python.motion_matching.pipeline import cli
from src.shared.python.motion_matching.pipeline.address_feet import FootTargets, model_feet_deg
args = cli.build_parser().parse_args(sys.argv[1:] or [
    "--static-seeds", "--fix-club-markers", "--capture", "driver",
    "--spec", "docs/development/full_body_models/full_body_spec_anthro_driver.json",
    "--out", "runs/r3_driver"])
res = cli.calibrate_run(args).cal_res
probe = FootTargets({"left": 0.0, "right": 0.0}, {"left": False, "right": False}, "probe")
print(model_feet_deg(res.kin, res.address2.q, probe))
PY
```

For the 7-iron, use `--static-seeds --zmp-filter --fix-club-markers --capture
iron --spec docs/development/full_body_models/full_body_spec_anthro_iron7.json`.
Add `--foot-progression off` or `--foot-progression capture` for the comparison
rows.
