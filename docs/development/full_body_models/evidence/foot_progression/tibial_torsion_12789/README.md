# Static Tibial-Torsion Foot-Yaw Offset (R3b, #12789)

Parent epic: #12678. Follow-up to R3 (#12683, PR #12731). The engine is MuJoCo,
and the stage is the calibrated address (inverse kinematics only). This is not
a dynamics or replay result.

## Problem

In `style` mode the 7-iron trail foot stopped at 13.5 deg against the owner's
20 deg target, because `hip_rotation_r` sat on its -40 deg external-rotation
limit. The ruling on #12683 forbids widening the hip limit. The leg chain has
no shank or foot yaw, so all toe-out came from the hip.

## Model Change

- **Quantity.** External tibial torsion is the angle about the shank long axis
  (knee centre to ankle centre) between the knee and ankle hinge axes.
  `tibial_torsion.malleolar_torsion_deg` measures it from the spec. The generic
  Rajagopal leg has 6.06 deg on both sides (`GENERIC_LEG_TORSION_DEG`, pinned
  by a test).
- **Source.** Seber S, Hazer B, Kose N, Gokturk E, Gunal I, Turgut A.
  Rotational profile of the lower extremity and foot progression angle:
  computerized tomographic examination of 50 male adults. Arch Orthop Trauma
  Surg. 2000;120:255-258. doi:10.1007/s004020050459. Tibiofibular torsion:
  mean 30.9 deg (right) and 29.1 deg (left), range 16-50 deg.
- **Offset.** `apply_tibial_torsion` turns each ankle's base frame
  (`joints[ankle_<side>].parent_to_base`) about the shank axis through the
  ankle centre. The foot bodies, foot markers and contact spheres turn with it.
  It adds no coordinate, and the ankle centre does not move.
- **Default.** 10 deg on both legs (`DEFAULT_TIBIAL_TORSION_OFFSET_DEG`). It
  is the smallest round value that puts the model's total torsion (16.06 deg)
  inside the documented adult range. No side asymmetry is claimed.
- **DbC.** The offset must be finite and within `OFFSET_RANGE_DEG` =
  [0, 43.94] deg, so the total stays at or below 50 deg. A second application
  raises. The record `tibial_torsion_deg` (per side, plus its source) is
  checked by `validate_full_body_spec`. An offset of 0 returns the document
  unchanged, without the record, so it is bit-identical.
- **Single source.** The spec document is the only place the offset lives.
  `spec_builder.compose_anthropometric_document` applies the default
  (`--tibial-torsion` on the builder CLI). `scripts/apply_tibial_torsion.py`
  applied it to the committed `full_body_spec_anthro_driver.json` and
  `full_body_spec_anthro_iron7.json`; those documents carry hand-tuned contact
  and grip parameters, so they are patched rather than rebuilt (as for #12057).
  MuJoCo, Drake, Pinocchio, OpenSim and MyoSuite all build the ankle from the
  spec placement, so no engine code changed.

## Measured Address

Canonical commands; angles in deg from the calcn-to-toes axis, positive toes
out. Before is the R3 summary at `16e8846229`; the 7-iron `style` row was
re-measured at `6345c8c2a7` before this change and reproduced 15.05 / 13.53
deg exactly. After is `6345c8c2a7` plus this change. Full numbers:
[`address_tibial_torsion_summary.json`](address_tibial_torsion_summary.json).

| Run              | Trail before | Trail after | `hip_rotation_r` before / after | Lead before / after | Address RMS before / after (mm) |
| ---------------- | ------------ | ----------- | ------------------------------- | ------------------- | ------------------------------- |
| 7-iron `style`   | 13.53        | **19.97**   | -40.00 (limit) / -36.57         | 15.05 / 15.07       | 12.19 / 12.18                   |
| 7-iron `capture` | -0.81        | -0.43       | -23.32 / -14.92                 | 15.28 / 15.06       | 9.71 / 10.71                    |
| driver `style`   | 19.80        | 19.74       | -34.38 / -26.58                 | 16.64 / 16.51       | 10.54 / 9.28                    |
| driver `capture` | 3.74         | 3.79        | -18.20 / -8.86                  | 16.52 / 16.43       | 9.62 / 9.83                     |

- The 7-iron and driver trail feet reach the 20 deg style target within
  0.3 deg, and `hip_rotation_r` is off its limit (3.4 deg spare on the
  7-iron). The issue band is 20 +/- 3 deg.
- The lead foot keeps its capture target in both modes. Lead hip internal
  rotation rises by 7-12 deg, to 17.3-26.4 deg, inside the +/-40 deg range.
- The 7-iron `capture`-mode address RMS rises by 1.0 mm and the driver
  `capture` RMS by 0.2 mm; the `style` RMS is unchanged on the 7-iron and
  1.3 mm better on the driver. No tolerance or range was changed.

## Cross-Engine

`test_every_engine_turns_the_feet_like_the_shared_kinematics` checks that each
engine's zero-pose foot yaw changes by the shared-kinematics change to
0.05 deg. On this host it ran for MuJoCo only; Drake, Pinocchio and OpenSim are
not installed natively, so those cases skip and remain to be verified on a host
with the engines (R10).

## Not Done Here

- The canonical ground-support receipts (`anthro_driver`, `anthro_iron`,
  `anthro_driver_seeds`, `anthro_iron_seeds_zmp`) chain to the base document
  hash, which this change alters. They are regenerated on ControlTower with
  R10 (#12692), after R4 (#12684). Until then
  `test_current_baseline_receipts_have_intact_provenance_chain` fails for them.
- The club-face fixtures (`tests/fixtures/club_face/spec_<club>.json`) are
  calibrated snapshots and were not regenerated.
- Re-render in every engine (R10).

## Reproduce

```bash
# Apply (idempotent) to the committed documents
python3 -m scripts.apply_tibial_torsion \
    docs/development/full_body_models/full_body_spec_anthro_driver.json \
    docs/development/full_body_models/full_body_spec_anthro_iron7.json
# Live acceptance (about 4 minutes per capture, PYTHONPATH with the repo root and src/)
python3 -m pytest tests/unit/motion_matching/test_tibial_torsion.py -m live_simulation
```

The address measurement command is the one in
[`../trail_toe_out_12678/README.md`](../trail_toe_out_12678/README.md#reproduce),
with `--foot-progression capture` for the capture rows.
