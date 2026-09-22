# MyoSuite Engine Notes (MS-51)

## Pinned `myo_sim`

The MyoHub [myo_sim](https://github.com/MyoHub/myo_sim) library is a git
submodule at `shared/models/myosuite/myo_sim`.

| Field                       | Value                                       |
| --------------------------- | ------------------------------------------- |
| Gitlink pin (authoritative) | `33f3ded946f55adbdcf963c99999587aadaf975f`  |
| `.gitmodules`               | URL/path only — not the commit pin          |
| Bootstrap                   | `scripts/setup_myosuite_models.ps1` / `.sh` |

## Golfer Scenes

Generator: `src.engines.physics_engines.myosuite.python.golfer_scene:generate_golfer_scene`

| Artifact       | Path                                                                     |
| -------------- | ------------------------------------------------------------------------ |
| Driver scene   | `shared/models/myosuite/golf/body/golfer_myobody_driver.xml`             |
| Iron scene     | `shared/models/myosuite/golf/body/golfer_myobody_iron.xml`               |
| Receipt        | `shared/models/myosuite/golf/body/golfer_myobody_receipt.json`           |
| Coordinate map | `src/engines/physics_engines/myosuite/python/coordinate_map_anthro.json` |

Composition base is `myobody_simpleupper.xml` (hands + feet). Full
`myobody.xml` lacks arms and cannot host dual-grip welds.

Each scene adds:

- Club body from shared `club_models.ClubSpec` (driver / iron7)
- Dual-grip site welds (`grip_weld_r` / `grip_weld_l`)
- Four foot contact sphere markers (heel + forefoot × left/right) with
  shared Hunt–Crossley parameter hints and `contype=0` (external shared
  contact law; stock MuJoCo contact is not claimed equivalent)

Scenes live under `shared/models/myosuite/golf/body/` so nested
`../../myo_sim/...` includes resolve the same way as stock `body/*.xml`.

## Coordinate Map Honesty

The anthro map is diagnostic. Mapped document coordinates resolve to named
MyoSuite joints. Omitted sources (root freejoint translations and one
scapular DOF) are listed in `omitted_source`. A partial map does **not**
prove dynamics equivalence or satisfy 15 mm marker parity.

## Inventory Status

MS-102 flagship packages `myosuite/driver` and `myosuite/iron` are `ready`
for structural smoke after native MuJoCo load of the generated scenes.
Still **not** claimed by MS-51:

- G1 / full-swing dynamics acceptance
- 15 mm marker parity (`parity_budget_qualified=false`)
- Golf-specific muscle–tendon calibration beyond upstream myo_sim

## Tests

```bash
python -m pytest tests/unit/engines/myosuite/test_golfer_scene.py -q
python -m pytest tests/myosuite/test_golfer_scene_native.py -q
```
