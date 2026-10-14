# Left Hip Axis Mirroring Reference

Start with the [maintained LaTeX reference](hip_axis_mirroring.tex). It is an
editable standalone research source (see `AGENTS.md`, Modeling Reference
Documentation) for OSV-6 (#11737). The engineering design manual source remains
`manuals/upstreamdrift`.

| Item                                                         | Status                                                                    |
| ------------------------------------------------------------ | ------------------------------------------------------------------------- |
| Left hip adduction/rotation mirrored like OpenSim            | Done, unit-tested (`test_hip_mirroring.py`)                               |
| Canonical calibrated receipts regenerated                    | Done, with a main control run                                             |
| Trail foot within 2 deg at address (MuJoCo, Drake)           | Met (within 0.2 deg)                                                      |
| Lead foot within 2 deg at address, mirror only               | **Not met** (15 to 32 deg, hip at its limit)                              |
| Both feet within 2 deg, `--foot-progression` (MuJoCo, Drake) | Met with the zero-twist and leg-marker corrections                        |
| Pinocchio and MyoSuite address toe-out (own FK)              | Met; MyoSuite through the calibrated hip retarget (#12052)                |
| Right knee flexes negative like the left (#12057)            | Done, unit-tested (`test_knee_axis_convention.py`)                        |
| Trail foot within 2 deg after #12057                         | **Not met** (driver -2.1, 7-iron -4.4 deg; trail hip rotation at -40 deg) |
| Two-hand golf grip definition & kinetics (OSV-2, OSV-7, 9)   | Met; neutral V-angle, compliant closure residual $\le 5$ mm across swing  |
| Head gaze stabilisation schedule & neck IK (OSV-3)           | Met; ball tracking to impact + 30 ms hold, quintic minimum-jerk release   |
| Neutral address foot progression parity (OSV-4, OSV-6)       | Met; captured ~20 deg toe-out per foot, mirrored hip rotation coordinates |
| Golfer realism showcase render CLI & manifest (OSV-5)        | Done, unit-tested (`test_render_golfer_realism_showcase.py`)              |

## Build and Reproduce

```bash
cd docs/research/hip_axis_mirroring && tectonic hip_axis_mirroring.tex
python3 -m pytest tests/unit/motion_matching/test_hip_mirroring.py -q
python3 -m pytest tests/unit/scripts/test_render_golfer_realism_showcase.py -q
```

The built PDF is not committed.
