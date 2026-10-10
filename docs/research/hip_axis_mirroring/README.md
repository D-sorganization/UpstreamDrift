# Left Hip Axis Mirroring Reference

Start with the [maintained LaTeX reference](hip_axis_mirroring.tex). It is an
editable standalone research source (see `AGENTS.md`, Modeling Reference
Documentation) for OSV-6 (#11737). The engineering design manual source remains
`manuals/upstreamdrift`.

| Item                                                    | Status                                                         |
| ------------------------------------------------------- | -------------------------------------------------------------- |
| Left hip adduction/rotation mirrored like OpenSim       | Done, unit-tested (`test_hip_mirroring.py`)                    |
| Canonical calibrated receipts regenerated               | Done, with a main control run                                  |
| Both feet within 2 deg, mirror only                     | Trail met; lead **not met** (15 to 32 deg, hip at its limit)   |
| Both feet within 2 deg, `--foot-progression` (MuJoCo, Drake) | Met with the zero-twist and leg-marker corrections        |
| Pinocchio and MyoSuite address toe-out (own FK)         | Met; MyoSuite through the calibrated hip retarget (#12052)     |
| Right knee flexion capped at 10 deg by the spec range   | Open, #12057                                                   |

## Build and Reproduce

```bash
cd docs/research/hip_axis_mirroring && tectonic hip_axis_mirroring.tex
python3 -m pytest tests/unit/motion_matching/test_hip_mirroring.py -q
```

The built PDF is not committed.
