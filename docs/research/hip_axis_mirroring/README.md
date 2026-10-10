# Left Hip Axis Mirroring Reference

Start with the [maintained LaTeX reference](hip_axis_mirroring.tex). It is an
editable standalone research source (see `AGENTS.md`, Modeling Reference
Documentation) for OSV-6 (#11737). The engineering design manual source remains
`manuals/upstreamdrift`.

| Item                                                | Status                                       |
| --------------------------------------------------- | -------------------------------------------- |
| Left hip adduction/rotation mirrored like OpenSim   | Done, unit-tested (`test_hip_mirroring.py`)  |
| Canonical calibrated receipts regenerated           | Done, with a main control run                |
| Trail foot within 2 deg at address (MuJoCo, Drake)  | Met (within 0.2 deg)                         |
| Lead foot within 2 deg at address                   | **Not met** (15 to 32 deg, hip at its limit) |
| Pinocchio and MyoSuite address toe-out              | Not available on this pathway                |
| Right knee flexes negative like the left (#12057)   | Done, unit-tested (`test_knee_axis_convention.py`) |
| Trail foot within 2 deg after #12057                | **Not met** (driver -2.1, 7-iron -4.4 deg; trail hip rotation at -40 deg) |

## Build and Reproduce

```bash
cd docs/research/hip_axis_mirroring && tectonic hip_axis_mirroring.tex
python3 -m pytest tests/unit/motion_matching/test_hip_mirroring.py -q
```

The built PDF is not committed.
