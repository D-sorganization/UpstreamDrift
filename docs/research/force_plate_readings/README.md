# Force-Plate Readings Reference

Start with the [maintained LaTeX reference](force_plate_readings.tex). It is an
editable standalone research source (see `AGENTS.md`, Modeling Reference
Documentation) for OF-05 (#13069) under epic #13061. The engineering design
manual source remains `manuals/upstreamdrift`.

| Item                                                          | Status                                        |
| ------------------------------------------------------------- | --------------------------------------------- |
| Linear interpolation on non-uniform samples, no extrapolation | Done, unit-tested (`test_readings.py`)        |
| Bias estimate, skew and composition                           | Done, unit-tested (`test_readings_budget.py`) |
| Clock rounding snapped onto samples                           | Done, unit-tested                             |
| CoP uncertainty in world axes under yaw                       | Done, unit-tested                             |
| Bias estimate valid across under-sampled impacts              | **Not met** (pinned limitation)               |
| Plate hardware measurement                                    | **Deferred** (OF-HW, epic #13061)             |
