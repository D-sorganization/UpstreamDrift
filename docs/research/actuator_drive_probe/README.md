# MuJoCo Actuator Drive Research Probe

This experimental seam compiles a copy of the existing plant MJCF with caller-supplied memoryless joint motors. For each explicit `(q, v, control)` it obtains MuJoCo `qfrc_actuator` after control and force limits and transmits that generalized force exactly once to the existing constrained KKT solve. It rejects a missing motor set, stateful actuator definitions, nonfinite inputs, shape mismatch, and force applied to a passive shaft coordinate.

The committed v2 model compiles with zero native actuators; this probe adds no actuators to the specification and provides no clinical limits, transmission parameters, or source provenance. Unit gears and ±1 limits in the tests are synthetic. The seam is a point-evaluation experiment, not a rollout, anatomical admission, complete actuator-state replay, or a production-ready hardware model. It does not authenticate caller-provided motor definitions. External loads remain the existing plant's independent forces.

Run headlessly from the repository root: `python -m pytest -q tests/unit/motion_matching/same_input/test_mujoco_actuator_probe.py`. See [calculation note](actuator_drive_probe.tex).
