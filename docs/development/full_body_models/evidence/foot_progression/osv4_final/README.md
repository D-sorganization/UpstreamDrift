# Address Foot Orientation: Final Evidence (#11730)

Fresh address solves on current `main` (after #12145: #12059 MyoSuite hip
retarget through calibrated hip frames, #12125 spec pelvis alignment) for
capture-A (reference driver) and capture-B (reference iron). Each
`engines.json` evaluates the one fitted address vector in the engine's own
forward kinematics (MuJoCo, Pinocchio, Drake, MyoSuite via the retarget map,
OpenSim Simbody FK of the exported model). Capture-O (owner driver) is private
and was not measurable on this host; its target stays unmeasured here.

Toe-out, degrees, model (error against the capture), tolerance 2 degrees:

| Engine    | A lead      | A trail      | B lead      | B trail      |
| --------- | ----------- | ------------ | ----------- | ------------ |
| MuJoCo    | 16.59 (+0.22) | 3.92 (-0.11) | 15.19 (+0.19) | -0.42 (-0.05) |
| Pinocchio | 16.59 (+0.22) | 3.92 (-0.11) | 15.19 (+0.19) | -0.42 (-0.05) |
| Drake     | 16.59 (+0.22) | 3.92 (-0.11) | 15.19 (+0.19) | -0.42 (-0.05) |
| OpenSim   | 16.59 (+0.22) | 3.92 (-0.11) | 15.19 (+0.19) | -0.42 (-0.05) |
| MyoSuite  | 16.55 (+0.18) | 3.88 (-0.15) | 15.15 (+0.15) | -0.46 (-0.09) |

Capture targets: A 16.37 / 4.03, B 15.00 / -0.37 (lead / trail).
`opensim_native_address.json` is OpenSim's own address fit
(`tour_matching.address`), an independent pathway that also lands within
0.02 degrees.

Reproduce (MyoSuite with the MyoSuite environment's Python, others with
`PYTHONPATH=src` so `shared.*` resolves to `src/shared`):

```bash
python3 -m src.shared.python.motion_matching.pipeline.cli --address-only \
  --spec docs/development/full_body_models/full_body_spec_anthro_driver.json \
  --static-seeds --capture driver --engine mujoco --foot-progression capture \
  --out RUN
python3 -m scripts.address_foot_progression_engines evaluate --run-dir RUN \
  --label driver --engines mujoco pinocchio drake opensim --out RUN/engines.json
myo-python -m scripts.address_foot_progression_engines evaluate --run-dir RUN \
  --label driver --engines myosuite --out RUN/engines.json
python3 -m scripts.render_address_engine_stills --run-dir RUN \
  --engines-json RUN/engines.json --engines mujoco pinocchio drake opensim myosuite \
  --out STILLS
```

This is the calibrated address (inverse kinematics), not a dynamics result.
Overhead stills are partly occluded by the torso; the projected foot axes
(red) against the straight-ahead reference (white) carry the measurement. The
MyoSuite arena overhead is rendered with `--no-axes --overhead-distance 4.5`:
its camera registration does not match the shared pinhole (feet appear about
0.8 m from the projected axes), so that still carries the text annotation only,
and the viewer shows the spec model in the arena rather than the `myolegs`
model used for the MyoSuite number.
