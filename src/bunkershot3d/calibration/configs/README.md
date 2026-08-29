# BunkerShot3D calibration configs

## `canonical.yaml`

Hand-authored reference configuration shared by all three backends. Its
`contact_model` values are literature/design defaults, not measurements.

## Removed: `sand_chrono.yaml`, `sand_mpm.yaml`, `sand_liggghts.yaml`

These three files were deleted in the fix for
[#7999](https://github.com/D-sorganization/UpstreamDrift/issues/7999). They were
presented as per-backend sand properties derived from angle-of-repose and
drained-shear-cell experiments. They were not:

- their `friction_coefficient` (0.5 in all three) was the algebraic fixed point
  of two hand-written linear formulas — `20 + 24f = 32` and `20 + 30f = 35` both
  invert to `f = 0.5` — and carried no information about any backend;
- their `restitution_coefficient` (0.679 / 0.417 / 0.454) were three independent
  uniform random draws. No experiment read that variable, so the objective was
  exactly flat in it and `differential_evolution` returned whichever member of
  its random population survived. Re-running the script produced different
  numbers every time, with a reported `error` of ~1e-26 either way;
- `calibrate_all.py` hardcoded `use_mock=True` inside an `except`, so the
  MuJoCo angle-of-repose path was unreachable from `__main__`.

Nothing in `src/` read these files.

## Regenerating

```bash
python -m bunkershot3d.calibration.calibrate_all --backends mujoco
```

`--use-mock` is available for fast tests. Files produced with it carry
`provenance.method: analytical-mock` and must not be cited as measured sand
properties. `restitution_coefficient` is copied from `canonical.yaml` and
reported under `provenance.not_calibrated`: no experiment in this package
measures it.

## The F1 Constitutive Calibration Record

`f1_continuum.yaml` is not checked in, for the same reason the three `sand_*.yaml` files above were
deleted: a calibration file sitting in the tree gets cited. Produce it when you
need it:

```bash
python -m bunkershot3d.calibration.f1_continuum
python -m bunkershot3d.calibration.f1_continuum --no-search   # closed form only
```

It records the friction angle fitted to the declared drained-shear-cell targets
through F1's own Drucker-Prager return map (issue #8733 section 6), **beside**
the Quikrete-analogue value it replaces, with the residual split into the part
the fit removed and the part no parameter value can remove.

The targets are declared numbers, not measurements of golf bunker sand. The
record says so in `provenance.honesty`, carries
`provenance.measured_on_bunker_sand: false`, and pins
`nasa_std_7009b_validation_levels_met: 0` of 4. Fitting F1 to a stated target
makes it self-consistent with that target; it does not validate it.

Only `friction_angle_deg` is calibrated. `provenance.not_calibrated` lists what
is not, and why — most importantly the elastic shear modulus, which cancels
exactly out of the drained limit ratio and keeps its Hardin & Richart (1963)
`ESTIMATED` label.
