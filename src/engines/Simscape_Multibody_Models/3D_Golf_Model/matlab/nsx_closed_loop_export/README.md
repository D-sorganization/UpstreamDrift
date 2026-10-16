# NSX-55 Simscape Closed-Loop Export

Exports `M`, `h`, `J_c` and `Jdot_v` from a Simscape Multibody model to the versioned
file `simscape-closed-loop-snapshot/v1`, read by
`src/shared/python/loop_nullspace/simscape_export.py`. Issue
[#12583](https://github.com/D-sorganization/UpstreamDrift/issues/12583), epic #12593.
Required release: **MATLAB R2025b** only; the exporter refuses to run in any other
release, and R2026a results are never acceptance evidence.

**Status.** The method is validated on a planar test rig (scope `verification_rig`, capability NONE). The golf models cannot be
exported yet; see `docs/development/planning/DV-12583.md`. The golf-model
`CLOSED_LOOP_SNAPSHOT` capability stays NONE.

## Files

| File | Purpose |
| --- | --- |
| `nsx_simscape_closed_loop_export.m` | Entry point: R2025b check, rig build, probing, SHA-256, JSON |
| `nsx_simscape_probe.m` | Forward-dynamics probe of an open-tree model at `(q, v)` |
| `nsx_build_planar_rig.m` | Builds `NSX_PlanarRig.slx`, the NSX-00 planar golfer as an open tree |

## Model and Conventions

- Rig: lead arm (shoulder, elbow), club joint, trail arm (shoulder, elbow), revolute
  joints about `+z`, relative angles, `q = [lead_shoulder lead_elbow club_angle
  trail_shoulder trail_elbow]`. Segments hang along `-y` at zero angle, so a segment of
  absolute angle `a` points along `u(a) = (sin a, -cos a)`. Gravity `(0, -g, 0)`.
- Units: m, kg, s, rad, N m. Parameters are the NSX-00 values, converted from exact
  rationals by `simscape_rig_spec.py`; rod inertia about the centre of mass is `m l^2 / 12`.
- Closure: the trail hand is not welded. A Transform Sensor reads
  `Phi(q) = H_R - G` (hand minus grip point, in World), matching the planar model, so
  `J_c = d Phi / dq` has rows `[fx fy]` and the grip frame is World axes at `G`.
  The grip rotation is a declared convention (identity), not a measurement.

## Method

With `tau` the joint input torques, `M a + h = tau` on the open tree. At `t = 0` with
position and velocity set as high-priority targets:

1. `acc0 = a(tau = 0) = -M^-1 h`; `a(tau = e_j) - acc0 = M^-1 e_j`. Stack the columns,
   `M = inv(M^-1)`, symmetrise, `h = -M acc0`.
2. `J_c e_j` is the sensed `Phi` velocity at `q` with velocity target `e_j`.
3. `Jdot_v = phiddot(tau = 0) - J_c acc0`, since `phiddot = J_c a + Jdot_v`.

No finite differences and no model internals are used: `2n + 1` Simscape runs per state.
The state `a` stored in the file is the requested acceleration from the spec (the
closure-consistent one), not a Simscape output. Declared inputs, not measured outputs:
rig parameters, bare-club inertia, topology, `a`.

## Reproduce

```bash
python3 -c "import json; from src.shared.python.loop_nullspace.simscape_rig_spec import planar_rig_spec; json.dump(planar_rig_spec(), open('rig_spec.json','w'))"
matlab -batch "addpath('<this folder>'); nsx_simscape_closed_loop_export('rig_spec.json','out')"
```

Use the R2025b executable explicitly. The `requires_matlab` test
`tests/unit/loop_nullspace/test_simscape_matlab_run.py` does this and loads the output.

## Evidence

`tests/fixtures/loop_nullspace/simscape/r2025b_planar_rig_export_v1.json` is the
checked-in export of the run on 2026-10-14 (MATLAB `25.2.0.3177638 (R2025b) Update 5`):
agreement with the analytic planar model is 4e-15 (`M`), 7e-15 (`h`), 3e-16 (`J_c`) and
5e-14 (`Jdot_v`) relative to each quantity's largest entry. The model file is not checked
in; its SHA-256 is recorded in the export. A rebuilt `.slx` has a different hash.
`synthetic_planar_closed_loop_v1.json` is a separate test-only fixture serialised from
the analytic model (`provenance_kind: synthetic_test_fixture`, capability NONE).

## Model Scope

Every export declares `provenance.model_scope`: `golf_model` or `verification_rig`. Only a
`golf_model` export from R2025b can grant `CLOSED_LOOP_SNAPSHOT` FULL; this exporter always
writes `verification_rig`. The checked-in R2025b rig export was re-labelled by adding
`model_scope: verification_rig` to its provenance. That is a declared re-labelling of our
own output: the recorded SHA-256, MATLAB version string and all data are unchanged. Its
capability is NONE and snapshots need `test_only=True`, like the synthetic fixture.

## Limits

- Open-tree models only; a model with an internal closed loop must be cut first.
- Planar rig only: no spatial (6-row) closure, no ground contact, no root joint.
- Validates exporter, format and loader. It does not qualify the golf model.
