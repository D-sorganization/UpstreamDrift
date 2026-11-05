# WSCG Update: ZTCF and ZVCF on the Best-Match MuJoCo Model

**Status: Draft.** Not for publication or submission. Release needs Dieter's
approval and the Assessment Board gate. Every result here is a model-scenario
result, not empirical validation.

## Summary

The WSCG 2024 presentation argued, in a planar two-hand MATLAB model, that
passive effects (momentum, gravity and shaft flex) account for much of the
interaction force at the hands. It also showed a zero-torque counterfactual
(ZTCF) that tracks the full response closely
([legacy audit](WSCG_2024_LEGACY_EVIDENCE_AUDIT.md)). This update repeats the
decomposition on the open MuJoCo model that best reproduces the WSCG model's
structure. It checks that the canonical ZTCF and zero-velocity counterfactual
(ZVCF) add up to the full response at every sample, and it pins every number
to one stored run.

## Methods

### Model

The model is `mujoco/advanced_biomechanical_golf_swing`, revision
`sha256:68979900daad7eda01f6f7b8b4e12086d444a322e99aba4acf57f86ed4fa7535`.
It is a joint-torque-driven body with a two-hand closure on the grip and a
flexible shaft: 42 position and 40 velocity coordinates. The run uses MuJoCo
3.8.0 with a 1 ms timestep and gravity of 9.80665 m/s² along −z. The initial
condition is the compiled reference pose with zero velocity. The run manifest
is the `manifest` block of
[`data/wscg_model_selection.json`](data/wscg_model_selection.json).

The torque program is deterministic. Each actuator gets a 2 Hz sine of
amplitude 0.3 in control units, phase-shifted by its index and clipped to its
control range, over a 0.3 s horizon. It exercises the decomposition; it is not
a golf swing.

### Matching Approach

Models were matched by structure, not fitted to the WSCG force series. Each
MuJoCo golf model scores one point for each of the four documented WSCG
features it reproduces: planar, joint-torque driven, two-hand closure and
flexible shaft. Ties go first to planar models, then to models with an existing
MuJoCo ZTCF/ZVCF test, then to fewer degrees of freedom. The selected model
scores 3 of 4; it is not planar. The runner-up,
`mujoco/golf_double_pendulum_params`, scores 2. It is planar, but with one grip
it cannot produce the separate lead- and trail-hand forces that WSCG reports.
The full ranking is in the selection record.

### Conventions

The counterfactuals follow the canonical definitions in
`src/shared/python/simulation_backends/ztcf_zvcf.py`, which match the original
MATLAB. All of them are evaluated pointwise, at the same configuration `q`:

- **Full**: state `(q, v)` with the applied control `u`.
- **ZTCF**: the drift, `(q, v)` with `u = 0` and the constraint reactions
  recomputed.
- **ZVCF**: the control component, `a_ZVCF = M⁻¹ (B u + Jᵀ (λ − λ₀))`. On this
  constrained model it carries the change in the closure reaction.

The identity tested is `a_full = a_ZTCF + a_ZVCF`, with gravity counted once.
Accelerations are generalized accelerations in MuJoCo joint coordinates
(rad/s² for hinges), positive along each joint axis as declared in the MJCF.
The declared tolerance is `1e-9` absolute plus `1e-10` relative to the largest
full acceleration at each sample.

### Limitations

- The model is 3D and WSCG is planar. Hand forces are not yet projected onto a
  declared swing plane, so no direct comparison with the WSCG force panels is
  made here.
- The initial condition is the compiled reference pose, not a matched address
  pose, and the torque program is not a golf swing. No swing event is claimed.
- The additivity identity holds pointwise only. Integrated counterfactual
  trajectories do not generally sum.
- The ZVCF share is a ratio of per-sample maximum magnitudes. It is not an
  energy or work partition, and it does not separate momentum from gravity or
  shaft flex.
- The force and torque vector overlay (#14439) is not included, because no
  stored MuJoCo run of this scenario exists yet to render it from.

## Results

ZTCF plus ZVCF reproduces the full response at all 30 samples of the pinned
rollout. The largest additivity residual is 2.9e-11 rad/s², inside the declared
tolerance. The velocity identity separating the drift from its zero-velocity
part holds to 1.0e-10 rad/s². At the median sample the ZVCF magnitude is 78%
of the full magnitude: on this torque program the applied torques, together
with the reaction change they cause, dominate the response. The WSCG model's
swing, in contrast, was dominated by its passive part. This is the opposite
regime, and no claim about a golf swing follows from it.

**Table 1.** Additivity on the pinned rollout, from [`data/wscg_ztcf_additivity.json`](data/wscg_ztcf_additivity.json) (run_id `97ecfd13ea5ef4386576e30f4161fe7799b746725494626636b027e8566a63a4`).

| Quantity                                   | Value          |
| ------------------------------------------ | -------------- |
| Samples                                    | 30 samples     |
| Max additivity residual                    | 2.9e-11 rad/s² |
| Max velocity-identity residual             | 1.0e-10 rad/s² |
| Median ZVCF share of the full acceleration | 78%            |

**Figure 1.** Lead-arm ZTCF, ZVCF, their sum and the full response, with the additivity residual against its tolerance band (run_id `97ecfd13ea5ef4386576e30f4161fe7799b746725494626636b027e8566a63a4`).

The figure is written on demand to `figures/fig_wscg_ztcf_additivity.svg` and
`.pdf` and is not checked in. Regenerate it with:

```bash
python3 -m scripts.research.proximal_distal_energy.wscg_ztcf_additivity_charts
```

## Reproduction

The run_id is the SHA-256 of the canonical run manifest, as
`src.sim_run.run_registry.RunRegistry.add` computes it. To regenerate the
records and check that they match:

```bash
python3 -m scripts.research.proximal_distal_energy.wscg_model_selection
python3 -m scripts.research.proximal_distal_energy.wscg_ztcf_additivity
python3 -m pytest tests/unit/research/test_wscg_model_selection.py tests/unit/research/test_wscg_ztcf_additivity.py tests/unit/research/test_wscg_update_draft.py
```

## Next Steps Before Submission

- Project lead- and trail-hand forces onto a declared swing plane and compare
  them with the WSCG force panels.
- Drive the model with a matched swing instead of the test torque program.
- Store the run and render the vector overlay from it.
- Obtain Dieter's approval and pass the Assessment Board gate.
