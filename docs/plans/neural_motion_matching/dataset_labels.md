# NM-02 Native Dataset Labels

Governing issue: [#10617](https://github.com/D-sorganization/UpstreamDrift/issues/10617)
(epic [#10603](https://github.com/D-sorganization/UpstreamDrift/issues/10603)).

Schema: `native-dataset-labels/1.0.0`

## What Landed

`DatasetGenerator` records complete, semantically typed channels instead of
silently writing zeros for unavailable quantities:

- Channel evidence (`available` / `unavailable` / `not_requested`) with reasons
- Native instantaneous accelerations vs interval finite-difference accelerations
- Requested vs applied controls (saturation)
- Model DoF layout (`n_q` / `n_v` / `n_u`) and root-force supervision gate
- Restore failure raises `StateError` (does not swallow)
- Dynamics residual helper `M a + h - u` (− contact when present)

First-wave adapters: `MockPhysicsEngine` (software contract) and analytical
`ODEBackend` driven double pendulum (native residual on the source clock).
Remaining engines stay under NM-09.

## Evidence

[`evidence/nm02_first_wave_label_receipts.json`](evidence/nm02_first_wave_label_receipts.json)

## Limitations

Software contracts and ODE residual checks only. No training, speed claims, or
qualification of MuJoCo / Drake / Pinocchio / OpenSim / Simscape corpora.
