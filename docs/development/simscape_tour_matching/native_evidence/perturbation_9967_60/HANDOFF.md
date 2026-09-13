# Native Torque Perturbation Propagation — Run 60

All six scalar replays completed from the original run19 initial position and
velocity. This is a sensitivity diagnostic on a rejected 0.85 s C3D candidate,
not an accepted swing or a representation-parity acceptance result.

## Reproducible Experiment

The existing `increment_native_bernstein` adds only LSInputX final coefficient
B6, degree 6, basis duration 0.85 s: zero, ±1e-6 Nm and ±1e-4 Nm, plus an
unchanged repeated baseline. Geometry, initial state, marker attachments and
all other controls remain identical. Native scalar DOP853 uses rtol 1e-12,
atol 1e-14 and max_step 0.000125 s. The explicit 120000 acceleration-call limit
includes output audits. No state resets, feedback or constraint projection
were introduced. The 307 output samples cover 0–0.85 s.

Run times were 7.35–7.91 s, with 87307–88135 acceleration calls per replay.
Every replay completed below budget. Sampled closure pose errors were below
2.79e-11 and closure rate errors below 7.38e-11. The repeated baseline was
bitwise identical in saved q, v, markers and joint angular velocities.

## Measured Responses

| Positive B6 Increment | Maximum Native Rate Change | Maximum Marker Displacement | Maximum Joint Angular Velocity Change |
| --------------------- | -------------------------: | --------------------------: | ------------------------------------: |
| 1e-6 Nm               |           0.00750466 rad/s |                5.38642e-7 m |                     0.000320987 rad/s |
| 1e-4 Nm               |             0.766049 rad/s |                5.08276e-5 m |                       0.0327832 rad/s |

The native-rate and joint-angular-velocity peaks occur at 0.786111 s. Marker
displacement peaks at 0.85 s. Joint angular velocity comes from the shared
`NativeJointStateAdapter`: relative joint rotation expressed in its native
parent frame, not world-frame segment velocity.

Central responses `(plus-minus)/(2*delta)` give peak native-rate gains of
7499.79 and 7649.34 rad/s/Nm at the two amplitudes. Corresponding physical
joint-angular-velocity gains are 320.759 and 327.585 rad/s/Nm; marker gains
are 0.540622 and 0.508051 m/Nm. Maximum-component even/odd response ratios
for native rates are 0.000652 and 0.001458. Central derivative estimates
between amplitudes differ by 1.96% for native rates, 2.06% for joint angular
velocities and 7.29% for markers.

These repeatable, approximately proportional, nearly antisymmetric responses
measure strong transition sensitivity in this control direction. They do not
establish exact derivatives: amplitude dependence can include nonlinearity
and integration error. Deterministic repetition is not an accuracy proof.
One coefficient direction is not a full parameter Jacobian, global chart
conditioning bound or cross-engine parity test. Preserve existing acceptance
gates. The next fitting work should assess trajectory/control conditioning
and avoid the rejected fixture's native-rate spike; this evidence does not
justify simply relaxing a gate or increasing computational budgets.

## Artifacts and Replay

- `report.json`: unchanged remote receipt and per-time response summaries.
- `summary.json`: explicit time vector, selected-time table and archive hash.
- `raw-run.zip`: original candidates, six complete state/marker/omega NPZs,
  central/even-response NPZ arrays, original report, exact executed diagnostic,
  geometry and runtime18 Python source/provenance.
- `run_perturbation.py`: convenience copy. Its only post-run change binds the
  synchronous factory closure explicitly to satisfy Ruff B023. The exact
  executed source remains `executed_run_perturbation.py` inside the archive.

Archive SHA256:
`50232847ef9d0309d422cf15c658cb58d12775db1c4ff96f729b09a2869efcaa`.
All original report artifact hashes were checked against archived bytes.
Original candidate canonical SHA256:
`b5b1c3823c86a21df323dc4e430366dc093495b069b3d25c361b0d007b8ff24f`.
Geometry raw SHA256:
`b817fea76407f71b29e42aeb188890df53d2d86874c7e5875d14d353f4a1a248`.

On ControlTower WSL (`ControlTower-Runner`), use a **new** output directory:

```bash
PYTHONPATH=/home/dieterolson/native-manifold-10043-18 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/dieterolson/simscape-pinocchio-9967/.venv/bin/python \
  /mnt/c/Users/diete/run_native_perturbation_9967_60.py \
  --model /mnt/c/Users/diete/native_geometry_spec_9967.json \
  --candidate /mnt/c/Users/diete/native-ms-fit-9967-19/returned-candidate.json \
  --output /mnt/c/Users/diete/native-perturbation-9967-60-repeat
```

Original immutable output is `/mnt/c/Users/diete/native-perturbation-9967-60`.
For another machine, extract the archived runtime and inputs and adjust only
paths; retain dependency versions in `runtime-provenance.json`. Do not overwrite
recorded evidence. The script refuses an existing output directory. The raw
ZIP must be explicitly force-added because repository ignore rules hide ZIPs.
No production source changed for this diagnostic; no additional integrations
remain active from this assignment.
