# Regularized Native Fitter Runtime 61

Prepared and qualified on ControlTower WSL `ControlTower-Runner` at
`/home/dieterolson/native-regularized-fit-9967-61`. No fitting was launched.
Root owns the planned bounded analytic B6 trial and its acceptance decisions.

The runtime clones immutable runtime18 and overlays 87 exact current source
and test files, including `prefix_fit.py`, `multi_shooting_fit.py`,
`residual_regularization.py`, `native_effort_penalty.py`, `native_sensitivity.py`,
`forward_sensitivity.py` and the refinement runner. All overlay hashes match
the final remote runtime. All archived Python files match receipt hashes.

## Qualification

- 87 focused fitter, regularization, effort penalty, refinement-control and
  sensitivity tests passed in 1.22 s.
- Nine existing real Pinocchio manifold tests passed in 0.43 s.
- Refinement runner `--help` passed and exposes the analytic Jacobian and
  effort/force/torque penalty arguments.
- Pinocchio 4.1.0, NumPy 2.5.3, SciPy 1.18.1.

Unknown `unit` pytest-mark warnings arise because this isolated runtime omits
the repository pytest configuration. Runtime18's documented outer namespace
scaffolding remains: existing math_utils and pose_interchange initializers
are retained; outer application/package initializers are absent. No numerical
function or physics provider was replaced. This is isolated provider/runtime
qualification, not full application bootstrap or swing-fit acceptance.

## Files and Reuse

`receipt.json` preserves exact commands, output, versions and all runtime
Python source hashes. `source-hashes.json` binds the current local overlay.
`source-overlay.zip` contains that overlay. `runtime-source.zip` contains all
final runtime Python sources and tests, including inherited dependencies.
Force-add both ZIP archives when committing. Final runtime archive SHA256:
`c65ce72406ab2e1b71a21d0ae82147423574fed226c000157e6da02807c4aad6`.

Run commands in the runtime directory using:

```bash
PYTHONPATH=/home/dieterolson/native-regularized-fit-9967-61 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/dieterolson/simscape-pinocchio-9967/.venv/bin/python \
  /home/dieterolson/native-regularized-fit-9967-61/docs/development/simscape_tour_matching/native_evidence/reproduction/refine_native_candidate.py \
  --help
```

Supply explicit original model/candidate/target inputs and a new output
directory for any authorized fitting trial. Preserve runtime61 unchanged;
new source changes require a new runtime identifier and qualification receipt.
No production edits or commits were performed for this runtime preparation.
