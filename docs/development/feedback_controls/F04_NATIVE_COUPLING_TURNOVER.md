# F04 Native Coupling Fixture Turnover

Child: UpstreamDrift #11952; parent: #11788. This slice was implemented
locally under the repository's fail-open coordination rule during the
2026-10-09 GitHub quota stop. After recovery, #11952 was checked free and
leased to `codex`. The stacked branch is `feat/f04-native-coupling-11788`,
based on the F02 native manifold feature branch. The parent F04 remains open.

## Executable Boundary

`src/engines/physics_engines/mujoco/python/native_control_coupling.py`
adapts the existing F04 bounded block-coordinate/joint tuner to F02's native
9/8/2 free-root/two-hinge controller. It consumes a frozen T01 teacher torque
bundle, native moving reference and time-varying gains; only two bounded
feedback-row scales change. Each exact simulated-state rollout uses the
existing F02 runner, records post-limit torques in the T01 bundle, and checks
fresh native replay of the complete integration state. The train/holdout
initial states are disjoint. The error uses MuJoCo tangent coordinates, so
the free-root quaternion is never subtracted as Euclidean coordinates.
Contact, grip reactions, muscles, external loads, measured state estimation
and human control are outside the fixture.

The separate symmetric motor interventions modify one executed direct-torque
sample and replay each sign from the same initial state and policy. The 2x2
terminal hip/knee response is an in-model physical input response; it is not
computed from residual covariance or feedback-gain loss sensitivity. This
allows a causal statement only about this named synthetic plant and
intervention, not about human neural loops.

## Native Result and Interpretation

Supported task-owned Windows runtime: MuJoCo 3.8.0, NumPy 2.5.3, SciPy
1.18.1. Training objective improved 0.01178597 to 0.00792137; the
post-training held-out baseline/accepted-controller joint-error RMSE was
0.11917156/0.09647448 rad. Both block stages accepted; joint refinement
reported `budget_exhausted`, so no convergence or global optimum is claimed.
The tuner counted 192 training plus four held-out native evaluations, 174
distinct applied-input identities, and maximum complete-state replay error
0.0. Total tuner wall/CPU time was 5.96/5.89 s, including failed solver
evaluations, diagnostics, trial preflight, replay and held-out assessment;
teacher creation and Python process startup are excluded. Timing changes with
host load and does not certify a production deadline.

The local frozen loss-response singular values were approximately 0.001923
and 0.000894, full rank for these two gain scales at this synthetic operating
point. The four paired input histories produced off-diagonal responses
-0.054917 and -0.061098 rad/N m. Gain/feedforward/model confounding,
population identifiability and statistical uncertainty remain unresolved;
the receipt says `unavailable_no_resampling_or_capture_noise_model`.

## Reproduction and Provenance

From this UpstreamDrift worktree, use the supported task-owned Python
interpreter and run:

```powershell
$env:F04_NATIVE_RECEIPT='1'
& 'C:/Users/diete/Repositories/.venv-feedback-opensim/Scripts/python.exe' -m pytest tests/unit/motion_matching/test_native_control_coupling.py -q --junitxml=docs/development/feedback_controls/.f04-native-coupling.xml
& 'C:/Users/diete/Repositories/.venv-feedback-opensim/Scripts/python.exe' -m pytest tests/unit/motion_matching/test_f04_native_coupling_receipt.py -q
& 'C:/Users/diete/Repositories/.venv-feedback-opensim/Scripts/python.exe' -m scripts.f04_native_coupling_receipt --junit docs/development/feedback_controls/.f04-native-coupling.xml --host-alias windows-owned-opensim-env --output docs/development/feedback_controls/F04_NATIVE_COUPLING_RECEIPT_MJ38.json
```

The committed JSON binds passed JUnit evidence, required model/initial
state/policy/time-grid identities, provider versions, the pinned Tools gitlink,
and exact source hashes. The intermediate `.f04-native-coupling.xml` is
untracked and must not be committed. `scripts/f04_native_coupling_receipt.py`
rejects incomplete/skipped tests, inconsistent identities, absent held-out
improvement, invalid local rank/uncertainty declarations, replay mismatch,
or missing cross-motor response. Its negative tests were written first and
failed before the new held-out field admission was implemented.

## Remaining Acceptance Work

Connect the same train-only tuner to a versioned private capture split and
native production contact/grip model after F03/F06/F09 admission. Freeze
measured marker/force/EMG clocks, frames and uncertainty model before fitting;
report cross-validated full-horizon replay, actuator/muscle input semantics,
model-parameter sensitivity, and all required engine rows. Compare total
time-to-accepted including failures and observation scoring. The provisional
chapter 21 and blocked calculation registry explicitly retain these gates.
