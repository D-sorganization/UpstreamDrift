# F03 Native Marker Fit Turnover (Child #11948; Parent #11787)

This local F03 slice extends the existing F05d BoxFDDP action with a
MuJoCo site-marker objective and tangent derivatives. It reuses F05c
native discrete Jacobians and the F05d nonlinear plan admission and Tools
T01 exact-input replay path. The branch `feat/f03-native-marker-fit-11787`
is stacked on local F02 `e8d8c5465eec4ac0ff83ff9b41bcc7e8969dcb03`,
which itself depends on the F05c feature stack. Parent F03 #11787 remains
open. At the October 9 GitHub rate-limit stop, no new child claim or PR
could be created. Repository_Management/AGENTS.md's fail-open lease-check
rule and the parent task's explicit F03 ownership authorization permitted
local work. After quota recovery, child #11948 was created, checked free and
leased to `codex`; the queued publication can proceed as a stacked, unarmed
PR. This coordination event changes no native physics or receipt data.

## Actual Result and Scope

The synthetic MuJoCo 3.8.0 model has floating-root $n_q=9$, $n_v=8$, two
direct hinge motors and two world-frame sites. A separately replayed
native teacher creates reachable moving marker targets; only the marker
positions, mask and exact native clock reach the optimizer. One site-time
sample is masked. An all-missing time row, wrong clock or missing site
fails closed. Every local site-gradient column and Gauss–Newton curvature
matches an independently finite-differenced native site Jacobian; the
chained action gradient agrees at both floating-root quaternion and hinge
tangent coordinates and at the motor input. Every optimized proposal is
rechecked by a full nonlinear native plan score against the zero-torque
fallback. The exported bundle contains actual post-limit ZOH torque and
complete initial native integration state; a fresh model executes it.

The source-bound `F03_NATIVE_MARKER_RECEIPT_MJ38.json` reports four of four
commands accepted and explicitly marks this particular four-step
trajectory fully accepted. Independent visible-marker RMSE is
`1.7134288601908426e-09 m` versus `0.008878453818013436 m` for a
zero-torque native baseline. The maximum full-integration-state replay
error is `0.0`. The single run's total wall/CPU are approximately
`0.56/0.24 s`: target validation, derivative preflight, all solve calls,
native execution, export, fresh replay, baseline scoring and validation
are included. Teacher generation and process startup are excluded.
The five-second per-call cooperative cap is diagnostic, not a native
step deadline. A later run can have fallback commands; the receipt
records accepted-command count and full-trajectory acceptance separately.

The observed targets are model-generated. No capture protocol, F09 clock
alignment, player geometry, marker placement, external contact/grip,
muscle excitation or full-swing observation accuracy is admitted. The
plant has no gravity/contact. This is an executable F03-to-F05/T01
integration step, not solver selection or production fitting. F03/F05/F09
and the all-model denominator remain open.

## Reproduction and Next Bindings

On the task-owned WSL runtime with MuJoCo 3.8.0, Crocoddyl 3.2.1,
Pinocchio 4.1.0 and the pinned Tools gitlink
`2e7665111b06f92ffbfe178b92d74d6a81c95388`:

```bash
F03_NATIVE_RECEIPT=1 python -m pytest -q tests/unit/motion_matching/test_native_marker_fit.py -o addopts="" -o junit_family=legacy --junitxml=docs/development/feedback_controls/.f03-native-marker.xml
python -m pytest -q tests/unit/motion_matching/test_f03_native_marker_receipt.py -o addopts=""
python -m scripts.f03_native_marker_receipt --junit docs/development/feedback_controls/.f03-native-marker.xml --host-alias wsl-task-overlay --tools-gitlink 2e7665111b06f92ffbfe178b92d74d6a81c95388 --output docs/development/feedback_controls/F03_NATIVE_MARKER_RECEIPT_MJ38.json
```

The `--tools-gitlink` value was obtained with the **Windows host's**
`git ls-tree HEAD vendor/ud-tools`; WSL cannot resolve this Windows-owned
worktree's `.git` pointer. The parser checks matching host Git when it is
available, and rejects missing/failed/skipped tests, malformed hashes,
missing mask, contradictory acceptance, missing improvement and bad replay.
The temporary JUnit file is not committed. The manual calculation and
remaining limitations are in provisional chapter 28. To advance beyond
the synthetic boundary, bind independently measured marker observations
with F09's source/output clocks, a calibrated anatomical marker-site map,
the production model and native contact policy; retain exact initial-state,
input and solver/provider hashes, held-out accuracy, and full-horizon
uninterrupted replay as separate gates.
