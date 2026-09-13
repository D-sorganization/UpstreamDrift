# Native Multi-Engine Matching Handoff

## Current Status

The full matching goal is OPEN. No accepted full-swing open-loop sixth-order
candidate exists. MATLAB R2025b is the required reference release. Keep the
original capture, physical model, initial state, actuator mapping and acceptance
criteria; quaternion conversion does not change the required native torque family.

Branch: feat/9967-native-simscape-pinocchio. Implementation through ae06e933b
is pushed. Issue #9967 owns native matching; #10043 owns representation work,
under #9921. Check/renew the lease before new issue work. Workspace:
C:/Users/diete/Repositories/Worktrees/UpstreamDrift-pinocchio-native.

## Latest Verified Result

Run58 is terminal (handle2605 exited0). Local-chart DOP853 on native spherical
Pinocchio PASSES all existing parity gates over the run19 0.85-second fixture:
marker maximum2.62982e-8 m; native q maximum1.28599e-7; native rate
maximum1.15849e-5; closure pose3.74372e-11 and rate9.40426e-11.
Elapsed37.6895 s/9972 RHS calls versus7.2147 s for the tighter scalar reference.
This is a same-input representation result, NOT a C3D fit or full R2025b parity.

Run59 is TERMINAL, original handle97788 exited0: same setup with smaller maximum
step. It FAILS native velocity parity (0.000587270); marker maximum1.15187e-7 m,
native q6.06676e-7, closure pose6.60249e-11/rate1.61521e-10. Elapsed61.2725 s.
Therefore58's isolated pass is not robust convergence or representation acceptance.
No root integration remains active. A parallel read-only pointwise acceleration
and same-state/history audit is in progress; collect its receipt before more fits.
Remote runtime /home/dieterolson/native-manifold-10043-18; driver
/mnt/c/Users/diete/compare_native_manifold_replay_9967_59.py; output
/mnt/c/Users/diete/native-manifold-replay-9967-59. Horizon0.85, methoddop853,
rtol1e-12, atol1e-14, max_step1/1440, max_evaluations100000. Scalar reference
rtol1e-12, atol1e-14, max_step0.000125. Poll the existing handle/process before
launching another run. All root runs through59 are terminal; none should restart.

## Why Earlier Attempts Stalled

- Static pose fitting and feedback tracking did not produce an open-loop sextic
  motion. Run45 feedback whole RMS35.2 mm becomes346.5 mm in time-only replay,
  and655.5 mm after global-sixth-order compression. Do not promote the initializer.
- An actual native gimbal singularity stopped run41. An independent inverse-angle
  branch bug caused run49 to apply a different native effort map. Branch metadata
  is now explicit and tested; true singular native actuator inverses remain rejected.
- Native rate errors are amplified near the shoulder chart. Run52 physical
  angular error0.000208 rad/s became native rate error0.00465 rad/s. Tight scalar
  reference54 and manifold tolerance refinement led to passed58. Gates were not
  relaxed. Run54/55 second-level budgets were insufficient even for their selected
  maximum step; those failures are not proof of solver nonconvergence.
- The Python manifold implementation remains slower than scalar Pinocchio. Keep
  scalar native dynamics available as the fitting baseline while qualifying
  alternate representations. Do not block fitting on all-engine alternate builders.

## Ordered Next Work

1. Collect the pointwise acceleration/history audit. Distinguish numerical
   sensitivity from a state/effort/dynamics defect before further tolerance runs.
   Do not accept isolated58 or weaken a gate to hide failed59.
2. Qualify the variant over further representative trajectories and against
   MATLAB R2025b before claiming full equivalence. Qualify tangent derivatives
   before using the manifold variant in an optimizer. Preserve existing scalar API.
3. Resume trajectory/control co-optimization using native constrained forward
   dynamics. Reuse native profile, sensitivity, marker and shooting providers.
   Start from integrated states, permit trajectory changes, and optimize a global
   degree-six native effort basis. Do not repeat tight static-node boxes or simply
   compress an arbitrary feedback torque trace. Existing seven-DOF bioptim tracking
   is a different physical model and cannot silently replace this native model.
4. Require uninterrupted original-state replay through1.813888889 s, all654 capture
   frames and valid observations, closure, effort audit and independent R2025b
   validation before accepting the final swing. Preserve per-marker/time residuals,
   overlay animation, coefficient/time-basis metadata and exact input hashes.
5. Implement/qualify remaining engine variants sequentially using canonical
   pose_interchange providers, without inferring dynamics parity from pose roundtrips.

## Engine and Representation Handoffs

- [Pinocchio Manifold](simscape_tour_matching/PINOCCHIO_MANIFOLD_HANDOFF.md): native
  spherical variant nq30/nv27, exact fixed transforms/inertia/weld reused. DOP853
  integrates local tangent coordinates and carries physical endpoints unchanged;
  no quaternion projection or target-state reset. Runtime18 passes96 tests with
  one optional MuJoCo skip; root independently verified generic and real Pin tests.
- [Canonical Representations](simscape_tour_matching/REPRESENTATION_HANDOFF.md):
  angle/quaternion/rate/convective-acceleration/effort conversions and fixed-frame
  transport. Preserve units, named frames, branch/winding and model identity.
  Moving-frame acceleration transport and #8867 convention consolidation remain open.
- [MuJoCo](mujoco_native_matching/HANDOFF.md) and
  [Drake](drake_native_matching/HANDOFF.md): native adapters and narrower receipts
  exist; alternate quaternion builders/full trajectory equivalence remain open.
  Drake's projected accelerations are not derivative-consistent trajectories.
- OpenSim epic #10003: plan/handoff branch docs/10003-opensim-matching-epic,
  documented plan commit1a68091b6. Implementation/runtime qualification is not
  accepted. Inspect its branch and requested planning check-in before proceeding.

## Evidence and Reproduction

Evidence root: simscape_tour_matching/native_evidence. Preserve raw ZIP archives;
formatted JSON raw hashes can differ without changing parsed physics. Original
native model SHA256 b817fea76407f71b29e42aeb188890df53d2d86874c7e5875d14d353f4a1a248.
Run19 fixture canonical candidate SHA256
b5b1c3823c86a21df323dc4e430366dc093495b069b3d25c361b0d007b8ff24f.
It is a rejected0.85 s C3D fit (whole30.79 mm, terminal99.99 mm), not the best full swing.

ControlTower: ssh alias controltower; WSL ControlTower-Runner; Python
/home/dieterolson/simscape-pinocchio-9967/.venv/bin/python; runtime18 PYTHONPATH
/home/dieterolson/native-manifold-10043-18; use one BLAS/OpenMP thread for comparable
runs. Driver compare_native_manifold_replay.py records independent scalar settings.
Raw run receipts identify exact archived source and inputs. Never overwrite runs.

[Convergence Review](simscape_tour_matching/CONVERGENCE_REVIEW_20260912.md) gives
strategy and delegation gates. [Historical Handoff](HANDOFF_HISTORY_20260912.md)
preserves previous matching and unrelated incoming project context. Its ACTIVE
statements are historical, not current process evidence. Update this concise
handoff and DEVELOPMENT_LOG with each implementation commit; append history only
when needed. Do not reintroduce competing current-status sections.
