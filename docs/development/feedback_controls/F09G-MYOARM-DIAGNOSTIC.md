# F09g Pinned MyoArm Native Diagnostic

This note records a bounded, unqualified native experiment against the public
MyoSim donor arm model. It is evidence for the direct MuJoCo command replay
kernel only. It does not register a production provider, establish official
MyoSuite execution, or qualify a golfer model.

## Source and Runtime Identity

The unchanged donor is
`UpstreamDrift/shared/models/myosuite/myo_sim/arm/myoarm.xml` at MyoSim git
commit `33f3ded946f55adbdcf963c99999587aadaf975f`. The donor's `VERSION` is
`0.1.0`; its license is Apache-2.0. The donor worktree was clean during the
experiment. The model entrypoint SHA-256 is
`7134c21c82a7d403f7900aa99f5a7d58d67515893ab23b38fcecb9ed59a29062`, and the
license file SHA-256 is
`1eb85fc97224598dad1852b5d6483bbcf0aa8608790dcc657a5a2a761ae9c8c6`.

The direct native runtime was MuJoCo `3.8.0`, not the official MyoSuite 3.0 /
MuJoCo 3.6 SDK environment. The loaded model SHA-256 is
`12c76dde5772be64b7c18aa97f0eeac94c97a214f2bae110ae3d4ad0bc3f6f1f`. The
experiment declares 304 donor resource files (59,534,623 bytes); the resource
set digest is
`0a841f5bca55a9ead70d21afb5fbe12ec657a747f2fdffbc1e23aa32539185b5`. This is
an explicit broad file inventory, not an independently discovered transitive
MJCF include/asset closure. Production binding remains blocked on proving that
closure and inspecting the exact SDK constructor/resource behavior.

## Compiled Plant and Initial Condition

MuJoCo compiled `nq=38`, `nv=38`, `nu=63`, `na=63`, `njnt=38`, `nbody=40`,
`ngeom=161`, `neq=11`, `npair=6`, and `nexclude=5`. All 63 actuators use the
compiled `mjDYN_MUSCLE / mjGAIN_MUSCLE / mjBIAS_MUSCLE / mjTRN_TENDON` law;
their controls are bounded to `[0, 1]`. This is a muscle-only upper-limb
candidate, unlike the golfer variants' 14 upper-limb unit-gear motors. It
cannot satisfy the required driver or iron model rows. The donor README calls
the model 27-DoF with 63 muscles.

At the source default state, all 11 equality constraints had zero residual,
but native forward dynamics reported two contacts: humerus/thorax penetration
of 20.64 mm with contact-frame force magnitude dominated by a 1,030 N normal
component, and thorax/radius penetration of 10.18 mm with zero force. Maximum
absolute `qacc` was 616.98. The run had no warnings; initial activation and
control were zero. These are visible adverse default-state diagnostics, not an
accepted assembly pose, equilibrium, or biomechanical result. No arm-specific
assembly solver or pose adjustment was used.

## Bounded Replay Result

The run used T01 `ACTUATOR_COMMAND`, with 63 ordered `actuator:<native-name>`
channels and native unitless control values. It did not call Gym actions,
environment `step`, observations, resets, or a controller. A `DELT1` command
of `0.05` was applied from sample rows 5 through 28; all other controls were
zero. The model's fixed step was 2 ms over a 60 ms interval, with 31
endpoint-inclusive state samples. Two independent full-state replays matched exactly. The
excited trajectory shared its first six samples with the zero-command replay,
then diverged; maximum activation difference was `0.0499990689`, and maximum
coordinate difference was `0.000636922` in the mixed joint-coordinate vector
(not a distance in metres).

Identity digests: provider
`c6309fff77ebbaab7eb21d2417c1d89490621d311f711cc03ce56bfe2dfed7dc`, initial
integration state
`a380d4609e1b34dd17fd9330f16dd2dcb7106b54dce393a6174e4b85e6cb55b34`, applied
input
`fffdb5dcbf9fc6243ff284abe410aa7aaa478677b63d0627cb5809879bd7be4b`, and
policy
`03d9a3bc37df480375985a6f9933fcd31878227fcca44cf662ab272d89f13891`.

## Exploratory Tangent Finite-Difference Probe

A reproducible one-step central finite-difference audit is preserved as the
runnable, source-hashed experiment file
[`f09g_myoarm_fd_audit.py.txt`](f09g_myoarm_fd_audit.py.txt); it is not a
maintained product module. Run it with `python f09g_myoarm_fd_audit.py.txt`.
Its MuJoCo 3.8 result is
[`F09G-MYOARM-FD-AUDIT-MJ38.json`](F09G-MYOARM-FD-AUDIT-MJ38.json). It retained
the stock configuration and velocity and set all 63 native activation states
and controls to `0.5`, so positive and negative perturbations stayed inside
their bounds. Before every rollout it restored a fresh `MjData` from the same
complete `mjSTATE_INTEGRATION` array and checked exact state equality. Global
callback slots were checked before and after forward/step calls. The same
command baseline was used for every plus/minus pair. Configuration input and
output differences used `mj_integratePos` and `mj_differentiatePos`; all output
positions use one shared nominal next-position reference. The native 2 ms
clock and MuJoCo warning counter were checked after each forward/step. The
baseline retained two contacts.

The finite matrices were highly step-size-sensitive in the stock state. `A`
and `B` stayed finite at each scale, but max-absolute `A` grew by about
tenfold for each tenfold reduction in perturbation size, while max-absolute
`B` remained near `0.11125`:

| State and control perturbation | max-absolute `A` | max-absolute `B` |
| ------------------------------ | ---------------: | ---------------: |
| `1e-5`                         |       `1.5313e5` |        `0.11125` |
| `1e-6`                         |       `1.5312e6` |        `0.11125` |
| `1e-7`                         |       `1.5312e7` |        `0.11125` |

For a deterministic L2-normalized state direction and input direction, the
relative error between the Jacobian directional prediction and an independent
centered rollout slope was large at every rollout half-step. At half-step
`1e-5`, the relative L2 errors for Jacobian perturbations `1e-5`, `1e-6`, and
`1e-7` were respectively `0.983`, `0.846`, and `1.866`. Across rollout
half-steps `1e-4`, `1e-5`, and `1e-6`, the corresponding triples were
`(0.846, 1.866, 23.64)`, `(0.983, 0.846, 1.866)`, and
`(0.998, 0.983, 0.846)`. The rollout slope magnitude itself scaled roughly
inversely with the half-step.

The most sensitive pair is tangent input column 29 and output velocity 29,
both `md3_flexion`. This joint starts exactly at its lower limit: compiled
range `[0, 1.5708] rad`, initial coordinate `0 rad`, zero limit margin,
`limited=true`, with friction loss `0`, armature `0.0001`, damping `0.05`, and
stiffness `0`. At epsilon `1e-5`, the plus trial is `1e-5 rad` inside the
range and the minus trial is `1e-5 rad` outside. Position reconstruction
errors are zero, and both trials restore the same complete state and control
baseline. The minus trial alone adds one `mjCNSTR_LIMIT_JOINT` row (20 versus
19 total constraint rows); both still have 11 equality and 8 pyramidal-contact
rows. The plus/minus next velocity is `-2.99274` versus `0.0697894`.

The force decomposition identifies the discontinuity: at dof 29, actuator
force is approximately `0.165898` versus `0.165894`, passive/applied/bias
forces are zero to reported precision, while constraint force is `0` versus
`0.412057`; pre-step acceleration is `-2807.79` versus `-280.734 rad/s^2`.
Equality-force maxima are `152.4215` versus `152.4444`. Contact geometry is
unchanged and maximum contact-normal force changes by about `0.291 N`. Muscle
and tendon quantities are smooth: initial actuator-length difference is
`6.13e-8 m`, largest estimated moment-arm difference is `5.00e-8 m/rad`
(FDP3), and sampled FDP3/EDC3 forces change only by approximately
`1.6e-7 N`/`3.8e-6 N`.

An in-memory diagnostic copy with only MuJoCo's joint-limit constraint
disabled removes the singular scaling: the central `md3_flexion` velocity
derivative is `1.808980` at all three epsilons (`1e-5`, `1e-6`, `1e-7`).
Disabling only equalities or only contacts does not remove the divergence.
These copies are causal diagnostics only; they do not modify the pinned XML,
represent an admissible plant, or qualify a controller. The evidence isolates
the observed finite-difference blow-up to a central perturbation straddling
the lower joint-limit boundary. It does not establish a stable derivative at
that boundary, and the original stock model state remains unsuitable for
controller qualification. No pose adjustment, assembly, contact
regularization, derivative acceptance threshold, feedback design, or
physiological interpretation was introduced.

Reproduce in the supported local MuJoCo 3.8 environment after setting
`UD_MYO_ARM_RESOURCE_ROOT` to the pinned `myo_sim` directory:

```powershell
python -m pytest tests/unit/engines/myosuite/test_native_direct_model_replay.py::test_pinned_myoarm_native_command_replay_is_exactly_repeatable -q -s
```

The test is opt-in and skips when the donor path is not supplied. This receipt
does not claim physiology, contact readiness, anatomical-marker accuracy,
capture performance, or production MyoSuite execution. It uses public model
assets and contains no private or raw mocap data.

## Implication for Muscle-Feedback Tangent Work

The current F05 native tangent derivative is explicitly a contact-free,
gravity-free, two-hinge plant with two bounded direct motors. Its Jacobians
have shape `[2*nv, 2*nv]` for tangent configuration error and velocity, and
`[2*nv, nu]` for direct motor torque inputs. It rejects muscle activation,
contact, equality constraints, plugins, and assistance. That result cannot be
reused as a muscle-feedback derivative.

A muscle tangent model must retain the configuration tangent (dimension `nv`,
not assume raw `nq` for manifold models), velocity (`nv`), and every dynamic
actuator state. This arm model has `na=63` activation states. The tangent state
therefore needs at least `2*nv + na` components before any additional tendon,
fiber, plugin, or wrapper state supported by a provider is considered. Its
input coordinates must also remain native muscle control/excitation channels,
with the compiled activation and tendon dynamics bound; they are not torque
columns. Such a derivative and feedback controller require a separate TDD
slice and admissible initial-state/contact policy. This diagnostic establishes
none of those qualification conditions.
