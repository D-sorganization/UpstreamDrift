# F05e Native Activation Manifold Turnover (#11958)

This child adds an activation-aware native action boundary for a smooth,
self-contained MuJoCo 3.8.0 floating-root muscle fixture. It uses Crocoddyl
3.2.1 but does not run BoxFDDP or select an optimizer. Parent F05 #11789
remains open. The canonical calculation is provisional manual chapter 36.

The physical optimizer state is $(q,v,a)$ with $n_x=16$; its manifold tangent
has $n_{dx}=15$. The native full `mjSTATE_INTEGRATION` restart payload has
50 values and is carried separately. MuJoCo configuration difference and
retraction handle the root quaternion; activation is a physical state with
an admitted interior, never silently clipped in the state operation.
Model admission requires built-in joint muscles, disabled warmstart and
autoreset, no contact/constraints/tendons/plugins/callbacks/assets/external
loads, and a time-invariant Euler step. The source XML hashes only the
self-contained entrypoint; the loaded-model and compiled-law hashes bind
compiled semantics. Models with file-bearing references are rejected, so
no unverified transitive asset closure is claimed.

The supported WSL MuJoCo 3.8/Crocoddyl 3.2.1 fixture has $n_q=8$, $n_v=7$,
$n_a=n_u=1$, $h=0.001$ s. Native `mjd_transitionFD` produces $15\times15$
and $15\times1$ matrices. The source-controlled receipt records nonzero
activation-to-joint-velocity and command-to-activation sensitivity, independent
centered native-direction errors at two interior perturbations, and zero
observed next-physical-state difference when only warmstart history/clock
changes under the admitted policy. Seeded state identity, quaternion sign,
retraction and Jacobian checks also pass. This local warmstart result is not
a general proof for contacting or delayed models.

Actual commands have unit `1` and `ACTUATOR_COMMAND` semantics; applied
muscle force is state dependent. The pinned Tools T01 bundle records ordered
channels, compiled-law/provider/source/model identity, full native initial
state, exact 1 ms input grid and held commands. A fresh native model reloads
and independently reproduces every integration-state row in the three-step
receipt (recorded maximum difference zero). The broader Tools compiled
actuator profile and F01/F09 comparison admission are separate, pending
authority; this slice does not claim them. No observation is fed to replay.

To reproduce from this checkout with MuJoCo 3.8.0, Crocoddyl 3.2.1 and the
pinned Tools T01 dependency:

```bash
python -m pytest --noconftest -o addopts= -q tests/unit/motion_matching/test_native_activation_manifold.py
python -m scripts.f05e_native_activation_receipt \
  --fixture tests/fixtures/feedback_controls/native_floating_muscle_11958.xml \
  --output docs/development/feedback_controls/F05E_NATIVE_ACTIVATION_RECEIPT_MJ38.json
python -m pytest -q tests/unit/motion_matching/test_f05e_native_receipt.py
python -m scripts.check_design_manual_governance
```

`F05E_NATIVE_ACTIVATION_RECEIPT_MJ38.json` binds source hashes and actual
native numerical results. Its kernel diagnostic wall time excludes process
startup, collection, and any solver; it is not time-to-accepted-control.
Unsupported active constraints and boundary states reject before derivative
work. There is no fallback controller in this child because there is no
candidate optimization to accept; future F05 solver integration must provide
an independently validated fallback and account for all preparation, failures,
export, replay and validation costs. This is synthetic software evidence,
not golfer anatomy, muscle physiological validation, contact/grip, private
capture, hard deadline, full swing, or six-engine qualification.
