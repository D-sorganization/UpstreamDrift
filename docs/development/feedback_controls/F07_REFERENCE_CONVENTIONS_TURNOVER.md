# F07 Native Reference-Conventions Turnover

Issue #11962 is a focused child of F07 #11791 and epic #11784. Branch
`feat/f07-reference-conventions-11962` is stacked on the unchanged
`feat/feedback-passive-readiness-11939` branch. The child adds a read-only
OpenSim 4.6 observer and explicit selective comparison; F07/F08 and all
six-engine/model qualification rows remain open. Canonical manual chapter 40
and the blocked calculation registry describe the method.

The TDD record began with a missing-module collection failure. Native tests
then found the missing DGF fiber-damping field and orientation-comparison field;
all eight current synthetic tests pass after implementing them. The analytic
slider verifies a 0.2 m body/path movement and a separate 0.3 m named-state
observation. A coupler fixture proves that OpenSim `initSystem()` itself
assembles inconsistent coordinate defaults; `post-init-system` replaces the
ambiguous prior `source-default` API wording. Complete named restoration is
continuous-value-only, at the initialized clock, with no added assembly.
The DGF fixture records passive-disable, rigid tendon, fiber damping and an
actually engaged wrap curve. MovingPathPoint and ConditionalPathPoint fixtures
verify two-state local/ground positions and activity/route membership; a rotated weld exposes an orientation
difference despite equal frame origins.

Native source diagnostics are local under the fleet workspace at
`docs/development/feedback_controls_planning/native_model_evaluation/reference_conventions_11962/`.
They contain all source-default/post-initSystem observations and an explicit
candidate-pair comparison. The unchanged Wilkinson XML SHA is
`a55c64341680551fb5a41be254bdfdb3b2be0ac789336ea44902b22fc9a83913`;
the unchanged Pose2Sim author XML SHA is
`53f84b9552b1c5eceee3768fa820b75bd675daeea473f918cffeb5c6337f0c13`.
The native observer counted respectively 520/318 muscles, 249/111 frames,
80/30 joints, 2/20 coordinate couplers and 76/89 wrap declarations.
The explicitly mapped `IL_L4_r` and `MF_m5_laminar_r` Pose2Sim paths were about
43 mm shorter in their unregistered pelvis frame comparison. Equal source
names do not prove anatomical correspondence. Missing visual assets were
reported by native loading from the owned XML copy; entrypoint XML SHA does
not certify transitive resource closure.

No source model, joint lock, muscle parameter, force law, control input,
assembly target, equilibrium or physiological limit was changed by the
observer. Native `initSystem()` still performs its own initialization assembly.
The full native state, official thoracolumbar donor version and neutral-pose
convention, source asset closure, subject anatomy, muscle policy, marker
registration, contact/grip and independent capture replay remain unresolved.
Earlier pose-scale and alternative-source research reports used the phrase
“source default” for actual post-initSystem/assembled native observations;
interpret their measured q and paths accordingly rather than as raw XML
coordinate defaults.

Reproduce serially in the pinned OpenSim runtime:

```text
python -m pytest -q -o addopts= tests/opensim/test_native_reference_conventions.py tests/opensim/test_native_constraint_state.py tests/opensim/test_native_passive_readiness.py
python -m scripts.check_design_manual_governance
```

The local staging `diagnostic.py` and `compare.py` reproduce the bounded public
source diagnostics with explicit source paths. They are not repository tests
and require the locally retained public source files. Generated manual release
and scientific acceptance remain governed and blocked. Next: obtain the exact
donor source/reference and signed correspondence evidence, audit resource
closure and physical state preparation, then reconsider geometry and muscle
parameter calibration without tuning around a source mismatch.
