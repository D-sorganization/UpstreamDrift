# F07 Declared Constrained Cold-Start Turnover

Issue #12136 is a focused child of F07 #11791 and epic #11784. Branch
`feat/f07-constrained-cold-start-12136` is based on the reviewed platform
integration commit `45ad4e372c5f32e12bd15a183ef09468b2024bd1`. The
implemented authority is `tour_matching/native_prepared_state.py`; canonical
calculation detail is chapter 41 of the UpstreamDrift manual. The existing
native constraint observer remains the measurement authority. A minimal
pre-initialization hook in `owned_native_source_state` installs the owned
time-only player without rewriting the source file.

TDD first failed on the missing module, then on actual source-controller
replacement. A new unreviewed CoordinateLimitForce test failed while the
mechanical player admitted it; a recursive force-set gate made it pass. The
future-step test first failed on the missing native time-only API; the
zero-order-held input player then passed the shared-prefix/future response.
An admission-identity negative changed only a source chart bound while
preserving the same input and trajectory: the separate policy digest changed.
The adapter source SHA is also recorded; original `PrescribedController` and
`Constant` require exact concrete classes. A mutable list masquerading as the
frozen controller-replacement triple first passed and then failed after the
declaration began requiring an immutable tuple.
The native suite now covers an explicit locked-coordinate target
that cannot be inferred from identical named values; a coupled coordinate
fixture with repeated and changed mechanical commands; mismatched constraint
enforcement, input coverage, an unreviewed non-actuator native force, source
digest and chart exit; and the actual
`physical_humerus_v2/candidate-129.osim` chart-positive prepared state. The
source default is outside the elevation chart. The source contains exactly one
constant-command PrescribedController and one CoordinateActuator. The derived
policy binds the original subtree SHA
`ee9237a7bceacdb1052658699d83dd57e7f245f112d48db1316551903679435a`,
its absolute path, constant value and sole actuator socket before replacing it
on an owned fresh model. Source XML SHA is
`e3ee3f6e031222b2d6919878f3a0a1b9c3bb24d7774903c13dc0d3c368bf7ae6`.
At the declared three-knot diagnostic clock, the original and replacement
mechanical trajectories agree in named q/u, derivatives, QErr/UErr and native
actuation; an owned zero-order-held future command step preserves the common
sampled prefix and changes the subsequent state. This is mechanical assistance, not
muscle excitation or a source-identical native state restart. The original
controller remains admitted for observation only unless this exact policy is
declared.

The exact local public-source fixture files live under the fleet workspace
`docs/development/feedback_controls_planning/native_model_evaluation/physical_humerus_v2`.
Set `UD_HUMERUS_V2_SOURCE` and `UD_HUMERUS_V2_RECEIPT` to their respective
candidate XML and receipt paths for the actual-source test; the fixture
otherwise skips without turning the source into a vendored model. The source
entrypoint hash does not cover external visual assets. Internal integrator
stage chart crossings, numerical/discrete restart history and arbitrary
contact/force/controller policies are outside this child. The unchanged
520-muscle source has six still-open production preparation blockers; no Moco
solve, physiological release or private-capture fit was attempted here.

Reproduce serially in the pinned OpenSim 4.6 runtime:

```text
python -m pytest -q -o addopts= tests/opensim/test_native_prepared_state.py tests/opensim/test_native_constraint_state.py
python -m scripts.check_design_manual_governance
```

The final focused native suite has 11 passes. The broader OpenSim regression
covering this observer, reference conventions, passive readiness, Moco runner
and muscle replay has **183 passes, 2 explicit skips**. Ruff, diff Mypy,
architecture/size/LoD/DRY checks and manual governance pass. The central
five-gate pre-PR runner passes; its generic environment lacks OpenSim and
therefore skips 29 affected native tests. The pinned native run above is the
actual execution evidence. The repository-wide legacy change-fragment
validator separately reports seven inherited fragments missing YAML
frontmatter; this issue's fragment validates in the scoped central gate.

The all-variant/six-engine denominator and original full-swing OpenSim muscle
excitation endpoint remain open. The next scientific step is source-valid
constraint/contact/passive and anatomical calibration of the unchanged
520-muscle model and frozen private marker protocol, without converting this
mechanical diagnostic into an excitation claim.
