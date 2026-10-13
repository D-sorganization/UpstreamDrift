# Legacy Gear-Spin Direction Reference

## Scope and Disposition

Issue [#12006](https://github.com/D-sorganization/UpstreamDrift/issues/12006)
corrects the horizontal spin sign in the public legacy impact helper. The
matched-delivery study uses coupled contact and does not call this helper.
Historical study outputs and their source hashes remain unchanged. This is a
separate research reference; its editable [LaTeX Source](LEGACY_GEAR_SIGN_REFERENCE.tex)
is not part of the canonical engineering manual.

## Frames, Inputs, and Equations

For the documented right-handed convention, X points forward, Z upward, and Y
left. The two contact coordinates are toe-positive horizontal offset h and
high-positive vertical offset v, in metres from face center. V is clubhead speed
in m/s, g is a dimensionless gear factor, and $k_h/k_v$ are heuristic scales in
$\mathrm{rad\,m^{-2}}$. With unit upward vector $\mathbf u$ and horizontal face-normal perpendicular
$\mathbf t = (\mathbf u \times \mathbf n)/|\mathbf u \times \mathbf n|$, the additive angular velocity is

$$\Delta\boldsymbol\omega = g h V k_h\,\mathbf u + g v V k_v\,\mathbf t.$$

For a forward face normal, t is +Y. The native Magnus direction is angular
velocity cross ball velocity. Thus positive h adds +Z spin and produces a
leftward contribution for a +X shot. Negative h reverses it. Positive v adds
topspin about +Y and reduces negative-Y backspin. The vertical term, scale,
contact model and native flight equations are unchanged by this repair.

The old horizontal term had the opposite sign. At h = 0.00635 m, V = 45 m/s,
g = 0.5 and $k_h = 350\,\mathrm{rad\,m^{-2}}$, the corrected contribution is +50.00625 rad/s
about Z; the old code returned $-50.00625\,\mathrm{rad/s}$. This is a frame/sign correction,
not a recalibration or a change in the magnitude model.

## Acceptance Evidence and Reproduction

Nine new tests failed before the production change: toe/heel Magnus direction
at three headings, public solver spin increments, and public-solver-to-native
landing direction. The native pre-fix test placed the heel shot at +11.293335 m
laterally while the centered shot stayed at zero, demonstrating the reversed
curvature. After the sign correction, those tests and the existing impact
regressions pass. An additional boundary regression replaces the legacy helper
with a function that raises and verifies the matched contact pathway still
produces a finite off-center landing.

The old assertions that called negative-Z toe spin a draw were corrected to
agree with the independently tested force direction. Their magnitude intervals
were reflected, not widened. Astra independently reviewed the sign and ran the
original nine new tests. The final relevant suite passes 58 tests, including
both native flight tests. The [Validation Receipt](LEGACY_GEAR_SIGN_VALIDATION.json)
binds source hashes and the [Final Test Log](LEGACY_GEAR_SIGN_TESTS.txt);
the [Pre-Fix Log](LEGACY_GEAR_SIGN_RED.txt) preserves the nine failing regressions.
LaTeX compilation remains unverified because the built-in compiler could not
retrieve its uncached bundle; editable source is preserved.

```bash
OPENBLAS_NUM_THREADS=1 /tmp/shot-pattern-validation-venv-20261009/bin/python -m pytest tests/unit/physics/test_legacy_gear_flight_direction.py tests/unit/physics/test_impact_physics_value_assertions.py tests/unit/shared_python/test_impact_model.py tests/unit/physics/test_impact_friction_axis_and_gear_offset.py -o addopts='' -q
```

Native flight tests skip explicitly if the required kernel cannot be imported;
the qualified local run must execute them. Pure vector and public solver tests
do not depend on a native flight installation.

## Limitations

The legacy model adds spin without solving the coupled club/ball rotational
collision. Its coefficients have no player or equipment calibration in this
study, and its angular momentum and energy consistency are not established by
this sign repair. It assumes an approximately forward face normal and a global
vertical spin axis; it is not an arbitrary-orientation contact law. No accuracy
claim, left-handed-delivery qualification, or universal net-curvature claim
follows from these tests. Bulge, roll and other launch/spin contributions can
change net curvature in a fuller model. Use the independently qualified coupled
contact pathway for the matched-delivery research.
