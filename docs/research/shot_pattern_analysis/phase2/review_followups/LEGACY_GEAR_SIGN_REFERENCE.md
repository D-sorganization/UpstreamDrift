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

## Rigid-Contact Gear Law for the Shot Card

Issue [#15443](https://github.com/D-sorganization/UpstreamDrift/issues/15443)
adds an opt-in replacement for the heuristic. When `ImpactParameters.rigid_gear_head`
is set, `ImpactSolverAPI` takes the gear increment from
`physics/impact_model/rigid_gear.py` instead:

$$\Delta\boldsymbol\omega = \boldsymbol\omega_b(b\,\mathbf n - h\,\mathbf e_h + v\,\mathbf e_u) - \boldsymbol\omega_b(b\,\mathbf n),$$

where $\boldsymbol\omega_b(\mathbf r)$ is the ball spin from the shared
finite-inertia contact solver (`physics/rigid_contact.py`). The solver uses the
contact lever $\mathbf r$ from the head CG, the head inertia tensor
$\mathbf R\,\mathrm{diag}(I_n, I_h, I_u)\,\mathbf R^\top$ in the face frame
(normal $\mathbf n$, heel $\mathbf e_h = \mathbf u \times \mathbf n$, face-up
$\mathbf e_u$), CG depth $b$, head mass, COR and friction. The face is flat.
Only the spin increment is applied; the offset's change to ball speed and
launch direction is not.

Head values per club are settings (`impact_parameters/club_impact_settings.json`,
`rigid_gear_heads`). They are the phase-two matched-delivery research
assumptions, not measured heads. Only the shot card and export flight path
(`flight_report.flight_for_impact`) opt in. Every other caller keeps the
legacy term and its historical outputs.

Evidence comes from `tests/unit/physics/test_rigid_gear_effect_15443.py`:

- A centred strike gives zero increment.
- Toe and heel strikes are antisymmetric, with the legacy sign.
- A CG in the face plane gives no side spin.
- A higher yaw inertia gives less gear.

For a measured driver heel strike (h = -3.4 mm, v = +1.4 mm, 51.4 m/s), the law
gives -143 rpm of side spin against the heuristic's -291 rpm. On the matched
driver swing, this moves the spin axis from 19.8° to 15.3° (D-plane alone:
10.3°). On the matched 7-iron (CG depth 10 mm), the gear tilt falls from 1.5° to
0.15°.

This is a software regression on a reviewed contact law, not an equipment
calibration: no launch-monitor comparison qualifies these head values. Face
bulge and roll are the next slice.

```bash
python3 -m pytest tests/unit/physics/test_rigid_gear_effect_15443.py tests/unit/impact_parameters -o addopts='' -q
```

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
