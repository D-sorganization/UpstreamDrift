# Inactive Sensitivity Padding Diagnostic

Manufactured equation x_dot=200*cos(200*t), x(0)=0 has exact solution sin(200\*t).
Every parameter sensitivity is exactly zero. At identical solver settings,
physical maximum error is2.60e-7 for1 column,8.46e-7 for27,2.23e-6 for81.
Adding mathematically inactive columns changes the augmented adaptive norm and
physical integration path. This does not establish a violation of a global
error guarantee: local solver tolerances are not global error bounds. It is
not evidence about native model mismatch or C3D fit quality.

Reproduce from repository root with:
`python -m docs.development.simscape_tour_matching.native_evidence.sensitivity_padding_9967_67.check_padding`.
The provider baseline is commitc52f2029f; report records NumPy/SciPy versions,
settings, source hash and full sampled states. All sensitivities remain zero.
No integrator behavior was changed for this diagnostic. Use this manufactured
case when qualifying separate physical-state error control; do not replace the
native physical or derivative agreement gates with this example.
