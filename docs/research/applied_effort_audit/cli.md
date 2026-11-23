# Recorded Solution Bundle Effort Audit CLI

Refs #15493. This CLI connects the strict Solution Bundle codec to the existing
`audit_applied_effort` calculation. The calculation, equations, SI units and
sampling limitations remain in [the reference](applied_effort_audit.tex).
It does not introduce a solver or new physics calculation.

Run from the repository root with the exact Tools submodule initialized:

```powershell
python3 -m scripts.audit_optimization_effort path/to/bundle --limits limits.json
```

The limits JSON maps every bundle joint name to exactly these three fields:

```json
{
  "lead_wrist_flex": {
    "lower_nm": -20,
    "upper_nm": 15,
    "slew_nm_s": 100
  }
}
```

These numbers illustrate the format; they are not sourced physiological limits.
The caller must declare justified rotational torque bounds in Nm and secant
slew bounds in Nm/s in the bundle's joint order, signs and frame. JSON ordering
does not matter; the report uses the bundle's order. Missing/extra joints,
unknown fields, booleans and nonfinite values are rejected.

The codec verifies bundle integrity before auditing `tau_applied`, including
feedback and saturation. Neither `tau_ff_ref` nor `tau_cmd` substitutes for this
channel. The report records solution identity, model hash, frame, sample count,
duration, supplied limits, signed per-joint margins and violating joints.
Exit status is 0 for sampled pass, 1 for a limit violation, and 2 for invalid
input; invalid input emits no success report. Output is strict JSON on stdout.

The result is a diagnostic of recorded samples. It does not establish
continuous-time slew, ZOH jump rates, between-sample feasibility, work, ROM,
tracking, contact, ballflight, clinical safety or forward replay. A sampled
pass must not admit an optimized swing by itself. No recorded baseline is
qualified by this change.

## Reproduction and Evidence

```powershell
python3 -m pytest tests/unit/scripts/test_audit_optimization_effort.py tests/unit/optimization/test_applied_effort_audit.py tests/unit/optimization/test_solution_bundle.py -q --no-cov -o addopts=''
python3 -m ruff check scripts/audit_optimization_effort.py tests/unit/scripts/test_audit_optimization_effort.py
python3 -m ruff format --check scripts/audit_optimization_effort.py tests/unit/scripts/test_audit_optimization_effort.py
```

Tests first failed with the CLI module missing. A constant three-joint signed
torque fixture checks exact margins; a changed applied-torque sample checks
secant slew. Zero feedforward and commanded channels cannot hide an applied
torque violation. Corrupt bundle bytes fail through the canonical codec.
These are software contract fixtures, not a new matched-swing experiment.
