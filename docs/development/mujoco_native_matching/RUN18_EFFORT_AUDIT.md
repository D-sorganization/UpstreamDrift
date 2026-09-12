# Run18 Effort Correction Audit

## Finding

Independently recovered 65 saturated degree-six Bernstein correction controls
across 23 of 27 channels: 33 lower and 32 upper. Coefficient reconstruction
agrees within 3.41e-13 and the active set exactly matches the prior bound audit.
All 65 active controls are at ±2; none of the eight ±10 controls is saturated.
These are increments from the original root-force02 polynomial, not total
physical or biological actuator limits. No optimization or dynamics ran.

The parent coverage metadata was extended from 0.8 to 0.85 seconds without
changing coefficients; its canonical identity matches run18 config.base.
The polynomial basis remains 0.8 seconds. Eleven torque channels with all
controls bounded ±2 have correction magnitudes above 2 after 0.8 seconds:
SpineInputX, SpineInputY, TorsoInput, LSInputX, LSInputY, LWInputY,
RScapInputX, RSInputX, RSInputZ, RWInputX and RWInputY. This is polynomial
extrapolation beyond the Bernstein convex-hull interval, not a bound violation.

## Saturated Entries and Actual Effort Ranges

B0 through B6 are correction controls. A plus or minus indicates upper or lower
bound respectively. Ranges cover 0 to 0.85 seconds and include endpoints and
numerically computed real derivative roots. Forces are world Cartesian commands;
torques are generalized joint efforts, not Cartesian moment components.

| Channel           | Unit | Saturated Controls                | Total Effort Range | Maximum Absolute Correction |
| ----------------- | ---- | --------------------------------- | ------------------ | --------------------------- |
| TranslationInputX | N    | B3+, B4+, B5+, B6+                | -41.773 to -2.141  | 2.000                       |
| TranslationInputY | N    | B1+, B2+, B3+, B4+, B5+, B6+      | -59.248 to 12.762  | 2.000                       |
| TranslationInputZ | N    | B1−, B2−, B3−, B4−, B5−, B6−      | 751.978 to 843.444 | 2.000                       |
| HipInputX         | Nm   | B0+, B1−, B2−, B3−, B4−, B5−, B6− | 6.840 to 64.152    | 2.000                       |
| HipInputY         | Nm   | B3+, B4+, B5+                     | 163.985 to 171.745 | 3.580                       |
| HipInputZ         | Nm   | B3+, B4+                          | -71.113 to -14.845 | 1.731                       |
| SpineInputX       | Nm   | B3−, B4−, B5−, B6+                | -10.024 to 20.454  | 3.755                       |
| SpineInputY       | Nm   | B3+, B4+, B6+                     | -63.189 to -41.648 | 3.347                       |
| TorsoInput        | Nm   | B4+, B6−                          | -75.020 to -14.740 | 3.539                       |
| LEInput           | Nm   | B5−                               | 6.642 to 13.766    | 2.523                       |
| LFInput           | Nm   | B3−, B5−                          | -3.996 to 4.362    | 4.909                       |
| LScapInputX       | Nm   | B1+                               | -6.639 to 6.946    | 1.036                       |
| LScapInputY       | Nm   | B4+                               | -11.810 to 14.240  | 6.477                       |
| LSInputX          | Nm   | B5−                               | -3.779 to 10.091   | 2.218                       |
| LSInputY          | Nm   | B2−, B3−, B5−                     | 11.358 to 19.072   | 2.907                       |
| LSInputZ          | Nm   | None                              | 4.522 to 11.198    | 0.452                       |
| LWInputX          | Nm   | B1+, B3+                          | -1.015 to 6.922    | 9.145                       |
| LWInputY          | Nm   | B4−, B6−                          | -10.526 to -2.826  | 3.204                       |
| REInput           | Nm   | None                              | 2.679 to 17.731    | 4.454                       |
| RFInput           | Nm   | B3−, B4+                          | -11.357 to 7.056   | 1.764                       |
| RScapInputX       | Nm   | B4−, B5−, B6−                     | 1.982 to 28.235    | 2.017                       |
| RScapInputY       | Nm   | B2+, B4−, B6+                     | -53.796 to -3.566  | 3.393                       |
| RSInputX          | Nm   | B2+, B6+                          | -1.315 to 9.039    | 3.599                       |
| RSInputY          | Nm   | None                              | -5.365 to -0.153   | 1.600                       |
| RSInputZ          | Nm   | B4+, B6−                          | -20.792 to 7.030   | 2.730                       |
| RWInputX          | Nm   | B4−, B5+, B6−                     | -3.445 to 9.272    | 4.054                       |
| RWInputY          | Nm   | None                              | 3.104 to 11.990    | 3.227                       |

## Time History and Interpretation

The machine-readable receipt includes each total and parent polynomial range,
correction ranges inside and beyond the basis interval, all seven control values
and bounds, plus values at 0, 0.2, 0.4, 0.6, 0.7, 0.8 and 0.85 seconds.
It also reports root-base force components using the existing NativeEffortProfile
rotation; these must not be confused with the world force commands.

Root world force X varies roughly −2.69 to −40.97 N across those samples;
Y changes from +3.24 to −59.25 N; Z from 765.47 to 843.44 N (minimum751.98 N).
The 16 saturated root-force controls out of21 show a heavily bounded numerical
search. Their ±2N corrections are small relative to the vertical command.
HipInputY remains approximately164–172Nm while its B3–B5 corrections sit at+2.
These observations justify auditing the effect of the imposed correction boxes;
they do not establish physical limits or prove that larger bounds will match.

For a future controlled continuation, first assess the currently active run19
without changing it. If a bound-release experiment is warranted, vary one
identified group (for example root-force corrections), preserve the common
starting point and other settings, and compare independent continuous replay,
closure and full acceptance metrics. Repeated saturation alone is not proof of
a reachable solution. Do not infer biological capacity from these numerical boxes.

## Reproduction

From this worktree with PYTHONPATH set to its root:

```powershell
python3 docs/development/mujoco_native_matching/audit_effort_bounds.py `
  --run <raw-root>/native-ms-fit-9967-18 `
  --parent <raw-root>/native-root-force-9967-02/returned-candidate.json `
  --model <raw-root>/gemini-ms-audit-20260912/native_geometry_spec_9967.json `
  --output <NEW-json-path>
```

Raw root is C:/Users/diete/Repositories/simscape-tour-checkpoints. The report
binds all input bytes and its script hash. The three extrema tests first failed
on the missing audit module and pass after implementation; Ruff passes.
The raw ZIP preserves exact receipts and inputs if repository formatting changes
adjacent JSON. No running optimizer, model, coefficient or physics file changed.
