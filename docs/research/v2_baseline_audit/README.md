# V2 Baseline Forward-Tracking Diagnostic

This experimental report runs the existing MuJoCo computed-torque tracker on
the controller target embedded in a same-input bundle. It requires a bundle
whose embedded model matches the pinned v2 spec. Bundles without the target
or with a different model produce an unavailable tracking result. Inputs and
private identifiers are not embedded in the output.

With the pinned Tools submodule initialized, run headlessly from the repo root:

```powershell
$env:PYTHONPATH = (Resolve-Path vendor/ud-tools/src).Path
$env:QT_QPA_PLATFORM = 'offscreen'
$env:MPLBACKEND = 'Agg'
$env:MUJOCO_GL = 'disable'
python -m scripts.audit_15493_baseline BUNDLE --out REPORT
```

Keep the bundle and report in an appropriate private location. The CLI writes
the report and exits with status 2 to signal that it is unqualified. The output is
always unqualified: target/ledger provenance, full anatomical ROM, speed,
actual total torque, work, finite-face contact, and flight remain unverified.
There is no ball force. The tracker preloads and projects its own initial state;
this is not an independent replay of the bundle's recorded initial state, an
Option B controller, or evidence of an optimized shot. See the accompanying
editable LaTeX reference for the calculation and limits.

A two-millisecond synthetic closure-consistent v2 input exercises the real
tracker in the test suite; it is not a golfer trajectory. No eligible pinned
v2 original-target bundle was found in the locally inspected private inputs,
so no actual best-match tracking result is claimed by this checkpoint.
