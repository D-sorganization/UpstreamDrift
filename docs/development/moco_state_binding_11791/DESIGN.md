# Native Moco State-Binding Design and Evidence

The calculation-level authority is [canonical chapter 35](../../../manuals/upstreamdrift/chapters/35-native-moco-initial-bindings.qmd).
This document is the implementation/evidence handoff for child #11949 under parent #11791, not a
second manual or scientific acceptance report.

## Boundary and Ordering

The existing `build_moco_study` accepts keyword-only `initial_bindings`.
`MocoInitialBindings` copies and freezes state bounds, fixed initial values and
scalar control bounds. Native discovery checks exact path coverage; declared
constraints are applied before solver initialization and guess creation.
Existing controller/PositionMotion semantics and multi-control actuators fail
explicitly. The optional legacy path remains exploratory. Nothing in this API
serializes complete native state or certifies a model's physiology.

Initial guesses may differ from fixed initial values; changing a guess cannot
change the initial constraints. Caller experiment bounds must be finite,
ordered and contain the supplied initial values. Fixed-width bounds are legal.
No suffix-based physiological limits, reserve insertion, tendon rigidification,
equilibration after restoration, or source-model processing is introduced.

## Evidence Inventory

`evidence/MANIFEST.json` hashes 54 copied historical files (21 original and 33 added). Raw receipts, logs
and executed scripts use `.txt` archive suffixes so code formatters cannot
silently change the evidence bytes. The script archive is not a maintained
Python entry point and must not be reformatted or represented as a new run.
The six NPZ files are small synthetic numeric outputs, not private captures.
The evidence-local `.gitattributes` disables Git text normalization for these
historical bytes, including original CRLF receipts/logs. The maintained manifest
and attribute policy remain ordinary normalized text.

The original 21 entries and their artifact bytes are preserved exactly. Their
original-grid optimized/native errors remain valid; their dense-grid diagnostics
are superseded because plain 101-point interpolation could omit control knots.
The original archive includes both missing-fiber-bound and plugin-search failures.

The authoritative corrected evidence is in `evidence/grid-preserving/`:

- [Executed Grid-Preserving Script](evidence/grid-preserving/grid_preserving_experiment.py.txt), SHA256
  `086490671ea3b6b50272f8f53a7f46094ffd936c4c1361b8b00d8016774fe812`.
- [10-Interval Receipt](evidence/grid-preserving/grid-preserving-final-mesh-10/receipt.json.txt).
- [20-Interval Receipt](evidence/grid-preserving/grid-preserving-final-mesh-20/receipt.json.txt).
- [40-Interval Receipt](evidence/grid-preserving/grid-preserving-final-mesh-40/receipt.json.txt).
- [80-Interval Receipt](evidence/grid-preserving/grid-preserving-final-mesh-80/receipt.json.txt).

Each corrected directory preserves its original synthetic model, teacher TRC,
constant-state guess, solution, numeric NPZ and serialized T01
`native_replay_bundle.json.txt`. Matching logs retain IPOPT residuals/iterations.
The exact bundle is serialized/saved and reloaded through the existing Tools
contract before independent F06 replay. Raw receipt/script/bundle text archives
are immutable evidence, not maintained executable sources or newly invented
complete-state contracts. Source-module and runtime hashes remain in receipts.

## Interpretation and Next Experiment

All four public-API solves succeeded and native replay repeated exactly. The
corrected dense grids preserve every original control knot and value, then add
requested output probes. The unions have 101/121/161/241 samples. Scoring uses the
same 101 physical-time probes on every mesh, not an average over each differing
union. Control-function probe error is 5.55112e-17 and retained-knot/probe time
difference is at most 1.38778e-17 s; original control knots are never retimed.

Activation discrepancy falls from .00723379 to .0000929729 across 10/20/40/80
intervals. Position error increases from 40 to 80; marker RMSE is also not
monotone. Tighter native integration alone does not explain the remaining state
disagreement. Teacher-state recovery is separate: the 80-interval final native
activation differs from the teacher by .01802976. No numerical or physiological
acceptance threshold is chosen after observing these results.

Next declare state-specific tolerances and continue bounded trajectory-feasibility
assessment, including native off-mesh force/derivative evidence. Keep Moco's spline
objective separate from rounded-TRC linear and unrounded native-teacher metrics.
Production anatomy/passive readiness, real capture bindings, contact/grip,
complete-state, uncertainty, frozen calibration/holdout and all-model parity
remain open. Old desktop preview files bind the old 20-mesh receipt and are not
corrected-run evidence.

## Reproduction Without Rewriting History

The archived script retains its historical absolute `REPO` path and refuses to
overwrite existing output directories. Reproduce in a **new empty output
directory**, using the existing Python 3.12/OpenSim 4.6 SDK. On the original host,
copy the corrected archive bytes to `grid_preserving_experiment.py` in that new directory; its
hash then equals the historical script hash. On another checkout, changing its
`REPO` assignment creates a new script identity and a new run, which must be
recorded rather than attributed to the old receipt.

From the repository root, this PowerShell example takes the existing provider
and a new output directory as explicit environment inputs:

```powershell
$nativePython = $env:FEEDBACK_NATIVE_PYTHON
$newRun = $env:FEEDBACK_MOCO_OUTPUT
if (-not $nativePython -or -not $newRun) { throw 'Set both explicit paths first' }
if (Test-Path -LiteralPath $newRun) { throw 'Use a new output directory' }
New-Item -ItemType Directory -Path $newRun | Out-Null
Copy-Item -LiteralPath docs/development/moco_state_binding_11791/evidence/grid-preserving/grid_preserving_experiment.py.txt -Destination (Join-Path $newRun 'grid_preserving_experiment.py')
$nativePackage = & $nativePython -c 'import pathlib, opensim; print(pathlib.Path(opensim.__file__).parent)'
$priorPath = $env:PATH
$priorCasadi = $env:CASADIPATH
try {
    $env:CASADIPATH = $nativePackage
    $env:PATH = "$nativePackage;$priorPath"
    $env:OPENBLAS_NUM_THREADS = '1'
    $env:OMP_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    & $nativePython (Join-Path $newRun 'grid_preserving_experiment.py') 10
    & $nativePython (Join-Path $newRun 'grid_preserving_experiment.py') 20
    & $nativePython (Join-Path $newRun 'grid_preserving_experiment.py') 40
    & $nativePython (Join-Path $newRun 'grid_preserving_experiment.py') 80
} finally {
    $env:PATH = $priorPath
    $env:CASADIPATH = $priorCasadi
}
```

Use a task-owned shell process for these settings. No package installation,
global DLL-path modification, or source/runtime substitution is needed. Native
tests use the same explicit SDK and `--noconftest -o addopts=` so repository mock
providers and default worker expansion cannot substitute for native execution.

```powershell
& $nativePython -m pytest --noconftest -o addopts= -q tests/opensim/test_moco_initial_bindings.py
```
