# OpenSim Activation-Property Probe — Prepared, Not Executed

This probe checks whether the offline scalar-activation fallback values match the properties exposed by the installed OpenSim 4.6 runtime for the exact synthetic source used by the 21-cell refinement experiment. It only constructs the model object and reads muscle/component properties. It must not initialize a system, assemble or realize a state, create a Manager or Moco study, integrate, optimize, or alter the model.

The immutable source is `C:/Users/diete/Repositories/Worktrees/feedback-controls-main-integration-12202/docs/development/feedback_controls/experiments/native_moco_refinement_solve_cecfcd98_root_20261011T080500Z/solution_control_replay_inputs_final/source.osim`, SHA-256 `6a098517bdba09b19b3eb99e4da7e795f27d10be8f42c0fca04fee20ce67e143`. Its two `Millard2012EquilibriumMuscle` objects are `flexor` and `extensor`. The XML contains neither activation time constant, deactivation time constant, nor minimum activation for either muscle. The offline analyzer therefore used inferred fallback values `(0.01 s, 0.04 s, 0.01)`; the native result must be reported as an observation and comparison, never as a source declaration.

Generated binding inspection (without importing OpenSim) confirms `Millard2012EquilibriumMuscle.safeDownCast`, muscle methods `getActivationTimeConstant()`, `getDeactivationTimeConstant()`, `getMinimumActivation()`, and `getActivationModel()`. The returned `MuscleFirstOrderActivationDynamicModel` exposes `get_activation_time_constant()`, `get_deactivation_time_constant()`, and `get_minimum_activation()`. The probe reads the `MuscleList` from `Model.getMuscles()`, verifies exactly two names before reading properties, then reads both getter surfaces and requires exact equality, finite values, positive time constants, and `0 <= minimum_activation < 1`. It requires `opensim.__version__ == "4.6"` and records the version, concrete class names, and absolute component paths. XML inspection uses `defusedxml.ElementTree`.

The prepared supervisor interpreter is `C:/Users/diete/AppData/Local/Programs/Python/Python313/python.exe`; its child SDK interpreter is `C:/Users/diete/Repositories/.venv-feedback-opensim/Scripts/python.exe`, with `CASADIPATH=C:/Users/diete/Repositories/.venv-feedback-opensim/Lib/site-packages/opensim`. The recorded fingerprints cover both executables, generated Python wrappers, Python extension modules, and selected OpenSim DLLs; they are a bounded fingerprint, not a closed dependency-runtime identity. The launcher records hashes before and after and fails closed if any captured input changes.

The reviewed execution envelope, if separately authorized, is one process-tree-contained invocation using the existing `tests/opensim/native_moco_ladder_harness.py::_supervise_process_tree`, with a 30-second wall limit, a 1-GiB sampled soft RSS threshold, six thread caps set to one, and owned-tree cleanup verification. The launcher checks source, probe, tests, launcher, interpreter, supervisor, and listed SDK fingerprints before and after. It refuses an existing output directory; the child also refuses an existing result file before importing OpenSim. Its child environment is allowlisted to Windows runtime paths, SDK `PATH`, `CASADIPATH`, `PYTHONHASHSEED`, and the six thread caps. This fingerprint is bounded and explicitly incomplete; it does not prove a closed dependency-runtime identity. This document and its fake-SDK tests are preparation only. No native SDK module has been imported or run for this task, so there are no measured native values yet.

Pure contract tests use fake binding-shaped objects and XML parsing only:

```powershell
python -m unittest C:\Users\diete\Repositories\docs\development\feedback_controls_planning\opensim_activation_property_probe_20261011\test_probe_activation_properties.py
```

After review, launch the supervisor with a new, nonexistent output folder:

```powershell
& 'C:\Users\diete\AppData\Local\Programs\Python\Python313\python.exe' 'C:\Users\diete\Repositories\docs\development\feedback_controls_planning\opensim_activation_property_probe_20261011\run_supervised_probe.py' --output-root 'C:\Users\diete\Repositories\docs\development\feedback_controls_planning\opensim_activation_property_probe_20261011\runs\activation_properties_6a098517_reviewed_run'
```

This command is prepared for review only and has not been run. The outer launcher imports the existing supervisor, verifies hashes, builds the minimal environment, and starts the SDK interpreter only as its supervised child. It writes a supervisor receipt even for worker failure or timeout, with before/after provenance and cleanup status.
