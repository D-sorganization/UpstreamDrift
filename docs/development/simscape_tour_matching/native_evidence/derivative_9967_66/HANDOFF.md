# Single-Direction Derivative Audit 66

## Result and Scope

All six requested original-state scalar replays completed below the120000 actual-acceleration-call budget. This audits only LSInputX B6 for candidate63 (`e057a7e977cfe2a163fe4c5e7e5767c17d5025b4ef7f33166672afbf658c358d`) against the exact stored64 analytic direction. It does not qualify the other80 Jacobian columns or accept an optimizer candidate.

The stored marker Jacobian is307x25x3x81. Metadata `first_control=4` means coordinate-major ascending B4/B5/B6; LSInputX B6 is column41 (`3*13+2`). Each perturbation uses `increment_native_bernstein`, preserving q0, qd0, marker labels/bodies/offsets exactly. `replay_candidate` uses the original native model with a thin acceleration counter; no numerical provider replacement, feedback, state reset or model change was introduced.

| h (Nm) | Whole Relative L2 Error | Early <=0.6 s | Transition >0.6 s | Maximum Component Error (m/Nm) |
| ------ | ----------------------- | ------------- | ----------------- | ------------------------------ |
| 1e-5   | 9.11254e-5              | 3.70477e-4    | 9.11065e-5        | 4.06399e-5                     |
| 1e-4   | 4.93792e-4              | 3.39036e-4    | 4.93796e-4        | 2.42390e-4                     |
| 1e-3   | 9.53003e-6              | 1.34865e-5    | 9.52990e-6        | 4.56084e-6                     |

Relative errors use norm(FD-analytic)/norm(analytic), across the specified time region and all marker components. All three whole-direction values are below the unchanged1e-3 threshold. Finite differences vary nonmonotonically with h; neighboring h direction differences are5.79788e-4 and5.03515e-4 relative to the preceding FD direction. This pattern does not establish asymptotic finite-difference convergence or validity of the complete Jacobian. Maximum analytic component is0.451053568 m/Nm. Full metrics and raw directional differences remain in the archive.

## Exact Run and Evidence

Replays used rtol1e-12, atol1e-14 and max_step0.000125. Counts range87631–87943 acceleration calls, including sample audits; every run remains below120000. Exact timestamps are in `report.json`, allowing correlation with independently running experiment65; elapsed times must not be interpreted as isolated hardware benchmarks. Original exec handle72187 terminated exit0.

`raw-run.zip` preserves all22 output entries, including exact executed `driver.py`, six perturbed candidates and raw state/marker trajectories, analytic direction, three FD/error arrays, original model/checkpoint,64receipt and immutable runtime sources. ZIP SHA256 is `696654a6e13647142e403d42edcdc3d5392ddf7ff7bb60f7e789bb24a611d4b9`. All22 entry hashes and downloaded archive hash were verified against `hashes.json`. The driver checked every runtime source against exact64's receipt before replay.

`audit_direction.py` is the local reproducer with an explicit per-run factory binding to satisfy Ruff B023. The exact executed source used the same synchronously consumed closure without that binding. It remains unchanged inside the archive. `summary.json` records both distinct hashes; the adjusted local reproducer passed Ruff/check-format but was not rerun. No production module changed.

`summary.json` records exact launch arguments and environment. Runtime remains `/home/dieterolson/native-regularized-fit-9967-61`; remote results remain `/mnt/c/Users/diete/native-derivative-9967-66`. Do not overwrite either or rerun into the same output directory. Preserve `raw-run.zip` with an explicit force-add if repository ignore rules exclude it. No optimizer was launched and no main handoff was edited by this subtask.
