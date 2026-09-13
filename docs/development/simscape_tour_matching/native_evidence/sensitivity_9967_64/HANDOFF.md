# Fixed-Candidate Sensitivity Diagnostic 64

**Passed the unchanged numerical agreement gate.** No optimizer ran and no
candidate changed. This rechecks run63's failing candidate
`e057a7e977cfe2a163fe4c5e7e5767c17d5025b4ef7f33166672afbf658c358d`.
The checkpoint hash and capture identity were verified before the call.

Arguments to the existing runtime61 `replay_marker_sensitivities` were
`first_control=4, rtol=1e-12, atol=1e-14, max_step=0.000125`, with the original
model, original initial state and all 307 capture times over 0–0.85 s. No
numerical function, threshold, state or runtime file was replaced.

Measured maximum primal marker component discrepancy is **8.9147861e-9 m**,
below the unchanged 1e-7 m bound. The sensitivity integration required
**1,222,535 evaluations and 409.086 s**. The recorded call/save interval is
419.190 s; total launch-to-terminal time is 466.239 s. The process completed
with exit 0 before the subsequently authorized 600 s wall limit. It was not
interrupted or restarted, and no evaluation cap was installed in runtime61.

The provider's independent primal replay remains at its existing rtol 1e-11
and atol 1e-13. Supplied tighter tolerances govern the augmented sensitivity
solve; the smaller max_step applies to both. Thus two integration controls
changed together. This supports an integration-accuracy contribution to
run63's gate failure but does not isolate which change resolved it, establish
the full Jacobian's accuracy, or qualify this rejected candidate as a swing.
The computational cost prevents assuming these settings are an efficient
optimizer default.

## Evidence and Next Work

`diagnose_sensitivity.py` is the exact executed diagnostic (Ruff check and
format check passed). `receipt.json` preserves input/source hashes, explicit
arguments, clock hash, outcome and timing. `summary.json` preserves the exact
launch vector/environment, terminal status and output hashes. `raw-run.zip`
contains full state/marker sensitivity arrays and primal marker samples,
original checkpoint/model/capture/script, stdout/stderr and runtime source.
All input and output hashes were checked after download. Archive SHA256:
`f26432cd579a5649a5e50550cb1c6ae76a26f956265e43254de1ec7030f4ba1f`.
Force-add the ignored ZIP when committing.

Runtime path is `/home/dieterolson/native-regularized-fit-9967-61`; output is
`/mnt/c/Users/diete/native-sensitivity-9967-64`. Preserve both unchanged. Root's
later optional evaluation-budget implementation is not part of this runtime
or evidence. A next separately qualified experiment can isolate smaller
max_step at the prior augmented tolerances with an explicit evaluation cap,
before deciding on another optimizer run. No further experiments were run
after64; all processes from this assignment are terminal.
