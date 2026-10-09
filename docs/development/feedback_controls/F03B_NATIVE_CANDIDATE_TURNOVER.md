# F03b Native Candidate Benchmark Turnover (#11824; Parent #11787)

This child adds `src/shared/python/motion_matching/native_candidate_benchmark.py`
and its focused tests. It extends the F03a sparse midpoint/shooting synthetic
fixture with an independent exact held-input reference and F06's fresh native
MuJoCo torque replay. The known plant is one hinge, one unit-gain motor, no
gravity or contact, and no capture data. Its purpose is to expose false
positive transcription and to account for native replay in candidate cost.

The implementation predeclares SI tolerances and rejects a candidate if its
actual post-bound torque or slew, transcription defect, native-to-exact gap,
native-to-node gap or synthetic observation RMSE exceeds the corresponding
limit. The applied input is independently checked rather than trusting
solver-reported residuals. Exact matrix-exponential truth is separate from
F03a RK45 and MuJoCo RK4. A coarse stiff case has a small midpoint defect but
fails native agreement; refining the native step reduces integrator error.
The execution history is ZOH at the actual native step, and the complete
initial native integration state and replay policy are bound by F06/T01.

`benchmark_native_candidates` runs predeclared starts serially and retains
failed attempts. Its per-attempt total includes synthetic preparation, solve,
native bundle build/verification and replay, and deterministic JSON receipt
export. It records cold/warm p50/p95, cumulative time to first accepted,
Python-tracked peak allocation, hardware/library versions, and receipt hashes.
It excludes capture decoding, private data preparation, full-body model
loading and native process memory; there is no hidden parallel search.
Shooting uses one constant input while collocation uses one per interval, so
these attempts do not establish a fair production winner.

The manual calculation is in
`manuals/upstreamdrift/chapters/20-native-candidate-benchmark.qmd`; `SPEC.md`
and calculation-registry entry `UP-D1-native-candidate-benchmark-inventory`
mark the result provisional. F03 parent #11787 remains open for the frozen
D02/D03 capture protocol, F09 observation-clock alignment, comparable solver
resources, native full-body/contact evidence, complete accepted-capture cost
and production selection. Native muscle/six-engine qualification is separate.

During development, Tools T01 was consumed from Luna's local Tools worktree by
explicitly preloading `sidekick.lab.mocap` on `PYTHONPATH`. That is a temporary
test seam, not portable evidence. Once T01 merges and F06 pins it, run the
normal repository test path without the override:

```powershell
python -m pytest tests/unit/motion_matching/test_native_candidate_benchmark.py -q
python scripts/ci/check_architecture_budget.py
python -m pytest tests/scripts/test_design_manual_governance_contract.py -q
```

Also run scoped Ruff, mypy and relevant F03a/F06 tests. Produce at least one
reproducible native receipt with the pinned Tools SHA before claiming this
child's portable gate is green. The public synthetic receipt must exclude
private capture identities and must report actual hardware without inferring
a universal speedup.
