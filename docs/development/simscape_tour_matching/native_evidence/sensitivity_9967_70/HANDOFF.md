# Grouped Sensitivity Diagnostic 70 — Terminal Agreement Failure

The exact failed run68 candidate was checked once with separate physical-state
and sensitivity error control. It still failed the unchanged marker agreement
gate: maximum absolute difference **1.02308673e-7 m**, at time **0.85 s**,
marker **WaistRBack**, axis **1** (Y), against limit **1e-7 m**. This is a
measured numerical agreement failure, not a budget exception. The call took
35.969 s, launch-to-terminal 40.103 s, exit 1. No threshold was weakened and
no retry or optimization followed.

Candidate canonical SHA256:
`c8885ace0b1021dea6e2fbd824286fd37eeff8cb91498147d2d585cb64bed039`.
Its hash and capture were verified before execution. Original model, initial
state and 307 capture times over 0–0.85 s were retained. Arguments were
first_control 4, rtol 1e-10, atol 1e-12, max_step 0.000125,
max_sensitivity_evaluations 100000 and separate_error_control true. Independent
primal replay remains at provider rtol 1e-11 / atol 1e-13. Agreement and closure
thresholds are unchanged. No successful result object or Jacobian arrays were
returned; the failure receipt must not be presented as validated derivatives.

## Frozen Runtime and Evidence

`/home/dieterolson/native-sensitivity-9967-70` clones runtime68 with exact
continuous_forward, forward_sensitivity, grouped_dop853 and native_sensitivity
source plus focused tests. The current optimization runner was excluded from
the overlay and was not executed. Qualification passed 49 grouped, budget,
sensitivity and real Pinocchio tests in 4.43 s. Six unknown unit-mark warnings
come from omitted repository pytest configuration. Inherited namespace
scaffolding remains; numerical providers were not shimmed. Root's subsequent
exception-local rename to marker_index is not in frozen70; exact source hashes
bind this receipt to the executed bytes, not later local files.

`runtime-receipt.json`, `overlay-hashes.json`, `source-overlay.zip` and
`runtime-source.zip` preserve qualification. Runtime archive SHA256:
`eeffabc9b3cfe7856ceda30dc09dccd1a5d01acf63dd028cf5e6ff36ef36c632`.
`diagnose_sensitivity.py`, `receipt.json`, `summary.json` and `raw-run.zip`
preserve exact call, source/input/clock hashes, launch environment, terminal
traceback and original checkpoint/model/capture. Raw archive SHA256:
`e4ac06f15c4137b06d95a73a2928826d3a3f85f7b5ae130a31d61c9ef8cdcfe7`.
All overlay/input/output hashes and qualification-versus-execution source
identity were verified after download. Force-add ignored ZIPs when committing.

Remote output `/mnt/c/Users/diete/native-sensitivity-9967-70` is terminal and
must remain unchanged. This narrowly missed gate does not justify accepting
the candidate or relaxing the threshold. Root owns any further bounded
diagnostic decision. No production edits, commits or additional experiments
were made after70 by this execution agent.
