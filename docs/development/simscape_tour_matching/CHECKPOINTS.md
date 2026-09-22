# Incremental Checkpoints and Replay (Current)

Canonical handoff: [AGENT_HANDOFF.md](../../../AGENT_HANDOFF.md).
Historical checkpoint narrative (prefix-100ms through early 0.75 s runs) lives in
[CHECKPOINTS_HISTORY.md](CHECKPOINTS_HISTORY.md) and must not be treated as
current resume instructions.

## Current Resume Entry (MS-61 / Run-103 Topology Qualification)

Best committed Simscape matched-swing evidence for the 0–0.85 s horizon remains
`native_evidence/two_window_fit_9967_102/` (MS-60). MS-61 qualifies topology and
full-marker terminal disclosure under
`native_evidence/two_window_fit_9967_103/`:

| Artifact                          | Path                                                                   |
| --------------------------------- | ---------------------------------------------------------------------- |
| Topology report (27-DOF, no neck) | `native_evidence/two_window_fit_9967_103/topology_report.json`         |
| Dual terminal breakdown           | `native_evidence/two_window_fit_9967_103/terminal_breakdown.json`      |
| Native gate (blocked)             | `native_evidence/two_window_fit_9967_103/native_gate.json`             |
| R2025b runtime/license receipt    | `native_evidence/two_window_fit_9967_103/runtime_license_receipt.json` |
| Pinocchio parity receipt          | `native_evidence/two_window_fit_9967_103/parity_receipt.json`          |
| Parent run-102 package            | `native_evidence/two_window_fit_9967_102/candidate.npz`                |

Headline dual terminals (from `terminal_breakdown.json`, derived from run-102):
full-marker **40.303 mm**, head-cluster **72.258 mm**, body-excluding-head
**33.673 mm** (diagnostic only). Full-body G1 terminal (≤ 35 mm) remains
**blocked**; do not accept the body-only figure as full-body success. Repair
task: MS-104 (#10378).

## Reproducible Commands

```powershell
PYTHONPATH=. python scripts/matlab/materialize_ms61_topology_receipts.py
powershell scripts/matlab/run_simscape_candidate.ps1 -Run two_window_fit_9967_103
# Licensed Fit (DeskComputer R2025b only; currently fail-closed until wired):
powershell scripts/matlab/run_simscape_candidate.ps1 -Run two_window_fit_9967_103 -Fit
# Parent replay:
powershell scripts/matlab/run_simscape_candidate.ps1 -Run two_window_fit_9967_102 -Replay
```

## Contract Reminders

- MATLAB release: **R2025b only** (no R2026a substitution).
- No inferred 1000-block license cap; runtime identity comes from run-102.
- Dual terminal disclosure is mandatory; head markers must never be hidden.
- Body-excluding-head diagnostics cannot satisfy full-body G1/G3.
