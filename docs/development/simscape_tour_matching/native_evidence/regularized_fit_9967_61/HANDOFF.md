# Regularized Native Trial 61

Terminal exit 0 after 75.396 s. The bounded trial improved the original run19
candidate but did not satisfy numerical acceptance. Optimizer convergence is
false: the specified maximum of three optimizer function evaluations was
reached. No R2025b or cross-engine acceptance is claimed.

| Metric              | Original Run19 | Returned Run61 |
| ------------------- | -------------: | -------------: |
| Whole Marker RMS    |      30.791 mm |      29.813 mm |
| Terminal Marker RMS |      99.989 mm |      75.846 mm |
| Club Cluster RMS    |      56.557 mm |      23.930 mm |
| Early Marker RMS    |      10.667 mm |      10.707 mm |
| Effort Penalty Cost |     1.55060051 |     1.55026095 |

All three analytic sensitivity/primal consistency gates passed. Their maximum
marker discrepancies were 5.66e-8, 1.29e-8 and 3.86e-8 m. Sensitivity integration
evaluation counts were 42479, 42383 and 42419. Five forward evaluation records
include additional audits; max-nfev is not an RHS evaluation budget. The small
penalty decrease does not establish a substantial reduction in physical effort.

## Exact Runtime and Inputs

Immutable runtime: `/home/dieterolson/native-regularized-fit-9967-61` on
ControlTower WSL `ControlTower-Runner`. Original model, candidate19 and marker
payload were used with original initial state. Source and input hashes plus
the exact argument vector and environment are in `launch.json`.

Flags were `--shaping sixth --max-nfev 3 --analytic-jacobian
--effort-penalty-weight 0.01`; defaults retain force scale 100 N, torque scale
20 Nm and amplitude scale 10. No thresholds, source files or runtime providers
were changed. The original process PID 2779075 terminated; no duplicate or
continuation run was launched.

Returned candidate:
`/mnt/c/Users/diete/native-regularized-fit-9967-61/returned-candidate.json`.
Canonical candidate SHA256:
`1df779e8cb3fd5536655cfda7b6cb2bd816c1a10cb39a8a9a964f957665de18f`.

The recorded near-bound count is zero, but the first/root-X parameter is
1.19570688 against the upper bound 1.2. Other parameters and exact values are
preserved in `summary.json`. A continuation must retain original candidate19
as the baseline and explicitly use this returned candidate as the restart;
do not silently redefine control bounds around the returned candidate.
Root owns the continuation decision. Existing acceptance gates remain in force.

## Archived Evidence

`raw-run.zip` contains every output, launch receipt, stdout/stderr, original
inputs and the exact runtime source archive. `summary.json` contains terminal
status, final metrics/parameters and all output hashes. Local archive and
every input/output hash were verified after download. Archive SHA256:
`b08e7bbab8d9f818dd21819f223272f7746a6ac6a15571a6927770b87bd83cce`.
Force-add the ZIP when committing. The exact replay command is the structured
argument array in `launch.json`; choose a new output directory for another
run and preserve this terminal evidence unchanged. No production edits or
commits were made by this run's execution agent.
