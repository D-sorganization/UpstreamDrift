# Engine and Conda Cache Contract

This contract defines cache identity, sharing scope, and disk-pressure behavior
for engine builds and Conda package downloads on self-hosted runners. It is the
policy reference for Repository_Management #2224 and its consumer
UpstreamDrift #13579 and implements the per-host BuildKit sharing goal from
#13208 (no per-runner `runner.tool_cache` BuildKit exports). It does not assert
that a cache is installed or that a host has passed rollout validation.

## Cache Identity

| Cache               | Logical key                                                                                                                                                        | Reused data                                                  |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------ |
| Engine image layers | `buildkit/v1/<repository>/<target>/<platform>`; BuildKit further identifies reusable layers by the build definition and input content.                             | BuildKit layer records for the selected target and platform. |
| Conda downloads     | `conda-pkgs/v1/<platform>/<channel-set>/<package-name>/<version>/<build>`; Conda's package metadata and archive checksums determine whether a package is reusable. | Downloaded and extracted package artifacts only.             |

The key is a logical namespace, not a request to create one directory or
archive per key. Keep target and platform boundaries explicit for engine
builds. For Conda, resolve environments from the repository's environment or
lock files for each job; do not treat the mutable environment prefix as a
package cache or restore it as a shared environment.

## Scope and Sharing

- On self-hosted runners, cache state is scoped to one physical runner host.
  All runners on that host may reuse the host's cache when they use the same
  service account and cache location. A runner name or runner index must not
  be part of a host-shared cache identity.
- Cache state must not be written to a path shared across different hosts.
  Each host has independent cache contents and lifecycle.
- Engine builds use the host's persistent BuildKit builder and its
  content-addressed store. The UpstreamDrift implementation names that
  builder `ud-buildkit`, serializes bootstrap and repair with a host-local
  `flock`, and falls back to a correct cold build when cache setup fails.
  BuildKit garbage collection owns layer eviction. Do not add per-runner
  `type=local` exports or a second promotion copy.
- Conda package caches may share immutable package artifacts on one host.
  Each job creates its own environment from the declared environment or lock
  file. Do not share a mutable `CONDA_PREFIX`, and do not use a cross-host
  network directory as the package cache.
- GitHub `actions/cache` keys have repository and branch scope, not runner-host
  scope. They do not satisfy this contract's self-hosted per-host sharing
  requirement and must not be presented as the host disk cache. Hosted-runner
  caches remain independent of this host-local contract.

## Disk Pressure and Eviction

The RM-OPS-017 disk-guard schema has no per-cache byte quota. Do not infer one
from its host-wide thresholds. The UpstreamDrift shared BuildKit builder does
have an explicit garbage-collection policy in
[`scripts/ci/buildkitd.ud.toml`](../../scripts/ci/buildkitd.ud.toml):
the transient source, checkout, and cache-mount policy keeps entries for 48
hours and sets a 5 GB `maxUsedSpace` threshold; the general policy sets
`reservedSpace` to 5 GB, `maxUsedSpace` to 30 GB, and `minFreeSpace` to 20%.
These are BuildKit garbage-collection policy values for that UpstreamDrift
builder only, not fleet-wide cache quotas. The overall host admission and
cleanup policy is in
[`config/runner_disk_guard.yaml`](https://github.com/D-sorganization/Repository_Management/blob/main/config/runner_disk_guard.yaml), schema
`runner-disk-guard/v1`:

| Guard                             | Configured value                              | Meaning                                                                                                          |
| --------------------------------- | --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `thresholds.trip_min_free_gb`     | 25.0 GiB                                      | Cordon when free space is at or below this threshold.                                                            |
| `thresholds.trip_used_percent`    | 92.0%                                         | Cordon when filesystem usage is at or above this threshold.                                                      |
| `thresholds.readmit_min_free_gb`  | 40.0 GiB                                      | Require at least this much free space to readmit a host.                                                         |
| `thresholds.readmit_used_percent` | 85.0%                                         | Require usage at or below this value to readmit a host.                                                          |
| `drain.timeout_seconds`           | 7200 seconds                                  | Maximum drain period for running jobs; `finish_running_jobs` is true.                                            |
| `cleanup.commands`                | Two allow-listed `runner-cleanup` invocations | Run after cordon and drain, including the `--disk-guard` cleanup and the work-dir, tool-cache, and Docker prune. |

These are host guard thresholds, not a reserved cache allowance. The
readmission values provide hysteresis; they are not cache eviction targets.
The documented BuildKit garbage-collection values are sourced from the
UpstreamDrift configuration above. Conda package-cache cleanup must use the
host's approved cleanup path and must not remove a running job's active
environment. The disk guard schema does not name a Conda-specific cache quota
or eviction command, and no numeric Conda cache quota is established here.

## Failure and Validation Behavior

Cache availability is an optimization. A failed or missing engine cache must
allow a cold build to continue and report why the cache was unavailable; it
must not change the build's correctness. Do not silently swallow a failed
cache save if doing so hides a persistent cache-corruption or disk-pressure
problem.

Before enabling or changing a cache implementation, validate its keys and
scope against this contract, inspect the actual disk-guard configuration, and
confirm its cleanup behavior preserves active jobs and any explicitly
protected persistent builder state. For UpstreamDrift #13579, acceptance also
requires a second consecutive run to show a cache hit and evidence that the
host remains within the configured disk guard. The document and merged
dependency #2221 alone do not establish that activation validation has
passed.

## Reviewed Issues and Implementations

- [Repository_Management #2221](https://github.com/D-sorganization/Repository_Management/issues/2221)
  supplies the merged `runner-disk-guard/v1` configuration. Its thresholds
  above define host-wide disk admission and readmission boundaries; they do not
  allocate a per-cache budget.
- [UpstreamDrift #13208](https://github.com/D-sorganization/UpstreamDrift/issues/13208)
  documents the observed per-runner BuildKit duplication and the decision to
  use one persistent builder per host. Its implementation and workflow changes
  merged as [#13277](https://github.com/D-sorganization/UpstreamDrift/pull/13277)
  and [#13430](https://github.com/D-sorganization/UpstreamDrift/pull/13430).
  The issue's remaining operational checks are not treated as complete by
  this contract.
- [Repository_Management #2138](https://github.com/D-sorganization/Repository_Management/issues/2138)
  records that `Swatinem/rust-cache` with `cache-bin: true` can delete the
  persistent host's `rustup` binary. Persistent host tools are outside
  disposable job cache ownership; cache actions must not delete or replace
  shared host tool binaries.
- [UpstreamDrift #13579](https://github.com/D-sorganization/UpstreamDrift/issues/13579)
  is the consumer that must demonstrate a cache hit on its second consecutive
  run and respect the host disk guard after implementing this contract.
