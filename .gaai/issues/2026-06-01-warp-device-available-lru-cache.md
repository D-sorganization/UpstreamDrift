---
title: "simulation_backends: `warp_device_available` lru_cache caches stale results in long-running processes"
labels: [bug, simulation-backends, observability, fleet-followup]
priority: medium
discovered_in: PR claude/test-coverage-improvements (branch)
discovered_at: 2026-06-01
reporter: claude
status: open
related_to: tests/unit/simulation_backends/test_capabilities_contract_extra.py
---

## Summary

`src/shared/python/simulation_backends/capabilities.py::warp_device_available` is
decorated with `@lru_cache(maxsize=1)`. The cached value is the _boolean return_ of
the probe (`True` or `False`). In a long-running process (e.g. the launcher
hosted in `src/launchers/`), this means:

- The first call performs the CUDA device probe via `wp.init()` /
  `wp.get_cuda_device_count()`.
- Every subsequent call returns that cached value, **even if the GPU
  availability has changed** (e.g. the user plugged in an external GPU,
  suspended the process, or moved it to a different machine).

For a CLI tool this is probably fine (one probe per process), but the launcher
process is long-lived and the value is user-visible: the status bar
"GPU available: True/False" indicator will go stale.

## Reproduction sketch

1. Start the launcher on a machine _with_ a CUDA device → `warp_device_available()` returns `True` and the value is cached.
2. `kill -STOP <pid>`; physically unplug the GPU; `kill -CONT <pid>`.
3. The status bar still reads "GPU available: True" because the cache has not been invalidated.

## Recommended fix

Replace the `lru_cache` with a TTL-bounded cache (e.g. 30 seconds) or expose
a `refresh()` helper. A simpler immediate fix is to add a `warp_device_available.cache_clear()`
call whenever the launcher loses / regains GPU focus.

This issue is _not_ a blocker for the test-improvement PR — the new
`@pytest.fixture(autouse=True)` in
`tests/unit/simulation_backends/test_capabilities_contract_extra.py::TestWarpDeviceAvailable::_isolate_warp_modules`
clears the cache between tests. But the long-running-process concern is real.

## Acceptance criteria

- `warp_device_available()` reflects GPU availability with a maximum staleness
  of N seconds (e.g. 30 s) once the launcher has been running for > N s.
- The existing `tests/unit/simulation_backends/test_capabilities_contract_extra.py`
  tests continue to pass without the autouse fixture needing additional changes
  (i.e. the new TTL helper has a public `cache_clear()`).

## Related

- `src/shared/python/simulation_backends/capabilities.py` — production code.
- `tests/unit/simulation_backends/test_capabilities_contract_extra.py` — new tests.
- `docs/adr/0023-mujoco-warp-backend.md` — design rationale for the warp backend.
