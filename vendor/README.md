# vendor/

Third-party and sibling-project code vendored into UpstreamDrift.

## `ud-tools/`

Submodule pointing at `D-sorganization/Tools`. Provides the shared
`upstream_drift_tools` Python package consumed across the GAAI fleet.

### C3D reader (#4484)

The vendored copy at
`vendor/ud-tools/src/shared/python/upstream_drift_tools/lab/bio/c3d_reader.py`
is a byte-identical mirror of the canonical reader at
`src/shared/python/upstream_drift_tools/lab/bio/c3d_reader.py` in this repo.
The duplication exists for one release while the cross-repo dependency
contract migrates so consumers can reach the canonical reader without going
through the vendored tree. It will be removed once that migration lands.

A drift sentinel test
(`tests/integration/test_vendor_c3d_drift.py`) compares the SHA-256 of the
two files and fails if they diverge. The test skips when the submodule is
not materialised (typical in fresh checkouts without
`--recurse-submodules`).

If you need to update the canonical reader, also re-vendor by syncing
`D-sorganization/Tools` and bumping the submodule pointer in the same PR.
Otherwise the sentinel will fail in CI.
