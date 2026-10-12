# Data Processing Provider Boundary

UpstreamDrift's legacy `sidekick.data_processing.core` and `.exceptions` paths
resolve to the pinned Tools provider through
`src/shared/python/sidekick_data_processing_bridge.py`. The shim preserves the
three legacy import spellings and class identity without copying the provider
implementation or extending either package's search path. Local I/O and format
modules remain separate consumers; `FileIOError` is exported by the public
package and provider core/exceptions, not by the I/O module.

The provider owns calculation and mutation behavior. Curve fitting uses its
sum-of-squares R², including R² of zero for a constant target. Rejected queries
restore the previous data. The provider imports the pinned
`shared.python.safe_pandas_eval` validator. Changes to these behaviors belong
in Tools and require a new pinned provider revision and consumer evidence here.

To reproduce the boundary checks from the repository root, run:

```sh
PYTHONPATH="$PWD" python -m pytest -q \
  tests/unit/repo_hygiene/test_data_processing_provider_bridge.py \
  tests/unit/test_data_processing_core_extended.py
```

The bridge fails with the existing seam resolution hint if the pinned Tools
tree is missing. It does not fall back to an UpstreamDrift duplicate.
