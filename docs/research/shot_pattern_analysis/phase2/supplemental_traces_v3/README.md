# Supplemental Flight Traces

These 1,500 paths replay 100 common supported shot IDs across five cells and
three patterns. They do not replace the original study data or statistics.
`receipt.json` records strict saved-path, row, source and runtime parity.
`archive.json` binds the deterministic gzip archive and its restored bytes.

Restore `trajectories.json` from `trajectories.json.gz` before invoking the
explicit supplemental rendering mode documented in `../presentation_v2/README.md`.
Verify its SHA256 against both archive and replay receipts before using it.
Never overwrite the frozen cells' `flight_samples.json` or manifests.
