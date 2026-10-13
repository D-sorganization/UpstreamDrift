# Audit Source Snapshots Before Architecture Cleanup

The manifest binds fifteen source files at its recorded Git commit, before
function-budget cleanup. Each gzip archive has a zero timestamp and separate
storage and decompressed-payload SHA-256 identities. The corresponding active
source paths remain present; these archives are evidence, not alternate runtime
entry points.

These are checkpoint copies, not a claim that every file matches the version
used in an earlier experiment. Earlier execution-specific source archives,
qualified environments and frozen receipt hashes retain their authority. A
future refactor must record its new source identity and must not renew an old
scientific acceptance receipt. Reconstruct historical execution from the exact
source and environment named by that execution's receipt.

Run `python3 -m pytest tests/research/test_shot_historical_source_snapshots.py`
to verify all storage and payload identities. The test also checks that every
active source path is retained. It intentionally does not require future
implementations to have the archived source hash.
