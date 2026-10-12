# Necromatcher Overlay Status

## Historical Execution and Download Availability

Issue #11246 separates a historical worker completion receipt from the present
download state. `execution_verified` checks the persisted successful execution
receipt. `download_available` additionally requires an unchanged artifact stat
baseline; `artifact_state` is `verified_stat_baseline` or
`changed_or_unverified`. Scientific acceptance remains rejected.

After an owned worker verifies the complete parent and output hashes, it saves
the exact relevant asset identities and metadata plus bounded file identities:
path, device, inode, size, modification time and change time. Metadata is sampled
before and after authoritative verification. Polling reads public session asset
metadata and at most nine artifact file stats, rejecting missing files,
symlinks, changed relevant asset metadata and undeclared output files. Adding
unrelated fit versions does not invalidate the baseline. Small request and
manifest hashes still protect the execution receipt; polling does not read the
MP4, capture archive or model bytes and does not fingerprint implementation
source files.

Stat identities detect ordinary mutations; they are not a cryptographic content
attestation. The download endpoint remains authoritative: it fully checks
immutable parents, declared output hashes and final ZIP members, rechecks them
after packaging, and compares stat identities across those checks.

## Legacy Receipt Procedure

Older completed exports have no artifact baseline. Their historical execution
can remain verified while download availability is unverified. Explicitly
request `GET /api/necromatcher/video-exports/{run_id}/download` for a known
successful historical run. A successful fully verified download saves the stat
baseline only after final package validation; subsequent polling and reopened
sessions can expose availability. Failed verification never repairs a receipt.
Web and native controls offer `Verify Stored Overlay Package` for a succeeded
historically verified run whose download readiness is unverified. This explicit
action calls the guarded download operation; changed parents or artifacts can
still be rejected. A ready run retains its ordinary download/save action. Failed,
cancelled, pending or historically unverified runs offer no stored verification.

## Producer Provenance

`producer_source_commit` reports the original request's execution stamp.
Historical completion does not establish equality with the current checkout or
current runtime. Current-checkout comparison is deferred; no current-source
hash or freshness claim is returned by polling. An older producer stamp does
not invalidate an otherwise intact historical overlay package.
