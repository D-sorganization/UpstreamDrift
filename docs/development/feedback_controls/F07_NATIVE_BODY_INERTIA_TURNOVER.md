# F07 Native Source-Body Inertia Turnover

Issue #12191 adds a necessary, read-only source-body gate. The production
entry point is `audit_native_body_inertia(source_model_path,
expected_source_sha256)` in `native_body_inertia.py`; consumers call
`receipt.require_admitted()` before claiming physical use. A failed body is
reported by native name and exact source/runtime identity. Diagnostic #12182
replay intentionally does not call this gate as an admission override.

The gate parses only an explicit source `BodySet/objects/Body` inventory,
rejects missing/duplicate/nonfinite properties, and compares every source
mass, body-frame COM and six inertia components to a fresh OpenSim 4.6 model
after `initSystem`. Hashes bind input source, adapter, shared validator source
and three native binaries; source and validator bytes are checked again
before returning. Positive-mass
bodies use the corrected #12183 principal-moment check. Massless routing
carriers need zero inertia and a native topology that OpenSim itself accepts;
the test fixture welds its carrier. No epsilon repair or tensor projection is
performed. Unknown XML body classes and unsupported source body fields reject.

TDD first failed because the module did not exist. The initial zero-mass
fixture then failed for an unwelded non-internal body, matching OpenSim's
native rule; it was replaced by an explicit welded carrier. The actual native
suite covers valid positive mass, 45-degree rotated triangle-invalid inertia,
source hash mutation, zero mass with nonzero inertia, zero-mass welded
readback, controlled native mass mismatch, and the unchanged exact source.
The public receipt records the final test command, source and runtime hashes.

For the exact derived 557-muscle XML SHA
`453e09c4e42dcc3f4b74a3b2efeff063e96c3820b3eee1efe95eef249579010d`,
all 40 Body properties read back exactly. Nine zero-mass/zero-inertia carriers
pass their separate policy. Bilateral clavicle and scapula tensors fail the
necessary physical principal-moment condition. Retained source ancestry is
in the public-source review under `buet557_inertia_xml_v1`; no private
capture identifiers or arrays are copied here. The author intent and an
accepted replacement dataset are unknown. Do not repair the four tensors
by projection, assume whole-system mass-matrix positivity is enough, or use
the diagnostic short replay as physiological acceptance.

The follow-on still needs an authoritative corrected physical source or a
separately justified mass/COM/inertia dataset, then passive-force, anatomy,
native contact/grip, prepared state and full-horizon capture qualification.
All 17 variants and six ecosystems remain in the denominator. Canonical
chapter52 is provisional, calculation approval empty and release blocked.

## Canonical Receipt Preservation

CI run 38093988214 passed 23,970 tests and failed only matched-swing ledger
freshness. A merge-time formatter had rewritten two unrelated canonical
ground-support receipts, changing their SHA-256 values. Restore their exact
`origin/main` bytes, which match the already recorded ledger hashes, and extend
the existing explicit immutable-evidence exclusions in `.prettierignore`.
The driver receipt retains `f2840c2018ee4fd51a24a6dcce3510dc265c5f67009fbec71ed9b2c4d45268fc`;
iron retains `f065ed2699e29aa75ebe274a46d2a9a54cfc897525ef166e06fb92f254e01cf7`.
No experiment, receipt field or scientific acceptance changes. Re-run the
existing ledger freshness and formatter-ignore governance tests; do not
rebaseline archived evidence to formatting changes.

Validation passed: existing `test_ledger_freshness` (one test),
`test_prettierignore_gate_owned_files.py` (two tests), and all five RM-6 central
gates. These checks repair publication provenance and do not rerun or qualify
the physical model experiments.

The remote advanced concurrently to `7877625369` with a generated-ledger hash
refresh. Preserve that commit through a normal merge, then run
`python -m src.shared.python.motion_matching ledger --write` against the restored
canonical receipt bytes. The 149-row generated ledger now indexes those original
bytes; no remote history or scientific result is discarded.

A subsequent integration of `main` at `7da0a29853` retains the new Drake IK
parity evidence. The two MuJoCo receipts were verified semantically identical
to their preserved originals before restoring their exact bytes. Regenerate
the combined ledger after this merge; newly changed Drake artifacts retain
their own current hashes and are not replaced by historical evidence.

The normal pre-push formatter also touched nine byte-bound JSON inputs and
receipts inherited from Drake parity PR #12216. Verify semantic equality,
restore their exact merged bytes and add nine explicit formatter exclusions.
Native revalidation passed 32 fixture cases and the separately supplied exact
557-source case (33 distinct passing cases); the host-only central runner
skipped native cases, so its skips are not used as native evidence. All five
central gates passed with `MYPYPATH` set to the repository root, preserving
canonical `src.*` package identity rather than duplicate root/src imports.
