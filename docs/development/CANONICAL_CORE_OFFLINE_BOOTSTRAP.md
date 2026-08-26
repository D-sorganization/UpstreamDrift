# Canonical Core Offline Dependency Bootstrap

Issue [#9122](https://github.com/D-sorganization/UpstreamDrift/issues/9122)
defines a narrow dependency-availability boundary for the Canonical Core
conformance and leaderboard jobs. It does not replace `requirements.lock`,
`requirements-dev.lock`, `environment.yml`, or the canonical dependency review
introduced by #9120.

## Failure being contained

The protected Canonical Core gate could reinstall NumPy 2.2.6 and SciPy 1.14.1
from bytes already present on a runner, then fail before collection when
Pydantic 2.12.5 was absent while the job was operating without an available
package index. Repeating ad hoc `pip install` commands in four job sites made
artifact availability an undocumented property of whichever runner accepted
the job.

The bootstrap therefore separates two authorities:

1. canonical dependency resolution approves package versions; and
2. the bootstrap manifest approves exact wheel bytes for one declared runtime.

Seeding a cache satisfies only the second authority. It cannot approve a new
version, dependency, platform, or interpreter.

## Source-controlled artifact contract

`config/ci/conformance-wheelhouse-v1.json` declares exactly seven wheel files:
Pydantic, pydantic-core, NumPy, SciPy, annotated-types, typing-extensions, and
typing-inspection. Every record contains the distribution and version, exact
filename, byte count, SHA-256 digest, and the official version-specific PyPI
JSON endpoint from which that metadata was checked.

The manifest is valid only for this runtime tuple:

```text
(python=3.11, implementation=cpython, system=Linux, machine=x86_64)
```

`config/ci/conformance-bootstrap-py311.lock` repeats each approved requirement
and digest in pip's `--require-hashes` format. Keeping the manifest and pip lock
independent makes accidental drift detectable before installation.

## Verification algorithm

`scripts/ci/conformance_dependency_bootstrap.py` applies the following order:

1. Parse the versioned manifest and require a non-empty artifact list.
2. Compare the actual runtime tuple with the declared tuple before accessing
   the cache directory.
3. Reject unsafe filenames, duplicate filenames, non-wheel records, malformed
   SHA-256 values, and non-positive byte counts.
4. Require a real, non-symlink wheelhouse directory.
5. Compare directory membership with the manifest as exact sets. Missing,
   extra, symlinked, or non-file entries fail the request.
6. Read each approved wheel, calculate
   \(h_i = \operatorname{SHA256}(b_i)\), and require
   \(h_i = h_{i,\mathrm{manifest}}\). Then require the observed byte count to
   equal the manifest count.
7. Parse the hash lock and require its normalized distribution/version/digest
   mapping to equal the manifest mapping.
8. Only after all proofs pass, invoke pip with `--no-index`, the verified local
   `--find-links` directory, `--require-hashes`, `--only-binary=:all:`, and
   `--force-reinstall`.

This order is fail closed. An unsupported runtime fails before artifact access;
an absent cache fails before pip; and no fallback index or alternate wheel is
permitted.

## Cache lifecycle

The composite action restores a cache key containing operating system,
architecture, the fixed CPython 3.11 tag, and the source manifest hash. A miss
is a job error with instructions to use the reviewed manual seed workflow. The
production action never downloads or seeds dependencies.

`.github/workflows/seed-conformance-wheelhouse.yml` is `workflow_dispatch`
only. Its first step restricts execution to OGLaptop runners 1 through 4. On a
cache miss it downloads only the hash-locked wheels, with an explicit PyPI
index, `--require-hashes`, `--only-binary=:all:`, and `--no-deps`; the verifier
must pass before the cache action may save anything. A cache hit is also
reverified. The workflow records that cache seeding is not package approval.

## Change procedure

Any change to a package, version, filename, digest, byte count, runtime tag, or
provenance endpoint requires all of the following in one reviewed change:

- update the canonical dependency authority first;
- obtain filename, size, and SHA-256 evidence from the official PyPI JSON API;
- update both the manifest and hash lock;
- update the focused fail-closed contract tests;
- re-run the bounded serial contract with explicit `-n 0` on an approved host;
- review and manually dispatch the seed workflow only after merge authority is
  established.

Never prewarm the whole pip cache, reuse a wheelhouse across runtime tuples,
add an index fallback to the production action, or treat a successful cache
seed as dependency approval.
