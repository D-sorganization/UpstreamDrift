# Publication Quality and Cross-Repository Authority

## Authority Decision

UpstreamDrift is the sole scientific source authority for this monograph. It
owns the computational source, registered evidence, claim review, rendered PDF,
release manifest, checksums, and publication-quality inspection. AffineDrift is
the generated publisher: it may render and serve the article, but it must pin an
exact UpstreamDrift revision and may not silently edit scientific claims. Tools
may consume the released evidence and expose bounded navigation or Sidekick
links; it is not a second scientific source.

This division is intentionally asymmetric. A deployed AffineDrift page is not
authoritative merely because it is live, and an UpstreamDrift PDF is not a
professional publication merely because its bytes are hash-pinned. Both source
identity and publication quality must pass.

## Two Independent Readiness Profiles

`publication_quality.py` inspects the complete PDF and emits one
`proximal-distal-publication-quality-v1` report bound to:

- the exact UpstreamDrift repository URL and 40-character revision;
- the SHA-256 digest of `release_manifest.json`;
- the SHA-256 digest, byte size, metadata, and page count of the PDF;
- every outline entry and PDF link;
- a render attempt and text-extraction check for every page; and
- the tagged-structure, font-resource, and fast-web-access posture.

The profiles answer different questions:

| Profile         | Blocking Meaning                                                                                                                         |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `computational` | The exact scientific PDF has correct identity and metadata, valid navigation, extractable content, and every page renders.               |
| `archival`      | The computational profile passes and the PDF is tagged, web-linearized, free of Type 3 resources, and free of unembedded font resources. |

Neither profile qualifies participant evidence or a human performance claim.
The governed held-out human protocol remains a separate scientific gate, as do
equipment calibration and external archive/PID deposition.

## Current Candidate Result

The 233-page candidate renders successfully on all 233 pages, exposes
extractable text on all 233 pages, contains 246 outline entries and 192 valid
external links, and is now linearized for fast web access. Its candidate PDF
SHA-256 is
`782621779730cd30003bc35c2d07b9999fe0e4aa6fe0025e3f08c1ec051ddd52`.

The computational profile passes. The archival profile remains deliberately
blocked because the PDF has no structure tag tree, contains 110 Type 3 font
resources from embedded figures, and uses two unembedded base-font resources.
Those facts are release findings, not waived successes. An archival release or
accessibility claim must wait for a regenerated, tagged document with accessible
figure fonts and a fresh full-page review.

## Reproduction and Validation

Install the publication-only dependencies without changing the core runtime:

```bash
python3 -m pip install -e '.[publication]'
```

`pikepdf` is MPL-2.0. PyMuPDF is dual-licensed under AGPL-3.0 or an Artifex
commercial license and is therefore recorded as `Needs review` in the license
ledger. Both remain release-only tooling; neither is bundled into the
application runtime or the PDF.

After Quarto produces the PDF, compact and linearize it while preserving the
page, external-link, and outline contract:

```bash
python3 -m scripts.research.proximal_distal_energy.optimize_article_pdf
```

Validate the computational candidate against the checked-out revision:

```bash
python3 -m scripts.research.proximal_distal_energy.publication_quality \
  --source-revision "$(git rev-parse HEAD)" \
  --profile computational \
  --report publication-quality-report.json
python3 -m scripts.research.proximal_distal_energy.qualify_open_release validate \
  --source-revision "$(git rev-parse HEAD)" \
  --publication-profile computational
```

Use `--profile archival` only when preparing a professional archival
publication; it currently fails closed on the disclosed accessibility gaps.
The generated JSON report is runtime evidence and is not committed into the
self-hashed release bundle.

## Protected Publication Contract

CI Standard's dedicated `publication-quality` job installs the release-only PDF
tooling, runs the complete release/claim/external-source/PDF validator when any
publication-authority path changes, and is explicitly aggregated into the sole
required `quality-gate`. A missing dependency or skipped PDF inspection can no
longer produce a protected green result. Cross-repository promotion must then
verify all of the following on the same candidate:

The claim-evidence authority hashes valid UTF-8 evidence after canonical CRLF
to LF normalization and hashes binary evidence byte-for-byte. The contract is
therefore stable across Windows and Linux checkouts while still failing closed
on semantic text changes and every binary-byte change.

1. UpstreamDrift's computational publication profile passes on an exact clean
   revision, and the release manifest and PDF digests are captured.
2. AffineDrift records that revision and both digests in generated publication
   metadata and fails closed if its copied/generated artifact drifts.
3. The deployed canonical HTML and PDF routes return successfully and expose
   the same source revision and PDF digest.
4. Tools and Sidekick link to the canonical publication metadata rather than
   embedding an independently editable scientific copy.

The same-named lightweight checks remain unable to satisfy this contract on the
aggregate's behalf.

## Handoff and Non-Overlap

This implementation belongs to UpstreamDrift issue #8451. Phase 0's portable
release authority, pinned publisher, and isolated benchmark environment landed
as UpstreamDrift #8791 (`2d0178c5beeaf3c9bc289f92f41d99620781f9e2`),
AffineDrift #3884 (`c36983cce5d0972c2ec115098f07470dea0f511d`), and
Tools #4586 (`2c0221258fc1711aa858151ba239ba1a39e677eb`). This slice is rebased
onto #8791, and its regenerated release authorities pass validation. After the
protected UpstreamDrift merge, rerun both publication profiles on the exact merge
and carry that revision and its final digests through
AffineDrift's generated publication contract. #8789 remains active for the
separate Docker/quarantine/baseline work; do not duplicate that scope.
