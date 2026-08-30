# Companion Publication and Acquisition

UpstreamDrift is the provider authority for the AffineDrift companion catalog.
This runbook covers delivery evidence only; it does not qualify a program,
workflow, screenshot, calculation, or engineering conclusion.

## Governed Workflow Evidence

From a clean exact-commit checkout, execute the provider-owned workflow
registry with:

```text
python3 -m scripts.companion_workflows --repo-root . execute-all --report dist/companion-workflows/execution-report.v1.json
```

The strict registry is `scripts/config/companion_workflows.v1.json`; it and all
referenced fixtures are hashed catalog inputs. Ten successful workflows cover
installation, launch resolution, simulation/analysis, import/export, program
export, counterfactual, report, and plot evidence. Four deterministic failure
fixtures must return their declared non-zero exits. The native OpenSim GUI
record is explicitly unavailable and cannot be executed.

The executor passes no shell string, rejects undeclared environment and
outputs, restricts paths to the repository, verifies artifact types and
digests, and refuses a stale source commit. CI runs every available record on
code pull requests and protected `main`, then uploads a commit-named 30-day
artifact. This is test evidence only. It is not a durable acquisition channel,
does not demonstrate native GUI interaction, and grants no scientific,
operational, engine, participant, or coaching qualification.

## One Generator, Two Delivery Channels

Both channels call the same public builder and verifier:

```text
python3 -m scripts.companion_publication build --authority <protected-main|tag> --output-dir dist/companion
python3 -m scripts.companion_publication verify-bundle --bundle-dir dist/companion
```

The builder delegates software-fact generation to
`scripts/companion_catalog.py`; it does not maintain another registry or
calculation manifest. It accepts only an official GitHub Actions `push` in
`D-sorganization/UpstreamDrift`, requires `GITHUB_SHA == HEAD`, and requires
either `refs/heads/main` or an exact `vX.Y.Z` tag. The catalog layer separately
requires a clean tree, committed local-only inputs, and an exact Tools gitlink.

The payload bundle contains exactly these files:

- `upstreamdrift-companion.v1.json` and its `.sha256`;
- `upstreamdrift-companion-v1.schema.json` and its `.sha256`;
- `upstreamdrift-companion-acquisition-v1.schema.json` and its `.sha256`; and
- `upstreamdrift-companion-compatibility-v1.json` and its `.sha256`.

Missing, renamed, extra, malformed, stale, or digest-mismatched files fail the
bundle verifier. The manifest's embedded source commit must match the exact CI
commit, and its schema/generator versions must match the policy.

## Protected-Main Artifact

Every protected `main` push produces one artifact named
`upstreamdrift-companion-<40-character-commit>` with 30-day retention. The
payloads are attested before upload. After GitHub returns the artifact ID, URL,
and archive digest, CI writes and attests a separate acquisition-evidence
artifact. That record includes the exact workflow run/attempt and explicitly
sets:

```text
channel = actions
durability = ephemeral
release = null
```

This is suitable for a reviewed temporary vendor pin. It is not a permanent
URL and must never be described as a release acquisition.

## Immutable Release Assets

A human-authorized signed `vX.Y.Z` tag uses the same builder at the tag commit.
The workflow uploads the payloads to a draft GitHub release with overwrites
disabled. It then reads the draft by numeric release ID and verifies exactly one
asset for every declared payload, including exact size. The acquisition record
uses numeric API identities of the form:

```text
https://api.github.com/repos/D-sorganization/UpstreamDrift/releases/assets/<asset-id>
```

Tag-based browser download links are included only as display links; they are
not the immutable identity. The record also contains the release ID/tag,
protected source commit, workflow run, schema/generator versions, payload sizes
and hashes, embedded manifest commit, and GitHub attestation ID/URL. The release
remains draft if any validation, attestation, or upload fails and becomes public
only after the acquisition record succeeds.

## Compatibility and Rollback

Schema `1.0.0` is current. No earlier formal schema exists, so the policy states
`previous_supported: []` rather than inventing history. A committed current
fixture validates, and future/incompatible fixtures must fail. The policy
validator requires a validating fixture for every entry added to
`previous_supported`, making the previous-version gate non-vacuous when a real
second supported version exists.

Consumers keep the prior immutable artifact and generated site until the new
pin, digest, schema, render, links, and accessibility checks pass together. A
failed replacement leaves the prior release intact; release assets are never
overwritten in place. Corrections use a new reviewed patch release.
