# Vulnerability Triage SLA

The response time for a reported or detected vulnerability depends on its
severity (CVSS) and the affected dependency tier.

| Severity \ Tier  | core         | extended    | experimental           | archived |
| ---------------- | ------------ | ----------- | ---------------------- | -------- |
| Critical (>=9.0) | 24h          | 48h         | 7d or remove from list | n/a      |
| High (7.0-8.9)   | 7d           | 14d         | 30d or remove          | n/a      |
| Medium (4.0-6.9) | 30d          | 60d         | 90d                    | n/a      |
| Low (<4.0)       | next release | best effort | best effort            | n/a      |

Response means one of the following exists before the deadline: a tracking issue,
a remediation PR, or a documented waiver that follows the waiver rules below.

## Dependency Tiers

- `core`: dependencies installed by `pip install upstream-drift`.
- `extended`: dependencies installed by `pip install "upstream-drift[all-engines]"`.
- `experimental`: biomechanics, pose, and RL extras that are opt-in and may carry
  higher upstream dependency risk.
- `archived`: legacy code and dependencies kept for historical reference only.

## Waiver Rules

- `core` tier waivers require approval by two CODEOWNERS for the affected module
  and must expire within 30 days.
- `extended` tier waivers require one CODEOWNER and must expire within 90 days.
- `experimental` tier waivers may be owner-approved and must expire within 180
  days. A waiver renewed twice triggers deprecation review.
- Every `pip-audit` waiver entry must include `id`, `package`, `tier`, `reason`,
  and `expires_at`.

## SBOM Artifacts

Releases publish SBOMs for `core`, `extended`, and `full` dependency surfaces in
CycloneDX JSON and SPDX-compatible JSON. Consumers can download the tier-specific
SBOMs from GitHub releases, for example:

```bash
gh release download v2.1.0 --pattern "*.cyclonedx.*.json"
```

## OSV Monitoring

The daily OSV monitor classifies findings by dependency tier and computes the SLA
deadline from the table above. Findings are converted into tracking records using
`scripts/security/triage_osv_findings.py`.
