# Criterion H: Security

**Repo:** UpstreamDrift
**Score:** 45/100
**Weight:** 10%
**Weighted Contribution:** 4.50

## Evidence

```json
{
  "secrets_raw": 10,
  "bandit_cfg": 0,
  "security_md": 1
}
```

## Findings

### P0: [UpstreamDrift] 10 potential hardcoded secrets detected

Audit source files for hardcoded credentials. Move to environment variables or secret manager (Vault, AWS Secrets Manager).
