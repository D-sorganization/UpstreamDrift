# Troubleshooting

This directory contains troubleshooting guides for the Golf Modeling Suite.

## Contents

| Document                                                 | Description                                           |
| -------------------------------------------------------- | ----------------------------------------------------- |
| [common-issues.md](common-issues.md)                     | Comprehensive guide for common problems and solutions |
| [cross_engine_deviations.md](cross_engine_deviations.md) | Known differences between physics engines             |

## Quick Links

### By Category

- **Installation Problems?** → [common-issues.md#installation-issues](common-issues.md#installation-issues)
- **Engine Not Working?** → [common-issues.md#engine-issues](common-issues.md#engine-issues)
- **GUI Crashes?** → [common-issues.md#guilauncher-issues](common-issues.md#guilauncher-issues)
- **Slow Performance?** → [common-issues.md#performance-issues](common-issues.md#performance-issues)
- **CI/CD Failures?** → [common-issues.md#cicd-issues](common-issues.md#cicd-issues)

### Quick Diagnostics

Run this command to get a diagnostic report:

```bash
python -c "from shared.python import EngineManager; print(EngineManager().get_diagnostic_report())"
```

### Still Need Help?

1. Check [GitHub Issues](https://github.com/D-sorganization/Golf_Modeling_Suite/issues)
2. Review [assessment documents](../assessments/)
3. Open a new issue with full details
