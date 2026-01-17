# Issue Tracking

This folder contains markdown documents that track known issues, bugs, and technical debt for the Golf Modeling Suite project.

## How to Use This System

### Adding New Issues

1. Add issues to the appropriate category file (or create a new one)
2. Use the standard format shown below
3. Include severity, status, and enough context for someone to understand the issue

### Issue Format

```markdown
### [SEVERITY] Brief Title
- **Status**: Open | In Progress | Resolved | Won't Fix
- **Component**: Module/file affected
- **Description**: What's wrong and why it matters
- **Suggested Fix**: How to resolve (if known)
- **GitHub Issue**: #123 (if created)
```

### Severity Levels

- **CRITICAL**: System broken, data loss, security vulnerability
- **HIGH**: Major functionality broken, significant user impact
- **MEDIUM**: Feature works but with issues, workarounds exist
- **LOW**: Minor issues, cosmetic, nice-to-have improvements

### Converting to GitHub Issues

To create a GitHub issue from an entry here:

```bash
# From repository root
gh issue create --title "Brief Title" --body "Description and context" --label "bug,severity:high"
```

Or use the GitHub web interface and copy the description.

### Syncing GitHub Issues Back

When a GitHub issue is created, update the entry here with the issue number.
When an issue is resolved, update status in both places.

## Issue Files

- [KNOWN_ISSUES.md](KNOWN_ISSUES.md) - Current known issues and bugs
- [TECHNICAL_DEBT.md](TECHNICAL_DEBT.md) - Code quality and refactoring items
- [ROADMAP_ISSUES.md](ROADMAP_ISSUES.md) - Planned improvements (linked to existing GitHub issues)
