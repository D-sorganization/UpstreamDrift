# Tools Dependency Migration Plan

## Issue
#2505: 301 shared filenames between UpstreamDrift and Tools repositories.
`text_editor.py` near-duplicate identified as primary DRY violation.

## Goal
Consume `upstream_drift_tools` (from the Tools repo) as a proper package
dependency instead of maintaining near-duplicate copies.

## Plan

### Phase 1: Audit (current)
- [x] Identify 301 shared filenames
- [x] Document primary duplicate: text_editor.py
- [ ] Map all duplicate symbols to upstream_drift_tools equivalents

### Phase 2: Dependency Setup
- [ ] Add upstream_drift_tools to pyproject.toml dependencies
- [ ] Configure editable install for local development: `pip install -e ../Tools`
- [ ] Update imports in UpstreamDrift to use upstream_drift_tools.*

### Phase 3: Deduplication
- [ ] Replace local text_editor.py with upstream_drift_tools.text_editor
- [ ] Remove 301 duplicate files once imports updated
- [ ] Add CI gate to prevent re-introduction of duplicates

## Commands
```bash
# Install shared tools locally
pip install -e ../Tools

# Verify no duplicate imports
python -c "import upstream_drift_tools; print(upstream_drift_tools.__version__)"
```
