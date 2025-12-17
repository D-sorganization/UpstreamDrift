# Project Template

This is a template repository for creating new projects with consistent structure, CI/CD workflows, and GitHub Copilot agents.

## Quick Start

```bash
# 1) Initialize repo + Git LFS
git init
git lfs install

# 2) Install pre-commit hooks
bash scripts/setup_precommit.sh

# 3) Protect main (on GitHub): require PR + status checks
# 4) Start safe WIP branch and code
git checkout -b chore/wip-$(date +%F)
```

## Project Structure

This template supports multiple programming languages and platforms:

```
Project_Template/
├── python/          # Python source code and tests
│   ├── src/         # Python source files
│   └── tests/       # Python test files
├── matlab/          # MATLAB source code and tests
│   ├── run_all.m    # Main MATLAB script
│   └── tests/       # MATLAB test files
├── javascript/      # JavaScript/TypeScript source code
│   ├── src/         # JavaScript/TypeScript source files
│   ├── tests/       # JavaScript/TypeScript test files
│   └── config/      # Configuration files (webpack, babel, etc.)
├── arduino/         # Arduino sketches and libraries
│   ├── src/         # Main Arduino sketches (.ino files)
│   ├── libraries/   # Custom Arduino libraries
│   └── examples/    # Example sketches
├── data/            # Data files
│   └── raw/         # Raw data files
├── docs/            # Documentation
├── scripts/         # Utility scripts
├── output/          # Generated output files
└── .github/         # GitHub configuration
    ├── agents/      # GitHub Copilot agents
    └── workflows/   # CI/CD workflows
```

## Language-Specific Setup

### Python

```bash
cd python
pip install -r requirements.txt
# Or use conda:
conda env create -f environment.yml
```

### MATLAB

- Open MATLAB and navigate to the `matlab/` directory
- Run `run_all.m` to execute all scripts
- Tests are in `matlab/tests/`

### JavaScript/TypeScript

```bash
cd javascript
npm install
npm test
```

See `javascript/README.md` for detailed setup instructions.

### Arduino

See `arduino/README.md` for setup instructions. Supports both Arduino IDE and PlatformIO.

## GitHub Copilot Agents

This template includes 6 specialized GitHub Copilot agents in `.github/agents/`:

1. **docs-agent** - Technical writing and documentation
2. **script-agent** - Cross-platform shell scripting
3. **security-agent** - Security analysis and branch protection
4. **git-workflow-agent** - Git operations and branch management
5. **ci-cd-agent** - CI/CD documentation and standards
6. **markdown-lint-agent** - Documentation quality enforcement

These agents provide intelligent assistance for repository management tasks. They are automatically available when using GitHub Copilot in this repository.

## Daily Safety

- Commit every ~30 minutes (`wip:` if tests fail).
- End-of-day snapshot: `bash scripts/snapshot.sh`.
- Big AI refactor? Create `backup/before-ai-<desc>` branch first.

## Reproducibility

- `matlab/run_all.m` should regenerate results.
- Python env pinned via `python/requirements.txt` or `python/environment.yml`.
- JavaScript dependencies in `javascript/package.json`.
- Arduino libraries documented in `arduino/README.md`.

## CI/CD

This template includes GitHub Actions workflows for:
- Code quality checks
- Linting and formatting
- Testing across multiple Python versions
- Security scanning
- Documentation validation

See `.github/workflows/` for workflow definitions.

## Cursor Settings Synchronization

This repository includes optimized Cursor settings for cross-computer development:

- **`cursor-settings.json`** - Complete settings with stall prevention and performance optimization
- **`CURSOR_SETTINGS_README.md`** - Detailed usage guide for syncing settings across computers

Copy these settings to your Cursor `settings.json` for consistent development experience.

## License

See `LICENSE` file for license information.
