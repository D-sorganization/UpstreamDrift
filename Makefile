# Golf Modeling Suite Makefile
# Provides common development tasks for the golf simulation framework
#
# Usage:
#   make help     - Show available targets
#   make lint     - Run all linters
#   make format   - Format code
#   make test     - Run tests
#   make clean    - Clean build artifacts

.PHONY: help lint format test test-unit test-int smoke clean install check all docs sync-deps sbom codemap codemap-watch codemap-mcp

# Default target
help:
	@echo "Golf Modeling Suite - Available targets:"
	@echo ""
	@echo "  make install   - Install dependencies"
	@echo "  make lint      - Run linters (ruff, mypy)"
	@echo "  make format    - Format code (ruff)"
	@echo "  make test      - Run pytest"
	@echo "  make test-unit - Run unit tests only"
	@echo "  make test-int  - Run integration tests only"
	@echo "  make smoke     - Run release smoke tests for available artifacts"
	@echo "  make check     - Run all checks (lint + test)"
	@echo "  make clean     - Remove build artifacts"
	@echo "  make docs      - Build documentation"
	@echo "  make sbom      - Generate core, extended, and full SBOMs"
	@echo "  make sync-deps - Regenerate Python lockfiles and environment.yml"
	@echo "  make codemap   - One-shot rebuild of .codemap/index.db for chat + MCP"
	@echo "  make codemap-watch - Run the live codemap watcher daemon (foreground)"
	@echo "  make codemap-mcp   - Run the codemap MCP server (stdio, for debugging)"
	@echo "  make all       - Install, format, lint, test"
	@echo ""

# Install dependencies (including dev tools: ruff, mypy, pytest)
install:
	pip install -r requirements.txt
	@if [ -f pyproject.toml ] || [ -f setup.py ]; then \
		echo "Installing package in editable mode with dev dependencies..."; \
		pip install -e ".[dev]"; \
	else \
		echo "Skipping editable install: no pyproject.toml or setup.py found."; \
	fi

# Run linters
lint:
	@echo "Running ruff check..."
	ruff check .
	@echo "Running mypy (errors are advisory; see CONTRIBUTING.md)..."
	mypy . --config-file pyproject.toml || true

# Format code
format:
	@echo "Running ruff format..."
	ruff format .
	@echo "Running ruff fix..."
	ruff check . --fix || true

# Regenerate dependency artifacts from pyproject.toml, the canonical Python source.
sync-deps:
	python3 -m pip install "pip-tools>=7.4" "tomli>=2.0.0; python_version<'3.11'"
	python3 -m piptools compile -o requirements.lock pyproject.toml
	python3 -m piptools compile --extra dev -o requirements-dev.lock pyproject.toml
	python3 scripts/sync_environment_yml.py

# Run all tests
test:
	@echo "Running pytest..."
	pytest tests/ -v --tb=short

# Run unit tests only
test-unit:
	@echo "Running unit tests..."
	pytest tests/unit/ -v --tb=short

# Run integration tests only
test-int:
	@echo "Running integration tests..."
	pytest tests/integration/ -v --tb=short

# Run smoke tests against locally built release artifacts
smoke:
	@echo "Running Python wheel smoke tests..."
	pytest tests/smoke/python_wheel
	@if command -v docker >/dev/null 2>&1 && [ -n "$$UPSTREAM_DRIFT_API_IMAGE" ]; then \
		echo "Running Docker API smoke tests..."; \
		pytest tests/smoke/docker_api; \
	else \
		echo "Skipping Docker API smoke tests; docker or UPSTREAM_DRIFT_API_IMAGE unavailable."; \
	fi
	@if [ -n "$$UPSTREAM_DRIFT_TAURI_BUNDLE" ] || [ -d ui/dist ]; then \
		echo "Running Tauri desktop smoke tests..."; \
		pytest tests/smoke/tauri_desktop; \
	else \
		echo "Skipping Tauri desktop smoke tests; no bundle path or ui/dist present."; \
	fi
	@echo "Running Rust crate smoke tests..."
	pytest tests/smoke/rust_crate

# Run all checks
check: lint test
	@echo "All checks complete."

# Build documentation
docs:
	@echo "Building documentation..."
	cd docs && make html || echo "Sphinx not configured"

sbom:
	@echo "Generating per-tier SBOMs..."
	bash scripts/security/generate_sbom.sh core
	bash scripts/security/generate_sbom.sh extended
	bash scripts/security/generate_sbom.sh full

# ── Code-map (chat / MCP) ─────────────────────────────────────────────────────
# One-shot rebuild of the local .codemap/index.db used by the in-app chat and
# external MCP agents. See docs/codemap-integration.md.
codemap:
	@echo "Rebuilding .codemap/index.db ..."
	@if command -v codemap >/dev/null 2>&1; then \
		codemap rebuild; \
	else \
		python3 -m shared.python.codemap.cli rebuild || \
			python3 -m src.shared.python.codemap.cli rebuild; \
	fi
	@echo "Code-map rebuild complete."

# Start the file-watcher daemon that keeps the index live during development.
# Run as: `make codemap-watch &` so it backgrounds in your dev shell.
codemap-watch:
	@echo "Starting codemap watcher (Ctrl-C to stop) ..."
	@if command -v codemap-watch >/dev/null 2>&1; then \
		codemap-watch; \
	else \
		python3 -m shared.python.codemap.cli watch || \
			python3 -m src.shared.python.codemap.cli watch; \
	fi

# Launch the MCP server so external CLI agents (Claude Code, Codex) can query
# the same index. Normally started by .mcp.json — only invoke manually for
# debugging.
codemap-mcp:
	@echo "Starting codemap MCP server (stdio) ..."
	@CODEMAP_REPO_ROOT=$$(pwd) codemap-mcp

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	find . -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".mypy_cache" -o -name ".ruff_cache" -o -name "*.egg-info" \) -print0 2>/dev/null | xargs -0 rm -rf || true
	find . -type f \( -name "*.pyc" -o -name "*_output.txt" -o -name "*_temp.txt" \) -delete 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/ 2>/dev/null || true
	@echo "Clean complete."

# Run everything
all: install format lint test
	@echo "All tasks complete."
