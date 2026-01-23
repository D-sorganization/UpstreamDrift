# Golf Modeling Suite Makefile
# Provides common development tasks for the golf simulation framework
#
# Usage:
#   make help     - Show available targets
#   make lint     - Run all linters
#   make format   - Format code
#   make test     - Run tests
#   make clean    - Clean build artifacts

.PHONY: help lint format test clean install check all docs

# Default target
help:
	@echo "Golf Modeling Suite - Available targets:"
	@echo ""
	@echo "  make install   - Install dependencies"
	@echo "  make lint      - Run linters (ruff, mypy)"
	@echo "  make format    - Format code (black, ruff)"
	@echo "  make test      - Run pytest"
	@echo "  make test-unit - Run unit tests only"
	@echo "  make test-int  - Run integration tests only"
	@echo "  make check     - Run all checks (lint + test)"
	@echo "  make clean     - Remove build artifacts"
	@echo "  make docs      - Build documentation"
	@echo "  make all       - Install, format, lint, test"
	@echo ""

# Install dependencies
install:
	pip install -r requirements.txt
	@if [ -f pyproject.toml ] || [ -f setup.py ]; then \
		echo "Installing package in editable mode..."; \
		pip install -e .; \
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
	@echo "Running black..."
	black .
	@echo "Running ruff format..."
	ruff format .
	@echo "Running ruff fix..."
	ruff check . --fix || true

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

# Run all checks
check: lint test
	@echo "All checks complete."

# Build documentation
docs:
	@echo "Building documentation..."
	cd docs && make html || echo "Sphinx not configured"

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
