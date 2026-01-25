#!/usr/bin/env python3
"""Golf Modeling Suite - API Server Startup Script.

Handles environment configuration, dependency validation, and server launch.

Refactored to address DRY and Orthogonality violations (Pragmatic Programmer).
"""

from __future__ import annotations

import os

import uvicorn

from api.server import app
from src.shared.python.launcher_utils import (
    check_python_dependencies,
    ensure_environment_var,
    get_repo_root,
    invoke_main,
)
from src.shared.python.logging_config import get_logger, setup_logging

setup_logging(use_simple_format=True)
logger = get_logger(__name__)


def _validate_security() -> bool:
    """Perform security validation on the environment.

    Returns:
        True if environment is secure enough to proceed.
    """
    try:
        from shared.python.env_validator import validate_environment

        logger.info("🔒 Validating security configuration...")
        results = validate_environment(raise_on_error=False)

        if results["critical_issues"]:
            logger.error("❌ CRITICAL SECURITY ISSUES FOUND:")
            for issue in results["critical_issues"]:
                logger.error(f"   - {issue}")

            if os.getenv("ENVIRONMENT", "development").lower() == "production":
                return False
            logger.warning("Continuing in development mode despite issues...")

        if results["warnings"]:
            for warning in results["warnings"][:3]:
                logger.warning(f"⚠️  {warning}")

        return True
    except ImportError:
        # Fallback security check
        if not os.getenv("GOLF_API_SECRET_KEY") and not os.getenv("SECRET_KEY"):
            logger.warning("⚠️  SECURITY: No SECRET_KEY detected. API Auth will fail.")
        return True


def setup_api_environment() -> tuple[str, int]:
    """Configure API environment variables and defaults.

    Decomposed from God function to improve Orthogonality.
    """
    root = get_repo_root()

    # Database setup
    db_path = root / "golf_modeling_suite.db"
    ensure_environment_var("DATABASE_URL", f"sqlite:///{db_path}", "Database URL")

    # Security validation
    _validate_security()

    # Network config
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))

    return host, port


def print_server_info(host: str, port: int) -> None:
    """Print connection information for the user."""
    logger.info("🚀 Golf Modeling Suite API server starting...")
    logger.info(f"   Docs: http://localhost:{port}/docs")
    logger.info(f"   Host: {host}:{port}")
    if not os.getenv("GOLF_ADMIN_PASSWORD"):
        logger.warning(
            "⚠️  GOLF_ADMIN_PASSWORD not set - see logs for generated password."
        )


def main() -> int:
    """Main API server entry point."""
    logger.info("🏌️ API Server Startup")

    # 1. Dependencies
    deps = ["fastapi", "uvicorn", "sqlalchemy"]
    if not check_python_dependencies(deps):
        return 1

    # 2. Environment
    host, port = setup_api_environment()

    # 3. Launch info
    print_server_info(host, port)

    # 4. Start uvicorn
    is_dev = os.getenv("ENVIRONMENT", "development").lower() == "development"
    uvicorn.run(app, host=host, port=port, reload=is_dev, log_level="info")
    return 0


if __name__ == "__main__":
    invoke_main(main)
