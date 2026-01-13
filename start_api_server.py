#!/usr/bin/env python3
"""
Simple startup script for Golf Modeling Suite API server.

This script handles the basic setup and starts the API server for interim use
without requiring complex deployment infrastructure.
"""

import importlib.util
import logging
import os
import sys
from pathlib import Path

import uvicorn

from api.server import app

# Configure logging for startup messages
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    required_modules = ["fastapi", "uvicorn", "sqlalchemy"]

    for module in required_modules:
        if importlib.util.find_spec(module) is None:
            logger.error("❌ Missing dependency: %s", module)
            logger.error(
                "Run: pip install fastapi uvicorn sqlalchemy pydantic[email] "
                "python-jose[cryptography] passlib[bcrypt] python-multipart slowapi"
            )
            return False

    logger.info("✅ API dependencies found")
    return True


def setup_environment() -> tuple[str, int]:
    """Set up environment variables with defaults."""
    # Set default database URL if not specified
    if not os.getenv("DATABASE_URL"):
        db_path = Path(__file__).parent / "golf_modeling_suite.db"
        os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
        logger.info("📁 Using SQLite database: %s", db_path)

    # SECURITY: Validate environment configuration
    try:
        from shared.python.env_validator import validate_environment

        logger.info("🔒 Validating security configuration...")

        # Validate without raising exception (we'll show warnings instead)
        results = validate_environment(raise_on_error=False)

        if results["critical_issues"]:
            logger.error("❌ CRITICAL SECURITY ISSUES FOUND:")
            for issue in results["critical_issues"]:
                logger.error(f"   - {issue}")

            environment = os.getenv("ENVIRONMENT", "development").lower()
            if environment == "production":
                logger.error("Cannot start server with critical security issues in production!")
                sys.exit(1)
            else:
                logger.warning("Continuing in development mode despite issues...")

        if results["warnings"]:
            logger.warning("⚠️  Security warnings:")
            for warning in results["warnings"][:5]:  # Show first 5
                logger.warning(f"   - {warning}")

        if results["valid"]:
            logger.info("✅ Security configuration validated")

    except ImportError:
        logger.warning("⚠️  Could not import env_validator (optional)")

        # Fallback to basic checks
        if not os.getenv("GOLF_API_SECRET_KEY") and not os.getenv("SECRET_KEY"):
            logger.warning(
                "⚠️  SECURITY: No GOLF_API_SECRET_KEY or SECRET_KEY environment variable set!"
            )
            logger.warning("   API authentication will fail. Set one for production.")
            logger.warning(
                "   Example: export GOLF_API_SECRET_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(64))')"
            )

    # Set default host and port
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))

    return host, port


def check_physics_engines() -> bool:
    """Check which physics engines are available."""
    try:
        from shared.python.engine_manager import EngineManager

        manager = EngineManager()
        available = manager.get_available_engines()

        logger.info("🔧 Available physics engines: %d", len(available))
        for engine in available:
            logger.info("   • %s", engine.value)

        if not available:
            logger.warning("⚠️  No physics engines available. Install at least MuJoCo:")
            logger.warning("   pip install mujoco>=3.3.0")

        return len(available) > 0

    except Exception as e:
        logger.warning("⚠️  Could not check physics engines: %s", e)
        return True  # Continue anyway


def start_server(host: str, port: int) -> None:
    """Start the API server."""
    logger.info("🚀 Starting Golf Modeling Suite API server...")
    logger.info("   Host: %s", host)
    logger.info("   Port: %d", port)
    logger.info("   API Documentation: http://localhost:%d/docs", port)
    logger.info("   Admin Interface: http://localhost:%d/redoc", port)
    logger.info("📝 Admin credentials:")
    logger.info("   Email: admin@golfmodelingsuite.com")
    if os.getenv("GOLF_ADMIN_PASSWORD"):
        logger.info("   Password: (set via GOLF_ADMIN_PASSWORD environment variable)")
    else:
        logger.warning(
            "   ⚠️  GOLF_ADMIN_PASSWORD not set - a random password will be generated."
        )
        logger.warning("   See server logs for temporary password.")
    logger.info("🛑 Press Ctrl+C to stop the server")

    try:
        # Start the server
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=True,  # Auto-reload on code changes
            log_level="info",
        )

    except KeyboardInterrupt:
        logger.info("👋 Server stopped by user")
    except Exception as e:
        logger.error("❌ Server error: %s", e)
        sys.exit(1)


def main() -> None:
    """Main startup function."""
    logger.info("🏌️ Golf Modeling Suite - API Server Startup")
    logger.info("=" * 50)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Setup environment
    host, port = setup_environment()

    # Check physics engines
    check_physics_engines()

    # Start server
    start_server(host, port)


if __name__ == "__main__":
    main()
