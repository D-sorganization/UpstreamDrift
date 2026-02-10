"""Environment variable validation for production deployments.

This module provides validation functions to ensure all required environment
variables are properly configured for secure production deployments.

Usage:
    from shared.python.env_validator import validate_environment

    # Validate all environment variables
    validate_environment()

    # Or validate specific components
    validate_api_security()
    validate_database_config()
"""

import secrets
from typing import TypedDict

from src.shared.python.environment import (
    get_admin_password,
    get_database_url,
    get_environment,
    get_secret_key,
    is_production,
)
from src.shared.python.error_utils import EnvironmentError as EnvironmentValidationError
from src.shared.python.logging_config import get_logger

logger = get_logger(__name__)


class APIKeyValidationResults(TypedDict):
    secret_key: bool
    environment: str | None
    admin_password: bool
    issues: list[str]
    warnings: list[str]


class DatabaseKeyValidationResults(TypedDict):
    database_url: bool
    database_type: str | None
    issues: list[str]
    warnings: list[str]


class EnvironmentValidationResults(TypedDict):
    environment: str
    api_security: APIKeyValidationResults
    database: DatabaseKeyValidationResults
    production_checklist: dict[str, bool]
    critical_issues: list[str]
    warnings: list[str]
    valid: bool


def validate_secret_key_strength(key: str, min_length: int = 64) -> None:
    """Validate that a secret key meets security requirements.

    Args:
        key: Secret key to validate
        min_length: Minimum required length (default: 64 characters)

    Returns:
        None

    Raises:
        EnvironmentValidationError: If key is insufficient
    """
    if not key:
        raise EnvironmentValidationError("Secret key is empty")

    if len(key) < min_length:
        raise EnvironmentValidationError(
            f"Secret key is too short: {len(key)} chars (minimum: {min_length})"
        )

    # Check it's not a placeholder
    if key == "UNSAFE-NO-SECRET-KEY-SET-AUTHENTICATION-WILL-FAIL":
        raise EnvironmentValidationError(
            "Secret key is using unsafe placeholder. "
            "Set GOLF_API_SECRET_KEY or SECRET_KEY environment variable."
        )

    # Check for common weak patterns
    weak_patterns = [
        "password",
        "secret",
        "key123",
        "changeme",
        "default",
        "test",
        "12345",
    ]

    key_lower = key.lower()
    for pattern in weak_patterns:
        if pattern in key_lower:
            logger.warning(
                f"Secret key contains weak pattern: '{pattern}'. "
                f"Consider generating a stronger key."
            )


def validate_api_security() -> APIKeyValidationResults:
    """Validate API security configuration.

    Returns:
        Dictionary with validation results

    Raises:
        EnvironmentValidationError: If critical security issues found
    """
    results: APIKeyValidationResults = {
        "secret_key": False,
        "environment": None,
        "admin_password": False,
        "issues": [],
        "warnings": [],
    }

    # Check environment using centralized function
    environment = get_environment()
    results["environment"] = environment

    if environment not in ["development", "staging", "production"]:
        results["warnings"].append(
            f"Unknown ENVIRONMENT value: '{environment}'. "
            f"Expected: development, staging, or production"
        )

    # Validate secret key using centralized function
    secret_key = get_secret_key()

    if not secret_key:
        if is_production():
            results["issues"].append(
                "CRITICAL: No GOLF_API_SECRET_KEY or SECRET_KEY set in production!"
            )
            raise EnvironmentValidationError(
                "GOLF_API_SECRET_KEY",
                "GOLF_API_SECRET_KEY is required for production",
            )
        else:
            results["warnings"].append(
                "No GOLF_API_SECRET_KEY set. Using unsafe placeholder for development."
            )
    else:
        try:
            validate_secret_key_strength(secret_key)
            results["secret_key"] = True
        except EnvironmentValidationError as e:
            if is_production():
                results["issues"].append(f"CRITICAL: {e}")
                raise
            else:
                results["warnings"].append(str(e))

    # Check admin password using centralized function
    admin_password = get_admin_password()

    if not admin_password:
        if is_production():
            results["warnings"].append(
                "No GOLF_ADMIN_PASSWORD set. A random password will be generated. "
                "Set this variable to use a custom admin password."
            )
        results["admin_password"] = False
    else:
        if len(admin_password) < 12:
            results["warnings"].append(
                f"Admin password is short ({len(admin_password)} chars). "
                f"Recommend at least 12 characters."
            )
        results["admin_password"] = True

    return results


def validate_database_config() -> DatabaseKeyValidationResults:
    """Validate database configuration.

    Returns:
        Dictionary with validation results
    """
    results: DatabaseKeyValidationResults = {
        "database_url": False,
        "database_type": None,
        "issues": [],
        "warnings": [],
    }

    database_url = get_database_url(default="")

    if not database_url:
        results["warnings"].append(
            "No DATABASE_URL set. Using default SQLite database."
        )
        results["database_url"] = False
        results["database_type"] = "sqlite"
        return results

    results["database_url"] = True

    # Determine database type
    if database_url.startswith("sqlite"):
        results["database_type"] = "sqlite"

        if is_production():
            results["warnings"].append(
                "SQLite is not recommended for production. "
                "Consider using PostgreSQL for better concurrency and reliability."
            )

    elif database_url.startswith("postgresql"):
        results["database_type"] = "postgresql"

    elif database_url.startswith("mysql"):
        results["database_type"] = "mysql"

    else:
        results["database_type"] = "unknown"
        results["warnings"].append(
            f"Unknown database type in DATABASE_URL: {database_url.split(':')[0]}"
        )

    # Check for credentials in URL (they should be there for remote databases)
    db_type = results["database_type"]
    if db_type in ["postgresql", "mysql"]:
        if database_url and "@" not in database_url:
            results["warnings"].append(
                f"{str(db_type).upper()} URL appears to be missing credentials"
            )

    return results


def validate_production_checklist() -> dict[str, bool]:
    """Validate production deployment checklist.

    Returns:
        Dictionary mapping checklist items to their status
    """
    checklist: dict[str, bool] = {}

    # Only enforce for production
    if not is_production():
        return checklist

    # Required for production
    checklist["secret_key_set"] = bool(get_secret_key())
    checklist["environment_set"] = is_production()
    checklist["https_recommended"] = True  # Can't check from env, assume true

    # Recommended for production
    database_url = get_database_url(default="")
    checklist["postgresql_database"] = database_url.startswith("postgresql")
    checklist["admin_password_set"] = bool(get_admin_password())

    return checklist


def generate_secure_key_command() -> str:
    """Generate command to create a secure secret key.

    Returns:
        Shell command to generate a secure key
    """
    # Generate a sample key to show format
    sample_key = secrets.token_urlsafe(64)

    return f"""
# Generate a secure secret key (64+ characters):
python3 -c "import secrets; print(secrets.token_urlsafe(64))"

# Example output (DO NOT USE THIS EXACT KEY):
# {sample_key}

# Set the environment variable:
export GOLF_API_SECRET_KEY="[your-generated-key]"

# Or add to .env file:
echo 'GOLF_API_SECRET_KEY=[your-generated-key]' >> .env
"""


def validate_environment(raise_on_error: bool = True) -> EnvironmentValidationResults:
    """Validate all environment configuration.

    Args:
        raise_on_error: If True, raise exception on critical errors (default: True)

    Returns:
        Dictionary with complete validation results

    Raises:
        EnvironmentValidationError: If critical issues found and raise_on_error=True
    """
    logger.info("Validating environment configuration...")

    results: EnvironmentValidationResults = {
        "environment": get_environment(),
        "api_security": {
            "secret_key": False,
            "environment": None,
            "admin_password": False,
            "issues": [],
            "warnings": [],
        },
        "database": {
            "database_url": False,
            "database_type": None,
            "issues": [],
            "warnings": [],
        },
        "production_checklist": {},
        "critical_issues": [],
        "warnings": [],
        "valid": True,
    }

    # Validate API security
    try:
        results["api_security"] = validate_api_security()
        results["warnings"].extend(results["api_security"]["warnings"])

        if results["api_security"]["issues"]:
            results["critical_issues"].extend(results["api_security"]["issues"])
            results["valid"] = False

    except EnvironmentValidationError as e:
        results["critical_issues"].append(str(e))
        results["valid"] = False
        if raise_on_error:
            raise

    # Validate database
    results["database"] = validate_database_config()
    results["warnings"].extend(results["database"]["warnings"])

    if results["database"]["issues"]:
        results["critical_issues"].extend(results["database"]["issues"])
        results["valid"] = False

    # Validate production checklist (if in production)
    if is_production():
        results["production_checklist"] = validate_production_checklist()

        # Check critical items
        if not results["production_checklist"].get("secret_key_set"):
            results["critical_issues"].append("Secret key not set for production!")
            results["valid"] = False

    # Log results
    if results["valid"]:
        logger.info("✓ Environment validation passed")

        if results["warnings"]:
            max_warnings = 5
            total_warnings = len(results["warnings"])
            logger.warning(
                f"Found {total_warnings} warnings (showing up to {max_warnings}):"
            )
            for warning in results["warnings"][:max_warnings]:
                logger.warning(f"  - {warning}")
            if total_warnings > max_warnings:
                logger.warning(f"  ... and {total_warnings - max_warnings} more")
    else:
        logger.error("✗ Environment validation FAILED")
        logger.error(f"Found {len(results['critical_issues'])} critical issues:")
        for issue in results["critical_issues"]:
            logger.error(f"  - {issue}")

        if raise_on_error:
            raise EnvironmentValidationError(
                f"Environment validation failed with {len(results['critical_issues'])} critical issues"
            )

    return results


def print_validation_report(results: EnvironmentValidationResults) -> None:
    """Log a formatted validation report.

    Args:
        results: Results from validate_environment()
    """
    logger.info("\n" + "=" * 80)
    logger.info("ENVIRONMENT VALIDATION REPORT")
    logger.info("=" * 80)

    # Environment
    logger.info(f"\nEnvironment: {results['environment'].upper()}")

    # API Security
    logger.info("\nAPI Security:")
    api_sec = results["api_security"]
    logger.info(
        f"  Secret Key:      {'✓ SET' if api_sec.get('secret_key') else '✗ NOT SET'}"
    )
    logger.info(
        f"  Admin Password:  {'✓ SET' if api_sec.get('admin_password') else '○ Optional'}"
    )

    # Database
    logger.info("\nDatabase:")
    db = results["database"]
    logger.info(
        f"  Database URL:    {'✓ SET' if db.get('database_url') else '○ Using default'}"
    )
    db_type_str = str(db.get("database_type", "unknown")).upper()
    logger.info(f"  Database Type:   {db_type_str}")

    # Production checklist (if applicable)
    if results["environment"] == "production":
        logger.info("\nProduction Checklist:")
        checklist = results["production_checklist"]
        for item, status in checklist.items():
            symbol = "✓" if status else "✗"
            logger.info(f"  {symbol} {item.replace('_', ' ').title()}")

    # Issues and warnings
    if results["critical_issues"]:
        logger.info(f"\n❌ CRITICAL ISSUES ({len(results['critical_issues'])}):")
        for issue in results["critical_issues"]:
            logger.info(f"  - {issue}")

    if results["warnings"]:
        logger.info(f"\n⚠️  WARNINGS ({len(results['warnings'])}):")
        for warning in results["warnings"]:
            logger.info(f"  - {warning}")

    # Overall status
    logger.info("\n" + "=" * 80)
    if results["valid"]:
        logger.info("✓ VALIDATION PASSED")
    else:
        logger.info("✗ VALIDATION FAILED")
    logger.info("=" * 80 + "\n")

    # Help for fixing issues
    if not results["valid"]:
        logger.info("\nTo fix issues:")
        logger.info(generate_secure_key_command())


if __name__ == "__main__":
    """Run environment validation when executed directly."""
    import sys

    try:
        results = validate_environment(raise_on_error=False)
        print_validation_report(results)

        # Exit with error code if validation failed
        sys.exit(0 if results["valid"] else 1)

    except Exception as e:
        logger.error(f"Validation error: {e}")
        sys.exit(1)
