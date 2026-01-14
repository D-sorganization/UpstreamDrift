#!/usr/bin/env python3
"""API Key Migration Script - SHA256 to Bcrypt.

This script migrates all existing API keys from SHA256 hashing to bcrypt hashing.

CRITICAL: This is a ONE-WAY migration. Old API keys will NOT work after migration.

Usage:
    python scripts/migrate_api_keys.py [options]

Options:
    --dry-run       Show what would be migrated without making changes
    --output FILE   Write new API keys to FILE (default: new_api_keys.txt)
    --database URL  Database connection string (default: from DATABASE_URL env)

Example:
    # Dry run first to see what will happen
    python scripts/migrate_api_keys.py --dry-run

    # Actual migration
    python scripts/migrate_api_keys.py --output new_keys.txt

    # Custom database
    python scripts/migrate_api_keys.py --database postgresql://user:pass@host/db

Security Notes:
    - This script generates NEW API keys (old ones cannot be recovered)
    - New keys are shown ONLY ONCE - save the output file securely
    - Distribute new keys to users immediately after migration
    - Revoke/delete old keys after confirming migration success

Author: Golf Modeling Suite Security Team
Date: January 2026
Version: 1.0.0
"""

import argparse
import logging
import os
import secrets
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from passlib.context import CryptContext
from sqlalchemy import create_engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import Session, sessionmaker

from api.auth.models import APIKey, User

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Password context for bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Migrate API keys from SHA256 to bcrypt hashing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (no changes)
  python scripts/migrate_api_keys.py --dry-run

  # Actual migration
  python scripts/migrate_api_keys.py

  # Save to custom file
  python scripts/migrate_api_keys.py --output /secure/path/keys.txt
        """,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="new_api_keys.txt",
        help="Output file for new API keys (default: new_api_keys.txt)",
    )

    parser.add_argument(
        "--database",
        type=str,
        default=None,
        help="Database URL (default: from DATABASE_URL environment variable)",
    )

    return parser.parse_args()


def get_database_session(database_url: str | None = None) -> Session:
    """Create database session.

    Args:
        database_url: Database connection string, or None to use environment variable

    Returns:
        SQLAlchemy session
    """
    if database_url is None:
        database_url = os.getenv("DATABASE_URL", "sqlite:///./golf_modeling_suite.db")

    try:
        url = make_url(database_url)
        masked_url = url.render_as_string(hide_password=True)
        logger.info(f"Connecting to database: {masked_url}")
    except Exception:
        # Fallback to simple logic if URL parsing fails
        logger.info(f"Connecting to database: {database_url.split('@')[-1]}")

    # Create engine
    if database_url.startswith("sqlite"):
        engine = create_engine(database_url, connect_args={"check_same_thread": False})
    else:
        engine = create_engine(database_url, pool_pre_ping=True)

    # Create session
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()


def generate_new_api_key() -> str:
    """Generate a new cryptographically secure API key.

    Returns:
        New API key with gms_ prefix
    """
    return f"gms_{secrets.token_urlsafe(32)}"


def migrate_api_keys(
    db_session: Session, dry_run: bool = False
) -> list[dict[str, Any]]:
    """Migrate all API keys from SHA256 to bcrypt.

    Args:
        db_session: Database session
        dry_run: If True, don't commit changes

    Returns:
        List of migration results with new API keys
    """
    # Get all API keys
    api_keys = db_session.query(APIKey).all()

    if not api_keys:
        logger.warning("No API keys found in database")
        return []

    logger.info(f"Found {len(api_keys)} API keys to migrate")

    migrations = []

    for idx, api_key_record in enumerate(api_keys, 1):
        # Get user info for context
        user = db_session.query(User).filter(User.id == api_key_record.user_id).first()
        user_email = user.email if user else "unknown"

        logger.info(
            f"[{idx}/{len(api_keys)}] Migrating key '{api_key_record.name}' "
            f"for user {user_email} (ID: {api_key_record.user_id})"
        )

        # Generate new API key
        new_key_value = generate_new_api_key()

        # Hash with bcrypt
        new_key_hash = pwd_context.hash(new_key_value)

        # Store migration info
        migration_info = {
            "key_id": api_key_record.id,
            "user_id": api_key_record.user_id,
            "user_email": user_email,
            "key_name": api_key_record.name,
            "old_hash_type": "SHA256",
            "new_hash_type": "bcrypt",
            "new_key": new_key_value,
            "created_at": api_key_record.created_at,
            "is_active": api_key_record.is_active,
        }

        migrations.append(migration_info)

        if not dry_run:
            # Update database record
            api_key_record.key_hash = new_key_hash
            logger.info("  ✓ Updated key hash to bcrypt")
        else:
            logger.info("  [DRY RUN] Would update key hash to bcrypt")

    if not dry_run:
        # Commit all changes
        db_session.commit()
        logger.info(f"✓ Successfully migrated {len(migrations)} API keys")
    else:
        logger.info(f"[DRY RUN] Would migrate {len(migrations)} API keys")

    return migrations


def save_migration_results(migrations: list[dict], output_file: str) -> None:
    """Save migration results to file.

    Args:
        migrations: List of migration info dictionaries
        output_file: Path to output file
    """
    output_path = Path(output_file)

    logger.info(f"Saving migration results to: {output_path.absolute()}")

    with open(output_path, "w") as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write("API KEY MIGRATION RESULTS\n")
        f.write(f"Migration Date: {datetime.now(UTC).isoformat()}\n")
        f.write(f"Total Keys Migrated: {len(migrations)}\n")
        f.write("=" * 80 + "\n\n")

        f.write("⚠️  SECURITY WARNING ⚠️\n")
        f.write("This file contains metadata for migrated API keys.\n")
        f.write("New API keys are NOT stored in this file for security.\n")
        f.write("- Distribute keys to users via secure channels ONLY\n")
        f.write("- Full keys were displayed ONCE on the console during migration\n")
        f.write("- Delete this file after distribution is verified\n")
        f.write("- Never commit this file to version control\n")
        f.write("- Old API keys are now INVALID\n\n")

        f.write("=" * 80 + "\n\n")

        # Write each migration
        for idx, migration in enumerate(migrations, 1):
            f.write(f"KEY {idx} of {len(migrations)}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Key ID:       {migration['key_id']}\n")
            f.write(f"Key Name:     {migration['key_name']}\n")
            f.write(f"User ID:      {migration['user_id']}\n")
            f.write(f"User Email:   {migration['user_email']}\n")
            f.write(
                f"Status:       {'ACTIVE' if migration['is_active'] else 'INACTIVE'}\n"
            )
            f.write(f"Created:      {migration['created_at']}\n")
            f.write(f"Old Hash:     {migration['old_hash_type']}\n")
            f.write(f"New Hash:     {migration['new_hash_type']}\n")
            f.write("\n")
            f.write("NEW API KEY:  [NOT STORED IN THIS FILE - SEE CONSOLE OUTPUT]\n")
            f.write("\n")
            f.write("Action Required:\n")
            f.write(
                f"  1. Send this key to {migration['user_email']} via secure channel\n"
            )
            f.write("  2. User must update their application with new key\n")
            f.write("  3. User should test authentication\n")
            f.write("\n")
            f.write("=" * 80 + "\n\n")

        # Write footer
        f.write("NEXT STEPS:\n")
        f.write("1. Distribute new API keys to all users\n")
        f.write("2. Verify all users can authenticate with new keys\n")
        f.write("3. Monitor authentication logs for failures\n")
        f.write("4. Delete this file securely (shred/secure delete)\n")
        f.write("5. Update documentation with new key format\n\n")

    # Set restrictive permissions (owner read/write only)
    output_path.chmod(0o600)

    logger.info(f"✓ Migration results saved to: {output_path.absolute()}")
    logger.info("✓ File permissions set to 0600 (owner read/write only)")


def main() -> int:
    """Main migration function.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    args = parse_arguments()

    logger.info("=" * 80)
    logger.info("API Key Migration: SHA256 → Bcrypt")
    logger.info("=" * 80)

    if args.dry_run:
        logger.warning("DRY RUN MODE - No changes will be made")

    try:
        # Get database session
        db_session = get_database_session(args.database)

        # Confirm migration (unless dry run)
        if not args.dry_run:
            logger.warning("\n⚠️  WARNING: This is a ONE-WAY migration!")
            logger.warning("Old API keys will be PERMANENTLY INVALIDATED")
            logger.warning(
                "New keys will be generated and must be distributed to users\n"
            )

            response = input("Proceed with migration? (yes/no): ")
            if response.lower() != "yes":
                logger.info("Migration cancelled by user")
                return 0

        # Perform migration
        logger.info("\nStarting migration...")
        migrations = migrate_api_keys(db_session, dry_run=args.dry_run)

        if not migrations:
            logger.info("No API keys to migrate")
            return 0

        # Save results
        if migrations:
            # Display new API keys once on console (not written to disk)
            if not args.dry_run:
                print("\n" + "=" * 80)
                print("NEW API KEYS (DISPLAYED ONCE - DO NOT CLOSE WINDOW UNTIL SAVED)")
                print("=" * 80)
                print(
                    "The following API keys are displayed ONLY in this console output.\n"
                    "They are NOT stored on disk. Distribute them to users via secure\n"
                    "channels and then securely discard this output.\n"
                )
                for idx, migration in enumerate(migrations, 1):
                    print(f"KEY {idx} of {len(migrations)}")
                    print("-" * 40)
                    print(f"Key ID:       {migration['key_id']}")
                    print(f"Key Name:     {migration['key_name']}")
                    print(f"User Email:   {migration['user_email']}")
                    print(f"NEW API KEY:  {migration['new_key']}")
                    print("-" * 40 + "\n")
                print("=" * 80 + "\n")

            # Create metadata-only version for disk storage (no raw keys)
            # This ensures CodeQL taint tracking is satisfied as sensitive keys
            # never reach the file-writing function scope.
            metadata_only_migrations = [
                {k: v for k, v in m.items() if k != "new_key"} for m in migrations
            ]
            save_migration_results(metadata_only_migrations, args.output)

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("MIGRATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Keys Migrated: {len(migrations)}")

        if not args.dry_run:
            logger.info(f"Results Saved To: {Path(args.output).absolute()}")
            logger.info("\n⚠️  CRITICAL NEXT STEPS:")
            logger.info("1. Read the output file and distribute new keys to users")
            logger.info("2. Verify users can authenticate with new keys")
            logger.info("3. Delete the output file securely after distribution")
            logger.info("4. Monitor logs for authentication failures")
        else:
            logger.info("\n[DRY RUN] No changes were made to the database")
            logger.info("Run without --dry-run to perform actual migration")

        logger.info("=" * 80)

        # Close database session
        db_session.close()

        return 0

    except KeyboardInterrupt:
        logger.warning("\nMigration interrupted by user")
        return 1

    except Exception as e:
        logger.error(f"\n❌ Migration failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
