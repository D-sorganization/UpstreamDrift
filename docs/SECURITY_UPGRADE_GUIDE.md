# Security Upgrade Guide - January 2026

This guide helps you upgrade from older versions to the security-enhanced version 1.0.0.

## Overview of Security Fixes

### Critical Fixes

1. **API Key Storage** (CRITICAL)
   - **Old**: SHA256 fast hash (vulnerable to brute force)
   - **New**: Bcrypt slow hash (industry standard)
   - **Impact**: ALL existing API keys must be regenerated

2. **JWT Token Generation** (MEDIUM)
   - **Old**: `datetime.utcnow()` (deprecated in Python 3.12+)
   - **New**: `datetime.now(timezone.utc)` (timezone-aware)
   - **Impact**: Improved compatibility with Python 3.12+

3. **Password Logging** (MEDIUM)
   - **Old**: Admin password logged in plaintext
   - **New**: No password logging, instructions provided instead
   - **Impact**: Prevents password leakage via logs

4. **CI Security Audit** (LOW)
   - **Old**: pip-audit advisory (non-blocking)
   - **New**: pip-audit blocking (fails on vulnerabilities)
   - **Impact**: Prevents vulnerable dependencies from being merged

5. **Archive Code** (HIGH)
   - **Old**: Unsafe eval() code accessible
   - **New**: Clear security warnings, excluded from statistics
   - **Impact**: Prevents accidental use of vulnerable legacy code

## Migration Steps

### Step 1: Backup Your Database

```bash
# SQLite
cp golf_modeling_suite.db golf_modeling_suite.db.backup

# PostgreSQL
pg_dump golf_db > golf_db_backup.sql
```

### Step 2: Update Code

```bash
git pull origin main
pip install --upgrade -r requirements.txt
```

### Step 3: Regenerate API Keys (REQUIRED)

**This is REQUIRED - old API keys will NOT work!**

#### Option A: Automatic Migration Script

We provide a migration script to help:

```python
# scripts/migrate_api_keys.py
import secrets
from api.database import SessionLocal
from api.auth.models import APIKey
from api.auth.security import security_manager
from passlib.context import CryptContext

def migrate_api_keys():
    """Regenerate all API keys with bcrypt hashing."""
    db = SessionLocal()
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    try:
        # Get all API keys
        api_keys = db.query(APIKey).all()

        print(f"Found {len(api_keys)} API keys to migrate")
        print("\n⚠️  WARNING: Save these new keys securely! They will only be shown ONCE.\n")

        migrations = []

        for key in api_keys:
            # Generate new API key
            new_key_value = f"gms_{secrets.token_urlsafe(32)}"

            # Hash with bcrypt
            new_key_hash = pwd_context.hash(new_key_value)

            # Update database
            key.key_hash = new_key_hash

            migrations.append({
                'user_id': key.user_id,
                'name': key.name,
                'old_id': key.id,
                'new_key': new_key_value
            })

        # Commit all changes
        db.commit()

        # Display new keys
        print("=" * 80)
        print("NEW API KEYS (save these securely!):")
        print("=" * 80)
        for m in migrations:
            print(f"\nUser ID: {m['user_id']}")
            print(f"Key Name: {m['name']}")
            print(f"New Key: {m['new_key']}")
            print("-" * 80)

        print(f"\n✅ Successfully migrated {len(migrations)} API keys")
        print("⚠️  Distribute new keys to users IMMEDIATELY")

    except Exception as e:
        db.rollback()
        print(f"❌ Migration failed: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    migrate_api_keys()
```

Run the migration:
```bash
python scripts/migrate_api_keys.py > new_api_keys.txt
chmod 600 new_api_keys.txt  # Protect the file
```

#### Option B: Manual Regeneration

For each user:

1. Revoke old API key via admin interface
2. Generate new API key
3. Provide new key to user securely (email, secure portal, etc.)

### Step 4: Update Environment Variables

Ensure these are set:

```bash
# Required for production
export GOLF_API_SECRET_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(64))')"
export GOLF_ADMIN_PASSWORD="[your-secure-admin-password]"
export ENVIRONMENT="production"
export DATABASE_URL="postgresql://user:pass@host/db"
```

Add to your `.env` file or systemd service:

```bash
# /etc/golf-modeling-suite/.env
GOLF_API_SECRET_KEY=your-64-char-secret-key-here
GOLF_ADMIN_PASSWORD=your-admin-password-here
ENVIRONMENT=production
DATABASE_URL=postgresql://user:pass@localhost/golf_db
```

### Step 5: Test Authentication

```bash
# Start the server
python start_api_server.py

# Test JWT authentication
curl -X POST http://localhost:8000/api/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin@golfmodelingsuite.com&password=your-password"

# Test API key authentication (use new key)
curl http://localhost:8000/api/health \
  -H "Authorization: Bearer gms_[your-new-api-key]"
```

### Step 6: Verify Security

```bash
# Run security audit
pip-audit

# Run security tests
pytest tests/integration/test_phase1_security_integration.py -v

# Check for secrets in code
git secrets --scan
```

## Breaking Changes

### API Key Format

- **Old**: Keys could be any format, hashed with SHA256
- **New**: Keys must start with `gms_`, hashed with bcrypt
- **Action**: Regenerate ALL API keys

### Database Schema

No schema changes required - `key_hash` column is reused with new hash format.

### Configuration

New required environment variables:
- `GOLF_API_SECRET_KEY` (64+ characters)
- `GOLF_ADMIN_PASSWORD` (for initial setup)

## Rollback Plan

If you need to rollback (NOT RECOMMENDED):

```bash
# Restore database backup
cp golf_modeling_suite.db.backup golf_modeling_suite.db

# Checkout previous version
git checkout v0.9.x

# Restore dependencies
pip install -r requirements.txt
```

**Warning**: Rollback exposes you to security vulnerabilities!

## Timeline

- **Immediate**: No grace period for API keys (security critical)
- **Week 1**: Migrate API keys
- **Week 2**: Verify all users have new keys
- **Week 3**: Revoke any remaining old-format keys

## Support

If you encounter issues during migration:

1. Check logs: `tail -f logs/api.log`
2. Verify environment variables are set
3. Test database connectivity
4. Review API key format (`gms_` prefix)
5. Create GitHub issue with migration errors

## Security Checklist

After migration, verify:

- [ ] All API keys regenerated and distributed
- [ ] `GOLF_API_SECRET_KEY` is 64+ characters
- [ ] `ENVIRONMENT=production` is set
- [ ] Database backups are working
- [ ] HTTPS/TLS is enabled
- [ ] Logs don't contain secrets
- [ ] pip-audit passes
- [ ] All security tests pass

## Post-Migration

1. **Monitor Authentication Failures**: Check for users using old API keys
2. **Review Logs**: Ensure no secrets are being logged
3. **Test Rate Limiting**: Verify it's working correctly
4. **Security Scan**: Run full security audit

## Questions?

- Review: `/SECURITY.md`
- Issues: https://github.com/D-sorganization/Golf_Modeling_Suite/issues
- Security: [Maintainer contact]

---

**Last Updated**: January 13, 2026
**Version**: 1.0.0
