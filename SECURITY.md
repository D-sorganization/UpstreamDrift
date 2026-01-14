# Security Policy

## Supported Versions

We actively support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in the Golf Modeling Suite, please report it to us responsibly.

### How to Report

1. **DO NOT** create a public GitHub issue for security vulnerabilities
2. Email security concerns to: [project maintainer email]
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if available)

### What to Expect

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 1 week
- **Fix Timeline**: Critical issues within 2 weeks, others within 30 days
- **Disclosure**: Coordinated disclosure after fix is available

## Security Measures

### Authentication & Authorization

- **Password Storage**: Bcrypt hashing (industry standard)
- **API Keys**: Bcrypt hashing with `gms_` prefix
- **JWT Tokens**: HS256 signing with secure secret keys
- **Session Management**: 30-minute access tokens, 30-day refresh tokens
- **Role-Based Access Control**: Admin, Enterprise, Professional, Researcher, Free tiers

### API Security

- **Rate Limiting**: Implemented via SlowAPI
- **Input Validation**: Pydantic models for all endpoints
- **SQL Injection**: Protected via SQLAlchemy ORM
- **XSS Protection**: Secure XML parsing with defusedxml
- **Path Traversal**: Validated in secure_subprocess module

### Dependency Security

- **Automated Auditing**: pip-audit runs in CI/CD (blocking)
- **Version Pinning**: Core dependencies version-locked
- **Regular Updates**: Monthly security patch reviews
- **Vulnerability Scanning**: Automated with Dependabot (if enabled)

### Code Security

- **Static Analysis**: Ruff, Black, MyPy in CI
- **Pre-commit Hooks**: Prevent secrets, large files
- **Archive Code**: Legacy code with vulnerabilities clearly marked (see `/engines/pendulum_models/archive/README_SECURITY_WARNING.md`)

## Security Best Practices for Users

### Environment Variables (CRITICAL)

Never commit these to version control:

```bash
# Authentication
export GOLF_API_SECRET_KEY="[64+ character random string]"
export GOLF_ADMIN_PASSWORD="[strong password]"

# Database
export DATABASE_URL="postgresql://user:pass@host/db"

# Environment
export ENVIRONMENT="production"
```

Generate secure secret keys:
```bash
# Linux/macOS
python3 -c "import secrets; print(secrets.token_urlsafe(64))"

# Or use openssl
openssl rand -base64 64
```

### Production Deployment Checklist

- [ ] Set `GOLF_API_SECRET_KEY` (64+ characters)
- [ ] Set `GOLF_ADMIN_PASSWORD` (strong password)
- [ ] Set `ENVIRONMENT=production`
- [ ] Use PostgreSQL (not SQLite) for production
- [ ] Enable HTTPS/TLS for API endpoints
- [ ] Configure firewall rules (only allow necessary ports)
- [ ] Set up log monitoring and alerting
- [ ] Enable database backups
- [ ] Review and set appropriate rate limits
- [ ] Disable debug mode (`echo=False` in database.py)

### API Key Management

**Creating API Keys** (Admin only):
```python
import secrets
api_key = f"gms_{secrets.token_urlsafe(32)}"
# Store hash in database, give plaintext to user ONCE
```

**Best Practices**:
- Rotate API keys every 90 days
- Use separate keys for different applications
- Revoke unused keys immediately
- Never commit API keys to version control
- Use environment variables or secret management systems

### File Upload Security

If you add file upload functionality:
- Validate file types (whitelist, not blacklist)
- Scan uploads with antivirus
- Store uploads outside web root
- Generate random filenames
- Set size limits
- Check for malware signatures

## Known Security Considerations

### Archive Directory

⚠️ **WARNING**: The `/engines/pendulum_models/archive/` directory contains legacy code with known vulnerabilities:

- **Unsafe eval() usage**: Code injection risk
- **No security updates**: Unmaintained code
- **Not for production**: Historical reference only

**See**: `/engines/pendulum_models/archive/README_SECURITY_WARNING.md`

### Expression Evaluation

**SECURE** (Use these):
- `simpleeval` library (already in dependencies)
- Validated mathematical expressions only

**INSECURE** (Never use):
- `eval()` or `exec()` on user input
- Code from archive directory

## Security Updates

### Recent Security Fixes (January 2026)

1. **API Key Hashing**: Upgraded from SHA256 to bcrypt
2. **JWT Timezone**: Fixed deprecated `datetime.utcnow()`
3. **Password Logging**: Removed plaintext password from logs
4. **CI Security Audit**: Made pip-audit blocking (was advisory)
5. **Archive Isolation**: Added security warnings to legacy code

### Upgrade Path

If you're using older versions with SHA256 API key hashing:

1. Generate new API keys for all users
2. Update to version 1.0.0+
3. Revoke old API keys
4. Users must use new keys with bcrypt hashing

**Note**: Old SHA256 hashed keys will NOT work after upgrade.

## Security Headers (for API deployments)

Add these headers to your web server configuration:

```nginx
# Nginx example
add_header X-Content-Type-Options "nosniff" always;
add_header X-Frame-Options "DENY" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
add_header Content-Security-Policy "default-src 'self'" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
```

## Compliance

This project follows:

- **OWASP Top 10**: Protection against common vulnerabilities
- **CWE/SANS Top 25**: Most dangerous software weaknesses
- **Python Security Best Practices**: As per Python Security team
- **FastAPI Security Guidelines**: Official FastAPI recommendations

## Security Testing

### Automated Tests

```bash
# Run security-focused tests
pytest tests/integration/test_phase1_security_integration.py -v

# Run static security analysis
ruff check . --select S  # Security rules

# Audit dependencies
pip-audit
```

### Manual Security Review

Before major releases:
1. Review all authentication/authorization code
2. Check for hardcoded secrets
3. Verify input validation on all endpoints
4. Test rate limiting effectiveness
5. Review database queries for injection risks
6. Check file permissions and access controls

## Contact

For security concerns: [Maintainer contact information]

For general issues: https://github.com/D-sorganization/Golf_Modeling_Suite/issues

---

**Last Updated**: January 13, 2026
**Version**: 1.0.0
