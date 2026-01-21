# Security: Audit URL open for permitted schemes (B310)

**Labels:** security, jules:sentinel

## Description
Bandit flagged usage of `urllib.request.urlopen` which might allow access to local files (file://) or other schemes if not sanitized.

## Locations
- `shared/python/standard_models.py:151`
- `tools/urdf_generator/model_library.py:233`

## Remediation
Validate the URL scheme before opening, ensuring it is http/https if that is the intent.
