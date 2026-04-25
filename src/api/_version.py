"""Single source of truth for the UpstreamDrift API version.

All version surfaces (server.py OpenAPI metadata, local_server.py, and the
root endpoint in routes/core.py) import __version__ from here so they stay
in sync with pyproject.toml.
"""

#: Canonical API version — must match pyproject.toml [project].version
__version__ = "2.1.0"
