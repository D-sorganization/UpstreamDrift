"""Tests for auth/dependencies — API key lookup field alignment (issue #2447).

Verifies that the dependency layer queries APIKey.key_prefix (the real ORM column),
not the non-existent APIKey.prefix_hash.
"""

import pytest

try:
    import sqlalchemy  # noqa: F401

    _HAS_SQLALCHEMY = True
except ImportError:
    _HAS_SQLALCHEMY = False

requires_sqlalchemy = pytest.mark.skipif(
    not _HAS_SQLALCHEMY, reason="sqlalchemy not installed"
)


class TestAPIKeyOrmFieldAlignment:
    """APIKey ORM model and dependency layer use the same column name."""

    @requires_sqlalchemy
    def test_apikey_has_key_prefix_column(self) -> None:
        """ORM model defines key_prefix column (not prefix_hash)."""
        from src.api.auth.models import APIKey

        col_names = {c.key for c in APIKey.__table__.columns}
        assert "key_prefix" in col_names

    @requires_sqlalchemy
    def test_apikey_does_not_have_prefix_hash_column(self) -> None:
        """ORM model does not define prefix_hash — that name is wrong."""
        from src.api.auth.models import APIKey

        col_names = {c.key for c in APIKey.__table__.columns}
        assert "prefix_hash" not in col_names

    def test_dependency_references_key_prefix_not_prefix_hash(self) -> None:
        """_lookup_api_key_by_prefix uses APIKey.key_prefix in its filter call."""
        from pathlib import Path

        dep_file = (
            Path(__file__).parents[3] / "src" / "api" / "auth" / "dependencies.py"
        )
        source = dep_file.read_text(encoding="utf-8")
        # After fix: the filter references key_prefix, not prefix_hash
        assert "key_prefix" in source, (
            "dependencies.py must filter on APIKey.key_prefix"
        )

    def test_dependency_does_not_reference_apikey_prefix_hash(self) -> None:
        """No ORM attribute access APIKey.prefix_hash — that column does not exist."""
        from pathlib import Path

        dep_file = (
            Path(__file__).parents[3] / "src" / "api" / "auth" / "dependencies.py"
        )
        source = dep_file.read_text(encoding="utf-8")
        # The local variable "prefix_hash" and function "compute_prefix_hash" are fine;
        # only the ORM column access APIKey.prefix_hash is wrong (column is key_prefix).
        assert "APIKey.prefix_hash" not in source, (
            "dependencies.py must not access APIKey.prefix_hash (non-existent column); "
            "use APIKey.key_prefix instead"
        )

    @requires_sqlalchemy
    def test_lookup_does_not_raise_attribute_error(self) -> None:
        """Filter on key_prefix succeeds; filter on prefix_hash raises AttributeError."""
        from unittest.mock import MagicMock

        from fastapi import HTTPException
        from sqlalchemy.orm import Session
        from src.api.auth.dependencies import _lookup_api_key_by_prefix

        mock_db = MagicMock(spec=Session)
        mock_chain = MagicMock()
        mock_db.query.return_value = mock_chain
        mock_chain.filter.return_value = mock_chain
        mock_chain.all.return_value = []  # No matching keys → 401

        # Should raise 401, NOT AttributeError for missing column
        with pytest.raises(HTTPException) as exc_info:
            _lookup_api_key_by_prefix("gms_12345678_dummy", mock_db)

        assert exc_info.value.status_code == 401
        # filter() must have been called (not short-circuited by AttributeError)
        assert mock_chain.filter.called
