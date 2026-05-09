"""Tests for src.shared.python.core.version (Issues #1949, #1744)."""

from __future__ import annotations

from src.shared.python.core.version import (
    FEATURES,
    PROFESSIONAL_FEATURES,
    SUPPORTED_ENGINES,
    __author__,
    __description__,
    __license__,
    __title__,
    __version__,
    __version_info__,
)

# ---------------------------------------------------------------------------
# Version string
# ---------------------------------------------------------------------------


class TestVersion:
    def test_version_is_string(self) -> None:
        assert isinstance(__version__, str)

    def test_version_non_empty(self) -> None:
        assert len(__version__) > 0

    def test_version_has_dots(self) -> None:
        assert "." in __version__

    def test_version_info_is_tuple(self) -> None:
        assert isinstance(__version_info__, tuple)

    def test_version_info_has_three_parts(self) -> None:
        assert len(__version_info__) == 3

    def test_version_info_values_are_ints(self) -> None:
        assert all(isinstance(v, int) for v in __version_info__)

    def test_version_string_matches_info(self) -> None:
        # "1.0.0" matches (1, 0, 0)
        parts = tuple(int(x) for x in __version__.split("."))
        assert parts == __version_info__


# ---------------------------------------------------------------------------
# Metadata strings
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_title_is_string(self) -> None:
        assert isinstance(__title__, str)

    def test_description_is_string(self) -> None:
        assert isinstance(__description__, str)

    def test_author_is_string(self) -> None:
        assert isinstance(__author__, str)

    def test_license_is_string(self) -> None:
        assert isinstance(__license__, str)

    def test_title_non_empty(self) -> None:
        assert len(__title__) > 0


# ---------------------------------------------------------------------------
# FEATURES dict
# ---------------------------------------------------------------------------


class TestFeatures:
    def test_version_is_dict(self) -> None:
        assert isinstance(FEATURES, dict)

    def test_version_non_empty(self) -> None:
        assert len(FEATURES) > 0

    def test_all_values_are_bool(self) -> None:
        assert all(isinstance(v, bool) for v in FEATURES.values())

    def test_api_server_feature_exists(self) -> None:
        assert "api_server" in FEATURES


# ---------------------------------------------------------------------------
# SUPPORTED_ENGINES
# ---------------------------------------------------------------------------


class TestSupportedEngines:
    def test_is_list(self) -> None:
        assert isinstance(SUPPORTED_ENGINES, list)

    def test_version_non_empty(self) -> None:
        assert len(SUPPORTED_ENGINES) > 0

    def test_mujoco_supported(self) -> None:
        assert "mujoco" in SUPPORTED_ENGINES

    def test_all_strings(self) -> None:
        assert all(isinstance(e, str) for e in SUPPORTED_ENGINES)


# ---------------------------------------------------------------------------
# PROFESSIONAL_FEATURES
# ---------------------------------------------------------------------------


class TestProfessionalFeatures:
    def test_is_list(self) -> None:
        assert isinstance(PROFESSIONAL_FEATURES, list)

    def test_version_non_empty(self) -> None:
        assert len(PROFESSIONAL_FEATURES) > 0

    def test_all_strings(self) -> None:
        assert all(isinstance(f, str) for f in PROFESSIONAL_FEATURES)
