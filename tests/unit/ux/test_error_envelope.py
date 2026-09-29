"""Tests for the UserFacingError envelope (epic #5968, Phase 5.1)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.shared.python.ux.error_envelope import (
    ErrorCatalog,
    UserFacingError,
    UserFacingErrorError,
    load_error_catalog,
)

pytestmark = pytest.mark.unit


# ---- UserFacingError construction -----------------------------------


def _good(**overrides) -> UserFacingError:
    base = {
        "code": "invalid_timestep",
        "title": "Timestep is out of range",
        "what_happened": "Timestep {value} s is below the minimum 1e-6 s.",
        "why": "Smaller-than-machine-epsilon timesteps cause integrator failure.",
        "how_to_fix": "Set timestep between 1e-6 and 1.0 (try 0.002 for MuJoCo).",
        "field_id": "simulation.timestep",
        "docs_url": None,
        "retriable": True,
    }
    base.update(overrides)
    return UserFacingError(**base)


def test_user_facing_error_constructs():
    err = _good()
    assert err.code == "invalid_timestep"
    assert err.field_id == "simulation.timestep"
    assert err.retriable is True


def test_user_facing_error_code_must_be_snake_case():
    with pytest.raises((ValueError, UserFacingErrorError)):
        _good(code="Invalid Timestep")


def test_user_facing_error_title_required():
    with pytest.raises((ValueError, UserFacingErrorError)):
        _good(title="")


def test_user_facing_error_what_happened_required():
    with pytest.raises((ValueError, UserFacingErrorError)):
        _good(what_happened="")


def test_user_facing_error_how_to_fix_required():
    with pytest.raises((ValueError, UserFacingErrorError)):
        _good(how_to_fix="")


def test_user_facing_error_field_id_must_be_dotted_when_present():
    with pytest.raises((ValueError, UserFacingErrorError)):
        _good(field_id="not dotted")


def test_user_facing_error_field_id_may_be_none():
    err = _good(field_id=None)
    assert err.field_id is None


def test_user_facing_error_is_frozen():
    err = _good()
    with pytest.raises((AttributeError, TypeError)):
        err.title = "x"  # type: ignore[misc]


def test_user_facing_error_format_substitutes_what_happened():
    err = _good()
    formatted = err.format(value=1e-12)
    assert "1e-12" in formatted.what_happened
    # other fields untouched
    assert formatted.why == err.why
    assert formatted.how_to_fix == err.how_to_fix


def test_user_facing_error_format_raises_on_missing_substitution():
    err = _good(what_happened="value is {value}, expected {expected}")
    with pytest.raises((KeyError, UserFacingErrorError)):
        err.format(value=0.5)


def test_user_facing_error_to_dict_includes_all_user_visible_fields():
    err = _good()
    d = err.to_dict()
    for key in (
        "code",
        "title",
        "what_happened",
        "why",
        "how_to_fix",
        "field_id",
        "docs_url",
        "retriable",
    ):
        assert key in d


def test_user_facing_error_from_dict_parses_string_boolean_fields():
    err = UserFacingError.from_dict(
        {
            "code": "invalid_timestep",
            "title": "Timestep is out of range",
            "what_happened": "Timestep 0 s is invalid.",
            "why": "Integrator cannot make progress.",
            "how_to_fix": "Pick a larger timestep.",
            "field_id": "simulation.timestep",
            "docs_url": None,
            "retriable": "false",
        }
    )
    assert err.retriable is False


def test_user_facing_error_from_dict_rejects_non_boolean_like_retriable():
    with pytest.raises(UserFacingErrorError, match="retriable must be"):
        UserFacingError.from_dict(
            {
                "code": "invalid_timestep",
                "title": "Timestep is out of range",
                "what_happened": "Timestep 0 s is invalid.",
                "why": "Integrator cannot make progress.",
                "how_to_fix": "Pick a larger timestep.",
                "field_id": "simulation.timestep",
                "docs_url": None,
                "retriable": "sometimes",
            }
        )


def test_user_facing_error_error_is_value_error_subclass():
    assert issubclass(UserFacingErrorError, ValueError)


# ---- ErrorCatalog loader --------------------------------------------


def _write_yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "error_messages.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def test_load_error_catalog_reads_yaml(tmp_path):
    yaml_text = """
    errors:
      - code: invalid_timestep
        title: Timestep is out of range
        what_happened: "Timestep {value} s is below the minimum."
        why: Integrator cannot make progress.
        how_to_fix: "Set timestep between 1e-6 and 1.0."
        field_id: simulation.timestep
        docs_url: ~
        retriable: true
    """
    catalog = load_error_catalog(_write_yaml(tmp_path, yaml_text))
    assert isinstance(catalog, ErrorCatalog)
    assert catalog.get("invalid_timestep").title == "Timestep is out of range"


def test_load_error_catalog_rejects_duplicate_codes(tmp_path):
    yaml_text = """
    errors:
      - code: x
        title: T
        what_happened: w
        why: y
        how_to_fix: f
        field_id: ~
        docs_url: ~
        retriable: true
      - code: x
        title: T
        what_happened: w
        why: y
        how_to_fix: f
        field_id: ~
        docs_url: ~
        retriable: true
    """
    with pytest.raises((ValueError, UserFacingErrorError)):
        load_error_catalog(_write_yaml(tmp_path, yaml_text))


def test_error_catalog_get_raises_on_unknown_code():
    catalog = ErrorCatalog(())
    with pytest.raises(KeyError):
        catalog.get("nope")


def test_error_catalog_contains_and_length():
    err = _good()
    catalog = ErrorCatalog((err,))
    assert "invalid_timestep" in catalog
    assert "nope" not in catalog
    assert len(catalog) == 1


def test_error_catalog_iter_is_deterministic_by_code():
    a = _good(code="a_first")
    b = _good(code="b_second")
    catalog = ErrorCatalog((b, a))
    assert [e.code for e in catalog] == ["a_first", "b_second"]
