"""Tests for the UX FieldMetadata registry (epic #5968, Phase 0.1).

TDD-first: these tests define the contract for ``FieldMetadata`` and the
loader/registry surface before implementation. They cover:

* Construction invariants (DbC preconditions raise on bad input).
* YAML round-trip — what the user writes in
  ``configs/ux/field_metadata.yaml`` is exactly what code sees.
* Registry semantics — lookup, iteration, producer/consumer graph,
  cycle detection (LOD: graph queries return tuples; callers do not
  reach into private state).
* Immutability — instances are hashable and frozen so they can be used
  as dict keys in coverage/ratchet logic without defensive copies (DRY).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.shared.python.ux.field_metadata import (
    FieldMetadata,
    FieldMetadataError,
    FieldRegistry,
    load_registry,
)

pytestmark = pytest.mark.unit


# ---- construction & invariants --------------------------------------


def _good_kwargs(**overrides):
    base = {
        "id": "simulation.timestep",
        "label": "Timestep",
        "short_help": "Integration step in seconds.",
        "long_help": "Smaller is more accurate but slower.",
        "units": "s",
        "valid_range": (1e-6, 1.0),
        "default": 0.002,
        "default_source": "MuJoCo recommended default for humanoid",
        "consumers": ("simulation.derived.fps",),
        "producers": (),
        "example": "0.002",
    }
    base.update(overrides)
    return base


def test_field_metadata_constructs_with_all_fields():
    fm = FieldMetadata(**_good_kwargs())
    assert fm.id == "simulation.timestep"
    assert fm.units == "s"
    assert fm.valid_range == (1e-6, 1.0)
    assert fm.consumers == ("simulation.derived.fps",)


def test_field_metadata_is_frozen_and_hashable():
    fm = FieldMetadata(**_good_kwargs())
    with pytest.raises((AttributeError, TypeError, Exception)):
        fm.label = "Mutated"  # type: ignore[misc]
    # hashable -> usable in sets / dict keys
    assert {fm, fm} == {fm}


def test_field_metadata_id_must_be_dotted_lowercase():
    with pytest.raises((ValueError, FieldMetadataError)):
        FieldMetadata(**_good_kwargs(id=""))
    with pytest.raises((ValueError, FieldMetadataError)):
        FieldMetadata(**_good_kwargs(id="Simulation.Timestep"))
    with pytest.raises((ValueError, FieldMetadataError)):
        FieldMetadata(**_good_kwargs(id="bad id with spaces"))


def test_field_metadata_short_help_is_capped():
    # Phase 0 spec: short_help <= 80 chars (tooltip-sized).
    too_long = "x" * 81
    with pytest.raises((ValueError, FieldMetadataError)):
        FieldMetadata(**_good_kwargs(short_help=too_long))


def test_field_metadata_numeric_range_is_ordered():
    with pytest.raises((ValueError, FieldMetadataError)):
        FieldMetadata(**_good_kwargs(valid_range=(1.0, 0.0)))


def test_field_metadata_default_must_be_within_numeric_range():
    with pytest.raises((ValueError, FieldMetadataError)):
        FieldMetadata(**_good_kwargs(default=10.0, valid_range=(0.0, 1.0)))


def test_field_metadata_enum_range_accepts_tuple_of_strings():
    fm = FieldMetadata(
        **_good_kwargs(
            id="actuator.control_type",
            units=None,
            valid_range=("constant", "polynomial", "pd_gains", "trajectory"),
            default="constant",
            consumers=(),
        )
    )
    assert fm.valid_range == ("constant", "polynomial", "pd_gains", "trajectory")


def test_field_metadata_default_must_be_in_enum():
    with pytest.raises((ValueError, FieldMetadataError)):
        FieldMetadata(
            **_good_kwargs(
                id="actuator.control_type",
                units=None,
                valid_range=("constant", "polynomial"),
                default="trajectory",
                consumers=(),
            )
        )


def test_field_metadata_to_dict_roundtrip():
    fm = FieldMetadata(**_good_kwargs())
    again = FieldMetadata.from_dict(fm.to_dict())
    assert again == fm


# ---- YAML loader ----------------------------------------------------


def _write_yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "field_metadata.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def test_load_registry_reads_yaml(tmp_path):
    yaml_text = """
    fields:
      - id: simulation.timestep
        label: Timestep
        short_help: Integration step in seconds.
        long_help: Smaller is more accurate but slower.
        units: s
        valid_range: [1.0e-6, 1.0]
        default: 0.002
        default_source: MuJoCo recommended for humanoid
        consumers: []
        producers: []
        example: "0.002"
    """
    reg = load_registry(_write_yaml(tmp_path, yaml_text))
    assert isinstance(reg, FieldRegistry)
    assert reg.get("simulation.timestep").default == 0.002


def test_load_registry_rejects_duplicate_ids(tmp_path):
    yaml_text = """
    fields:
      - id: simulation.timestep
        label: A
        short_help: ok
        long_help: ok
        units: s
        valid_range: [0.0, 1.0]
        default: 0.5
        default_source: t
        consumers: []
        producers: []
        example: "0.5"
      - id: simulation.timestep
        label: B
        short_help: ok
        long_help: ok
        units: s
        valid_range: [0.0, 1.0]
        default: 0.5
        default_source: t
        consumers: []
        producers: []
        example: "0.5"
    """
    with pytest.raises((ValueError, FieldMetadataError)):
        load_registry(_write_yaml(tmp_path, yaml_text))


def test_load_registry_validates_consumer_ids_exist(tmp_path):
    yaml_text = """
    fields:
      - id: simulation.timestep
        label: T
        short_help: ok
        long_help: ok
        units: s
        valid_range: [0.0, 1.0]
        default: 0.5
        default_source: t
        consumers: [simulation.does_not_exist]
        producers: []
        example: "0.5"
    """
    with pytest.raises((ValueError, FieldMetadataError)):
        load_registry(_write_yaml(tmp_path, yaml_text))


def test_load_registry_detects_cycle(tmp_path):
    yaml_text = """
    fields:
      - id: a.one
        label: A
        short_help: ok
        long_help: ok
        units: ~
        valid_range: [0.0, 1.0]
        default: 0.5
        default_source: t
        consumers: [a.two]
        producers: []
        example: "0.5"
      - id: a.two
        label: B
        short_help: ok
        long_help: ok
        units: ~
        valid_range: [0.0, 1.0]
        default: 0.5
        default_source: t
        consumers: [a.one]
        producers: []
        example: "0.5"
    """
    with pytest.raises((ValueError, FieldMetadataError)):
        load_registry(_write_yaml(tmp_path, yaml_text))


# ---- registry surface (LoD: returns flat tuples, no chain access) ----


def _two_field_registry() -> FieldRegistry:
    a = FieldMetadata(**_good_kwargs(id="a.upstream", consumers=("a.downstream",)))
    b = FieldMetadata(
        **_good_kwargs(
            id="a.downstream",
            consumers=(),
            producers=("a.upstream",),
            default=0.001,
        )
    )
    return FieldRegistry((a, b))


def test_registry_get_raises_on_missing():
    reg = _two_field_registry()
    with pytest.raises(KeyError):
        reg.get("not.a.real.field")


def test_registry_iter_fields_is_deterministic():
    reg = _two_field_registry()
    ids = [f.id for f in reg.iter_fields()]
    assert ids == sorted(ids)  # deterministic ordering is part of the contract


def test_registry_consumers_returns_full_metadata_not_just_ids():
    reg = _two_field_registry()
    downstream = reg.consumers_of("a.upstream")
    assert tuple(f.id for f in downstream) == ("a.downstream",)


def test_registry_producers_returns_full_metadata_not_just_ids():
    reg = _two_field_registry()
    upstream = reg.producers_of("a.downstream")
    assert tuple(f.id for f in upstream) == ("a.upstream",)


def test_registry_contains_supports_in_operator():
    reg = _two_field_registry()
    assert "a.upstream" in reg
    assert "nope" not in reg


def test_registry_length_matches_field_count():
    reg = _two_field_registry()
    assert len(reg) == 2


def test_field_metadata_error_is_value_error_subclass():
    # Domain-specific error must still satisfy `except ValueError:` for
    # generic input-validation callers (DRY against existing handlers).
    assert issubclass(FieldMetadataError, ValueError)
