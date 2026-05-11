"""Tests for ``body_part_viz.bindings``.

Covers:
- :class:`BindingKind` enum members and string round-trip.
- :class:`MarkerBinding` invariants (per-kind marker count, name validity,
  rest dimensions, quaternion unit norm).
"""

from __future__ import annotations

import pytest

from src.shared.python.body_part_viz.bindings import (
    BindingKind,
    MarkerBinding,
)

# ---------------------------------------------------------------------------
# BindingKind enum
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_binding_kind_values() -> None:
    assert BindingKind.BETWEEN_TWO.value == "between_two"
    assert BindingKind.CLUSTER.value == "cluster"
    assert BindingKind.ON_MARKER.value == "on_marker"


@pytest.mark.unit
def test_binding_kind_string_roundtrip() -> None:
    """BindingKind inherits ``str`` for transparent JSON serialisation."""
    for kind in BindingKind:
        assert kind == BindingKind(kind.value)
        assert str(kind.value) == kind.value


# ---------------------------------------------------------------------------
# MarkerBinding — happy paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_between_two_happy_path() -> None:
    b = MarkerBinding(
        kind=BindingKind.BETWEEN_TWO,
        marker_names=("shoulder_l", "elbow_l"),
        rest_dimensions=(0.30,),
    )
    assert b.kind is BindingKind.BETWEEN_TWO
    assert b.marker_names == ("shoulder_l", "elbow_l")
    assert b.rest_dimensions == (0.30,)
    assert b.rest_orientation_quat == (1.0, 0.0, 0.0, 0.0)


@pytest.mark.unit
def test_cluster_happy_path_three_markers() -> None:
    b = MarkerBinding(
        kind=BindingKind.CLUSTER,
        marker_names=("a", "b", "c"),
        rest_dimensions=(0.10, 0.05, 0.05),
    )
    assert b.kind is BindingKind.CLUSTER


@pytest.mark.unit
def test_cluster_happy_path_more_markers() -> None:
    """CLUSTER accepts arbitrary marker counts >= 3."""
    b = MarkerBinding(
        kind=BindingKind.CLUSTER,
        marker_names=("a", "b", "c", "d", "e"),
    )
    assert len(b.marker_names) == 5


@pytest.mark.unit
def test_on_marker_happy_path() -> None:
    b = MarkerBinding(
        kind=BindingKind.ON_MARKER,
        marker_names=("head_top",),
        rest_dimensions=(0.20, 0.20, 0.25),
    )
    assert b.marker_names == ("head_top",)


# ---------------------------------------------------------------------------
# MarkerBinding — frozen invariant
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_marker_binding_is_frozen() -> None:
    b = MarkerBinding(BindingKind.ON_MARKER, ("head",))
    with pytest.raises(Exception):  # noqa: B017 - FrozenInstanceError
        b.kind = BindingKind.BETWEEN_TWO  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MarkerBinding — DbC failures: marker count
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_between_two_rejects_one_marker() -> None:
    with pytest.raises(ValueError, match="exactly 2 markers"):
        MarkerBinding(BindingKind.BETWEEN_TWO, ("solo",))


@pytest.mark.unit
def test_between_two_rejects_three_markers() -> None:
    with pytest.raises(ValueError, match="exactly 2 markers"):
        MarkerBinding(BindingKind.BETWEEN_TWO, ("a", "b", "c"))


@pytest.mark.unit
def test_cluster_rejects_two_markers() -> None:
    with pytest.raises(ValueError, match="at least 3 markers"):
        MarkerBinding(BindingKind.CLUSTER, ("a", "b"))


@pytest.mark.unit
def test_on_marker_rejects_two_markers() -> None:
    with pytest.raises(ValueError, match="exactly 1 marker"):
        MarkerBinding(BindingKind.ON_MARKER, ("a", "b"))


# ---------------------------------------------------------------------------
# MarkerBinding — DbC failures: marker names
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_marker_names_must_be_strings() -> None:
    with pytest.raises(ValueError, match="must be non-empty strings"):
        MarkerBinding(BindingKind.BETWEEN_TWO, ("a", 42))  # type: ignore[arg-type]


@pytest.mark.unit
def test_marker_names_must_be_non_empty() -> None:
    with pytest.raises(ValueError, match="must be non-empty"):
        MarkerBinding(BindingKind.BETWEEN_TWO, ("a", ""))


# ---------------------------------------------------------------------------
# MarkerBinding — DbC failures: kind type
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_kind_must_be_binding_kind_enum() -> None:
    with pytest.raises(TypeError, match="kind must be BindingKind"):
        MarkerBinding("between_two", ("a", "b"))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# MarkerBinding — DbC failures: rest dimensions
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rest_dimensions_rejects_negative() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        MarkerBinding(
            BindingKind.BETWEEN_TWO,
            ("a", "b"),
            rest_dimensions=(-0.1,),
        )


@pytest.mark.unit
def test_rest_dimensions_rejects_zero() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        MarkerBinding(
            BindingKind.BETWEEN_TWO,
            ("a", "b"),
            rest_dimensions=(0.0,),
        )


@pytest.mark.unit
def test_rest_dimensions_rejects_inf() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        MarkerBinding(
            BindingKind.BETWEEN_TWO,
            ("a", "b"),
            rest_dimensions=(float("inf"),),
        )


@pytest.mark.unit
def test_rest_dimensions_rejects_nan() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        MarkerBinding(
            BindingKind.BETWEEN_TWO,
            ("a", "b"),
            rest_dimensions=(float("nan"),),
        )


# ---------------------------------------------------------------------------
# MarkerBinding — DbC failures: quaternion
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_quaternion_must_be_unit_norm() -> None:
    with pytest.raises(ValueError, match="unit-norm"):
        MarkerBinding(
            BindingKind.ON_MARKER,
            ("a",),
            rest_orientation_quat=(2.0, 0.0, 0.0, 0.0),
        )


@pytest.mark.unit
def test_quaternion_accepts_identity() -> None:
    MarkerBinding(
        BindingKind.ON_MARKER,
        ("a",),
        rest_orientation_quat=(1.0, 0.0, 0.0, 0.0),
    )


@pytest.mark.unit
def test_quaternion_accepts_rotated_unit() -> None:
    """A 90° rotation about z: q = (cos(45°), 0, 0, sin(45°))."""
    import math

    c = math.cos(math.radians(45))
    s = math.sin(math.radians(45))
    MarkerBinding(
        BindingKind.ON_MARKER,
        ("a",),
        rest_orientation_quat=(c, 0.0, 0.0, s),
    )


@pytest.mark.unit
def test_quaternion_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="4-tuple"):
        MarkerBinding(
            BindingKind.ON_MARKER, ("a",), rest_orientation_quat=(1.0, 0.0, 0.0)
        )  # type: ignore[arg-type]


@pytest.mark.unit
def test_quaternion_rejects_inf_components() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        MarkerBinding(
            BindingKind.ON_MARKER,
            ("a",),
            rest_orientation_quat=(float("inf"), 0.0, 0.0, 0.0),
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_positive_rest_dimensions_accepts_empty() -> None:
    """Empty tuple is valid (some shapes have no rest dimensions)."""
    MarkerBinding(BindingKind.ON_MARKER, ("a",), rest_dimensions=())


@pytest.mark.unit
def test_check_positive_rest_dimensions_accepts_all_positive() -> None:
    MarkerBinding(BindingKind.CLUSTER, ("a", "b", "c"), rest_dimensions=(0.1, 0.2, 0.3))
