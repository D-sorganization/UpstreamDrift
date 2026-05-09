"""Unit tests for body_part_viz contracts and dataclasses."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.body_part_viz import (
    BindingKind,
    BodyPartShape,
    FittedShape,
    MarkerBinding,
    ShapeFitter,
    ShapeRenderer,
    ShapeTheme,
)


# ---------- BindingKind --------------------------------------------------


def test_binding_kind_round_trip_through_str() -> None:
    for kind in BindingKind:
        as_str = str(kind)
        assert as_str == kind.value
        assert BindingKind(as_str) is kind


# ---------- MarkerBinding ------------------------------------------------


def test_marker_binding_between_two_happy_path() -> None:
    binding = MarkerBinding(
        kind=BindingKind.BETWEEN_TWO,
        marker_names=("a", "b"),
        rest_dimensions=(0.5,),
    )
    assert binding.marker_names == ("a", "b")


def test_marker_binding_between_two_wrong_count_raises() -> None:
    with pytest.raises(ValueError, match="BETWEEN_TWO"):
        MarkerBinding(kind=BindingKind.BETWEEN_TWO, marker_names=("a",))
    with pytest.raises(ValueError, match="BETWEEN_TWO"):
        MarkerBinding(kind=BindingKind.BETWEEN_TWO, marker_names=("a", "b", "c"))


def test_marker_binding_cluster_happy_path() -> None:
    binding = MarkerBinding(kind=BindingKind.CLUSTER, marker_names=("a", "b", "c"))
    assert binding.kind is BindingKind.CLUSTER


def test_marker_binding_cluster_too_few_raises() -> None:
    with pytest.raises(ValueError, match="CLUSTER"):
        MarkerBinding(kind=BindingKind.CLUSTER, marker_names=("a", "b"))


def test_marker_binding_on_marker_happy_path() -> None:
    binding = MarkerBinding(kind=BindingKind.ON_MARKER, marker_names=("h",))
    assert binding.marker_names == ("h",)


def test_marker_binding_on_marker_wrong_count_raises() -> None:
    with pytest.raises(ValueError, match="ON_MARKER"):
        MarkerBinding(kind=BindingKind.ON_MARKER, marker_names=("h1", "h2"))


def test_marker_binding_negative_rest_dimensions_raises() -> None:
    with pytest.raises(ValueError, match="rest_dimensions"):
        MarkerBinding(
            kind=BindingKind.BETWEEN_TWO,
            marker_names=("a", "b"),
            rest_dimensions=(-1.0,),
        )


def test_marker_binding_zero_rest_dimensions_raises() -> None:
    with pytest.raises(ValueError, match="rest_dimensions"):
        MarkerBinding(
            kind=BindingKind.BETWEEN_TWO,
            marker_names=("a", "b"),
            rest_dimensions=(0.0,),
        )


def test_marker_binding_non_unit_quaternion_raises() -> None:
    with pytest.raises(ValueError, match="unit-norm"):
        MarkerBinding(
            kind=BindingKind.BETWEEN_TWO,
            marker_names=("a", "b"),
            rest_orientation_quat=(2.0, 0.0, 0.0, 0.0),
        )


def test_marker_binding_quaternion_wrong_length_raises() -> None:
    with pytest.raises(ValueError, match="rest_orientation_quat"):
        MarkerBinding(
            kind=BindingKind.BETWEEN_TWO,
            marker_names=("a", "b"),
            rest_orientation_quat=(1.0, 0.0, 0.0),  # type: ignore[arg-type]
        )


def test_marker_binding_quaternion_non_finite_raises() -> None:
    with pytest.raises(ValueError, match="finite"):
        MarkerBinding(
            kind=BindingKind.BETWEEN_TWO,
            marker_names=("a", "b"),
            rest_orientation_quat=(float("nan"), 0.0, 0.0, 0.0),
        )


def test_marker_binding_kind_wrong_type_raises() -> None:
    with pytest.raises(TypeError, match="kind"):
        MarkerBinding(kind="between_two", marker_names=("a", "b"))  # type: ignore[arg-type]


def test_marker_binding_marker_names_list_normalized_to_tuple() -> None:
    # Lists are accepted but normalized to an immutable tuple so the caller
    # cannot mutate the binding after construction (issue #4775).
    src = ["a", "b"]
    binding = MarkerBinding(kind=BindingKind.BETWEEN_TWO, marker_names=src)  # type: ignore[arg-type]
    assert isinstance(binding.marker_names, tuple)
    assert binding.marker_names == ("a", "b")
    src.append("c")
    assert binding.marker_names == ("a", "b")


def test_marker_binding_marker_names_str_raises() -> None:
    # A bare string would otherwise be silently treated as a sequence of
    # single-character marker names (issue #4775).
    with pytest.raises(TypeError, match="marker_names"):
        MarkerBinding(kind=BindingKind.BETWEEN_TWO, marker_names="ab")  # type: ignore[arg-type]


def test_marker_binding_marker_names_non_iterable_raises() -> None:
    with pytest.raises(TypeError, match="marker_names"):
        MarkerBinding(kind=BindingKind.BETWEEN_TWO, marker_names=42)  # type: ignore[arg-type]


def test_marker_binding_empty_marker_name_raises() -> None:
    with pytest.raises(ValueError, match="marker_names"):
        MarkerBinding(kind=BindingKind.BETWEEN_TWO, marker_names=("a", ""))


def test_marker_binding_rest_dimensions_not_tuple_raises() -> None:
    with pytest.raises(TypeError, match="rest_dimensions"):
        MarkerBinding(
            kind=BindingKind.BETWEEN_TWO,
            marker_names=("a", "b"),
            rest_dimensions=[1.0],  # type: ignore[arg-type]
        )


def test_marker_binding_rest_dimensions_non_numeric_raises() -> None:
    with pytest.raises(TypeError, match="rest_dimensions"):
        MarkerBinding(
            kind=BindingKind.BETWEEN_TWO,
            marker_names=("a", "b"),
            rest_dimensions=("oops",),  # type: ignore[arg-type]
        )


def test_marker_binding_quaternion_non_numeric_raises() -> None:
    with pytest.raises(TypeError, match="rest_orientation_quat"):
        MarkerBinding(
            kind=BindingKind.BETWEEN_TWO,
            marker_names=("a", "b"),
            rest_orientation_quat=("a", "b", "c", "d"),  # type: ignore[arg-type]
        )


# ---------- ShapeTheme ---------------------------------------------------


def test_shape_theme_defaults() -> None:
    theme = ShapeTheme()
    assert 0.0 <= theme.opacity <= 1.0
    assert theme.edge_width >= 0.0


def test_shape_theme_rejects_opacity_above_one() -> None:
    with pytest.raises(ValueError, match="opacity"):
        ShapeTheme(opacity=1.5)


def test_shape_theme_rejects_opacity_below_zero() -> None:
    with pytest.raises(ValueError, match="opacity"):
        ShapeTheme(opacity=-0.1)


def test_shape_theme_rejects_negative_edge_width() -> None:
    with pytest.raises(ValueError, match="edge_width"):
        ShapeTheme(edge_width=-0.5)


def test_shape_theme_rejects_empty_color() -> None:
    with pytest.raises(ValueError, match="color"):
        ShapeTheme(color="")


def test_shape_theme_rejects_invalid_color() -> None:
    with pytest.raises(ValueError, match="matplotlib colour"):
        ShapeTheme(color="not-a-real-colour-name")


def test_shape_theme_rejects_invalid_edge_color() -> None:
    with pytest.raises(ValueError, match="edge_color"):
        ShapeTheme(edge_color="definitely-not-a-colour")


def test_shape_theme_rejects_empty_edge_color() -> None:
    with pytest.raises(ValueError, match="edge_color"):
        ShapeTheme(edge_color="")


def test_shape_theme_rejects_non_finite_opacity() -> None:
    with pytest.raises(ValueError, match="opacity"):
        ShapeTheme(opacity=float("nan"))


def test_shape_theme_rejects_non_numeric_opacity() -> None:
    with pytest.raises(TypeError, match="opacity"):
        ShapeTheme(opacity="0.5")  # type: ignore[arg-type]


def test_shape_theme_rejects_non_numeric_edge_width() -> None:
    with pytest.raises(TypeError, match="edge_width"):
        ShapeTheme(edge_width="0.5")  # type: ignore[arg-type]


def test_shape_theme_rejects_non_finite_edge_width() -> None:
    with pytest.raises(ValueError, match="edge_width"):
        ShapeTheme(edge_width=float("inf"))


def test_shape_theme_rejects_non_bool_flat_shaded() -> None:
    with pytest.raises(TypeError, match="flat_shaded"):
        ShapeTheme(flat_shaded="yes")  # type: ignore[arg-type]


def test_shape_theme_rejects_empty_group() -> None:
    with pytest.raises(ValueError, match="group"):
        ShapeTheme(group="")


# ---------- Protocol runtime checks --------------------------------------


class _StubShape:
    shape_id = "stub"
    rest_dimensions: tuple[float, ...] = (1.0,)

    def vertices_at_rest(self) -> np.ndarray:
        return np.zeros((0, 3))

    def faces(self) -> np.ndarray:
        return np.zeros((0, 3), dtype=np.int64)

    def transform(self, fitted: FittedShape) -> np.ndarray:
        return np.zeros((0, 3))


class _StubFitter:
    def fit(
        self,
        shape: BodyPartShape,
        binding: MarkerBinding,
        markers_xyz: dict[str, np.ndarray],
    ) -> FittedShape:
        n = 1
        return FittedShape(
            shape_id=shape.shape_id,
            binding=binding,
            centroid=np.zeros((n, 3)),
            rotation_matrix=np.broadcast_to(np.eye(3), (n, 3, 3)).copy(),
            scale=np.ones((n, 3)),
            valid_mask=np.ones((n,), dtype=bool),
        )


class _StubRenderer:
    def add_shape(
        self,
        shape: BodyPartShape,
        fitted: FittedShape,
        theme: ShapeTheme,
    ) -> str:
        return "h0"

    def update_frame(self, handle: str, frame_idx: int) -> None:
        return None

    def set_visible(self, handle: str, visible: bool) -> None:
        return None

    def remove(self, handle: str) -> None:
        return None


def test_stub_shape_satisfies_protocol() -> None:
    assert isinstance(_StubShape(), BodyPartShape)


def test_stub_fitter_satisfies_protocol() -> None:
    assert isinstance(_StubFitter(), ShapeFitter)


def test_stub_renderer_satisfies_protocol() -> None:
    assert isinstance(_StubRenderer(), ShapeRenderer)


# ---------- FittedShape --------------------------------------------------


def _ok_binding() -> MarkerBinding:
    return MarkerBinding(kind=BindingKind.BETWEEN_TWO, marker_names=("a", "b"))


def _ok_fitted(n: int = 4) -> FittedShape:
    return FittedShape(
        shape_id="stub",
        binding=_ok_binding(),
        centroid=np.zeros((n, 3)),
        rotation_matrix=np.broadcast_to(np.eye(3), (n, 3, 3)).copy(),
        scale=np.ones((n, 3)),
        valid_mask=np.ones((n,), dtype=bool),
    )


def test_fitted_shape_happy_path() -> None:
    fitted = _ok_fitted()
    assert fitted.centroid.shape == (4, 3)


def test_fitted_shape_mismatched_rotation_matrix_raises() -> None:
    binding = _ok_binding()
    with pytest.raises(ValueError, match="rotation_matrix"):
        FittedShape(
            shape_id="stub",
            binding=binding,
            centroid=np.zeros((4, 3)),
            rotation_matrix=np.broadcast_to(np.eye(3), (5, 3, 3)).copy(),
            scale=np.ones((4, 3)),
            valid_mask=np.ones((4,), dtype=bool),
        )


def test_fitted_shape_valid_mask_dtype_must_be_bool() -> None:
    binding = _ok_binding()
    with pytest.raises(TypeError, match="dtype"):
        FittedShape(
            shape_id="stub",
            binding=binding,
            centroid=np.zeros((3, 3)),
            rotation_matrix=np.broadcast_to(np.eye(3), (3, 3, 3)).copy(),
            scale=np.ones((3, 3)),
            valid_mask=np.ones((3,), dtype=np.int64),
        )


def test_fitted_shape_empty_shape_id_raises() -> None:
    binding = _ok_binding()
    with pytest.raises(ValueError, match="shape_id"):
        FittedShape(
            shape_id="",
            binding=binding,
            centroid=np.zeros((1, 3)),
            rotation_matrix=np.broadcast_to(np.eye(3), (1, 3, 3)).copy(),
            scale=np.ones((1, 3)),
            valid_mask=np.ones((1,), dtype=bool),
        )


def test_fitted_shape_binding_wrong_type_raises() -> None:
    with pytest.raises(TypeError, match="binding"):
        FittedShape(
            shape_id="stub",
            binding="not-a-binding",  # type: ignore[arg-type]
            centroid=np.zeros((1, 3)),
            rotation_matrix=np.broadcast_to(np.eye(3), (1, 3, 3)).copy(),
            scale=np.ones((1, 3)),
            valid_mask=np.ones((1,), dtype=bool),
        )


def test_fitted_shape_centroid_wrong_shape_raises() -> None:
    binding = _ok_binding()
    with pytest.raises(ValueError, match="centroid"):
        FittedShape(
            shape_id="stub",
            binding=binding,
            centroid=np.zeros((4, 2)),
            rotation_matrix=np.broadcast_to(np.eye(3), (4, 3, 3)).copy(),
            scale=np.ones((4, 3)),
            valid_mask=np.ones((4,), dtype=bool),
        )


def test_fitted_shape_centroid_not_ndarray_raises() -> None:
    binding = _ok_binding()
    with pytest.raises(TypeError, match="centroid"):
        FittedShape(
            shape_id="stub",
            binding=binding,
            centroid=[[0.0, 0.0, 0.0]],  # type: ignore[arg-type]
            rotation_matrix=np.broadcast_to(np.eye(3), (1, 3, 3)).copy(),
            scale=np.ones((1, 3)),
            valid_mask=np.ones((1,), dtype=bool),
        )


def test_fitted_shape_scale_wrong_shape_raises() -> None:
    binding = _ok_binding()
    with pytest.raises(ValueError, match="scale"):
        FittedShape(
            shape_id="stub",
            binding=binding,
            centroid=np.zeros((4, 3)),
            rotation_matrix=np.broadcast_to(np.eye(3), (4, 3, 3)).copy(),
            scale=np.ones((4, 2)),
            valid_mask=np.ones((4,), dtype=bool),
        )


def test_fitted_shape_valid_mask_wrong_shape_raises() -> None:
    binding = _ok_binding()
    with pytest.raises(ValueError, match="valid_mask"):
        FittedShape(
            shape_id="stub",
            binding=binding,
            centroid=np.zeros((4, 3)),
            rotation_matrix=np.broadcast_to(np.eye(3), (4, 3, 3)).copy(),
            scale=np.ones((4, 3)),
            valid_mask=np.ones((5,), dtype=bool),
        )


def test_fitted_shape_non_positive_scale_on_valid_frame_raises() -> None:
    binding = _ok_binding()
    scale = np.ones((2, 3))
    scale[0, 0] = 0.0
    with pytest.raises(ValueError, match="positive"):
        FittedShape(
            shape_id="stub",
            binding=binding,
            centroid=np.zeros((2, 3)),
            rotation_matrix=np.broadcast_to(np.eye(3), (2, 3, 3)).copy(),
            scale=scale,
            valid_mask=np.ones((2,), dtype=bool),
        )


def test_fitted_shape_nan_scale_on_valid_frame_raises() -> None:
    binding = _ok_binding()
    scale = np.ones((2, 3))
    scale[1, 2] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        FittedShape(
            shape_id="stub",
            binding=binding,
            centroid=np.zeros((2, 3)),
            rotation_matrix=np.broadcast_to(np.eye(3), (2, 3, 3)).copy(),
            scale=scale,
            valid_mask=np.ones((2,), dtype=bool),
        )


def test_fitted_shape_zero_frames_allowed() -> None:
    binding = _ok_binding()
    fitted = FittedShape(
        shape_id="stub",
        binding=binding,
        centroid=np.zeros((0, 3)),
        rotation_matrix=np.zeros((0, 3, 3)),
        scale=np.zeros((0, 3)),
        valid_mask=np.zeros((0,), dtype=bool),
    )
    assert fitted.centroid.shape == (0, 3)


def test_fitted_shape_invalid_frames_skip_scale_check() -> None:
    binding = _ok_binding()
    scale = np.zeros((2, 3))  # zeros would fail if validated
    fitted = FittedShape(
        shape_id="stub",
        binding=binding,
        centroid=np.zeros((2, 3)),
        rotation_matrix=np.broadcast_to(np.eye(3), (2, 3, 3)).copy(),
        scale=scale,
        valid_mask=np.zeros((2,), dtype=bool),
    )
    assert not bool(fitted.valid_mask.any())
