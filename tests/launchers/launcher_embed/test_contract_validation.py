"""Unit tests for :class:`EmbedCapabilities` invariants and the protocol."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src.shared.python.launcher_embed import EmbedCapabilities, EmbeddableTool

pytestmark = [pytest.mark.unit]


class TestEmbedCapabilitiesDefaults:
    def test_defaults(self) -> None:
        caps = EmbedCapabilities()
        assert caps.supports_embedded is True
        assert caps.prefers_dock is False
        assert caps.min_size == (640, 480)
        assert caps.requires_separate_qapplication is False

    def test_custom_values(self) -> None:
        caps = EmbedCapabilities(
            supports_embedded=False,
            prefers_dock=True,
            min_size=(100, 200),
            requires_separate_qapplication=True,
        )
        assert caps.supports_embedded is False
        assert caps.prefers_dock is True
        assert caps.min_size == (100, 200)
        assert caps.requires_separate_qapplication is True

    def test_frozen(self) -> None:
        caps = EmbedCapabilities()
        with pytest.raises(FrozenInstanceError):
            caps.supports_embedded = False  # type: ignore[misc]


class TestEmbedCapabilitiesValidation:
    def test_min_size_not_tuple_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a tuple"):
            EmbedCapabilities(min_size=[640, 480])  # type: ignore[arg-type]

    def test_min_size_not_tuple_with_dict(self) -> None:
        with pytest.raises(ValueError, match="must be a tuple"):
            EmbedCapabilities(min_size="640x480")  # type: ignore[arg-type]

    def test_min_size_wrong_length_raises(self) -> None:
        with pytest.raises(ValueError, match="2-tuple"):
            EmbedCapabilities(min_size=(640,))  # type: ignore[arg-type]

    def test_min_size_three_elements_raises(self) -> None:
        with pytest.raises(ValueError, match="2-tuple"):
            EmbedCapabilities(min_size=(640, 480, 1))  # type: ignore[arg-type]

    def test_min_size_non_int_raises(self) -> None:
        with pytest.raises(ValueError, match="must contain ints"):
            EmbedCapabilities(min_size=(640.0, 480))  # type: ignore[arg-type]

    def test_min_size_bool_rejected(self) -> None:
        with pytest.raises(ValueError, match="must contain ints"):
            EmbedCapabilities(min_size=(True, True))  # type: ignore[arg-type]

    def test_min_size_string_element_rejected(self) -> None:
        with pytest.raises(ValueError, match="must contain ints"):
            EmbedCapabilities(min_size=("640", 480))  # type: ignore[arg-type]

    def test_min_size_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            EmbedCapabilities(min_size=(0, 480))

    def test_min_size_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            EmbedCapabilities(min_size=(640, -1))


class _ConformingTool:
    tool_id = "conforming"

    def embed_capabilities(self) -> EmbedCapabilities:
        return EmbedCapabilities()

    def create_main_widget(self, parent):  # noqa: ANN001
        return object()

    def cleanup(self) -> None:
        pass

    def is_dirty(self) -> bool:
        return False


class _NonConforming:
    tool_id = "nope"


class TestEmbeddableToolProtocol:
    def test_conforming_instance_passes_isinstance(self) -> None:
        assert isinstance(_ConformingTool(), EmbeddableTool)

    def test_nonconforming_fails(self) -> None:
        assert not isinstance(_NonConforming(), EmbeddableTool)

    def test_protocol_methods_return_none_when_called(self) -> None:
        # Calling the protocol method bodies directly (ellipsis) -- covers
        # the ``...`` lines on the Protocol class for coverage purposes.
        # Protocols are not directly instantiable; call via a subclass.
        class P(EmbeddableTool):  # type: ignore[misc]
            tool_id = "p"

        # Calling the inherited protocol stubs returns None (ellipsis body).
        p = P.__new__(P)
        assert EmbeddableTool.embed_capabilities(p) is None
        assert EmbeddableTool.create_main_widget(p, None) is None
        assert EmbeddableTool.cleanup(p) is None
        assert EmbeddableTool.is_dirty(p) is None
